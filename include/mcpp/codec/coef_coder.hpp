// Coefficient model: entropy-codes quantized DCT indices in scan order.
//
// Rationale (design memory mcpp-original-codec-facts): code coefficients in
// ascending-band scan order; per position emit significance (zero/nonzero) under
// band-dependent contexts; once only zeros remain, emit EOB. Nonzero coeffs get
// a sign (bypass) and a magnitude (adaptive unary prefix + Exp-Golomb-style
// remainder). PURE INTEGER over the range coder => exact roundtrip.
//
// The model state (BitContexts) must be IDENTICAL on encode and decode. We keep
// it in one struct so both directions instantiate the same contexts; the codec
// later seeds these from trained priors (consteval), but defaults (p=0.5) work.
#ifndef MCPP_CODEC_COEF_CODER_HPP
#define MCPP_CODEC_COEF_CODER_HPP

#include "mcpp/codec/quant.hpp"
#include "mcpp/codec/range_coder.hpp"
#include "mcpp/codec/scan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace mcpp::codec {

// Number of band buckets for context selection (DC + a few frequency bands).
inline constexpr int kBandBuckets = 8;
inline constexpr int kMagContexts = 16;
// total adaptive contexts in a CoefModel: sig + eob + mag_cont
inline constexpr int kNumPriors = kBandBuckets + kBandBuckets + kMagContexts;  // 32

// Per-context initial P(bit==0), 11-bit fixed point. TRAINED priors replace the
// flat p=0.5 (=1024) default: a well-calibrated start means the per-block coder
// (contexts reset every block — self-contained invariant) doesn't waste bits
// adapting from scratch. Layout: [0,8)=sig, [8,16)=eob, [16,32)=mag_cont.
struct Priors {
    std::array<std::uint16_t, kNumPriors> p0{};
};

// Flat (untrained) priors: p=0.5 everywhere. Kept for reference/testing.
consteval Priors flat_priors() {
    Priors pr{};
    for (auto& v : pr.p0) v = std::uint16_t(kProbInit);  // 1024
    return pr;
}

// TRAINED priors — FROZEN, baked once from representative PHercParis4 scroll data
// (bench/gen_priors, 32768 blocks pooled across q={1,2,3,4,6,9,12,20}). These
// seed every block's contexts (blocks reset per-block — self-contained), so the
// adaptive coder starts calibrated instead of from p=0.5. ONE-TIME/build-time;
// the codec does NO runtime calibration. Layout: [0,8)=sig [8,16)=eob [16,32)=mag.
// Regenerate with bench/gen_priors if the data distribution changes.
inline constexpr Priors kTrainedPriors{{{
     164,  752, 1333, 1625, 1807, 1933, 1990, 1733,
       1,    1,    3,    4,   13,   80, 1347, 2047,
    1042,  713,  866,  972, 1041, 1086, 1104, 1069,
     959,  873,  852,  995, 1316, 1891, 2047, 1024,
}}};

// The codec default IS the trained table.
consteval Priors default_priors() { return kTrainedPriors; }

// Adaptive context model shared by encoder and decoder. Reset per block (seeded
// from priors) — blocks are self-contained (no cross-block entropy state).
struct CoefModel {
    std::array<BitContext, kBandBuckets> sig{};
    std::array<BitContext, kBandBuckets> eob{};
    std::array<BitContext, kMagContexts> mag_cont{};

    CoefModel() { seed(default_priors()); }
    explicit CoefModel(const Priors& pr) { seed(pr); }

    void seed(const Priors& pr) {
        for (int i = 0; i < kBandBuckets; ++i) sig[std::size_t(i)].p0 = pr.p0[std::size_t(i)];
        for (int i = 0; i < kBandBuckets; ++i) eob[std::size_t(i)].p0 = pr.p0[std::size_t(kBandBuckets+i)];
        for (int i = 0; i < kMagContexts; ++i) mag_cont[std::size_t(i)].p0 = pr.p0[std::size_t(2*kBandBuckets+i)];
    }

    static int bucket_for_band(int band, int max_band) {
        if (max_band <= 0) return 0;
        int b = band * kBandBuckets / (max_band + 1);
        return (b < kBandBuckets) ? b : kBandBuckets - 1;
    }
};

namespace detail {

// Encode magnitude m >= 1 as: unary prefix of "number of extra bits" using
// adaptive mag_cont contexts, then that many bypass remainder bits.
// (Exp-Golomb-like: value v=m-1, k = bit_width(v), prefix codes k, remainder = v.)
inline void encode_magnitude(RangeEncoder& enc, CoefModel& mdl, std::uint32_t m) {
    std::uint32_t v = m - 1;            // m>=1 -> v>=0
    int k = 0;
    while ((v >> k) != 0) ++k;          // number of significant bits of v (0 if v==0)
    // unary code k with adaptive continuation
    int i = 0;
    for (; i < k; ++i) enc.encode_bit(mdl.mag_cont[std::size_t(i < 16 ? i : 15)], 1);
    enc.encode_bit(mdl.mag_cont[std::size_t(i < 16 ? i : 15)], 0);  // terminator
    // remainder: k low bits of v (bypass)
    for (int b = k - 1; b >= 0; --b) enc.encode_bypass(int((v >> b) & 1u));
}

inline std::uint32_t decode_magnitude(RangeDecoder& dec, CoefModel& mdl) {
    int k = 0;
    // Cap the unary prefix: a real magnitude fits in 31 bits. On adversarial
    // input the range coder can emit a runaway of 1-bits past end-of-buffer;
    // capping keeps decode bounded (hardened-decoder invariant).
    while (k < 31 && dec.decode_bit(mdl.mag_cont[std::size_t(k < 16 ? k : 15)]) == 1) ++k;
    std::uint32_t v = 0;
    for (int b = 0; b < k; ++b) v = (v << 1) | std::uint32_t(dec.decode_bypass());
    return v + 1;  // m = v+1
}

}  // namespace detail

// Encode an N^Rank block of quantized indices into the range coder.
template <std::size_t N, std::size_t Rank>
void encode_coeffs(RangeEncoder& enc, CoefModel& mdl, const std::int32_t* idx) {
    const auto& scan = scan_for<N, Rank>();
    constexpr std::size_t total = block_total<N, Rank>();
    const int maxb = max_band<N, Rank>();

    // last nonzero scan position (so we can EOB after it).
    std::size_t last_nz = 0;
    bool any = false;
    for (std::size_t s = 0; s < total; ++s)
        if (idx[scan[s]] != 0) { last_nz = s; any = true; }

    for (std::size_t s = 0; s < total; ++s) {
        const std::uint16_t pos = scan[s];
        const int band = coeff_band<N, Rank>(pos);
        const int bk = CoefModel::bucket_for_band(band, maxb);
        const std::int32_t c = idx[pos];

        // significance bit (1 = nonzero)
        enc.encode_bit(mdl.sig[std::size_t(bk)], c != 0 ? 1 : 0);
        if (c != 0) {
            enc.encode_bypass(c < 0 ? 1 : 0);                    // sign
            detail::encode_magnitude(enc, mdl, std::uint32_t(c < 0 ? -c : c));
            // EOB after each nonzero: 1 = more nonzeros follow, 0 = done.
            const bool more = any && (s < last_nz);
            enc.encode_bit(mdl.eob[std::size_t(bk)], more ? 1 : 0);
            if (!more) return;  // remaining are all zero
        }
    }
}

// Decode an N^Rank block of quantized indices from the range coder.
template <std::size_t N, std::size_t Rank>
void decode_coeffs(RangeDecoder& dec, CoefModel& mdl, std::int32_t* idx) {
    const auto& scan = scan_for<N, Rank>();
    constexpr std::size_t total = block_total<N, Rank>();
    const int maxb = max_band<N, Rank>();

    for (std::size_t i = 0; i < total; ++i) idx[i] = 0;

    for (std::size_t s = 0; s < total; ++s) {
        const std::uint16_t pos = scan[s];
        const int band = coeff_band<N, Rank>(pos);
        const int bk = CoefModel::bucket_for_band(band, maxb);

        const int sig = dec.decode_bit(mdl.sig[std::size_t(bk)]);
        if (sig) {
            const int neg = dec.decode_bypass();
            const std::uint32_t m = detail::decode_magnitude(dec, mdl);
            idx[pos] = neg ? -std::int32_t(m) : std::int32_t(m);
            const int more = dec.decode_bit(mdl.eob[std::size_t(bk)]);
            if (!more) return;  // rest are zero
        }
    }
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_COEF_CODER_HPP
