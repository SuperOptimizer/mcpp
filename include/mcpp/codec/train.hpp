// Prior training: estimate per-context P(bit==0) over a corpus of quantized
// blocks, so encode/decode can seed each block's contexts with calibrated priors
// instead of flat p=0.5. Entropy coding is lossless, so better priors improve
// ratio with ZERO quality change.
//
// We re-walk the same coding logic as coef_coder but, instead of arithmetic-
// coding each bit, tally (count0, count1) per context. The optimal initial prior
// is the corpus-wide empirical frequency p0 = 2048 * count0 / (count0+count1).
#ifndef MCPP_CODEC_TRAIN_HPP
#define MCPP_CODEC_TRAIN_HPP

#include "mcpp/codec/coef_coder.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/codec/scan.hpp"

#include <array>
#include <cstdint>

namespace mcpp::codec {

// Per-context bit tallies during training.
struct PriorTally {
    std::array<std::uint64_t, kNumPriors> c0{};
    std::array<std::uint64_t, kNumPriors> c1{};

    void tally(int ctx, int bit) { (bit ? c1 : c0)[std::size_t(ctx)]++; }

    // context index helpers (must match CoefModel layout)
    static int sig_ctx(int bk) { return bk; }
    static int eob_ctx(int bk) { return kBandBuckets + bk; }
    static int mag_ctx(int i)  { return 2*kBandBuckets + (i < kMagContexts ? i : kMagContexts-1); }
};

// Tally the contexts a block's quantized indices would exercise (mirrors
// encode_coeffs / encode_magnitude exactly, but counts instead of codes).
template <std::size_t N, std::size_t Rank>
void train_block(PriorTally& t, const std::int32_t* idx) {
    const auto& scan = scan_for<N, Rank>();
    constexpr std::size_t total = block_total<N, Rank>();
    const int maxb = max_band<N, Rank>();

    std::size_t last_nz = 0; bool any = false;
    for (std::size_t s = 0; s < total; ++s)
        if (idx[scan[s]] != 0) { last_nz = s; any = true; }

    for (std::size_t s = 0; s < total; ++s) {
        const std::uint16_t pos = scan[s];
        const int bk = CoefModel::bucket_for_band(coeff_band<N, Rank>(pos), maxb);
        const std::int32_t c = idx[pos];
        t.tally(PriorTally::sig_ctx(bk), c != 0 ? 1 : 0);
        if (c != 0) {
            // sign is bypass (not modeled) — skip
            // magnitude unary prefix
            std::uint32_t m = std::uint32_t(c < 0 ? -c : c);
            std::uint32_t v = m - 1;
            int k = 0; while ((v >> k) != 0) ++k;
            for (int i = 0; i < k; ++i) t.tally(PriorTally::mag_ctx(i), 1);
            t.tally(PriorTally::mag_ctx(k), 0);  // terminator
            // remainder bits are bypass — skip
            const bool more = any && (s < last_nz);
            t.tally(PriorTally::eob_ctx(bk), more ? 1 : 0);
            if (!more) break;
        }
    }
}

// Convert accumulated tallies into a Priors table. Contexts never seen keep the
// flat default. p0 clamped to [1, 2047] (range coder requires a valid prob).
inline Priors finalize_priors(const PriorTally& t) {
    Priors pr = default_priors();
    for (int i = 0; i < kNumPriors; ++i) {
        std::uint64_t n0 = t.c0[std::size_t(i)], n1 = t.c1[std::size_t(i)];
        std::uint64_t n = n0 + n1;
        if (n == 0) continue;  // unseen -> keep default 1024
        std::uint32_t p0 = std::uint32_t((n0 * kProbOne + n/2) / n);  // round
        if (p0 < 1) p0 = 1; if (p0 > kProbOne - 1) p0 = kProbOne - 1;
        pr.p0[std::size_t(i)] = std::uint16_t(p0);
    }
    return pr;
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_TRAIN_HPP
