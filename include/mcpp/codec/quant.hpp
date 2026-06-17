// Band-weighted dead-zone quantization of f32 DCT coefficients.
//
// Rationale (design memory mcpp-scope + mcpp-original-codec-facts): lossy step is
// a dead-zone quantizer with band-weighted steps. Coefficient c at band b:
//
//   index i  = quantize(c)   (integer, the entropy-coded symbol)
//   recon  c' = dequantize(i)
//
//   step(b)  = q_step * band_weight(b)       q_step = base * quality dial
//   dead-zone: the zero bin is widened by MC_DZ_FRAC of a step, so small
//              coefficients map to 0 (the dominant compression lever).
//   reconstruction: mid-bin offset (DZ_FRAC-anchored) so c' sits in the bin.
//
// Constants are the original's *starting-shape* tuning, carried as named knobs to
// be re-fit for the f32 pipeline (NOT frozen). All compute is f32; -ffast-math
// expected. Quant indices are integers, so the index stream itself is exact; the
// loss is entirely in the f32 step arithmetic, validated by tolerance.
//
// Band model: for a separable DCT block the "band" of a coefficient is a measure
// of its frequency. We use the sum of per-axis frequency indices (L1 distance
// from DC), normalized — cheap, separable, and matches the original's band_of.
#ifndef MCPP_CODEC_QUANT_HPP
#define MCPP_CODEC_QUANT_HPP

#include "mcpp/codec/dct_tables.hpp"  // detail::const_* not needed; for parity only

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace mcpp::codec {

// ---- tuning constants (starting-shape; re-fit later, not frozen) ----------
inline constexpr float kDeadZoneFrac = 0.80f;   // zero-bin widening (MC_DZ_FRAC)
inline constexpr float kReconOffset  = 0.40f;   // mid-bin reconstruction offset
inline constexpr float kHfExp        = 0.65f;   // band-weight exponent (MC_HF_EXP)
inline constexpr float kBaseStep     = 1.0f;    // base quant step at q=1

// Per-coefficient band weight: 1 at DC, growing with normalized frequency so
// higher-frequency coefficients get coarser steps. `norm_freq` in [0,1].
constexpr float band_weight(float norm_freq) {
    // 1 + (HF growth). At norm_freq=0 -> 1 (DC unweighted); rises sublinearly.
    return 1.0f + std::pow(norm_freq, kHfExp) * 3.0f;
}

// Compute the L1 band (sum of per-axis freq indices) for a coefficient at linear
// position `lin` in an N^Rank block, and the max possible band for normalization.
template <std::size_t N, std::size_t Rank>
constexpr int coeff_band(std::size_t lin) {
    int band = 0;
    for (std::size_t a = 0; a < Rank; ++a) {
        band += int(lin % N);   // last-axis-fastest digit
        lin /= N;
    }
    return band;
}

template <std::size_t N, std::size_t Rank>
constexpr int max_band() { return int(Rank * (N - 1)); }

// The quantization step for a coefficient at band `band`, given quality `q`.
// Larger q => larger step => more compression, lower quality.
template <std::size_t N, std::size_t Rank>
constexpr float step_for(int band, float q) {
    const float nf = float(band) / float(max_band<N, Rank>());
    return kBaseStep * q * band_weight(nf);
}

// ---- scalar dead-zone quantize / dequantize -------------------------------

// Quantize one coefficient to an integer index using a dead-zone quantizer.
// |c| below the dead-zone threshold maps to 0.
constexpr std::int32_t quantize(float c, float step) {
    if (step <= 0.0f) return 0;
    const float a = std::fabs(c);
    // dead-zone: subtract DZ_FRAC*step worth of width from the zero bin.
    const float t = a / step;
    if (t < kDeadZoneFrac) return 0;
    // bins beyond the dead zone
    std::int32_t mag = std::int32_t((t - kDeadZoneFrac) + 1.0f);
    return (c < 0.0f) ? -mag : mag;
}

// Reconstruct a coefficient from its integer index (mid-bin).
constexpr float dequantize(std::int32_t i, float step) {
    if (i == 0) return 0.0f;
    const std::int32_t mag = (i < 0) ? -i : i;
    // inverse of quantize: place at (mag-1 + DZ_FRAC + recon_offset) * step
    const float r = (float(mag - 1) + kDeadZoneFrac + kReconOffset) * step;
    return (i < 0) ? -r : r;
}

// ---- whole-block quant / dequant -----------------------------------------

// Quantize an N^Rank block of DCT coefficients in place... but we need both the
// integer indices (for entropy coding) and to keep the float buffer free. So the
// block versions write indices to a separate int buffer.

template <std::size_t N, std::size_t Rank>
void quantize_block(const float* coeffs, std::int32_t* indices, float q) {
    constexpr std::size_t total = [] {
        std::size_t t = 1; for (std::size_t a = 0; a < Rank; ++a) t *= N; return t;
    }();
    for (std::size_t i = 0; i < total; ++i) {
        const int band = coeff_band<N, Rank>(i);
        indices[i] = quantize(coeffs[i], step_for<N, Rank>(band, q));
    }
}

template <std::size_t N, std::size_t Rank>
void dequantize_block(const std::int32_t* indices, float* coeffs, float q) {
    constexpr std::size_t total = [] {
        std::size_t t = 1; for (std::size_t a = 0; a < Rank; ++a) t *= N; return t;
    }();
    for (std::size_t i = 0; i < total; ++i) {
        const int band = coeff_band<N, Rank>(i);
        coeffs[i] = dequantize(indices[i], step_for<N, Rank>(band, q));
    }
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_QUANT_HPP
