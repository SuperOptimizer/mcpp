// dtype <-> f32 conversion at the codec boundary.
//
// Rationale (design memory mcpp-scope): f32 compute EVERYWHERE. Every input dtype
// is converted to f32 on ingest; on decode the f32 reconstruction is rounded to
// nearest and clamped (saturated) to the output dtype range. dtype affects ONLY
// these edges — the entire transform/quant/entropy interior is dtype-agnostic f32.
//
// Float inputs are SCRUBBED through mcpp::fpbits::sanitize at ingest (the scrub
// discipline) so the fast-math hot path never sees NaN/Inf.
#ifndef MCPP_CODEC_CONVERT_HPP
#define MCPP_CODEC_CONVERT_HPP

#include "mcpp/core/dtype.hpp"
#include "mcpp/core/fpbits.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace mcpp::codec {

// Load one sample of type T as f32 (scrubbing non-finite floats at the boundary).
template <Sample T>
constexpr float load_f32(T v) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return fpbits::sanitize(v);
    } else if constexpr (std::is_same_v<T, float16>) {
        return fpbits::sanitize(float(v));
    } else {
        return float(v);  // integer types convert exactly into f32's range concern
    }
}

// Store one f32 reconstruction back to type T: round-to-nearest + saturate.
template <Sample T>
constexpr T store_from_f32(float r) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return r;
    } else if constexpr (std::is_same_v<T, float16>) {
        return float16(r);
    } else {
        // integer: round to nearest, then clamp to [min,max] of T.
        constexpr float lo = float(std::numeric_limits<T>::min());
        constexpr float hi = float(std::numeric_limits<T>::max());
        float rr = std::nearbyint(r);
        rr = std::clamp(rr, lo, hi);
        return T(rr);
    }
}

// Bulk load/store helpers over a block.
template <Sample T>
void load_block(const T* src, float* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = load_f32<T>(src[i]);
}

template <Sample T>
void store_block(const float* src, T* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = store_from_f32<T>(src[i]);
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_CONVERT_HPP
