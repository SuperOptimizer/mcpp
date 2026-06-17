// Component (cross-channel) transforms.
//
// Rationale (design memory mcpp-scope): for C-component data the codec optionally
// decorrelates across channels before the per-plane DCT. A ComponentTransform is
// a pluggable, concept-constrained policy. Only TWO ship at launch:
//   - Identity : C=1 grayscale/CT volume; C=3 uncorrelated xyz coordinate surface
//                (independent per-plane DCT). The default.
//   - YCoCg    : C=3 correlated RGB (the sole correlated case). Decorrelates so
//                the per-plane DCT compresses better.
// The policy stays pluggable for extensibility (e.g. a future normal-map
// transform) but no speculative transforms are added.
//
// Transforms operate IN f32 on a set of C parallel planes (one value per channel
// per pixel/voxel). forward() is applied before per-plane DCT; inverse() after
// per-plane inverse DCT.
#ifndef MCPP_CODEC_COMPONENT_TRANSFORM_HPP
#define MCPP_CODEC_COMPONENT_TRANSFORM_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <span>

namespace mcpp::codec {

// A ComponentTransform maps C f32 channel-values <-> C decorrelated f32 values,
// in place, for one pixel/voxel. Channel count is a compile-time property.
template <class P, std::size_t C>
concept ComponentTransform = requires(std::array<float, C>& px) {
    { P::template forward<C>(px) } -> std::same_as<void>;
    { P::template inverse<C>(px) } -> std::same_as<void>;
};

// Identity: no cross-channel mixing. Valid for any C.
struct Identity {
    template <std::size_t C>
    static void forward(std::array<float, C>&) noexcept {}
    template <std::size_t C>
    static void inverse(std::array<float, C>&) noexcept {}
};

// YCoCg: RGB <-> YCoCg-R style decorrelation (C must be 3). f32, lossy-by-design.
//   Co = R - B
//   t  = B + Co/2
//   Cg = G - t
//   Y  = t + Cg/2
struct YCoCg {
    template <std::size_t C>
        requires (C == 3)
    static void forward(std::array<float, C>& px) noexcept {
        const float r = px[0], g = px[1], b = px[2];
        const float co = r - b;
        const float t  = b + co * 0.5f;
        const float cg = g - t;
        const float y  = t + cg * 0.5f;
        px[0] = y; px[1] = co; px[2] = cg;
    }
    template <std::size_t C>
        requires (C == 3)
    static void inverse(std::array<float, C>& px) noexcept {
        const float y = px[0], co = px[1], cg = px[2];
        const float t = y - cg * 0.5f;
        const float g = cg + t;
        const float b = t - co * 0.5f;
        const float r = b + co;
        px[0] = r; px[1] = g; px[2] = b;
    }
};

static_assert(ComponentTransform<Identity, 1>);
static_assert(ComponentTransform<Identity, 3>);
static_assert(ComponentTransform<YCoCg, 3>);

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_COMPONENT_TRANSFORM_HPP
