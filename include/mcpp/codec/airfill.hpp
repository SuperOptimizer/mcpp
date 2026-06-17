// Mask-aware air fill (encode-side energy shaping).
//
// Rationale (design memory mcpp-original-codec-facts + mcpp-scope): "air" voxels
// (value 0) are not real signal; a sharp air/material boundary injects spurious
// high-frequency energy into the DCT, wasting bits. The encoder therefore FILLS
// air voxels with a smooth harmonic extension of the surrounding material BEFORE
// the transform (red-black SOR solving Laplace's equation on the air region with
// material as Dirichlet boundary). The fill is purely to shape coefficient
// energy — it is DISCARDED on decode, where air is forced back to exactly 0 via
// the mask (see mask.hpp). So the fill values never need to be exact or
// deterministic; only the mask matters for correctness.
//
// All f32; -ffast-math fine (the fill is lossy/heuristic by nature).
#ifndef MCPP_CODEC_AIRFILL_HPP
#define MCPP_CODEC_AIRFILL_HPP

#include "mcpp/codec/scan.hpp"  // block_total

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpp::codec {

inline constexpr float kSorOmega = 1.6f;   // over-relaxation factor (original)
inline constexpr int   kSorSweeps = 8;     // iteration budget

// Fill air voxels (mask[i]==0) of an N^Rank block in place with a harmonic
// extension of the material (mask[i]!=0) values, via red-black SOR. Material
// voxels are held fixed (Dirichlet boundary). Block layout is last-axis-fastest
// (matches the codec's flat block buffers).
//
// mask: 1 byte per voxel, nonzero = material (fixed), zero = air (to fill).
template <std::size_t N, std::size_t Rank>
void air_fill(float* block, const std::uint8_t* mask) {
    constexpr std::size_t total = block_total<N, Rank>();

    // Seed air voxels with the mean of material (a stable starting point).
    double msum = 0.0; std::size_t mcount = 0;
    for (std::size_t i = 0; i < total; ++i)
        if (mask[i]) { msum += block[i]; ++mcount; }
    const float seed = mcount ? float(msum / double(mcount)) : 0.0f;
    for (std::size_t i = 0; i < total; ++i)
        if (!mask[i]) block[i] = seed;
    if (mcount == 0 || mcount == total) return;  // all-air or all-material: nothing to relax

    // Precompute per-axis strides.
    std::array<std::size_t, Rank> stride{};
    {
        std::size_t s = 1;
        for (std::size_t a = Rank; a-- > 0;) { stride[a] = s; s *= N; }
    }

    auto coord_digit = [&](std::size_t lin, std::size_t a) {
        return (lin / stride[a]) % N;
    };

    // Red-black Gauss-Seidel SOR sweeps over air voxels only.
    for (int sweep = 0; sweep < kSorSweeps; ++sweep) {
        for (int color = 0; color < 2; ++color) {
            for (std::size_t i = 0; i < total; ++i) {
                if (mask[i]) continue;  // material is fixed
                // checkerboard color = parity of coordinate-sum
                std::size_t psum = 0;
                for (std::size_t a = 0; a < Rank; ++a) psum += coord_digit(i, a);
                if (int(psum & 1u) != color) continue;

                // average of in-bounds neighbors along each axis
                float acc = 0.0f; int cnt = 0;
                for (std::size_t a = 0; a < Rank; ++a) {
                    const std::size_t d = coord_digit(i, a);
                    if (d > 0)     { acc += block[i - stride[a]]; ++cnt; }
                    if (d < N - 1) { acc += block[i + stride[a]]; ++cnt; }
                }
                if (cnt) {
                    const float gs = acc / float(cnt);
                    block[i] += kSorOmega * (gs - block[i]);  // over-relaxation
                }
            }
        }
    }
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_AIRFILL_HPP
