// sample_quad_volume — sample a volume along a parametric surface + its normal.
//
// Rationale (design memory mcpp-sampling): THE parametric-surface sampler. Given
// a 2D quad surface (a grid of (u,v)->{x,y,z} world coords) and per-(u,v) normals,
// march along the normal over a stack of layers, trilinear-sampling the volume at
// each, writing LAYER-MAJOR output [layer][v][u]. Pure read-phase consumer (reads
// through a Sampler<View>; misses are recorded by the View). Both strict and
// best-LOD modes are a caller concern at the View level; here we sample what the
// View provides.
//
// This is the ink-detection input and feeds rendering. The surface grid is itself
// a decoded codec case-4 parametric surface — one data type, two roles.
#ifndef MCPP_SAMPLING_SURFACE_SAMPLE_HPP
#define MCPP_SAMPLING_SURFACE_SAMPLE_HPP

#include "mcpp/sampling/sampler.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpp::sampling {

// A quad surface: gw x gh grid of {z,y,x} world coords + matching {nz,ny,nx}
// normals (unit). Flat, row-major [v*gw + u].
struct QuadSurface {
    const std::array<float, 3>* coords;   // gw*gh
    const std::array<float, 3>* normals;  // gw*gh
    std::uint32_t gw, gh;
};

// Sample `nlayers` layers from t0 stepping by dt along the normal at each grid
// point. Output is layer-major: out[(layer*gh + v)*gw + u]. Trilinear.
template <VolumeView View>
void sample_quad_volume(Sampler<View>& smp, const QuadSurface& q,
                        float t0, float dt, std::uint32_t nlayers,
                        std::vector<float>& out) {
    out.assign(std::size_t(nlayers) * q.gh * q.gw, 0.0f);
    for (std::uint32_t L = 0; L < nlayers; ++L) {
        const float t = t0 + float(L) * dt;
        for (std::uint32_t v = 0; v < q.gh; ++v)
            for (std::uint32_t u = 0; u < q.gw; ++u) {
                const std::size_t gi = std::size_t(v) * q.gw + u;
                const auto& c = q.coords[gi];
                const auto& n = q.normals[gi];
                const float z = c[0] + t * n[0];
                const float y = c[1] + t * n[1];
                const float x = c[2] + t * n[2];
                out[(std::size_t(L) * q.gh + v) * q.gw + u] =
                    smp.sample(z, y, x, Filter::trilinear);
            }
    }
}

}  // namespace mcpp::sampling

#endif  // MCPP_SAMPLING_SURFACE_SAMPLE_HPP
