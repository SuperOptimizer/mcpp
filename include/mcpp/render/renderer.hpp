// Renderer — read-phase BSP frame consumer over a Surface + VolumeView.
//
// Rationale (design memory mcpp-rendering-geometry): render a Surface by, for each
// output pixel (mapped to grid coords u,v), marching the surface normal over
// [t0,t1] step dt, trilinear-sampling the volume via Sampler<View>, and reducing
// with a Compositor policy. Pure read-phase consumer: reads what the View has,
// misses recorded by the View (render-now-refine-later). Band-partition over
// output rows is the BSP parallelization (each band writes disjoint pixels and
// accumulates its own misses) — exposed as render_band so a coordinator can fan
// out; render() does the whole image.
#ifndef MCPP_RENDER_RENDERER_HPP
#define MCPP_RENDER_RENDERER_HPP

#include "mcpp/geom/surface.hpp"
#include "mcpp/render/compositor.hpp"
#include "mcpp/sampling/sampler.hpp"

#include <cstdint>
#include <vector>

namespace mcpp::render {

struct RenderParams {
    float t0 = 0.0f, t1 = 0.0f, dt = 1.0f;  // normal-march range
};

// Render rows [row0, row1) of the surface grid into `img` (row-major, full grid
// width). img must be sized gw*gh. Compositor is per-pixel (reset each pixel).
template <geom::Surface Surf, mcpp::sampling::VolumeView View, Compositor Comp>
void render_band(const Surf& surf, mcpp::sampling::Sampler<View>& smp,
                 const RenderParams& rp, std::uint32_t row0, std::uint32_t row1,
                 float* img) {
    const auto ext = surf.extent();
    const std::uint32_t gw = ext[0];
    Comp comp;
    for (std::uint32_t v = row0; v < row1; ++v)
        for (std::uint32_t u = 0; u < gw; ++u) {
            const geom::Vec3f c = surf.coord(float(u), float(v));
            const geom::Vec3f n = surf.normal(float(u), float(v));
            comp.reset();
            for (float t = rp.t0; t <= rp.t1; t += rp.dt) {
                const geom::Vec3f p = c + n * t;
                comp.add(smp.sample(p.z, p.y, p.x, mcpp::sampling::Filter::trilinear));
            }
            img[std::size_t(v) * gw + u] = comp.value();
        }
}

// Render the whole surface grid.
template <geom::Surface Surf, mcpp::sampling::VolumeView View, Compositor Comp>
void render(const Surf& surf, mcpp::sampling::Sampler<View>& smp,
            const RenderParams& rp, std::vector<float>& img) {
    const auto ext = surf.extent();
    img.assign(std::size_t(ext[0]) * ext[1], 0.0f);
    render_band<Surf, View, Comp>(surf, smp, rp, 0, ext[1], img.data());
}

}  // namespace mcpp::render

#endif  // MCPP_RENDER_RENDERER_HPP
