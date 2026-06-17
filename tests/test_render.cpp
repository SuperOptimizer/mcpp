#include "mcpp/geom/surface.hpp"
#include "mcpp/geom/vec.hpp"
#include "mcpp/render/compositor.hpp"
#include "mcpp/render/renderer.hpp"
#include "mcpp/sampling/sampler.hpp"
#include "mcpp/sampling/volume_view.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace mcpp;
using namespace mcpp::geom;
using namespace mcpp::render;
using mcpp::sampling::DenseView;
using mcpp::sampling::Sampler;

// ---- geometry math --------------------------------------------------------

MCPP_TEST("geom: vec ops, dot, cross, normalize") {
    Vec3f a{1, 0, 0}, b{0, 1, 0};
    MCPP_CHECK(dot(a, b) == 0.0f);
    MCPP_CHECK((a + b) == (Vec3f{1, 1, 0}));
    Vec3f c = cross(a, b);                 // z-axis cross y-axis
    MCPP_CHECK(length(c) > 0.99f && length(c) < 1.01f);  // unit-ish
    Vec3f n = normalize(Vec3f{0, 3, 4});
    MCPP_CHECK(std::fabs(length(n) - 1.0f) < 1e-5f);
}

MCPP_TEST("geom: PlaneSurface from_normal_and_up has orthonormal basis") {
    auto p = PlaneSurface::from_normal_and_up({10,10,10}, {1,0,0}, {0,0,1}, 8, 8);
    Vec3f bu = p.basis_u(), bv = p.basis_v(), n = p.normal(0,0);
    MCPP_CHECK(std::fabs(length(bu) - 1.0f) < 1e-5f);
    MCPP_CHECK(std::fabs(length(bv) - 1.0f) < 1e-5f);
    MCPP_CHECK(std::fabs(dot(bu, bv)) < 1e-5f);  // orthogonal
    MCPP_CHECK(std::fabs(dot(bu, n))  < 1e-5f);
    MCPP_CHECK(std::fabs(dot(bv, n))  < 1e-5f);
    // coord(0,0) == origin
    MCPP_CHECK(p.coord(0,0) == (Vec3f{10,10,10}));
}

MCPP_TEST("geom: QuadSurface bilinear interpolation + extent") {
    // 2x2 grid forming a tilted quad
    std::vector<Vec3f> pts{ {0,0,0}, {0,0,10}, {0,10,0}, {0,10,10} };
    QuadSurface q(pts, 2, 2);
    auto e = q.extent();
    MCPP_CHECK(e[0] == 2 && e[1] == 2);
    // midpoint (0.5,0.5) bilinearly = average of 4 corners = (0,5,5)
    Vec3f mid = q.coord(0.5f, 0.5f);
    MCPP_CHECK(std::fabs(mid.y - 5.0f) < 1e-4f && std::fabs(mid.x - 5.0f) < 1e-4f);
}

// ---- compositors ----------------------------------------------------------

MCPP_TEST("render: compositor policies reduce correctly") {
    MaxComposite mx; mx.reset(); for (float v : {3.f,9.f,1.f,7.f}) mx.add(v);
    MCPP_CHECK(mx.value() == 9.0f);
    MinComposite mn; mn.reset(); for (float v : {3.f,9.f,1.f,7.f}) mn.add(v);
    MCPP_CHECK(mn.value() == 1.0f);
    MeanComposite me; me.reset(); for (float v : {2.f,4.f,6.f}) me.add(v);
    MCPP_CHECK(std::fabs(me.value() - 4.0f) < 1e-5f);
}

// ---- end-to-end rendering -------------------------------------------------

// volume: a bright slab at z in [20,24], value 200; else 0.
static std::vector<std::uint8_t> slab_volume(std::uint32_t n) {
    std::vector<std::uint8_t> v(std::size_t(n)*n*n, 0);
    for (std::uint32_t z = 20; z <= 24 && z < n; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x)
                v[(std::size_t(z)*n + y)*n + x] = 200;
    return v;
}

MCPP_TEST("render: plane through a slab, MAX composite hits the bright slab") {
    const std::uint32_t n = 64;
    auto vol = slab_volume(n);
    DenseView<std::uint8_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint8_t>> smp(view);

    // a plane at z=22 (inside the slab), spanning y/x via basis, normal +z.
    // coord(u,v) = origin + u*bu + v*bv; put origin at (22,10,10), bu=+x, bv=+y.
    PlaneSurface p({22,10,10}, /*bu*/{0,0,1}, /*bv*/{0,1,0}, /*n*/{1,0,0}, 16, 16);
    RenderParams rp{-1.0f, 1.0f, 1.0f};  // march z in [21,23], all inside slab

    std::vector<float> img;
    mcpp::render::render<PlaneSurface, DenseView<std::uint8_t>, MaxComposite>(p, smp, rp, img);
    MCPP_CHECK(img.size() == 16u*16u);
    // every pixel marched through the bright slab -> max == 200
    for (float px : img) MCPP_CHECK(std::fabs(px - 200.0f) < 1.0f);
}

MCPP_TEST("render: plane OUTSIDE the slab composites to ~0") {
    const std::uint32_t n = 64;
    auto vol = slab_volume(n);
    DenseView<std::uint8_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint8_t>> smp(view);
    PlaneSurface p({40,10,10}, {0,0,1}, {0,1,0}, {1,0,0}, 16, 16);  // z=40, far from slab
    RenderParams rp{-1.0f, 1.0f, 1.0f};
    std::vector<float> img;
    mcpp::render::render<PlaneSurface, DenseView<std::uint8_t>, MaxComposite>(p, smp, rp, img);
    for (float px : img) MCPP_CHECK(px == 0.0f);
}

MCPP_TEST("render: band partition matches whole-image render (BSP disjoint)") {
    const std::uint32_t n = 64;
    auto vol = slab_volume(n);
    DenseView<std::uint8_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint8_t>> smp(view);
    PlaneSurface p({22,5,5}, {0,0,1}, {0,1,0}, {1,0,0}, 20, 20);
    RenderParams rp{-1.0f, 1.0f, 1.0f};

    std::vector<float> whole;
    mcpp::render::render<PlaneSurface, DenseView<std::uint8_t>, MeanComposite>(p, smp, rp, whole);

    // render in two bands into a fresh image; must match the whole render.
    std::vector<float> banded(20u*20u, -1.0f);
    render_band<PlaneSurface, DenseView<std::uint8_t>, MeanComposite>(p, smp, rp, 0, 10, banded.data());
    render_band<PlaneSurface, DenseView<std::uint8_t>, MeanComposite>(p, smp, rp, 10, 20, banded.data());
    for (std::size_t i = 0; i < whole.size(); ++i)
        MCPP_CHECK(std::fabs(whole[i] - banded[i]) < 1e-4f);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
