#include "mcpp/geom/vec.hpp"
#include "mcpp/mcs/mcs.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace mcpp;
using namespace mcpp::mcs;
using geom::Vec3f;

// A smooth parametric surface: (u,v) -> a slowly varying {z,y,x} sheet.
static SurfacePoints make_surface(std::uint32_t gw, std::uint32_t gh, float base) {
    SurfacePoints pts(std::size_t(gw) * gh);
    for (std::uint32_t v = 0; v < gh; ++v)
        for (std::uint32_t u = 0; u < gw; ++u)
            pts[std::size_t(v)*gw + u] = Vec3f{
                base + float(u) * 0.5f + float(v) * 0.25f,   // z
                100.0f + float(v) * 0.5f,                    // y
                200.0f + float(u) * 0.5f };                  // x
    return pts;
}

MCPP_TEST("mcs: build/read TOC — count, dims preserved") {
    McsBuilder b;
    b.add(SurfaceDesc{100, 80, 0.1f}, make_surface(100, 80, 10.0f));
    b.add(SurfaceDesc{40, 200, 0.1f}, make_surface(40, 200, 50.0f));
    auto img = b.finish();

    McsReader r(img.data(), img.size());
    MCPP_CHECK(r.valid());
    MCPP_CHECK(r.count() == 2);
    auto e0 = r.entry(0); MCPP_CHECK(e0.gw == 100 && e0.gh == 80);
    auto e1 = r.entry(1); MCPP_CHECK(e1.gw == 40 && e1.gh == 200);
}

MCPP_TEST("mcs: surface coords reconstruct within sub-voxel tolerance") {
    const std::uint32_t gw = 130, gh = 70;  // spans multiple 64^2 tiles
    auto orig = make_surface(gw, gh, 5.0f);
    McsBuilder b;
    b.add(SurfaceDesc{gw, gh, 0.05f}, orig);  // tight q for subvoxel coords
    auto img = b.finish();

    McsReader r(img.data(), img.size());
    SurfacePoints got = r.decode(0);
    MCPP_CHECK(got.size() == orig.size());

    double maxerr = 0.0;
    for (std::size_t i = 0; i < orig.size(); ++i) {
        maxerr = std::max(maxerr, double(std::fabs(orig[i].z - got[i].z)));
        maxerr = std::max(maxerr, double(std::fabs(orig[i].y - got[i].y)));
        maxerr = std::max(maxerr, double(std::fabs(orig[i].x - got[i].x)));
    }
    // smooth coordinate surface at tight q -> sub-voxel accuracy
    MCPP_CHECK_LE(maxerr, 1.0);
}

MCPP_TEST("mcs: two surfaces decode independently and correctly") {
    auto s0 = make_surface(70, 70, 0.0f);
    auto s1 = make_surface(70, 70, 1000.0f);  // very different z range
    McsBuilder b;
    b.add(SurfaceDesc{70, 70, 0.05f}, s0);
    b.add(SurfaceDesc{70, 70, 0.05f}, s1);
    auto img = b.finish();

    McsReader r(img.data(), img.size());
    auto g0 = r.decode(0), g1 = r.decode(1);
    // surface 0's z near 0..; surface 1's z near 1000.. — not swapped/mixed
    MCPP_CHECK(std::fabs(g0[0].z - s0[0].z) < 1.0f);
    MCPP_CHECK(std::fabs(g1[0].z - s1[0].z) < 1.0f);
    MCPP_CHECK(g1[0].z > 900.0f);
}

MCPP_TEST("mcs: compresses below raw f32 grid size") {
    const std::uint32_t gw = 128, gh = 128;
    auto s = make_surface(gw, gh, 0.0f);
    McsBuilder b;
    b.add(SurfaceDesc{gw, gh, 0.1f}, s);
    auto img = b.finish();
    const std::size_t raw = std::size_t(gw) * gh * 3 * sizeof(float);
    MCPP_CHECK(img.size() < raw);  // smooth surface compresses
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
