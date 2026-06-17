#include "mcpp/sampling/sampler.hpp"
#include "mcpp/sampling/surface_sample.hpp"
#include "mcpp/sampling/volume_view.hpp"
#include "mcpp/test/test.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace mcpp;
using namespace mcpp::sampling;

// A dense volume with a known linear field f(z,y,x) = z*100 + y*10 + x (clamped
// to a value, but we keep it small so it fits u16 exactly for exactness checks).
static std::vector<std::uint16_t> linear_volume(std::uint32_t n) {
    std::vector<std::uint16_t> v(std::size_t(n)*n*n);
    for (std::uint32_t z = 0; z < n; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x)
                v[(std::size_t(z)*n + y)*n + x] = std::uint16_t(z*100 + y*10 + x);
    return v;
}

MCPP_TEST("sampler: nearest returns exact voxel values") {
    const std::uint32_t n = 40;
    auto vol = linear_volume(n);
    DenseView<std::uint16_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint16_t>> smp(view);

    MCPP_CHECK(smp.sample(0, 0, 0, Filter::nearest) == 0.0f);
    MCPP_CHECK(smp.sample(3, 2, 1, Filter::nearest) == float(3*100 + 2*10 + 1));
    MCPP_CHECK(smp.sample(20, 19, 18, Filter::nearest) == float(20*100 + 19*10 + 18));
}

MCPP_TEST("sampler: trilinear of a LINEAR field is exact at fractional coords") {
    const std::uint32_t n = 40;
    auto vol = linear_volume(n);
    DenseView<std::uint16_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint16_t>> smp(view);

    // f is linear so trilinear interpolation is exact: f(z,y,x)=z*100+y*10+x.
    auto expect = [](float z, float y, float x){ return z*100 + y*10 + x; };
    for (auto [z,y,x] : std::array<std::array<float,3>,4>{{
            {{5.5f, 6.0f, 7.0f}}, {{10.25f, 3.75f, 2.5f}},
            {{1.5f, 1.5f, 1.5f}}, {{20.0f, 20.0f, 20.9f}} }}) {
        float got = smp.sample(z, y, x, Filter::trilinear);
        MCPP_CHECK(std::fabs(got - expect(z,y,x)) < 0.05f);  // exact up to f32
    }
}

MCPP_TEST("sampler: out-of-bounds and block edges read as zero") {
    const std::uint32_t n = 32;
    auto vol = linear_volume(n);
    DenseView<std::uint16_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint16_t>> smp(view);
    MCPP_CHECK(smp.sample(-1, 0, 0, Filter::nearest) == 0.0f);
    MCPP_CHECK(smp.sample(0, 0, float(n), Filter::nearest) == 0.0f);  // x == n OOB
    // sampling across the 16-block boundary (x=15.5) must be continuous/correct
    float across = smp.sample(5, 5, 15.5f, Filter::trilinear);
    float expect = 5*100 + 5*10 + 15.5f;
    MCPP_CHECK(std::fabs(across - expect) < 0.05f);
}

MCPP_TEST("sampler: CachedView records misses, no IO") {
    const std::uint32_t n = 256;
    cache::Cache<std::uint16_t, 64> cache;  // empty
    std::vector<BlockKey> misses;
    CachedView<std::uint16_t, 64> view(cache, {n,n,n}, Lod{0}, &misses);
    Sampler<CachedView<std::uint16_t,64>> smp(view);

    // sampling an empty cache -> 0, and the touched block is recorded as a miss
    float v = smp.sample(40, 40, 40, Filter::nearest);  // block (2,2,2)
    MCPP_CHECK(v == 0.0f);
    MCPP_CHECK(misses.size() == 1);
    MCPP_CHECK(misses[0] == BlockKey::make(Lod{0}, 2, 2, 2));
}

MCPP_TEST("sampler: last-block memo avoids re-resolving within a block") {
    const std::uint32_t n = 256;
    cache::Cache<std::uint16_t, 64> cache;
    std::vector<BlockKey> misses;
    CachedView<std::uint16_t, 64> view(cache, {n,n,n}, Lod{0}, &misses);
    Sampler<CachedView<std::uint16_t,64>> smp(view);

    // many samples within the SAME 16^3 block -> memo means only ONE miss recorded
    for (int i = 0; i < 50; ++i) smp.sample(float(i % 16), 5, 5, Filter::nearest);
    MCPP_CHECK(misses.size() == 1);  // all in block (0,0,0); resolved once
}

MCPP_TEST("surface: sample_quad_volume marches the normal, layer-major output") {
    const std::uint32_t n = 64;
    auto vol = linear_volume(n);  // f = z*100+y*10+x
    DenseView<std::uint16_t> view(vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1});
    Sampler<DenseView<std::uint16_t>> smp(view);

    // a 2x2 flat surface at z=20, normal +z; march layers 0,1,2 (t=0,1,2).
    std::array<std::array<float,3>,4> coords{{
        {{20,10,10}}, {{20,10,11}}, {{20,11,10}}, {{20,11,11}} }};
    std::array<std::array<float,3>,4> normals{{
        {{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{1,0,0}} }};
    QuadSurface q{coords.data(), normals.data(), 2, 2};

    std::vector<float> out;
    sample_quad_volume(smp, q, /*t0*/0.0f, /*dt*/1.0f, /*nlayers*/3, out);
    MCPP_CHECK(out.size() == 3u * 2 * 2);

    // layer L at grid (v=0,u=0) is coords (20+L,10,10) -> (20+L)*100+10*10+10
    auto at = [&](std::uint32_t L, std::uint32_t v, std::uint32_t u){
        return out[(std::size_t(L)*2 + v)*2 + u]; };
    MCPP_CHECK(std::fabs(at(0,0,0) - float(20*100 + 100 + 10)) < 0.1f);
    MCPP_CHECK(std::fabs(at(1,0,0) - float(21*100 + 100 + 10)) < 0.1f);
    MCPP_CHECK(std::fabs(at(2,1,1) - float(22*100 + 110 + 11)) < 0.1f);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
