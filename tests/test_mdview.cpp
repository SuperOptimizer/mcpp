#include "mcpp/core/mdview.hpp"
#include "mcpp/test/test.hpp"

#include <array>
#include <cstdint>
#include <vector>

using namespace mcpp;

MCPP_TEST("mdview: 3D zfast offset matches data[lz + ly*ez + lx*ez*ey]") {
    // extents [x, y, z] outermost->innermost so z is fastest (VC convention).
    // We label axes (ax=x, ay=y, az=z); zfast => z innermost (last axis fastest).
    const std::size_t ex = 4, ey = 3, ez = 2;  // axis0=x, axis1=y, axis2=z
    std::vector<int> buf(ex * ey * ez);
    volume_view<int> v(buf.data(), {ex, ey, ez});
    MCPP_CHECK(v.size() == ex * ey * ez);

    // Fill via operator() and verify linear layout: last axis (z) fastest.
    for (std::size_t x = 0; x < ex; ++x)
        for (std::size_t y = 0; y < ey; ++y)
            for (std::size_t z = 0; z < ez; ++z)
                v(x, y, z) = int(x * 100 + y * 10 + z);

    // Expected linear offset = z + y*ez + x*ez*ey  (z fastest).
    for (std::size_t x = 0; x < ex; ++x)
        for (std::size_t y = 0; y < ey; ++y)
            for (std::size_t z = 0; z < ez; ++z) {
                std::size_t off = z + y * ez + x * ez * ey;
                MCPP_CHECK(buf[off] == int(x * 100 + y * 10 + z));
            }
}

MCPP_TEST("mdview: 2D plane access") {
    const std::size_t ev = 5, eu = 7;  // axis0=v, axis1=u; u fastest
    std::vector<float> buf(ev * eu, 0.0f);
    plane_view<float> p(buf.data(), {ev, eu});
    p(2, 3) = 9.0f;
    MCPP_CHECK(buf[2 * eu + 3] == 9.0f);  // off = u + v*eu
    MCPP_CHECK(p(2, 3) == 9.0f);
}

MCPP_TEST("mdview: operator[] with index array") {
    std::vector<int> buf(2 * 2 * 2, 0);
    volume_view<int> v(buf.data(), {2, 2, 2});
    std::array<std::size_t, 3> idx{1, 1, 1};
    v[idx] = 42;
    MCPP_CHECK(buf[1 + 1 * 2 + 1 * 4] == 42);
}

MCPP_TEST("mdview: const-correct view over const data") {
    std::vector<int> buf{0, 1, 2, 3, 4, 5, 6, 7};
    mdview<const int, 3, zfast_layout> v(buf.data(), {2, 2, 2});
    MCPP_CHECK(v(0, 0, 1) == 1);   // z fastest
    MCPP_CHECK(v(1, 0, 0) == 4);   // x slowest: off = 0 + 0*2 + 1*4
}

MCPP_TEST("mdview: constexpr offset computation") {
    constexpr std::array<std::size_t, 3> ext{4, 3, 2};
    constexpr std::array<std::size_t, 3> idx{1, 2, 1};
    constexpr std::size_t off = zfast_layout::offset<3>(ext, idx);
    static_assert(off == 1 + 2 * 2 + 1 * 2 * 3);  // z + y*ez + x*ez*ey = 1+4+6 = 11
    MCPP_CHECK(off == 11);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
