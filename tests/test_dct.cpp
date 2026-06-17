#include "mcpp/codec/dct.hpp"
#include "mcpp/codec/dct_tables.hpp"
#include "mcpp/test/test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

using namespace mcpp;
using namespace mcpp::codec;

// ---- consteval cos sanity (compile-time) ---------------------------------

MCPP_TEST("dct: constexpr cos matches known values") {
    static_assert(detail::const_cos(0.0) > 0.999999);
    // cos(pi) ~ -1, cos(pi/2) ~ 0
    MCPP_CHECK(std::fabs(detail::const_cos(detail::kPi) + 1.0) < 1e-9);
    MCPP_CHECK(std::fabs(detail::const_cos(detail::kPi / 2.0)) < 1e-9);
    MCPP_CHECK(std::fabs(detail::const_sqrt(2.0) - 1.4142135623730951) < 1e-12);
}

// ---- matrix orthonormality: C * C^T == I ---------------------------------

template <std::size_t N>
static void check_orthonormal(const std::array<float, N * N>& C, double tol) {
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            double dot = 0.0;
            for (std::size_t n = 0; n < N; ++n)
                dot += double(C[i * N + n]) * double(C[j * N + n]);
            double expect = (i == j) ? 1.0 : 0.0;
            MCPP_CHECK(std::fabs(dot - expect) < tol);
        }
}

MCPP_TEST("dct: 16-point matrix is orthonormal") {
    check_orthonormal<16>(kDct16, 1e-5);
}

MCPP_TEST("dct: 64-point matrix is orthonormal") {
    check_orthonormal<64>(kDct64, 1e-4);
}

// ---- 3D 16^3 roundtrip ---------------------------------------------------

static std::vector<float> make_block_3d(unsigned seed) {
    std::vector<float> b(16 * 16 * 16);
    // simple deterministic pseudo-random fill in [0,255]
    unsigned s = seed ? seed : 1u;
    for (auto& v : b) {
        s = s * 1664525u + 1013904223u;
        v = float((s >> 8) & 0xFF);
    }
    return b;
}

MCPP_TEST("dct: 3D 16^3 forward->inverse roundtrips within f32 tolerance") {
    auto orig = make_block_3d(12345);
    auto work = orig;
    forward_dct3_16(work.data());
    inverse_dct3_16(work.data());
    std::span<const float> a{orig}, b{work};
    // Orthonormal transform pair is the identity up to f32 rounding only.
    double mx = mcpp::test::max_abs_error<float>(a, b);
    MCPP_CHECK_LE(mx, 2.0e-2);   // ~f32 accumulation over 3 length-16 passes
    double p = mcpp::test::psnr<float>(a, b, 255.0);
    MCPP_CHECK_GE(p, 80.0);      // effectively lossless
}

MCPP_TEST("dct: 3D constant block -> energy only in DC") {
    std::vector<float> b(16 * 16 * 16, 50.0f);  // flat block
    forward_dct3_16(b.data());
    // DC coefficient is index 0; for value V over N^3 with orthonormal DCT,
    // DC = V * (N^3)^(1/2-ish)... just assert all AC are ~0 and DC carries energy.
    MCPP_CHECK(std::fabs(b[0]) > 1.0f);  // DC nonzero
    double ac_energy = 0.0;
    for (std::size_t i = 1; i < b.size(); ++i) ac_energy += double(b[i]) * b[i];
    MCPP_CHECK_LE(ac_energy, 1e-2);      // all AC negligible
}

MCPP_TEST("dct: 3D Parseval — transform preserves energy") {
    auto orig = make_block_3d(777);
    double e_spatial = 0.0;
    for (float v : orig) e_spatial += double(v) * v;
    auto work = orig;
    forward_dct3_16(work.data());
    double e_freq = 0.0;
    for (float v : work) e_freq += double(v) * v;
    // orthonormal transform: ||X||^2 == ||x||^2
    MCPP_CHECK_LE(std::fabs(e_spatial - e_freq) / e_spatial, 1e-4);
}

// ---- 2D 64^2 roundtrip ---------------------------------------------------

static std::vector<float> make_block_2d(unsigned seed) {
    std::vector<float> b(64 * 64);
    unsigned s = seed ? seed : 1u;
    for (auto& v : b) {
        s = s * 1664525u + 1013904223u;
        v = float((s >> 8) & 0xFF);
    }
    return b;
}

MCPP_TEST("dct: 2D 64^2 forward->inverse roundtrips within f32 tolerance") {
    auto orig = make_block_2d(54321);
    auto work = orig;
    forward_dct2_64(work.data());
    inverse_dct2_64(work.data());
    std::span<const float> a{orig}, b{work};
    double mx = mcpp::test::max_abs_error<float>(a, b);
    MCPP_CHECK_LE(mx, 5.0e-2);   // larger N => more accumulation, still tiny
    double p = mcpp::test::psnr<float>(a, b, 255.0);
    MCPP_CHECK_GE(p, 78.0);
}

MCPP_TEST("dct: 2D smooth ramp compacts energy into low frequencies") {
    std::vector<float> b(64 * 64);
    for (std::size_t v = 0; v < 64; ++v)
        for (std::size_t u = 0; u < 64; ++u)
            b[v * 64 + u] = float(u + v);   // smooth gradient
    forward_dct2_64(b.data());
    // Energy in the top-left 8x8 low-freq corner vs total.
    double low = 0.0, total = 0.0;
    for (std::size_t k = 0; k < 64; ++k)
        for (std::size_t l = 0; l < 64; ++l) {
            double e = double(b[k * 64 + l]) * b[k * 64 + l];
            total += e;
            if (k < 8 && l < 8) low += e;
        }
    MCPP_CHECK_GE(low / total, 0.99);  // smooth signal -> nearly all energy low
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
