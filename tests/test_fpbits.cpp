#include "mcpp/core/fpbits.hpp"
#include "mcpp/test/test.hpp"

#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace mcpp;

// These must hold even under -ffast-math, which is exactly the point.
// IMPORTANT: under Clang -ffinite-math-only, a NaN materialized into a *float*
// can be assumed-finite by the optimizer. The realistic ingest scenario is data
// arriving in a memory buffer; make_f32 routes the bits through a volatile load
// + memcpy so the optimizer cannot constant-reason the result is finite.
static float make_f32(std::uint32_t bits) {
    volatile std::uint32_t src = bits;
    std::uint32_t b = src;
    float f;
    std::memcpy(&f, &b, sizeof f);
    return f;
}

MCPP_TEST("fpbits: finite values classified correctly") {
    MCPP_CHECK(fpbits::is_finite(0.0f));
    MCPP_CHECK(fpbits::is_finite(-1.5f));
    MCPP_CHECK(fpbits::is_finite(3.14159f));
    MCPP_CHECK(!fpbits::is_nan(1.0f));
    MCPP_CHECK(!fpbits::is_inf(1.0f));
    MCPP_CHECK(fpbits::classify(0.0f) == fpbits::fpclass::zero);
    MCPP_CHECK(fpbits::classify(1.0f) == fpbits::fpclass::normal);
}

MCPP_TEST("fpbits: canonical bits-API detects NaN/Inf") {
    // The canonical, always-reliable primitives operate on integer bits.
    MCPP_CHECK(fpbits::is_nan_bits(0x7FC0'0000u));
    MCPP_CHECK(fpbits::is_nan_bits(0x7F80'0001u));
    MCPP_CHECK(fpbits::is_inf_bits(0x7F80'0000u));
    MCPP_CHECK(!fpbits::is_finite_bits(0x7F80'0000u));
    MCPP_CHECK(fpbits::classify_bits(0x7FC0'0000u) == fpbits::fpclass::nan);
}

MCPP_TEST("fpbits: detects NaN via float overload (survives fast-math)") {
    const float qnan = make_f32(0x7FC0'0000u);  // quiet NaN from memory
    const float snan = make_f32(0x7F80'0001u);  // signaling NaN from memory
    MCPP_CHECK(fpbits::is_nan(qnan));
    MCPP_CHECK(fpbits::is_nan(snan));
    MCPP_CHECK(!fpbits::is_finite(qnan));
    MCPP_CHECK(!fpbits::is_inf(qnan));
    MCPP_CHECK(fpbits::classify(qnan) == fpbits::fpclass::nan);
}

MCPP_TEST("fpbits: detects Inf via bits (survives fast-math)") {
    const float pinf = make_f32(0x7F80'0000u);
    const float ninf = make_f32(0xFF80'0000u);
    MCPP_CHECK(fpbits::is_inf(pinf));
    MCPP_CHECK(fpbits::is_inf(ninf));
    MCPP_CHECK(!fpbits::is_nan(pinf));
    MCPP_CHECK(!fpbits::is_finite(pinf));
    MCPP_CHECK(fpbits::classify(ninf) == fpbits::fpclass::inf);
}

MCPP_TEST("fpbits: subnormal classified") {
    const float sub = make_f32(0x0000'0001u);  // smallest positive subnormal
    MCPP_CHECK(fpbits::classify(sub) == fpbits::fpclass::subnormal);
    MCPP_CHECK(fpbits::is_finite(sub));
}

MCPP_TEST("fpbits: sanitize scrubs non-finite to fallback") {
    const float qnan = make_f32(0x7FC0'0000u);
    const float pinf = make_f32(0x7F80'0000u);
    MCPP_CHECK(fpbits::sanitize(qnan) == 0.0f);
    MCPP_CHECK(fpbits::sanitize(pinf, -1.0f) == -1.0f);
    MCPP_CHECK(fpbits::sanitize(2.5f) == 2.5f);  // finite passes through
}

MCPP_TEST("fpbits: constexpr-evaluable at compile time") {
    static_assert(fpbits::is_finite(1.0f));
    static_assert(fpbits::is_inf(std::numeric_limits<float>::infinity()));
    static_assert(fpbits::sanitize(0.0f) == 0.0f);
    static_assert(fpbits::classify(0.0f) == fpbits::fpclass::zero);
    MCPP_CHECK(true);
}

MCPP_TEST("fpbits: f64 NaN/Inf via bits") {
    const double dnan = std::bit_cast<double>(std::uint64_t{0x7FF8'0000'0000'0000ull});
    const double dinf = std::bit_cast<double>(std::uint64_t{0x7FF0'0000'0000'0000ull});
    MCPP_CHECK(fpbits::is_nan(dnan));
    MCPP_CHECK(fpbits::is_inf(dinf));
    MCPP_CHECK(fpbits::is_finite(2.0));
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
