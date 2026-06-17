#include "mcpp/core/dtype.hpp"
#include "mcpp/test/test.hpp"

#include <cstdint>
#include <type_traits>

using namespace mcpp;

MCPP_TEST("dtype: sizes correct") {
    MCPP_CHECK(dtype_size(Dtype::u8)  == 1);
    MCPP_CHECK(dtype_size(Dtype::s8)  == 1);
    MCPP_CHECK(dtype_size(Dtype::u16) == 2);
    MCPP_CHECK(dtype_size(Dtype::f16) == 2);
    MCPP_CHECK(dtype_size(Dtype::u32) == 4);
    MCPP_CHECK(dtype_size(Dtype::f32) == 4);
}

MCPP_TEST("dtype: signed/float predicates") {
    MCPP_CHECK(!dtype_is_signed(Dtype::u8));
    MCPP_CHECK(dtype_is_signed(Dtype::s16));
    MCPP_CHECK(dtype_is_signed(Dtype::f32));
    MCPP_CHECK(dtype_is_float(Dtype::f16));
    MCPP_CHECK(!dtype_is_float(Dtype::u32));
}

MCPP_TEST("dtype: round-trip type <-> tag") {
    static_assert(dtype_of_v<std::uint8_t>  == Dtype::u8);
    static_assert(dtype_of_v<std::int16_t>  == Dtype::s16);
    static_assert(dtype_of_v<float>         == Dtype::f32);
    static_assert(dtype_of_v<float16>       == Dtype::f16);
    MCPP_CHECK(true);
}

MCPP_TEST("dtype: compute_t is f32 for every supported type") {
    static_assert(std::is_same_v<compute_t<std::uint8_t>,  float>);
    static_assert(std::is_same_v<compute_t<std::uint32_t>, float>);
    static_assert(std::is_same_v<compute_t<float16>,           float>);
    static_assert(std::is_same_v<compute_t<float>,         float>);
    MCPP_CHECK(true);
}

MCPP_TEST("dtype: Sample concept accepts exactly the 8 types") {
    static_assert(Sample<std::uint8_t>);
    static_assert(Sample<std::int32_t>);
    static_assert(Sample<float16>);
    static_assert(Sample<float>);
    static_assert(!Sample<double>);        // 64-bit deferred
    static_assert(!Sample<std::uint64_t>); // 64-bit deferred
    static_assert(!Sample<bool>);
    MCPP_CHECK(true);
}

MCPP_TEST("dtype: strong Lod type and caps") {
    Lod a{3}, b{3}, c{5};
    MCPP_CHECK(a == b);
    MCPP_CHECK(a < c);
    MCPP_CHECK(kMaxLods == 8);
    MCPP_CHECK(kMaxDim == (1u << 20));
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
