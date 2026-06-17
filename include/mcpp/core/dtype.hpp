// mcpp dtype vocabulary + strong types.
//
// Rationale (design memory mcpp-scope): the codec is generic over a CLOSED set
// of 8 sample types (NO 64-bit for now — u64/s64/f64 deferred, a non-breaking
// later addition). Compute is always f32 (compute_t<T> == float for every T);
// f16 is the only possible future narrow path, via this seam. Dtype is resolved
// to the right monomorphization ONCE at the open()/header boundary via a variant
// visitor — the runtime tag below is what that visitor switches on.
#ifndef MCPP_CORE_DTYPE_HPP
#define MCPP_CORE_DTYPE_HPP

#include <compare>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace mcpp {

// f16: C++26 std::float16_t lives in <stdfloat> but is unevenly available on
// libstdc++ today. _Float16 is supported by both Clang 21 and GCC 15 as an
// extension and is what we use until <stdfloat> is reliable on both.
using float16 = _Float16;

// ---- runtime dtype tag (the open()-time dispatch switch) ------------------

enum class Dtype : std::uint8_t {
    u8, u16, u32,
    s8, s16, s32,
    f16, f32,
    // NOTE: u64/s64/f64 intentionally absent (deferred). Adding them later is a
    // non-breaking extension: more visitor arms + f64 consteval tables.
};

inline constexpr int dtype_count = 8;

// Byte size of a sample of the given dtype.
constexpr std::size_t dtype_size(Dtype d) noexcept {
    switch (d) {
        case Dtype::u8:  case Dtype::s8:                return 1;
        case Dtype::u16: case Dtype::s16: case Dtype::f16: return 2;
        case Dtype::u32: case Dtype::s32: case Dtype::f32: return 4;
    }
    return 0;  // unreachable for valid tags
}

constexpr bool dtype_is_float(Dtype d) noexcept {
    return d == Dtype::f16 || d == Dtype::f32;
}

constexpr bool dtype_is_signed(Dtype d) noexcept {
    return d == Dtype::s8 || d == Dtype::s16 || d == Dtype::s32 ||
           dtype_is_float(d);
}

// ---- compile-time mapping C++ type <-> Dtype tag -------------------------

template <class T> struct dtype_of;
template <> struct dtype_of<std::uint8_t>  { static constexpr Dtype value = Dtype::u8;  };
template <> struct dtype_of<std::uint16_t> { static constexpr Dtype value = Dtype::u16; };
template <> struct dtype_of<std::uint32_t> { static constexpr Dtype value = Dtype::u32; };
template <> struct dtype_of<std::int8_t>   { static constexpr Dtype value = Dtype::s8;  };
template <> struct dtype_of<std::int16_t>  { static constexpr Dtype value = Dtype::s16; };
template <> struct dtype_of<std::int32_t>  { static constexpr Dtype value = Dtype::s32; };
template <> struct dtype_of<float16>           { static constexpr Dtype value = Dtype::f16; };
template <> struct dtype_of<float>         { static constexpr Dtype value = Dtype::f32; };

template <class T>
inline constexpr Dtype dtype_of_v = dtype_of<T>::value;

// The concept that constrains every codec instantiation to the supported set.
template <class T>
concept Sample = requires { dtype_of<T>::value; };

// compute_t: ALL supported types compute in f32 (project decision: f32
// everywhere, no integer compute). f16 is the only future narrow path and
// would be introduced here, gated on native-f16 hardware.
template <Sample T>
using compute_t = float;

// SIMD lane count for the (f32) compute type, derived at compile time.
// Conservative default; real targets refine this per ISA in the SIMD seam.
template <Sample T>
consteval int lanes_for() { return 8; }  // AVX2 f32 width; NEON/AVX-512 override later

// ---- strong types --------------------------------------------------------
// Thin wrappers so LOD/coords cannot be silently transposed. Trivially copyable,
// register-resident; comparisons defaulted.

struct Lod {
    std::uint8_t v = 0;
    friend constexpr bool operator==(Lod, Lod) = default;
    friend constexpr auto operator<=>(Lod, Lod) = default;
};

inline constexpr int kMaxLods = 8;        // LOD0..LOD7
inline constexpr std::uint32_t kMaxDim = 1u << 20;  // hard per-axis cap (2^20)

}  // namespace mcpp

#endif  // MCPP_CORE_DTYPE_HPP
