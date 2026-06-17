// mcpp::fpbits — IEEE-754 inspection that SURVIVES -ffast-math.
//
// Rationale (design memory mcpp-scope): -ffast-math (specifically
// -ffinite-math-only) lets the compiler ASSUME no NaN/Inf, so std::isnan /
// std::isinf / (x != x) fold to constant false and become useless.
//
// CRITICAL LESSON (found by test_fpbits under clang-release):
//   A naive `bit_cast<u32>(x)` on a FLOAT-typed argument is NOT enough under
//   Clang -ffinite-math-only. Clang propagates "this float is finite" into the
//   bit_cast and folds the exponent check to false BEFORE the cast — so a real
//   NaN loaded from memory is reported finite. (GCC happens not to, but that is
//   luck, not a guarantee.)
//
// THE RULES:
//   1. The CANONICAL primitives operate on raw integer bits (`*_bits`). Prefer
//      these: keep untrusted float data as bytes and inspect the bits directly.
//   2. The float-typed overloads route the value through an opaque compiler
//      barrier (`detail::bits_of`) so the optimizer cannot assume finiteness
//      before the cast. Verified correct under Clang AND GCC -ffast-math.
//   3. During constant evaluation there is no fast-math (consteval is strict
//      IEEE), so the barrier is skipped via `if consteval`.
//
// Discipline: SCRUB float input at the ingest boundary once (sanitize), then
// TRUST the fast-math hot path.
#ifndef MCPP_CORE_FPBITS_HPP
#define MCPP_CORE_FPBITS_HPP

#include <bit>
#include <cstdint>

namespace mcpp::fpbits {

enum class fpclass { zero, subnormal, normal, inf, nan };

namespace detail {
// Runtime-only opaque barriers. Inline asm disqualifies a function from constant
// evaluation entirely (even guarded by `if consteval`), so these are kept
// separate and only called from the runtime branch below.
[[gnu::always_inline]] inline std::uint32_t barrier32(std::uint32_t b) noexcept {
    asm volatile("" : "+r"(b));  // optimizer can no longer assume b's value
    return b;
}
[[gnu::always_inline]] inline std::uint64_t barrier64(std::uint64_t b) noexcept {
    asm volatile("" : "+r"(b));
    return b;
}

// Returns the IEEE bits of `x`, preventing the optimizer from carrying a
// finiteness assumption across the conversion at runtime. At compile time
// (consteval) there is no fast-math, so a plain bit_cast is both correct and
// constexpr-valid.
constexpr std::uint32_t bits_of(float x) noexcept {
    if consteval {
        return std::bit_cast<std::uint32_t>(x);
    } else {
        return barrier32(std::bit_cast<std::uint32_t>(x));
    }
}
constexpr std::uint64_t bits_of(double x) noexcept {
    if consteval {
        return std::bit_cast<std::uint64_t>(x);
    } else {
        return barrier64(std::bit_cast<std::uint64_t>(x));
    }
}
}  // namespace detail

// ===========================================================================
// f32 — canonical bit-based primitives
// ===========================================================================

inline constexpr std::uint32_t kF32ExpMask  = 0x7F80'0000u;
inline constexpr std::uint32_t kF32MantMask = 0x007F'FFFFu;
inline constexpr std::uint32_t kF32SignMask = 0x8000'0000u;

constexpr bool is_nan_bits(std::uint32_t b) noexcept {
    return (b & kF32ExpMask) == kF32ExpMask && (b & kF32MantMask) != 0u;
}
constexpr bool is_inf_bits(std::uint32_t b) noexcept {
    return (b & kF32ExpMask) == kF32ExpMask && (b & kF32MantMask) == 0u;
}
constexpr bool is_finite_bits(std::uint32_t b) noexcept {
    return (b & kF32ExpMask) != kF32ExpMask;
}
constexpr fpclass classify_bits(std::uint32_t b) noexcept {
    const auto exp  = b & kF32ExpMask;
    const auto mant = b & kF32MantMask;
    if (exp == kF32ExpMask) return mant ? fpclass::nan : fpclass::inf;
    if (exp == 0u)          return mant ? fpclass::subnormal : fpclass::zero;
    return fpclass::normal;
}

// f32 — float-typed convenience (barrier-protected against fast-math folding)
constexpr bool is_nan(float x) noexcept    { return is_nan_bits(detail::bits_of(x)); }
constexpr bool is_inf(float x) noexcept    { return is_inf_bits(detail::bits_of(x)); }
constexpr bool is_finite(float x) noexcept { return is_finite_bits(detail::bits_of(x)); }
constexpr fpclass classify(float x) noexcept { return classify_bits(detail::bits_of(x)); }

// Ingest-boundary scrub: map any non-finite value to a finite fallback.
constexpr float sanitize(float x, float fallback = 0.0f) noexcept {
    return is_finite(x) ? x : fallback;
}

// ===========================================================================
// f64 — canonical bit-based primitives
// ===========================================================================

inline constexpr std::uint64_t kF64ExpMask  = 0x7FF0'0000'0000'0000ull;
inline constexpr std::uint64_t kF64MantMask = 0x000F'FFFF'FFFF'FFFFull;

constexpr bool is_nan_bits(std::uint64_t b) noexcept {
    return (b & kF64ExpMask) == kF64ExpMask && (b & kF64MantMask) != 0u;
}
constexpr bool is_inf_bits(std::uint64_t b) noexcept {
    return (b & kF64ExpMask) == kF64ExpMask && (b & kF64MantMask) == 0u;
}
constexpr bool is_finite_bits(std::uint64_t b) noexcept {
    return (b & kF64ExpMask) != kF64ExpMask;
}

constexpr bool is_nan(double x) noexcept    { return is_nan_bits(detail::bits_of(x)); }
constexpr bool is_inf(double x) noexcept    { return is_inf_bits(detail::bits_of(x)); }
constexpr bool is_finite(double x) noexcept { return is_finite_bits(detail::bits_of(x)); }

constexpr double sanitize(double x, double fallback = 0.0) noexcept {
    return is_finite(x) ? x : fallback;
}

}  // namespace mcpp::fpbits

#endif  // MCPP_CORE_FPBITS_HPP
