// consteval DCT basis tables.
//
// Rationale (design memory mcpp-scope): the separable DCT is one rank-generic
// kernel; the per-axis 1-D transform length is a compile-time property
// (block_side: 16 for 3D, 64 for 2D). The cosine basis is generated at compile
// time via consteval so there are no runtime init, no lazy tables, no mutexes.
//
// We use the orthonormal DCT-II (forward) / DCT-III (inverse) pair. With the
// orthonormal scaling, the inverse is exactly the transpose of the forward
// matrix, so a single NxN table serves both directions.
//
//   forward (DCT-II):  X[k] = sum_n  C[k][n] * x[n]
//   inverse (DCT-III): x[n] = sum_k  C[k][n] * X[k]        (= transpose)
//
//   C[k][n] = alpha(k) * sqrt(2/N) * cos( pi*(2n+1)*k / (2N) )
//   alpha(0) = 1/sqrt(2), alpha(k>0) = 1
//
// Compile-time cos: we provide our own constexpr cosine (std::cos is not
// constexpr on either toolchain). Accuracy only needs to be ~f32; we evaluate
// in double via range-reduction + Taylor and store float.
#ifndef MCPP_CODEC_DCT_TABLES_HPP
#define MCPP_CODEC_DCT_TABLES_HPP

#include <array>
#include <cstddef>

namespace mcpp::codec {

namespace detail {

// --- constexpr math (double precision, evaluated at compile time) ----------

inline constexpr double kPi = 3.14159265358979323846264338327950288;

// Reduce angle to [-pi, pi] for Taylor accuracy.
constexpr double reduce_angle(double x) {
    const double two_pi = 2.0 * kPi;
    // bring into [-pi, pi]
    while (x >  kPi) x -= two_pi;
    while (x < -kPi) x += two_pi;
    return x;
}

// constexpr cosine via Taylor series after range reduction. ~1e-15 on [-pi,pi].
constexpr double const_cos(double x) {
    x = reduce_angle(x);
    double term = 1.0;   // x^0 / 0!
    double sum  = 1.0;
    const double x2 = x * x;
    // 18 terms is far more than enough for double precision on [-pi,pi].
    for (int n = 1; n <= 18; ++n) {
        term *= -x2 / (double((2 * n - 1)) * double(2 * n));
        sum  += term;
    }
    return sum;
}

constexpr double const_sqrt(double v) {
    if (v <= 0.0) return 0.0;
    double g = v;                 // Newton-Raphson
    for (int i = 0; i < 60; ++i) g = 0.5 * (g + v / g);
    return g;
}

}  // namespace detail

// The orthonormal DCT matrix C[k][n] for length N, as a flat row-major
// (k-major) array of floats: C[k*N + n].
template <std::size_t N>
consteval std::array<float, N * N> make_dct_matrix() {
    std::array<float, N * N> m{};
    const double inv_sqrt2 = 1.0 / detail::const_sqrt(2.0);
    const double scale     = detail::const_sqrt(2.0 / double(N));
    for (std::size_t k = 0; k < N; ++k) {
        const double alpha = (k == 0) ? inv_sqrt2 : 1.0;
        for (std::size_t n = 0; n < N; ++n) {
            const double c = detail::const_cos(
                detail::kPi * double(2 * n + 1) * double(k) / double(2 * N));
            m[k * N + n] = float(alpha * scale * c);
        }
    }
    return m;
}

// The two block lengths the codec uses. Generated once, at compile time.
inline constexpr auto kDct16 = make_dct_matrix<16>();   // 3D blocks (16^3)
inline constexpr auto kDct64 = make_dct_matrix<64>();   // 2D blocks (64^2)

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_DCT_TABLES_HPP
