// Separable N-point DCT over an N^Rank block, in f32.
//
// Rationale (design memory mcpp-scope): ONE rank-generic separable-DCT kernel.
// 3D = DCT(z) then DCT(y) then DCT(x) over a 16^3 block; 2D = DCT(v) then DCT(u)
// over a 64^2 block. The forward is DCT-II; the inverse is DCT-III (= transpose
// of the orthonormal matrix). All compute in f32 (compute_t is always float);
// -ffast-math is expected, so this is intentionally non-deterministic across
// ISAs and validated by TOLERANCE, never byte-identity.
//
// This is the clean, correct reference kernel (dense matrix apply per axis). The
// fast factored DCT-64 and SIMD widths come later as drop-in replacements behind
// this same interface; correctness first.
#ifndef MCPP_CODEC_DCT_HPP
#define MCPP_CODEC_DCT_HPP

#include "mcpp/codec/dct_tables.hpp"

#include <array>
#include <cstddef>

namespace mcpp::codec {

// Pick the orthonormal DCT matrix for a given length at compile time.
template <std::size_t N>
consteval const std::array<float, N * N>& dct_matrix_for() {
    if constexpr (N == 16) return kDct16;
    else if constexpr (N == 64) return kDct64;
    else static_assert(N == 16 || N == 64, "unsupported DCT length");
}

namespace detail {

// Apply the 1-D length-N transform along one axis of an N^Rank flat buffer.
// `forward` selects DCT-II (true) vs DCT-III (false, the transpose).
//
// The buffer is row-major with the LAST axis fastest (matches zfast_layout for
// a cubic block where all extents == N). For a given axis `ax`, the stride
// between successive elements along that axis is N^(Rank-1-ax).
template <std::size_t N, std::size_t Rank>
constexpr void axis_transform(float* buf, std::size_t ax, bool forward) {
    const auto& C = dct_matrix_for<N>();

    // stride along axis ax
    std::size_t stride = 1;
    for (std::size_t a = Rank - 1; a > ax; --a) stride *= N;

    // total elements, and number of 1-D lines = total / N
    std::size_t total = 1;
    for (std::size_t a = 0; a < Rank; ++a) total *= N;
    const std::size_t lines = total / N;

    std::array<float, N> in{};
    std::array<float, N> out{};

    // Iterate over every 1-D line along axis ax. A line is identified by a base
    // offset whose digit at position `ax` is 0; we enumerate lines by walking
    // all offsets and skipping those whose ax-digit != 0.
    std::size_t processed = 0;
    for (std::size_t base = 0; processed < lines; ++base) {
        // is the ax-digit of `base` zero? digit = (base / stride) % N
        if ((base / stride) % N != 0) continue;
        ++processed;

        for (std::size_t i = 0; i < N; ++i) in[i] = buf[base + i * stride];

        if (forward) {
            // X[k] = sum_n C[k][n] in[n], computed output-parallel (SAXPY): for
            // each input n, scatter-add in[n]*C[k][n] across all outputs k. The
            // outer-n / inner-k shape is the one the compiler vectorizes well
            // (same structure as the fast inverse below), unlike the inner-n
            // horizontal reduction which compiled ~6x slower.
            for (std::size_t k = 0; k < N; ++k) out[k] = 0.0f;
            for (std::size_t n = 0; n < N; ++n) {
                const float s = in[n];
                for (std::size_t k = 0; k < N; ++k) out[k] += s * C[k * N + n];
            }
        } else {
            // x[n] = sum_k C[k][n] X[k]   (transpose apply)
            for (std::size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < N; ++k) acc += C[k * N + n] * in[k];
                out[n] = acc;
            }
        }

        for (std::size_t i = 0; i < N; ++i) buf[base + i * stride] = out[i];
    }
}

}  // namespace detail

// Forward separable DCT-II, in place, over an N^Rank cubic block.
template <std::size_t N, std::size_t Rank>
constexpr void forward_dct(float* block) {
    for (std::size_t ax = 0; ax < Rank; ++ax)
        detail::axis_transform<N, Rank>(block, ax, /*forward=*/true);
}

// Inverse separable DCT-III, in place, over an N^Rank cubic block.
template <std::size_t N, std::size_t Rank>
constexpr void inverse_dct(float* block) {
    for (std::size_t ax = 0; ax < Rank; ++ax)
        detail::axis_transform<N, Rank>(block, ax, /*forward=*/false);
}

// Convenience: the codec's two configurations.
constexpr void forward_dct3_16(float* block) { forward_dct<16, 3>(block); }
constexpr void inverse_dct3_16(float* block) { inverse_dct<16, 3>(block); }
constexpr void forward_dct2_64(float* block) { forward_dct<64, 2>(block); }
constexpr void inverse_dct2_64(float* block) { inverse_dct<64, 2>(block); }

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_DCT_HPP
