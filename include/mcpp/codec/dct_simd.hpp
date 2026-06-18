// Explicit-SIMD multi-line DCT-16 for the 3D codec path.
//
// The separable 16-point transform has 256 independent 1-D lines per axis. The
// fast structure packs 16 DIFFERENT lines into the lanes of a 16-wide register
// and runs the matmul one coefficient at a time: out[k] += C[k][n] * in_lines[n]
// — 16 independent FMA chains, contiguous vector loads, no horizontal reduction,
// no gather. This auto-vec-beats the scalar SAXPY by ~2.3-2.6x (measured).
//
// Backends (compile-time selected, widest first): AVX-512 (16-wide, ideal — a
// whole DCT-16 output vector is one zmm), AVX2 (8-wide), NEON (4-wide), and the
// scalar SAXPY (portable fallback, also used for the 2 axes that don't pack
// contiguously and for non-16 lengths). Every backend is validated against the
// scalar oracle in tests within f32 tolerance.
//
// Per-axis contiguity (block index = z*256 + y*16 + x):
//   axis z (stride 256) and axis y (stride 16): 16 lines packed along x are
//     contiguous -> vector loads. Use the SIMD kernel.
//   axis x (stride 1): the transform axis IS the contiguous one, so 16 lines are
//     NOT lane-contiguous (gather/transpose both measured SLOWER than scalar).
//     Use the scalar SAXPY for x — its contiguous loads vectorize fine.
#ifndef MCPP_CODEC_DCT_SIMD_HPP
#define MCPP_CODEC_DCT_SIMD_HPP

#include "mcpp/codec/dct_tables.hpp"

#include <array>
#include <cstddef>

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define MCPP_DCT_SIMD 512
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define MCPP_DCT_SIMD 256
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define MCPP_DCT_SIMD 128
#else
#  define MCPP_DCT_SIMD 0
#endif

namespace mcpp::codec::simd {

inline constexpr bool kHaveDct16Simd = (MCPP_DCT_SIMD >= 128);

// Scalar SAXPY for one axis (used for the x-axis and as the universal fallback).
// `apply_C`: forward applies C (out[k]=sum_n C[k][n]in[n]); inverse applies C^T.
inline void axis_scalar16(float* buf, std::size_t stride, bool forward) {
    const auto& C = kDct16;
    constexpr std::size_t N = 16, total = 4096, lines = total / N;
    float in[16], out[16];
    std::size_t processed = 0;
    for (std::size_t base = 0; processed < lines; ++base) {
        if ((base / stride) % N != 0) continue;
        ++processed;
        for (std::size_t i = 0; i < N; ++i) in[i] = buf[base + i * stride];
        if (forward) {
            for (std::size_t k = 0; k < N; ++k) out[k] = 0.0f;
            for (std::size_t n = 0; n < N; ++n) {
                const float s = in[n];
                for (std::size_t k = 0; k < N; ++k) out[k] += s * C[k * N + n];
            }
        } else {
            for (std::size_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < N; ++k) acc += C[k * N + n] * in[k];
                out[n] = acc;
            }
        }
        for (std::size_t i = 0; i < N; ++i) buf[base + i * stride] = out[i];
    }
}

#if MCPP_DCT_SIMD == 512
// 16 lines in the 16 lanes of a zmm. `base` is the first line's base; the 16
// lines are at base+0..15 (contiguous). `axstride` steps along the transform axis.
inline void axis16_lines512(float* B, int base, int axstride, bool forward) {
    const auto& C = kDct16;
    __m512 acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = _mm512_setzero_ps();
    if (forward) {  // out[k] = sum_n C[k][n] in[n]
        for (int n = 0; n < 16; ++n) {
            __m512 inn = _mm512_loadu_ps(B + base + n * axstride);
            for (int k = 0; k < 16; ++k)
                acc[k] = _mm512_fmadd_ps(_mm512_set1_ps(C[k * 16 + n]), inn, acc[k]);
        }
    } else {        // out[n] = sum_k C[k][n] in[k]
        for (int k = 0; k < 16; ++k) {
            __m512 ink = _mm512_loadu_ps(B + base + k * axstride);
            for (int n = 0; n < 16; ++n)
                acc[n] = _mm512_fmadd_ps(_mm512_set1_ps(C[k * 16 + n]), ink, acc[n]);
        }
    }
    for (int i = 0; i < 16; ++i) _mm512_storeu_ps(B + base + i * axstride, acc[i]);
}

// 3D DCT-16: z & y axes via the 16-line SIMD kernel, x axis via scalar.
inline void dct3d16(float* B, bool forward) {
    for (int y = 0; y < 16; ++y) axis16_lines512(B, y * 16, 256, forward);   // axis z
    for (int z = 0; z < 16; ++z) axis16_lines512(B, z * 256, 16, forward);   // axis y
    axis_scalar16(B, 1, forward);                                            // axis x
}
inline constexpr bool kDct3d16 = true;

#elif MCPP_DCT_SIMD == 256
// AVX2: 8 lines per group (two groups of 8 cover the 16 contiguous lines).
inline void axis8_lines256(float* B, int base, int axstride, bool forward) {
    const auto& C = kDct16;
    __m256 acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = _mm256_setzero_ps();
    if (forward) {
        for (int n = 0; n < 16; ++n) {
            __m256 inn = _mm256_loadu_ps(B + base + n * axstride);
            for (int k = 0; k < 16; ++k)
                acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(C[k * 16 + n]), inn, acc[k]);
        }
    } else {
        for (int k = 0; k < 16; ++k) {
            __m256 ink = _mm256_loadu_ps(B + base + k * axstride);
            for (int n = 0; n < 16; ++n)
                acc[n] = _mm256_fmadd_ps(_mm256_set1_ps(C[k * 16 + n]), ink, acc[n]);
        }
    }
    for (int i = 0; i < 16; ++i) _mm256_storeu_ps(B + base + i * axstride, acc[i]);
}
inline void dct3d16(float* B, bool forward) {
    for (int y = 0; y < 16; ++y) { axis8_lines256(B, y*16, 256, forward); axis8_lines256(B, y*16+8, 256, forward); }
    for (int z = 0; z < 16; ++z) { axis8_lines256(B, z*256, 16, forward); axis8_lines256(B, z*256+8, 16, forward); }
    axis_scalar16(B, 1, forward);
}
inline constexpr bool kDct3d16 = true;

#elif MCPP_DCT_SIMD == 128
// NEON: 4 lines per group (four groups of 4 cover the 16 contiguous lines).
inline void axis4_lines_neon(float* B, int base, int axstride, bool forward) {
    const auto& C = kDct16;
    float32x4_t acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(0.0f);
    if (forward) {
        for (int n = 0; n < 16; ++n) {
            float32x4_t inn = vld1q_f32(B + base + n * axstride);
            for (int k = 0; k < 16; ++k)
                acc[k] = vmlaq_n_f32(acc[k], inn, C[k * 16 + n]);
        }
    } else {
        for (int k = 0; k < 16; ++k) {
            float32x4_t ink = vld1q_f32(B + base + k * axstride);
            for (int n = 0; n < 16; ++n)
                acc[n] = vmlaq_n_f32(acc[n], ink, C[k * 16 + n]);
        }
    }
    for (int i = 0; i < 16; ++i) vst1q_f32(B + base + i * axstride, acc[i]);
}
inline void dct3d16(float* B, bool forward) {
    for (int y = 0; y < 16; ++y) for (int o = 0; o < 16; o += 4) axis4_lines_neon(B, y*16+o, 256, forward);
    for (int z = 0; z < 16; ++z) for (int o = 0; o < 16; o += 4) axis4_lines_neon(B, z*256+o, 16, forward);
    axis_scalar16(B, 1, forward);
}
inline constexpr bool kDct3d16 = true;

#else
inline void dct3d16(float* B, bool forward) {  // pure scalar fallback
    axis_scalar16(B, 256, forward);
    axis_scalar16(B, 16, forward);
    axis_scalar16(B, 1, forward);
}
inline constexpr bool kDct3d16 = true;
#endif

}  // namespace mcpp::codec::simd

#endif  // MCPP_CODEC_DCT_SIMD_HPP
