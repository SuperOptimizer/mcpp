// Generic block codec: codec<T, N, Rank, C, Transform>.
//
// The user-facing per-block codec for real data. Ties the f32 core to the 8
// dtypes and the C-component dimension:
//
//   encode: interleaved T samples
//             -> load to f32 + scrub
//             -> component transform (forward, per pixel)
//             -> deinterleave into C planes
//             -> per-plane (forward DCT -> quantize -> entropy code)
//   decode: per-plane (entropy -> dequant -> inverse DCT)
//             -> interleave
//             -> component transform (inverse, per pixel)
//             -> round/clamp to T
//
// Rationale (design memory mcpp-scope): dtype is ORTHOGONAL to use case; one
// generic codec covers {8 dtypes} x {ranks} x {C} x {transform}, monomorphized.
// dtype touches only the load/store edges; the interior is f32. NO 3D
// multichannel (volumes are C=1) — but the wrapper is written generically and
// that constraint is enforced by the caller / config, not hardcoded here.
#ifndef MCPP_CODEC_CODEC_HPP
#define MCPP_CODEC_CODEC_HPP

#include "mcpp/codec/block.hpp"
#include "mcpp/codec/component_transform.hpp"
#include "mcpp/codec/convert.hpp"
#include "mcpp/core/dtype.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpp::codec {

// Block codec over T samples, length-N transform, Rank dims, C components,
// component transform policy CT.
template <Sample T, std::size_t N, std::size_t Rank, std::size_t C = 1,
          class CT = Identity>
    requires ComponentTransform<CT, C>
struct Codec {
    static constexpr std::size_t plane = block_total<N, Rank>();  // samples/plane
    static constexpr std::size_t count = plane * C;               // interleaved

    // Encode one interleaved C-channel block. Appends to `out`. Returns a small
    // header of per-plane byte lengths (so decode can split the planes) plus the
    // concatenated plane payloads. We prepend a u32 length per plane.
    static void encode(const T* src, float q, std::vector<std::uint8_t>& out) {
        // 1. load + scrub + component-transform, deinterleaving into C planes
        std::array<std::vector<float>, C> planes;
        for (auto& p : planes) p.resize(plane);

        std::array<float, C> px{};
        for (std::size_t i = 0; i < plane; ++i) {
            for (std::size_t c = 0; c < C; ++c) px[c] = load_f32<T>(src[i * C + c]);
            CT::template forward<C>(px);
            for (std::size_t c = 0; c < C; ++c) planes[c][i] = px[c];
        }

        // 2. per-plane encode, each prefixed by its u32 byte length
        for (std::size_t c = 0; c < C; ++c) {
            std::vector<std::uint8_t> pbytes;
            encode_block<N, Rank>(planes[c].data(), q, pbytes);
            const std::uint32_t n = std::uint32_t(pbytes.size());
            out.push_back(std::uint8_t(n & 0xFF));
            out.push_back(std::uint8_t((n >> 8) & 0xFF));
            out.push_back(std::uint8_t((n >> 16) & 0xFF));
            out.push_back(std::uint8_t((n >> 24) & 0xFF));
            out.insert(out.end(), pbytes.begin(), pbytes.end());
        }
    }

    // Decode one interleaved C-channel block from `payload`.
    static void decode(const std::uint8_t* payload, std::size_t len, float q, T* dst) {
        std::array<std::vector<float>, C> planes;
        for (auto& p : planes) p.resize(plane);

        std::size_t off = 0;
        for (std::size_t c = 0; c < C; ++c) {
            // read u32 plane length
            std::uint32_t n = std::uint32_t(payload[off]) |
                              (std::uint32_t(payload[off + 1]) << 8) |
                              (std::uint32_t(payload[off + 2]) << 16) |
                              (std::uint32_t(payload[off + 3]) << 24);
            off += 4;
            decode_block<N, Rank>(payload + off, n, q, planes[c].data());
            off += n;
        }
        (void)len;

        // interleave + inverse component transform + store to T
        std::array<float, C> px{};
        for (std::size_t i = 0; i < plane; ++i) {
            for (std::size_t c = 0; c < C; ++c) px[c] = planes[c][i];
            CT::template inverse<C>(px);
            for (std::size_t c = 0; c < C; ++c) dst[i * C + c] = store_from_f32<T>(px[c]);
        }
    }
};

// The concrete configurations the project actually uses (closed list).
template <Sample T> using VolumeCodec  = Codec<T, 16, 3, 1, Identity>; // 3D grayscale volume (main)
template <Sample T> using GrayImgCodec = Codec<T, 64, 2, 1, Identity>; // 2D grayscale image
template <Sample T> using RgbImgCodec  = Codec<T, 64, 2, 3, YCoCg>;    // 2D RGB image
template <Sample T> using SurfaceCodec = Codec<T, 64, 2, 3, Identity>; // 2D xyz parametric surface

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_CODEC_HPP
