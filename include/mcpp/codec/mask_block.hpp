// Mask-aware block codec: enforces AIR BIT-EXACTNESS.
//
// Rationale (design memory mcpp-scope, KEPT invariant): air voxels must decode to
// EXACTLY 0, and material must never become 0. This is enforced by an INTEGER
// mask compare applied AFTER the f32 inverse transform — outside the float path,
// so it survives -ffast-math. The encoder air-fills (airfill.hpp) only to shape
// DCT energy; the fill is discarded here.
//
// Layout of a mask-aware block payload:
//   [ range-coded mask bitstream (1 adaptive bit/voxel) | finish ]
//   [ u32 coeff payload length | coeff payload (block.hpp) ]
//
// The mask is coded losslessly (integer => exact). The coefficient payload is the
// ordinary self-contained block payload. Decode recovers the mask, decodes the
// coefficients, inverse-transforms, then forces air->0 and material->away-from-0.
#ifndef MCPP_CODEC_MASK_BLOCK_HPP
#define MCPP_CODEC_MASK_BLOCK_HPP

#include "mcpp/codec/airfill.hpp"
#include "mcpp/codec/block.hpp"
#include "mcpp/codec/range_coder.hpp"
#include "mcpp/codec/scan.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpp::codec {

// Material floor: decoded material is clamped to be at least this (so it never
// collapses to 0 and gets misread as air). 1 in u8 terms; callers using other
// dtypes scale via the codec's store stage.
inline constexpr float kMaterialFloor = 1.0f;

// Framed layout: [u32 mask_len][mask bytes][u32 coeff_len][coeff bytes]

template <std::size_t N, std::size_t Rank>
std::size_t encode_mask_block_framed(const float* voxels, const std::uint8_t* mask,
                                     float q, std::vector<std::uint8_t>& out) {
    constexpr std::size_t total = block_total<N, Rank>();
    const std::size_t before = out.size();

    auto put_u32 = [&](std::uint32_t v) {
        out.push_back(std::uint8_t(v & 0xFF));
        out.push_back(std::uint8_t((v >> 8) & 0xFF));
        out.push_back(std::uint8_t((v >> 16) & 0xFF));
        out.push_back(std::uint8_t((v >> 24) & 0xFF));
    };

    // mask bytes
    std::vector<std::uint8_t> mbytes;
    {
        RangeEncoder enc(mbytes);
        BitContext ctx;
        for (std::size_t i = 0; i < total; ++i) enc.encode_bit(ctx, mask[i] ? 1 : 0);
        enc.finish();
    }
    // coeff bytes (air-filled)
    std::vector<float> work(voxels, voxels + total);
    air_fill<N, Rank>(work.data(), mask);
    std::vector<std::uint8_t> cbytes;
    encode_block<N, Rank>(work.data(), q, cbytes);

    put_u32(std::uint32_t(mbytes.size()));
    out.insert(out.end(), mbytes.begin(), mbytes.end());
    put_u32(std::uint32_t(cbytes.size()));
    out.insert(out.end(), cbytes.begin(), cbytes.end());
    return out.size() - before;
}

// Decode a framed mask-aware block. Forces air->0 (exact) and material away from
// 0, both via integer mask compares outside the float transform.
//
// HARDENED (fuzz-driven, design memory mcpp-scope): all framing reads are
// bounds-checked against `len`. On ANY malformed input the block decodes to all
// zeros rather than reading out of bounds — decode of untrusted bytes never
// crashes. (The RangeDecoder already returns 0 past end-of-buffer.)
template <std::size_t N, std::size_t Rank>
void decode_mask_block_framed(const std::uint8_t* payload, std::size_t len,
                              float q, float* voxels) {
    constexpr std::size_t total = block_total<N, Rank>();
    for (std::size_t i = 0; i < total; ++i) voxels[i] = 0.0f;  // safe default

    auto get_u32 = [&](std::size_t at) {
        return std::uint32_t(payload[at]) | (std::uint32_t(payload[at + 1]) << 8) |
               (std::uint32_t(payload[at + 2]) << 16) | (std::uint32_t(payload[at + 3]) << 24);
    };

    // framing: [u32 mlen][mlen bytes][u32 clen][clen bytes], all within `len`.
    std::size_t off = 0;
    if (off + 4 > len) return;
    const std::uint32_t mlen = get_u32(off); off += 4;
    if (mlen > len || off + mlen > len) return;
    const std::uint8_t* mptr = payload + off; off += mlen;
    if (off + 4 > len) return;
    const std::uint32_t clen = get_u32(off); off += 4;
    if (clen > len || off + clen > len) return;
    const std::uint8_t* cptr = payload + off; off += clen;

    // 1. recover the mask exactly
    std::vector<std::uint8_t> mask(total);
    {
        RangeDecoder dec(mptr, mlen);
        BitContext ctx;
        for (std::size_t i = 0; i < total; ++i)
            mask[i] = std::uint8_t(dec.decode_bit(ctx));
    }

    // 2. decode coefficients -> f32 reconstruction
    decode_block<N, Rank>(cptr, clen, q, voxels);

    // 3. ENFORCE AIR BIT-EXACTNESS outside the float transform:
    //    air -> exactly 0; material -> never 0 (floor).
    for (std::size_t i = 0; i < total; ++i) {
        if (mask[i] == 0) {
            voxels[i] = 0.0f;                       // exact air
        } else if (voxels[i] < kMaterialFloor) {
            voxels[i] = kMaterialFloor;             // material never collapses to air
        }
    }
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_MASK_BLOCK_HPP
