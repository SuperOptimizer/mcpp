// Block orchestration: the complete per-block encode/decode pipeline.
//
//   encode: voxels(f32) -> forward DCT -> quantize -> coef coder -> bytes
//   decode: bytes -> coef coder -> dequantize -> inverse DCT -> voxels(f32)
//
// Rationale (design memory): blocks are SELF-CONTAINED — one block decodes from
// its own payload alone, with no cross-block entropy state (the CoefModel is
// reset per block). The entropy/quant index stream is exact; the f32 transform
// is lossy + non-deterministic, so block roundtrip is validated by TOLERANCE
// against the dequantized reconstruction, and the index stream by exact ==.
//
// This is the first END-TO-END codec milestone: real compressed bytes in, voxels
// out. Mask-aware air handling, max-error correction, and the generic
// codec<T,Rank,C> wrapper layer on top of this later.
#ifndef MCPP_CODEC_BLOCK_HPP
#define MCPP_CODEC_BLOCK_HPP

#include "mcpp/codec/coef_coder.hpp"
#include "mcpp/codec/dct.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/codec/range_coder.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpp::codec {

// Encode one N^Rank f32 block at quality q. Appends the payload to `out` and
// returns the number of bytes written.
template <std::size_t N, std::size_t Rank>
std::size_t encode_block(const float* voxels, float q, std::vector<std::uint8_t>& out) {
    constexpr std::size_t total = block_total<N, Rank>();

    // forward transform (on a scratch copy; caller's buffer untouched)
    std::vector<float> coeffs(voxels, voxels + total);
    forward_dct<N, Rank>(coeffs.data());

    // quantize to integer indices
    std::vector<std::int32_t> idx(total);
    quantize_block<N, Rank>(coeffs.data(), idx.data(), q);

    // entropy code
    const std::size_t before = out.size();
    RangeEncoder enc(out);
    CoefModel mdl;  // reset per block (self-contained)
    encode_coeffs<N, Rank>(enc, mdl, idx.data());
    enc.finish();
    return out.size() - before;
}

// Decode one N^Rank block from `payload` at quality q into `voxels` (f32).
template <std::size_t N, std::size_t Rank>
void decode_block(const std::uint8_t* payload, std::size_t len, float q, float* voxels) {
    constexpr std::size_t total = block_total<N, Rank>();

    std::vector<std::int32_t> idx(total);
    RangeDecoder dec(payload, len);
    CoefModel mdl;  // identical reset state as the encoder
    decode_coeffs<N, Rank>(dec, mdl, idx.data());

    dequantize_block<N, Rank>(idx.data(), voxels, q);
    inverse_dct<N, Rank>(voxels);
}

// Decode just the quantized indices (no dequant/inverse) — used by tests and the
// max-error correction pass to inspect the exact integer stream.
template <std::size_t N, std::size_t Rank>
void decode_block_indices(const std::uint8_t* payload, std::size_t len, std::int32_t* idx) {
    RangeDecoder dec(payload, len);
    CoefModel mdl;
    decode_coeffs<N, Rank>(dec, mdl, idx);
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_BLOCK_HPP
