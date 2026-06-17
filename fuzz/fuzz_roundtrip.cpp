// Fuzz target: encode arbitrary bytes as a u8 volume block, then decode.
//
// Roundtrip must never crash, and air bit-exactness must hold: any voxel that was
// air (0) on input decodes to exactly 0. This fuzzes the ENCODE path (mask build,
// air-fill, DCT, quant, entropy) plus decode.
#include "mcpp/codec/mask_block.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    constexpr std::size_t total = mcpp::codec::block_total<16, 3>();
    // build a 16^3 block from the fuzz bytes (zero-padded); mask from value!=0.
    std::vector<float> vox(total, 0.0f);
    std::vector<std::uint8_t> mask(total, 0);
    for (std::size_t i = 0; i < total && i < size; ++i) {
        vox[i] = float(data[i]);
        mask[i] = data[i] ? 1 : 0;
    }

    std::vector<std::uint8_t> bytes;
    mcpp::codec::encode_mask_block_framed<16, 3>(vox.data(), mask.data(), 4.0f, bytes);

    std::vector<float> recon(total);
    mcpp::codec::decode_mask_block_framed<16, 3>(bytes.data(), bytes.size(), 4.0f, recon.data());

    // air bit-exactness: every air voxel decodes to exactly 0.
    for (std::size_t i = 0; i < total; ++i)
        if (mask[i] == 0 && recon[i] != 0.0f) __builtin_trap();
    return 0;
}
