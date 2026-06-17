// Fuzz target: decode arbitrary bytes as a mask-aware block.
//
// The hardened-decoder invariant (design memory mcpp-scope): decode of corrupted
// / untrusted bytes must NEVER crash (no OOB, no SIGSEGV) — it may produce
// garbage voxels, but it must stay in bounds. This harness feeds arbitrary bytes
// to decode_mask_block_framed and asserts only that it returns.
//
// Build: clang++ -std=c++26 -fsanitize=fuzzer,address,undefined -I include \
//          fuzz/fuzz_block_decode.cpp -o fuzz_block_decode
#include "mcpp/codec/mask_block.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    // Try a range of quality values; decode must never crash on any input.
    std::vector<float> out(mcpp::codec::block_total<16, 3>());
    for (float q : {0.5f, 4.0f, 32.0f}) {
        mcpp::codec::decode_mask_block_framed<16, 3>(data, size, q, out.data());
    }
    return 0;
}
