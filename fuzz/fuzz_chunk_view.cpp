// Fuzz target: parse arbitrary bytes as a chunk and read every block.
//
// The chunk offset table + xxh3 verify + per-block span extraction must never
// OOB on corrupted bytes. ChunkView trusts the archive's own bytes on the happy
// path, but this proves it stays in bounds on garbage (defense in depth).
#include "mcpp/archive/chunk.hpp"

#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    // need at least the fixed header to form a view; clamp.
    if (size < mcpp::archive::kChunkHeaderBytes) return 0;
    mcpp::archive::ChunkView v(data, size);
    (void)v.stored_hash();
    (void)v.quality();
    (void)v.verify();
    // walk every block; block(k) must clamp to bounds.
    for (std::size_t k = 0; k < mcpp::archive::kBlocksPerChunk; ++k) {
        if (v.block_present(k)) {
            auto span = v.block(k);
            // touch the span bounds (ASan catches OOB if block() mis-sized it)
            volatile std::uint8_t sink = 0;
            if (!span.empty()) sink = span.front() ^ span.back();
            (void)sink;
        }
    }
    return 0;
}
