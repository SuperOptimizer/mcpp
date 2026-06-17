// Coordinator — the BSP mutation-phase orchestrator.
//
// Rationale (design memory mcpp-architecture + mcpp-cache): the cache is a DUMB
// store; ALL coordination lives here, OUTSIDE it. Consumers accumulate their own
// misses during the frozen read phase; at the tick boundary the coordinator:
//   1. gathers miss lists from all consumers
//   2. merges + dedups (and would dedup against in-flight for streaming)
//   3. classifies each via archive occupancy (ALL_ZERO / DONT_KNOW / REAL_DATA)
//   4. decodes REAL_DATA blocks (per-block, faulting only their pages)
//   5. applies the fresh pairs to the cache in one batched apply()
//
// This is the local (non-streaming) coordinator. The streaming subsystem extends
// the DONT_KNOW path with remote fetch via a ByteSource; the rest is identical.
//
// First a CANONICAL ORDER is imposed on the gathered keys (sort by BlockKey
// bits) so the coordination is deterministic regardless of consumer/worker count
// (design review fix: "coordination is deterministic" requires a canonical order).
#ifndef MCPP_COORDINATOR_COORDINATOR_HPP
#define MCPP_COORDINATOR_COORDINATOR_HPP

#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/chunk.hpp"
#include "mcpp/cache/block_key.hpp"
#include "mcpp/cache/cache.hpp"
#include "mcpp/codec/convert.hpp"
#include "mcpp/codec/mask_block.hpp"
#include "mcpp/core/dtype.hpp"

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace mcpp::coordinator {

using cache::BlockKey;
using cache::FreshPair;

// Maps a cache BlockKey (16^3-block coords) to its (chunk coord, in-chunk block
// index) within the archive. A cache block is a 16^3 block; a chunk is 16 blocks
// per axis (256^3). So chunk coord = bcoord / 16, in-chunk index from bcoord % 16.
struct BlockAddr {
    std::uint64_t cz, cy, cx;     // chunk coords
    std::size_t bidx;             // in-chunk block index [0,4096)
};

inline BlockAddr to_block_addr(BlockKey k) {
    const std::uint32_t bz = k.bz(), by = k.by(), bx = k.bx();
    const std::uint32_t lz = bz % archive::kBlocksPerAxis;
    const std::uint32_t ly = by % archive::kBlocksPerAxis;
    const std::uint32_t lx = bx % archive::kBlocksPerAxis;
    return BlockAddr{
        bz / archive::kBlocksPerAxis, by / archive::kBlocksPerAxis,
        bx / archive::kBlocksPerAxis,
        (std::size_t(lz) * archive::kBlocksPerAxis + ly) * archive::kBlocksPerAxis + lx
    };
}

// The coordinator for a single dtype T cache + archive pair.
template <Sample T, std::size_t Capacity>
class Coordinator {
public:
    Coordinator(archive::Archive& arc, cache::Cache<T, Capacity>& cache)
        : arc_(arc), cache_(cache) {}

    // One BSP superstep: gather the (already-deduped-by-caller-or-not) miss keys,
    // resolve them via the archive, and apply to the cache. `misses` is the union
    // of all consumers' frozen-phase misses. Returns the number of blocks filled.
    std::size_t resolve(std::span<const BlockKey> misses) {
        // 1. canonical order + dedup
        scratch_keys_.assign(misses.begin(), misses.end());
        std::sort(scratch_keys_.begin(), scratch_keys_.end(),
                  [](BlockKey a, BlockKey b) { return a.bits < b.bits; });
        scratch_keys_.erase(std::unique(scratch_keys_.begin(), scratch_keys_.end()),
                            scratch_keys_.end());

        // 2. classify + decode. Decoded blocks live in decode_store_ (stable
        //    addresses across the batch); zero blocks get nullptr.
        decode_store_.clear();
        decode_store_.reserve(scratch_keys_.size());
        fresh_.clear();
        fresh_.reserve(scratch_keys_.size());

        for (BlockKey k : scratch_keys_) {
            const BlockAddr a = to_block_addr(k);
            const Lod lod = k.lod();
            const auto st = arc_.state(lod, a.cz, a.cy, a.cx);
            if (st == archive::ChunkState::all_zero ||
                st == archive::ChunkState::dont_know) {
                // local archive: nothing to fetch. ALL_ZERO -> zero block;
                // DONT_KNOW (only in partial/streaming) also treated as zero here.
                fresh_.push_back({k, nullptr});
                continue;
            }
            // REAL_DATA: decode the specific block (faults only its pages).
            archive::ChunkView v = arc_.chunk(lod, a.cz, a.cy, a.cx);
            if (!v.block_present(a.bidx)) {
                fresh_.push_back({k, nullptr});  // ALL_ZERO block within chunk
                continue;
            }
            auto payload = v.block(a.bidx);
            std::vector<float> fblk(codec::block_total<16, 3>());
            codec::decode_mask_block_framed<16, 3>(payload.data(), payload.size(),
                                                   v.quality(), fblk.data());
            // convert f32 -> T into stable storage
            std::vector<T> tblk(fblk.size());
            for (std::size_t i = 0; i < fblk.size(); ++i)
                tblk[i] = codec::store_from_f32<T>(fblk[i]);
            decode_store_.push_back(std::move(tblk));
            fresh_.push_back({k, decode_store_.back().data()});
        }

        // 3. one batched apply at the thaw boundary
        cache_.thaw();
        cache_.touch(scratch_keys_);             // re-request-implies-hot
        cache_.apply(std::span<const FreshPair<T>>(fresh_));
        return fresh_.size();
    }

private:
    archive::Archive& arc_;
    cache::Cache<T, Capacity>& cache_;
    std::vector<BlockKey> scratch_keys_;
    std::vector<std::vector<T>> decode_store_;   // stable block storage for apply()
    std::vector<FreshPair<T>> fresh_;
};

}  // namespace mcpp::coordinator

#endif  // MCPP_COORDINATOR_COORDINATOR_HPP
