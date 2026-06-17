// Chunk format: within-chunk byte layout.
//
// Rationale (design memory mcpp-archive): a 256^3 chunk = 4096 self-contained
// 16^3 blocks. Layout at chunk_base (page-aligned):
//
//   [chunk header]
//     xxh3        u64   integrity over the stored chunk bytes
//     quality     f32   per-chunk q
//     flags       u32   reserved
//     offset_table u32 x 4097   block k payload offset from chunk_base;
//                                0xFFFFFFFF => block k is ALL_ZERO (no payload);
//                                entry 4096 = end-of-payload terminator
//   [dense-packed block payloads]  (back-to-back, no gaps; page-aligned start)
//
// A single block is read by faulting only its payload pages: offset_table[k] ..
// offset_table[k+1] (skipping ALL_ZERO sentinels for the length). Within-chunk
// blocks are 2-state (ALL_ZERO sentinel vs real); the 3-state DONT_KNOW lives
// only in the per-chunk occupancy map.
#ifndef MCPP_ARCHIVE_CHUNK_HPP
#define MCPP_ARCHIVE_CHUNK_HPP

#include "mcpp/archive/layout.hpp"
#include "mcpp/codec/mask_block.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace mcpp::archive {

inline constexpr std::uint32_t kAllZeroSentinel = 0xFFFF'FFFFu;
inline constexpr std::size_t   kOffsetTableLen  = kBlocksPerChunk + 1;  // 4097

// Fixed chunk-header size (before the offset table): xxh3(8) + quality(4) + flags(4).
inline constexpr std::size_t kChunkHeaderFixed = 8 + 4 + 4;
// Full header including the offset table.
inline constexpr std::size_t kChunkHeaderBytes =
    kChunkHeaderFixed + kOffsetTableLen * sizeof(std::uint32_t);  // 20 + 16388 = 16408

inline std::uint64_t align_up(std::uint64_t v, std::uint64_t a) {
    return (v + a - 1) / a * a;
}

// xxh3-64 — minimal portable implementation (integrity only, not canonical).
// (A vendored xxHash can drop in later; this is a correct 64-bit hash with the
// xxh3 avalanche finalizer over the input, sufficient for corruption detection.)
inline std::uint64_t xxh3_64(const std::uint8_t* p, std::size_t n) {
    // FNV-1a-style accumulation + xxh3 avalanche; deterministic, fast enough.
    std::uint64_t h = 0x9E3779B185EBCA87ull ^ (std::uint64_t(n) * 0xC2B2AE3D27D4EB4Full);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        std::uint64_t k;
        std::memcpy(&k, p + i, 8);
        h ^= k * 0xFF51AFD7ED558CCDull;
        h = (h << 31) | (h >> 33);
        h *= 0xC2B2AE3D27D4EB4Full;
    }
    for (; i < n; ++i) { h ^= std::uint64_t(p[i]) * 0x100000001B3ull; h = (h << 23) | (h >> 41); }
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 29; h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 32;
    return h;
}

// Build a chunk's bytes from 4096 already-encoded block payloads. A block with
// an empty payload span is ALL_ZERO (sentinel). Returns the chunk byte buffer
// (header + offset table + dense payloads), with xxh3 filled in.
class ChunkBuilder {
public:
    void set_quality(float q) { quality_ = q; }

    // Add block k's payload (k in [0,4096)). Empty span => ALL_ZERO.
    void set_block(std::size_t k, std::span<const std::uint8_t> payload) {
        blocks_[k].assign(payload.begin(), payload.end());
        present_[k] = !payload.empty();
    }

    std::vector<std::uint8_t> finish() {
        // compute offsets
        std::vector<std::uint32_t> table(kOffsetTableLen, kAllZeroSentinel);
        std::uint64_t cursor = kChunkHeaderBytes;  // payloads start after header
        for (std::size_t k = 0; k < kBlocksPerChunk; ++k) {
            if (present_[k]) {
                table[k] = std::uint32_t(cursor);
                cursor += blocks_[k].size();
            }  // else stays sentinel
        }
        table[kBlocksPerChunk] = std::uint32_t(cursor);  // end terminator

        std::vector<std::uint8_t> out(cursor, 0);
        // header
        float q = quality_;
        std::uint32_t flags = 0;
        std::memcpy(out.data() + 8, &q, 4);
        std::memcpy(out.data() + 12, &flags, 4);
        std::memcpy(out.data() + kChunkHeaderFixed, table.data(),
                    table.size() * sizeof(std::uint32_t));
        // payloads
        for (std::size_t k = 0; k < kBlocksPerChunk; ++k)
            if (present_[k])
                std::memcpy(out.data() + table[k], blocks_[k].data(), blocks_[k].size());
        // xxh3 over everything AFTER the hash field (offset 8..end)
        std::uint64_t h = xxh3_64(out.data() + 8, out.size() - 8);
        std::memcpy(out.data(), &h, 8);
        return out;
    }

private:
    float quality_ = 1.0f;
    std::array<std::vector<std::uint8_t>, kBlocksPerChunk> blocks_{};
    std::array<bool, kBlocksPerChunk> present_{};
};

// Read-only view over a chunk's bytes (in the mmap). Bounds-light: trusts the
// archive's own bytes (happy-path); untrusted-input hardening is the streaming
// layer's concern.
class ChunkView {
public:
    ChunkView(const std::uint8_t* base, std::size_t len) : base_(base), len_(len) {}

    std::uint64_t stored_hash() const {
        std::uint64_t h; std::memcpy(&h, base_, 8); return h;
    }
    float quality() const { float q; std::memcpy(&q, base_ + 8, 4); return q; }

    // The chunk's true byte length = the offset-table terminator (end of the last
    // payload). The mmap slot is larger (fixed stride, sparse tail); we must hash
    // only the real bytes, exactly as the builder did.
    std::size_t chunk_len() const { return std::size_t(offset(kBlocksPerChunk)); }

    bool verify() const {
        const std::size_t n = chunk_len();
        if (n < kChunkHeaderBytes || n > len_) return false;
        return xxh3_64(base_ + 8, n - 8) == stored_hash();
    }

    std::uint32_t offset(std::size_t k) const {
        std::uint32_t v;
        std::memcpy(&v, base_ + kChunkHeaderFixed + k * sizeof(std::uint32_t), 4);
        return v;
    }
    bool block_present(std::size_t k) const { return offset(k) != kAllZeroSentinel; }

    // Payload span for block k (empty if ALL_ZERO). Length is derived from the
    // next non-sentinel offset (the terminator guarantees a successor).
    std::span<const std::uint8_t> block(std::size_t k) const {
        const std::uint32_t o = offset(k);
        if (o == kAllZeroSentinel) return {};
        // find next present offset (or terminator)
        std::uint32_t next = offset(kBlocksPerChunk);  // terminator default
        for (std::size_t j = k + 1; j <= kBlocksPerChunk; ++j) {
            std::uint32_t oj = offset(j);
            if (oj != kAllZeroSentinel) { next = oj; break; }
        }
        return { base_ + o, std::size_t(next - o) };
    }

private:
    const std::uint8_t* base_;
    std::size_t len_;
};

}  // namespace mcpp::archive

#endif  // MCPP_ARCHIVE_CHUNK_HPP
