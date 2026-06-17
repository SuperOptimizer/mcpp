// 3-state per-chunk occupancy map (2 bits/chunk).
//
// Rationale (design memory mcpp-archive): occupancy is the authoritative "what
// exists" oracle. States renumbered so the sparse-zero default is the SAFE one:
//   0 = DONT_KNOW / ABSENT  (sparse-mmap zero default: streaming = not fetched,
//                            offline = must-be-filled)
//   1 = ALL_ZERO            (queried source of truth, confirmed empty)
//   2 = REAL_DCT_DATA       (must decode)
//   3 = RESERVED
// A fresh sparse-mmap'd occupancy region reads as all-DONT_KNOW (correct). The
// map is a flat 2-bit array indexed by the global linear slot index across all
// LODs. 3-state is ONLY at chunk granularity; within a chunk, blocks are 2-state.
#ifndef MCPP_ARCHIVE_OCCUPANCY_HPP
#define MCPP_ARCHIVE_OCCUPANCY_HPP

#include <cstddef>
#include <cstdint>

namespace mcpp::archive {

enum class ChunkState : std::uint8_t {
    dont_know = 0,   // sparse-zero default; not fetched / must be filled
    all_zero  = 1,   // confirmed empty
    real_data = 2,   // must decode
    reserved  = 3,
};

// Non-owning 2-bit-per-entry view over a byte buffer. 4 entries per byte.
class OccupancyMap {
public:
    OccupancyMap() = default;
    OccupancyMap(std::uint8_t* bytes, std::uint64_t count)
        : bytes_(bytes), count_(count) {}

    // Number of 2-bit entries this map holds.
    std::uint64_t count() const noexcept { return count_; }

    // Bytes needed to hold `count` 2-bit entries.
    static std::uint64_t bytes_for(std::uint64_t count) noexcept {
        return (count + 3) / 4;
    }

    ChunkState get(std::uint64_t i) const noexcept {
        const std::uint64_t byte = i >> 2;
        const unsigned shift = unsigned(i & 3u) * 2u;
        return ChunkState((bytes_[byte] >> shift) & 0x3u);
    }

    void set(std::uint64_t i, ChunkState s) noexcept {
        const std::uint64_t byte = i >> 2;
        const unsigned shift = unsigned(i & 3u) * 2u;
        const std::uint8_t mask = std::uint8_t(0x3u << shift);
        bytes_[byte] = std::uint8_t((bytes_[byte] & ~mask) |
                                    (std::uint8_t(std::uint8_t(s) & 0x3u) << shift));
    }

private:
    std::uint8_t* bytes_ = nullptr;
    std::uint64_t count_ = 0;
};

}  // namespace mcpp::archive

#endif  // MCPP_ARCHIVE_OCCUPANCY_HPP
