// BlockKey — packed 8-byte key for the decoded-block cache.
//
// Rationale (design memory mcpp-cache): a key is (lod, bz, by, bx). Block coords
// are <= 2^16/axis (dims cap 2^20 / 16-block), LOD <= 3 bits. So the whole key
// packs into one u64: [lod:3 | bz:16 | by:16 | bx:16] (51 bits used, 13 spare).
// It is a trivially-copyable 8-byte strong type that IS a u64 (register-resident,
// fast hash/compare) but reads like a class (named accessors).
#ifndef MCPP_CACHE_BLOCK_KEY_HPP
#define MCPP_CACHE_BLOCK_KEY_HPP

#include "mcpp/core/dtype.hpp"

#include <compare>
#include <cstdint>

namespace mcpp::cache {

struct BlockKey {
    std::uint64_t bits = 0;

    // layout: bx[0:16] by[16:32] bz[32:48] lod[48:51]
    static constexpr BlockKey make(Lod lod, std::uint32_t bz, std::uint32_t by,
                                   std::uint32_t bx) {
        // coords must fit 16 bits (2^16 blocks/axis = 2^20 voxels / 16).
        return BlockKey{ (std::uint64_t(lod.v & 0x7u) << 48) |
                         (std::uint64_t(bz & 0xFFFFu) << 32) |
                         (std::uint64_t(by & 0xFFFFu) << 16) |
                          std::uint64_t(bx & 0xFFFFu) };
    }

    constexpr std::uint32_t bx()  const { return std::uint32_t(bits & 0xFFFFu); }
    constexpr std::uint32_t by()  const { return std::uint32_t((bits >> 16) & 0xFFFFu); }
    constexpr std::uint32_t bz()  const { return std::uint32_t((bits >> 32) & 0xFFFFu); }
    constexpr Lod           lod() const { return Lod{ std::uint8_t((bits >> 48) & 0x7u) }; }

    // hash: a good 64-bit avalanche so open-addressing probes spread well.
    constexpr std::uint64_t hash() const {
        std::uint64_t h = bits;
        h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull;
        h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ull;
        h ^= h >> 33;
        return h;
    }

    friend constexpr bool operator==(BlockKey, BlockKey) = default;
    friend constexpr auto operator<=>(BlockKey, BlockKey) = default;
};

static_assert(sizeof(BlockKey) == 8);

// A sentinel "empty" key for open-addressed slots. bits==0 is a valid key
// (lod0, block 0,0,0), so we reserve an otherwise-impossible value: lod=7 (>
// kMaxLods-1) with all coords max — never produced by make() for a real volume
// since num_lods <= 8 means valid lods are 0..7, but coords 0xFFFF*16 = 2^20
// which is the cap... use the top spare bit instead.
inline constexpr std::uint64_t kEmptyKeyBit = std::uint64_t(1) << 63;
inline constexpr BlockKey kEmptyKey{ kEmptyKeyBit };

}  // namespace mcpp::cache

#endif  // MCPP_CACHE_BLOCK_KEY_HPP
