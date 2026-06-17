// Archive geometry + static slot mapping (.mcp).
//
// Rationale (design memory mcpp-archive): 3D-volumetric only, z/y/x, dims 3xu32
// capped at 2^20/axis. Chunk = 256^3 (16^3 blocks). LODs are externally provided
// 2x2x2 downscales (LOD0 finest, max 8), pre-aligned (lower coord = higher/2).
// A chunk's position is a PURE FUNCTION of (lod, cz, cy, cx) — the static slot
// mapping — with NO dynamic index. LOD regions are laid out coarse-first.
//
// Everything here is integer arithmetic, derived from dims (single source of
// truth): per-LOD chunk counts and region bases are computed, not stored.
#ifndef MCPP_ARCHIVE_LAYOUT_HPP
#define MCPP_ARCHIVE_LAYOUT_HPP

#include "mcpp/core/dtype.hpp"
#include "mcpp/core/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace mcpp::archive {

inline constexpr std::uint32_t kChunkSide = 256;          // voxels per chunk axis
inline constexpr std::uint32_t kBlockSide = 16;           // voxels per block axis
inline constexpr std::uint32_t kBlocksPerAxis = kChunkSide / kBlockSide;  // 16
inline constexpr std::size_t   kBlocksPerChunk =
    std::size_t(kBlocksPerAxis) * kBlocksPerAxis * kBlocksPerAxis;        // 4096

// ceil(a / b) for unsigned.
constexpr std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) {
    return (a + b - 1) / b;
}

// Volume geometry: dims (z,y,x) at LOD0 + number of LODs.
struct Geometry {
    std::array<std::uint32_t, 3> dims{};  // [z, y, x] at LOD0
    std::uint8_t num_lods = 1;

    constexpr bool valid() const {
        if (num_lods < 1 || num_lods > kMaxLods) return false;
        for (auto d : dims) if (d == 0 || d > kMaxDim) return false;
        return true;
    }

    // dims at a given LOD (2x2x2 downscale, ceil; pre-aligned coord/2 per axis).
    constexpr std::array<std::uint32_t, 3> dims_at(std::uint8_t lod) const {
        std::array<std::uint32_t, 3> d = dims;
        for (auto& v : d) {
            std::uint32_t r = v >> lod;
            v = r ? r : 1u;  // never degenerate to 0
        }
        return d;
    }

    // chunk-grid extents (number of chunks along each axis) at a LOD.
    constexpr std::array<std::uint64_t, 3> chunk_extents(std::uint8_t lod) const {
        auto d = dims_at(lod);
        return { ceil_div(d[0], kChunkSide), ceil_div(d[1], kChunkSide),
                 ceil_div(d[2], kChunkSide) };
    }

    // number of chunk slots at a LOD.
    constexpr std::uint64_t chunks_at(std::uint8_t lod) const {
        auto e = chunk_extents(lod);
        return e[0] * e[1] * e[2];
    }

    // total chunk slots across all LODs (size of the occupancy map domain).
    constexpr std::uint64_t total_chunks() const {
        std::uint64_t t = 0;
        for (std::uint8_t l = 0; l < num_lods; ++l) t += chunks_at(l);
        return t;
    }

    // base slot index of a LOD's region in the global flat slot space.
    // Layout is COARSE-FIRST: the coarsest LOD (num_lods-1) comes first.
    constexpr std::uint64_t lod_base(std::uint8_t lod) const {
        std::uint64_t base = 0;
        for (std::uint8_t l = std::uint8_t(num_lods); l-- > lod + 1u;)
            base += chunks_at(l);
        return base;
    }

    // The STATIC SLOT MAPPING: global slot index of a chunk. Pure arithmetic.
    constexpr std::uint64_t slot_index(Lod lod, std::uint64_t cz, std::uint64_t cy,
                                       std::uint64_t cx) const {
        auto e = chunk_extents(lod.v);
        const std::uint64_t in_lod = (cz * e[1] + cy) * e[2] + cx;  // row-major z,y,x
        return lod_base(lod.v) + in_lod;
    }
};

}  // namespace mcpp::archive

#endif  // MCPP_ARCHIVE_LAYOUT_HPP
