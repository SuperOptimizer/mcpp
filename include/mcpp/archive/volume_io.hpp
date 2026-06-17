// Volume-level convenience: write a whole volume into an archive LOD, and read
// back an arbitrary sub-region.
//
// This is the usable API a consumer (ingest tool, VC3D) calls. It sits on the
// chunk/block primitives: a volume is tiled into 256^3 chunks, each chunk into
// 4096 16^3 blocks; air voxels (value 0) form the per-block mask so air handling
// + ALL_ZERO chunk/block elision happen automatically.
//
// Rank-3, single-channel (the archive is 3D grayscale only). Voxels are f32 here
// (the codec compute type); the dtype<->f32 conversion is the codec's edge — the
// volume API works in the natural sample type T and converts per block.
//
// Conventions: coordinates are [z,y,x]; the input volume is a dense buffer with
// caller-supplied strides (in elements). A chunk that is entirely air is stored
// as ALL_ZERO (no payload); a block that is entirely air is the ALL_ZERO
// sentinel within its chunk.
#ifndef MCPP_ARCHIVE_VOLUME_IO_HPP
#define MCPP_ARCHIVE_VOLUME_IO_HPP

#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/chunk.hpp"
#include "mcpp/codec/convert.hpp"
#include "mcpp/codec/mask_block.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace mcpp::archive {

// A dense 3D volume view the caller provides: pointer + dims[z,y,x] + element
// strides[z,y,x] (in elements). T is the sample type.
template <Sample T>
struct VolumeSrc {
    const T* data;
    std::array<std::uint32_t, 3> dims;     // z, y, x
    std::array<std::uint64_t, 3> stride;   // element strides for z, y, x

    T at(std::uint32_t z, std::uint32_t y, std::uint32_t x) const {
        return data[z * stride[0] + y * stride[1] + x * stride[2]];
    }
};

// Write one whole volume into LOD `lod` of the archive at quality q.
// Tiles into 256^3 chunks; each chunk into 4096 16^3 blocks with air masks.
template <Sample T>
void write_volume(Archive& a, Lod lod, const VolumeSrc<T>& src, float q) {
    const auto cext = a.geometry().chunk_extents(lod.v);
    const auto ldims = a.geometry().dims_at(lod.v);
    // the caller's src dims must match the archive's LOD dims
    MCPP_ASSERT(src.dims[0] == ldims[0] && src.dims[1] == ldims[1] &&
                src.dims[2] == ldims[2], "write_volume: dims mismatch for LOD");

    constexpr std::size_t B = kBlockSide;          // 16

    std::vector<float> blk(B * B * B);
    std::vector<std::uint8_t> mask(B * B * B);

    for (std::uint64_t cz = 0; cz < cext[0]; ++cz)
    for (std::uint64_t cy = 0; cy < cext[1]; ++cy)
    for (std::uint64_t cx = 0; cx < cext[2]; ++cx) {
        ChunkBuilder cb; cb.set_quality(q);
        bool chunk_has_material = false;

        for (std::size_t bz = 0; bz < kBlocksPerAxis; ++bz)
        for (std::size_t by = 0; by < kBlocksPerAxis; ++by)
        for (std::size_t bx = 0; bx < kBlocksPerAxis; ++bx) {
            // fill this 16^3 block from the volume (zero-padded at edges)
            bool block_has_material = false;
            for (std::size_t lz = 0; lz < B; ++lz)
            for (std::size_t ly = 0; ly < B; ++ly)
            for (std::size_t lx = 0; lx < B; ++lx) {
                const std::uint64_t vz = cz * kChunkSide + bz * B + lz;
                const std::uint64_t vy = cy * kChunkSide + by * B + ly;
                const std::uint64_t vx = cx * kChunkSide + bx * B + lx;
                const std::size_t li = (lz * B + ly) * B + lx;
                float v = 0.0f;
                if (vz < src.dims[0] && vy < src.dims[1] && vx < src.dims[2])
                    v = codec::load_f32<T>(src.at(std::uint32_t(vz), std::uint32_t(vy),
                                                  std::uint32_t(vx)));
                blk[li] = v;
                const bool air = (v == 0.0f);
                mask[li] = air ? 0 : 1;
                if (!air) block_has_material = true;
            }

            const std::size_t bidx = (bz * kBlocksPerAxis + by) * kBlocksPerAxis + bx;
            if (!block_has_material) {
                cb.set_block(bidx, {});  // ALL_ZERO sentinel
            } else {
                std::vector<std::uint8_t> bytes;
                codec::encode_mask_block_framed<16, 3>(blk.data(), mask.data(), q, bytes);
                cb.set_block(bidx, bytes);
                chunk_has_material = true;
            }
        }

        if (chunk_has_material)
            a.write_chunk(lod, cz, cy, cx, cb.finish(), ChunkState::real_data);
        else
            a.write_chunk(lod, cz, cy, cx, {}, ChunkState::all_zero);
    }
}

// Read an arbitrary sub-region [z0,z0+dz) x [y0..] x [x0..] from LOD `lod` into a
// dense output buffer `out` with element strides `ostride` (z,y,x). Out-of-data
// chunks (ALL_ZERO or DONT_KNOW) yield zeros. Decodes only the chunks/blocks the
// region overlaps (faulting only their pages).
template <Sample T>
void read_region(Archive& a, Lod lod,
                 std::uint32_t z0, std::uint32_t y0, std::uint32_t x0,
                 std::uint32_t dz, std::uint32_t dy, std::uint32_t dx,
                 T* out, const std::array<std::uint64_t, 3>& ostride) {
    constexpr std::size_t B = kBlockSide;
    // zero the output first (air / absent regions stay zero)
    for (std::uint32_t z = 0; z < dz; ++z)
        for (std::uint32_t y = 0; y < dy; ++y)
            for (std::uint32_t x = 0; x < dx; ++x)
                out[z * ostride[0] + y * ostride[1] + x * ostride[2]] = T(0);

    // iterate over chunks overlapping the region
    const std::uint64_t cz0 = z0 / kChunkSide, cz1 = (z0 + dz - 1) / kChunkSide;
    const std::uint64_t cy0 = y0 / kChunkSide, cy1 = (y0 + dy - 1) / kChunkSide;
    const std::uint64_t cx0 = x0 / kChunkSide, cx1 = (x0 + dx - 1) / kChunkSide;

    std::vector<float> blk(B * B * B);

    for (std::uint64_t cz = cz0; cz <= cz1; ++cz)
    for (std::uint64_t cy = cy0; cy <= cy1; ++cy)
    for (std::uint64_t cx = cx0; cx <= cx1; ++cx) {
        if (a.state(lod, cz, cy, cx) != ChunkState::real_data) continue;  // zeros
        ChunkView v = a.chunk(lod, cz, cy, cx);

        for (std::size_t bz = 0; bz < kBlocksPerAxis; ++bz)
        for (std::size_t by = 0; by < kBlocksPerAxis; ++by)
        for (std::size_t bx = 0; bx < kBlocksPerAxis; ++bx) {
            const std::size_t bidx = (bz * kBlocksPerAxis + by) * kBlocksPerAxis + bx;
            // block's voxel origin in this LOD
            const std::uint64_t boz = cz * kChunkSide + bz * B;
            const std::uint64_t boy = cy * kChunkSide + by * B;
            const std::uint64_t box = cx * kChunkSide + bx * B;
            // skip blocks that don't overlap the region
            if (boz + B <= z0 || boz >= z0 + dz) continue;
            if (boy + B <= y0 || boy >= y0 + dy) continue;
            if (box + B <= x0 || box >= x0 + dx) continue;

            if (!v.block_present(bidx)) continue;  // ALL_ZERO block -> stays zero
            auto payload = v.block(bidx);
            codec::decode_mask_block_framed<16, 3>(payload.data(), payload.size(),
                                                   v.quality(), blk.data());

            // scatter the overlapping voxels into out
            for (std::size_t lz = 0; lz < B; ++lz) {
                const std::uint64_t vz = boz + lz;
                if (vz < z0 || vz >= z0 + dz) continue;
                for (std::size_t ly = 0; ly < B; ++ly) {
                    const std::uint64_t vy = boy + ly;
                    if (vy < y0 || vy >= y0 + dy) continue;
                    for (std::size_t lx = 0; lx < B; ++lx) {
                        const std::uint64_t vx = box + lx;
                        if (vx < x0 || vx >= x0 + dx) continue;
                        const std::size_t li = (lz * B + ly) * B + lx;
                        const std::uint64_t oz = vz - z0, oy = vy - y0, ox = vx - x0;
                        out[oz * ostride[0] + oy * ostride[1] + ox * ostride[2]] =
                            codec::store_from_f32<T>(blk[li]);
                    }
                }
            }
        }
    }
}

}  // namespace mcpp::archive

#endif  // MCPP_ARCHIVE_VOLUME_IO_HPP
