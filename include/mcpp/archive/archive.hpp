// The .mcp archive: file header + static sparse-mmap chunk store.
//
// File layout (all page-aligned regions):
//   [0)            file header (magic, version, dtype, dims, num_lods, offsets)
//   [meta)         user-metadata carve-out (large, free-form)
//   [occupancy)    3-state 2-bit/chunk map, own region (sparse)
//   [chunks)       fixed-stride chunk slots: slot s at chunks_base + s*kSlotStride
//
// Rationale (design memory mcpp-archive): the chunk slot is fixed-stride so
// slot_index -> byte offset is pure arithmetic. The stride is the worst-case
// chunk size (header + 4096 max block payloads), page-aligned. Sparsity makes the
// unused tail of each slot cost zero disk. Oversize = impossible by construction
// (lossy-shrinking codec) => assert-die.
#ifndef MCPP_ARCHIVE_ARCHIVE_HPP
#define MCPP_ARCHIVE_ARCHIVE_HPP

#include "mcpp/archive/chunk.hpp"
#include "mcpp/archive/layout.hpp"
#include "mcpp/archive/occupancy.hpp"
#include "mcpp/core/dtype.hpp"
#include "mcpp/core/error.hpp"
#include "mcpp/os/mmap.hpp"

#include <cstdint>
#include <cstring>
#include <string>

namespace mcpp::archive {

inline constexpr std::uint32_t kMagic   = 0x3050'434Du;  // "MCP0" little-endian-ish
inline constexpr std::uint32_t kVersion = 8;

// Worst-case bytes for one encoded block payload. A 16^3 block has 4096 samples;
// the codec is lossy-and-shrinking, but to size the fixed slot safely we bound a
// payload by raw f32 samples + generous entropy/coding overhead. (Real payloads
// are far smaller; the slack is sparse holes = free.)
inline constexpr std::uint64_t kMaxBlockPayload =
    std::uint64_t(codec::block_total<16, 3>()) * 4 + 4096;  // 4096*4 + slack
// Worst-case chunk = header + 4096 worst-case block payloads, page-aligned.
inline std::uint64_t slot_stride() {
    std::uint64_t raw = kChunkHeaderBytes + kBlocksPerChunk * kMaxBlockPayload;
    return align_up(raw, 4096);
}

// Fixed file header (kept within the first page).
struct Header {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint8_t  dtype;       // Dtype tag
    std::uint8_t  num_lods;
    std::uint8_t  pad0[2];
    std::uint32_t dims[3];     // z, y, x at LOD0
    std::uint64_t meta_off;
    std::uint64_t meta_len;
    std::uint64_t occupancy_off;
    std::uint64_t chunks_off;
    std::uint64_t slot_stride;
};

inline constexpr std::uint64_t kHeaderRegion = 4096;
inline constexpr std::uint64_t kMetaRegion   = 128 * 1024;  // 128 KiB user metadata

class Archive {
public:
    // Create a new archive sized for `geom` and `dtype`.
    static Archive create(const std::string& path, Geometry geom, Dtype dtype) {
        MCPP_ASSERT(geom.valid(), "archive: invalid geometry");
        Archive a;
        a.geom_ = geom;
        a.dtype_ = dtype;
        a.stride_ = slot_stride();

        const std::uint64_t occ_off = kHeaderRegion + kMetaRegion;
        const std::uint64_t occ_bytes =
            align_up(OccupancyMap::bytes_for(geom.total_chunks()), 4096);
        const std::uint64_t chunks_off = occ_off + occ_bytes;
        const std::uint64_t logical = chunks_off + geom.total_chunks() * a.stride_;

        a.file_ = os::MappedFile::open(path, logical, os::Access::read_write);
        a.occ_off_ = occ_off;
        a.chunks_off_ = chunks_off;

        // write header
        Header h{};
        h.magic = kMagic; h.version = kVersion;
        h.dtype = std::uint8_t(dtype); h.num_lods = geom.num_lods;
        h.dims[0] = geom.dims[0]; h.dims[1] = geom.dims[1]; h.dims[2] = geom.dims[2];
        h.meta_off = kHeaderRegion; h.meta_len = kMetaRegion;
        h.occupancy_off = occ_off; h.chunks_off = chunks_off; h.slot_stride = a.stride_;
        std::memcpy(a.file_.data(), &h, sizeof(h));
        a.file_.advise(chunks_off, logical - chunks_off, os::MappedFile::Advice::random);
        return a;
    }

    // Open an existing archive (read-write by default).
    static Archive open(const std::string& path, os::Access access = os::Access::read_write) {
        Archive a;
        a.file_ = os::MappedFile::open(path, 0, access);  // size from file
        Header h{};
        std::memcpy(&h, a.file_.data(), sizeof(h));
        MCPP_ASSERT(h.magic == kMagic, "archive: bad magic");
        MCPP_ASSERT(h.version == kVersion, "archive: version mismatch");
        a.geom_.dims[0] = h.dims[0]; a.geom_.dims[1] = h.dims[1]; a.geom_.dims[2] = h.dims[2];
        a.geom_.num_lods = h.num_lods;
        a.dtype_ = Dtype(h.dtype);
        a.occ_off_ = h.occupancy_off; a.chunks_off_ = h.chunks_off; a.stride_ = h.slot_stride;
        return a;
    }

    const Geometry& geometry() const { return geom_; }
    Dtype dtype() const { return dtype_; }

    OccupancyMap occupancy() {
        return OccupancyMap(file_.data() + occ_off_, geom_.total_chunks());
    }

    ChunkState state(Lod lod, std::uint64_t cz, std::uint64_t cy, std::uint64_t cx) {
        return occupancy().get(geom_.slot_index(lod, cz, cy, cx));
    }

    // Write a chunk's bytes into its static slot and mark occupancy. `bytes` must
    // fit the slot stride (guaranteed by construction; assert otherwise).
    void write_chunk(Lod lod, std::uint64_t cz, std::uint64_t cy, std::uint64_t cx,
                     const std::vector<std::uint8_t>& bytes, ChunkState st) {
        const std::uint64_t slot = geom_.slot_index(lod, cz, cy, cx);
        MCPP_ASSERT(bytes.size() <= stride_, "archive: chunk exceeds slot stride");
        if (st == ChunkState::real_data && !bytes.empty()) {
            std::uint8_t* dst = file_.data() + chunks_off_ + slot * stride_;
            std::memcpy(dst, bytes.data(), bytes.size());
            file_.sync(chunks_off_ + slot * stride_, bytes.size());
        }
        occupancy().set(slot, st);
    }

    // Read-only view of a chunk's bytes (only valid if state == real_data).
    ChunkView chunk(Lod lod, std::uint64_t cz, std::uint64_t cy, std::uint64_t cx) {
        const std::uint64_t slot = geom_.slot_index(lod, cz, cy, cx);
        const std::uint8_t* base = file_.data() + chunks_off_ + slot * stride_;
        return ChunkView(base, std::size_t(stride_));
    }

    // Pointer to the user-metadata region.
    std::uint8_t* metadata() { return file_.data() + kHeaderRegion; }
    std::uint64_t metadata_size() const { return kMetaRegion; }

private:
    os::MappedFile file_;
    Geometry geom_;
    Dtype dtype_ = Dtype::u8;
    std::uint64_t occ_off_ = 0;
    std::uint64_t chunks_off_ = 0;
    std::uint64_t stride_ = 0;
};

}  // namespace mcpp::archive

#endif  // MCPP_ARCHIVE_ARCHIVE_HPP
