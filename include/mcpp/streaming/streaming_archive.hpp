// StreamingArchive — local sparse .mcp as a write-through cache of a remote.
//
// Rationale (design memory mcpp-streaming): the local .mcp is a partial mirror.
// On a DONT_KNOW chunk, fetch the WHOLE chunk from the remote ByteSource (one
// ranged read), write it into the same static slot locally, flip occupancy. Air
// chunks cost ZERO network because the remote occupancy map is fetched up front.
// Robust: any fetch failure leaves the chunk DONT_KNOW and returns an Error —
// streaming I/O never aborts (the one exception to abort-and-die).
//
// Assumes the remote object has the IDENTICAL static layout (same Geometry =>
// same slot arithmetic). For this reference implementation the remote is itself
// a serialized .mcp-shaped byte image: [occupancy region][chunk slots], whose
// offsets we know from the shared Geometry. (S3/https packed-wire optimization
// layers on later; this proves the write-through + robustness model.)
#ifndef MCPP_STREAMING_STREAMING_ARCHIVE_HPP
#define MCPP_STREAMING_STREAMING_ARCHIVE_HPP

#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/occupancy.hpp"
#include "mcpp/streaming/byte_source.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace mcpp::streaming {

// Layout of the remote byte image (shared with the local archive's chunk store):
//   [ occupancy bytes (one per chunk slot domain) ][ fixed-stride chunk slots ]
// The remote ships its occupancy map first; we fetch it at attach() so the
// client knows ALL_ZERO vs REAL_DATA without fetching chunk data.
struct RemoteLayout {
    std::uint64_t occupancy_off = 0;
    std::uint64_t occupancy_bytes = 0;
    std::uint64_t chunks_off = 0;
    std::uint64_t slot_stride = 0;
};

template <ByteSource Src>
class StreamingArchive {
public:
    StreamingArchive(archive::Archive& local, Src& remote, RemoteLayout layout)
        : local_(local), remote_(remote), layout_(layout) {}

    // Fetch the remote occupancy map up front and seed the local occupancy with
    // the authoritative ALL_ZERO / REAL_DATA states. After this, ALL_ZERO chunks
    // are known and never fetched. Robust: returns Error on transport failure.
    Result<void> attach() {
        auto r = remote_.read(layout_.occupancy_off, layout_.occupancy_bytes);
        if (!r) return std::unexpected(r.error());
        const Bytes& occ = *r;
        archive::OccupancyMap rmap(const_cast<std::uint8_t*>(occ.data()),
                                   local_.geometry().total_chunks());
        archive::OccupancyMap lmap = local_.occupancy();
        const std::uint64_t n = local_.geometry().total_chunks();
        for (std::uint64_t i = 0; i < n; ++i) {
            auto st = rmap.get(i);
            // Seed ALL_ZERO authoritatively; REAL_DATA stays DONT_KNOW locally
            // (not yet fetched) so reads trigger a fetch; ALL_ZERO costs no net.
            if (st == archive::ChunkState::all_zero)
                lmap.set(i, archive::ChunkState::all_zero);
            // REAL_DATA / DONT_KNOW remain local DONT_KNOW until fetched.
        }
        attached_ = true;
        return {};
    }

    // Ensure a chunk is locally present (fetch if DONT_KNOW). Returns the local
    // state after the attempt. Robust: a failed fetch leaves it DONT_KNOW.
    Result<archive::ChunkState> ensure_chunk(Lod lod, std::uint64_t cz,
                                             std::uint64_t cy, std::uint64_t cx) {
        const std::uint64_t slot = local_.geometry().slot_index(lod, cz, cy, cx);
        archive::OccupancyMap lmap = local_.occupancy();
        const auto st = lmap.get(slot);
        if (st == archive::ChunkState::all_zero ||
            st == archive::ChunkState::real_data)
            return st;  // already known locally, no fetch

        // DONT_KNOW: fetch the whole chunk slot from the remote.
        const std::uint64_t off = layout_.chunks_off + slot * layout_.slot_stride;
        auto r = remote_.read(off, layout_.slot_stride);
        if (!r) {
            ++fetch_failures_;
            return std::unexpected(r.error());  // stays DONT_KNOW; caller degrades
        }
        // write into the local slot + flip occupancy to REAL_DATA.
        const Bytes& bytes = *r;
        local_.write_chunk(lod, cz, cy, cx, bytes, archive::ChunkState::real_data);
        return archive::ChunkState::real_data;
    }

    bool attached() const { return attached_; }
    std::uint64_t fetch_failures() const { return fetch_failures_; }

private:
    archive::Archive& local_;
    Src& remote_;
    RemoteLayout layout_;
    bool attached_ = false;
    std::uint64_t fetch_failures_ = 0;
};

}  // namespace mcpp::streaming

#endif  // MCPP_STREAMING_STREAMING_ARCHIVE_HPP
