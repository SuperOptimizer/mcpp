#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/volume_io.hpp"
#include "mcpp/streaming/byte_source.hpp"
#include "mcpp/streaming/streaming_archive.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <vector>

using namespace mcpp;
using namespace mcpp::archive;
using namespace mcpp::streaming;

static std::string tmp_path(const char* tag) {
    return std::string("/tmp/mcpp_stream_") + tag + "_" + std::to_string(::getpid()) + ".mcp";
}

static std::vector<std::uint8_t> make_volume_r(std::uint32_t n, float rfrac) {
    std::vector<std::uint8_t> v(std::size_t(n) * n * n, 0);
    const float c = float(n) / 2.0f, r = float(n) * rfrac;
    for (std::uint32_t z = 0; z < n; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x) {
                float dx = float(x) - c, dy = float(y) - c, dz = float(z) - c;
                if (std::sqrt(dx*dx + dy*dy + dz*dz) < r)
                    v[(std::size_t(z)*n + y)*n + x] = std::uint8_t(120);
            }
    return v;
}

// Build a "remote" complete archive, then snapshot its occupancy+chunks regions
// into a byte image for a MemorySource. Returns (image, layout, geometry).
struct RemoteFixture {
    std::vector<std::uint8_t> image;
    RemoteLayout layout;
    Geometry geom;
};

static RemoteFixture make_remote_r(std::uint32_t n, float rfrac) {
    const std::string rpath = tmp_path("remote");
    Geometry geom{{n, n, n}, 1};
    auto vol = make_volume_r(n, rfrac);
    {
        Archive a = Archive::create(rpath, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }
    // Read the remote file raw and slice out occupancy + chunks regions. We know
    // the offsets via the same construction the archive uses.
    Archive ra = Archive::open(rpath, os::Access::read_only);
    (void)ra;
    // Re-derive offsets exactly as Archive::create does.
    const std::uint64_t occ_off = archive::kHeaderRegion + archive::kMetaRegion;
    const std::uint64_t occ_bytes =
        archive::align_up(OccupancyMap::bytes_for(geom.total_chunks()), 4096);
    const std::uint64_t chunks_off = occ_off + occ_bytes;
    const std::uint64_t stride = archive::slot_stride();
    const std::uint64_t total_len = chunks_off + geom.total_chunks() * stride;

    // load the whole remote file into memory (it is sparse on disk but read() of
    // holes yields zeros, which is exactly the remote byte image we want).
    std::FILE* f = std::fopen(rpath.c_str(), "rb");
    std::vector<std::uint8_t> image(total_len, 0);
    std::size_t rd = std::fread(image.data(), 1, total_len, f); (void)rd;
    std::fclose(f);
    ::unlink(rpath.c_str());

    RemoteFixture rf;
    rf.image = std::move(image);
    rf.geom = geom;
    rf.layout = RemoteLayout{occ_off, occ_bytes, chunks_off, stride};
    return rf;
}

MCPP_TEST("streaming: attach seeds ALL_ZERO; fetch populates REAL_DATA chunk") {
    const std::uint32_t n = 256;  // 1 chunk/axis
    RemoteFixture rf = make_remote_r(n, 0.30f);

    const std::string lpath = tmp_path("local");
    Archive local = Archive::create(lpath, rf.geom, Dtype::u8);
    MemorySource remote(rf.image);
    StreamingArchive<MemorySource> sa(local, remote, rf.layout);

    // before attach: everything DONT_KNOW
    MCPP_CHECK(local.state(Lod{0}, 0, 0, 0) == ChunkState::dont_know);

    auto ar = sa.attach();
    MCPP_CHECK(ar.has_value());
    MCPP_CHECK(sa.attached());

    // the single chunk has material (central sphere) -> remote says REAL_DATA, so
    // local stays DONT_KNOW until fetched.
    MCPP_CHECK(local.state(Lod{0}, 0, 0, 0) == ChunkState::dont_know);

    auto er = sa.ensure_chunk(Lod{0}, 0, 0, 0);
    MCPP_CHECK(er.has_value());
    MCPP_CHECK(*er == ChunkState::real_data);
    MCPP_CHECK(local.state(Lod{0}, 0, 0, 0) == ChunkState::real_data);

    // the fetched-into-local chunk decodes coherently
    ChunkView v = local.chunk(Lod{0}, 0, 0, 0);
    MCPP_CHECK(v.verify());  // integrity survived the remote->local copy
    ::unlink(lpath.c_str());
}

// A small blob placed in ONE octant (near chunk 0,0,0) so the opposite corner
// chunks are definitively pure air.
static RemoteFixture make_remote_corner_blob(std::uint32_t n) {
    const std::string rpath = tmp_path("remote");
    Geometry geom{{n, n, n}, 1};
    std::vector<std::uint8_t> vol(std::size_t(n) * n * n, 0);
    const float cx = 64, cy = 64, cz = 64, r = 40;  // blob near origin corner
    for (std::uint32_t z = 0; z < n; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x) {
                float dx = float(x) - cx, dy = float(y) - cy, dz = float(z) - cz;
                if (std::sqrt(dx*dx + dy*dy + dz*dz) < r)
                    vol[(std::size_t(z)*n + y)*n + x] = std::uint8_t(120);
            }
    {
        Archive a = Archive::create(rpath, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }
    const std::uint64_t occ_off = archive::kHeaderRegion + archive::kMetaRegion;
    const std::uint64_t occ_bytes =
        archive::align_up(OccupancyMap::bytes_for(geom.total_chunks()), 4096);
    const std::uint64_t chunks_off = occ_off + occ_bytes;
    const std::uint64_t stride = archive::slot_stride();
    const std::uint64_t total_len = chunks_off + geom.total_chunks() * stride;
    std::FILE* f = std::fopen(rpath.c_str(), "rb");
    std::vector<std::uint8_t> image(total_len, 0);
    std::size_t rd = std::fread(image.data(), 1, total_len, f); (void)rd;
    std::fclose(f);
    ::unlink(rpath.c_str());
    RemoteFixture rf;
    rf.image = std::move(image); rf.geom = geom;
    rf.layout = RemoteLayout{occ_off, occ_bytes, chunks_off, stride};
    return rf;
}

MCPP_TEST("streaming: ALL_ZERO chunks need no fetch (zero network)") {
    const std::uint32_t n = 512;  // 8 chunks; opposite corner is pure air
    RemoteFixture rf = make_remote_corner_blob(n);
    const std::string lpath = tmp_path("local2");
    Archive local = Archive::create(lpath, rf.geom, Dtype::u8);
    MemorySource remote(rf.image);
    StreamingArchive<MemorySource> sa(local, remote, rf.layout);
    MCPP_CHECK(sa.attach().has_value());

    // find a corner chunk that the remote marked ALL_ZERO; ensure_chunk on it
    // must NOT fetch (returns all_zero immediately).
    bool found_air = false;
    for (std::uint64_t cz = 0; cz < 2 && !found_air; ++cz)
        for (std::uint64_t cy = 0; cy < 2 && !found_air; ++cy)
            for (std::uint64_t cx = 0; cx < 2 && !found_air; ++cx) {
                if (local.state(Lod{0}, cz, cy, cx) == ChunkState::all_zero) {
                    auto er = sa.ensure_chunk(Lod{0}, cz, cy, cx);
                    MCPP_CHECK(er.has_value() && *er == ChunkState::all_zero);
                    found_air = true;
                }
            }
    MCPP_CHECK(found_air);  // attach correctly seeded at least one ALL_ZERO chunk
    ::unlink(lpath.c_str());
}

MCPP_TEST("streaming: ROBUST — transient fetch failure leaves chunk DONT_KNOW, no abort") {
    const std::uint32_t n = 256;
    RemoteFixture rf = make_remote_r(n, 0.30f);
    const std::string lpath = tmp_path("local3");
    Archive local = Archive::create(lpath, rf.geom, Dtype::u8);
    MemorySource remote(rf.image);
    StreamingArchive<MemorySource> sa(local, remote, rf.layout);
    MCPP_CHECK(sa.attach().has_value());

    remote.fail_next();  // simulate transport failure on the next read
    auto er = sa.ensure_chunk(Lod{0}, 0, 0, 0);
    MCPP_CHECK(!er.has_value());                 // returned an Error, did NOT abort
    MCPP_CHECK(er.error() == Error::network);
    MCPP_CHECK(local.state(Lod{0}, 0, 0, 0) == ChunkState::dont_know);  // unchanged
    MCPP_CHECK(sa.fetch_failures() == 1);

    // retry succeeds (robust: the failure was transient)
    auto er2 = sa.ensure_chunk(Lod{0}, 0, 0, 0);
    MCPP_CHECK(er2.has_value() && *er2 == ChunkState::real_data);
    ::unlink(lpath.c_str());
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
