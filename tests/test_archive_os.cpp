#include "mcpp/archive/layout.hpp"
#include "mcpp/archive/occupancy.hpp"
#include "mcpp/os/mmap.hpp"
#include "mcpp/test/test.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace mcpp;
using namespace mcpp::archive;

static std::string tmp_path(const char* tag) {
    return std::string("/tmp/mcpp_test_") + tag + "_" +
           std::to_string(::getpid()) + ".mcp";
}

// ---- sparse mmap ----------------------------------------------------------

MCPP_TEST("os: sparse file — huge logical size, tiny real disk usage") {
    const std::string path = tmp_path("sparse");
    const std::uint64_t logical = std::uint64_t(8) << 30;  // 8 GiB logical
    {
        auto m = os::MappedFile::open(path, logical, os::Access::read_write);
        MCPP_CHECK(m.valid());
        MCPP_CHECK(m.size() == logical);
        // write a few bytes near the END (offset ~7.5 GiB) — must NOT allocate
        // the 7.5 GiB gap before it.
        std::uint64_t off = (std::uint64_t(7) << 30);
        m.data()[off] = 0xAB;
        m.data()[off + 1] = 0xCD;
        m.sync(off, 4096);
    }
    // stat: logical size is 8 GiB, but allocated blocks are tiny (sparse).
    struct stat st{};
    MCPP_CHECK(::stat(path.c_str(), &st) == 0);
    MCPP_CHECK(std::uint64_t(st.st_size) == logical);    // logical 8 GiB
    // st_blocks is 512-byte units; sparse file uses far less than logical.
    std::uint64_t allocated = std::uint64_t(st.st_blocks) * 512;
    MCPP_CHECK(allocated < (std::uint64_t(64) << 20));   // < 64 MiB actually on disk
    ::unlink(path.c_str());
}

MCPP_TEST("os: holes read as zero; writes persist across reopen") {
    const std::string path = tmp_path("persist");
    const std::uint64_t logical = std::uint64_t(1) << 20;  // 1 MiB
    {
        auto m = os::MappedFile::open(path, logical, os::Access::read_write);
        // an untouched region reads as zero (hole)
        MCPP_CHECK(m.data()[500000] == 0);
        m.data()[12345] = 0x7E;
        m.sync(0, logical);
    }
    {
        auto m = os::MappedFile::open(path, logical, os::Access::read_only);
        MCPP_CHECK(m.data()[12345] == 0x7E);  // persisted
        MCPP_CHECK(m.data()[12346] == 0);     // still a hole
    }
    ::unlink(path.c_str());
}

// ---- static slot mapping --------------------------------------------------

MCPP_TEST("layout: dims_at does 2x2x2 downscale, never zero") {
    Geometry g{{1024, 512, 300}, 4};
    MCPP_CHECK(g.valid());
    auto d1 = g.dims_at(1);
    MCPP_CHECK(d1[0] == 512 && d1[1] == 256 && d1[2] == 150);
    auto d3 = g.dims_at(3);
    MCPP_CHECK(d3[0] == 128 && d3[1] == 64 && d3[2] == 37);  // 300>>3 = 37
}

MCPP_TEST("layout: slot mapping is a bijection over all chunk slots") {
    Geometry g{{700, 600, 500}, 3};  // small enough to enumerate
    std::uint64_t total = g.total_chunks();
    std::vector<char> seen(total, 0);

    for (std::uint8_t lod = 0; lod < g.num_lods; ++lod) {
        auto e = g.chunk_extents(lod);
        for (std::uint64_t cz = 0; cz < e[0]; ++cz)
            for (std::uint64_t cy = 0; cy < e[1]; ++cy)
                for (std::uint64_t cx = 0; cx < e[2]; ++cx) {
                    std::uint64_t s = g.slot_index(Lod{lod}, cz, cy, cx);
                    MCPP_CHECK(s < total);
                    MCPP_CHECK(!seen[s]);  // no collisions
                    seen[s] = 1;
                }
    }
    for (char c : seen) MCPP_CHECK(c);  // every slot hit exactly once (onto)
}

MCPP_TEST("layout: coarse-first — coarsest LOD occupies the lowest slots") {
    Geometry g{{2048, 2048, 2048}, 4};
    // coarsest LOD = 3 starts at slot 0
    MCPP_CHECK(g.lod_base(3) == 0);
    // each finer LOD starts after the coarser ones
    MCPP_CHECK(g.lod_base(2) == g.chunks_at(3));
    MCPP_CHECK(g.lod_base(0) > g.lod_base(1));  // finest is last/highest
}

MCPP_TEST("layout: 2^20 cap respected; oversize rejected") {
    Geometry ok{{kMaxDim, kMaxDim, kMaxDim}, 8};
    MCPP_CHECK(ok.valid());
    Geometry bad{{kMaxDim + 1, 10, 10}, 1};
    MCPP_CHECK(!bad.valid());
}

// ---- occupancy map --------------------------------------------------------

MCPP_TEST("occupancy: sparse-zero default is DONT_KNOW") {
    std::vector<std::uint8_t> buf(OccupancyMap::bytes_for(100), 0);  // zeroed = sparse
    OccupancyMap m(buf.data(), 100);
    for (std::uint64_t i = 0; i < 100; ++i)
        MCPP_CHECK(m.get(i) == ChunkState::dont_know);
}

MCPP_TEST("occupancy: 2-bit get/set round-trips for all states") {
    std::vector<std::uint8_t> buf(OccupancyMap::bytes_for(8), 0);
    OccupancyMap m(buf.data(), 8);
    m.set(0, ChunkState::real_data);
    m.set(1, ChunkState::all_zero);
    m.set(2, ChunkState::dont_know);
    m.set(3, ChunkState::reserved);
    m.set(7, ChunkState::real_data);
    MCPP_CHECK(m.get(0) == ChunkState::real_data);
    MCPP_CHECK(m.get(1) == ChunkState::all_zero);
    MCPP_CHECK(m.get(2) == ChunkState::dont_know);
    MCPP_CHECK(m.get(3) == ChunkState::reserved);
    MCPP_CHECK(m.get(7) == ChunkState::real_data);
    // neighbors within the same byte are independent
    MCPP_CHECK(m.get(4) == ChunkState::dont_know);
}

MCPP_TEST("occupancy: bytes_for packs 4 entries per byte") {
    MCPP_CHECK(OccupancyMap::bytes_for(1) == 1);
    MCPP_CHECK(OccupancyMap::bytes_for(4) == 1);
    MCPP_CHECK(OccupancyMap::bytes_for(5) == 2);
    MCPP_CHECK(OccupancyMap::bytes_for(4096) == 1024);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
