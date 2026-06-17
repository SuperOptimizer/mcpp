#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/volume_io.hpp"
#include "mcpp/cache/cache.hpp"
#include "mcpp/coordinator/coordinator.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <vector>

using namespace mcpp;
using namespace mcpp::coordinator;
using namespace mcpp::archive;
using cache::BlockKey;

static std::string tmp_path(const char* tag) {
    return std::string("/tmp/mcpp_coord_") + tag + "_" + std::to_string(::getpid()) + ".mcp";
}

// Volume: air background + a central material sphere.
static std::vector<std::uint8_t> make_volume(std::uint32_t n) {
    std::vector<std::uint8_t> v(std::size_t(n) * n * n, 0);
    const float c = float(n) / 2.0f, r = float(n) * 0.30f;
    for (std::uint32_t z = 0; z < n; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x) {
                float dx = float(x) - c, dy = float(y) - c, dz = float(z) - c;
                if (std::sqrt(dx*dx + dy*dy + dz*dz) < r)
                    v[(std::size_t(z)*n + y)*n + x] = std::uint8_t(120);
            }
    return v;
}

MCPP_TEST("coordinator: resolves misses -> populates cache from archive") {
    const std::string path = tmp_path("rt");
    const std::uint32_t n = 256;  // one chunk/axis = 16 blocks/axis
    auto vol = make_volume(n);
    Geometry geom{{n, n, n}, 1};
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }

    Archive a = Archive::open(path, os::Access::read_only);
    cache::Cache<std::uint8_t, 4096> cache;
    Coordinator<std::uint8_t, 4096> coord(a, cache);

    // A consumer "misses" a central block (material) and a corner block (air).
    // central voxel ~ (128,128,128) -> block (8,8,8); corner block (0,0,0).
    BlockKey center = BlockKey::make(Lod{0}, 8, 8, 8);
    BlockKey corner = BlockKey::make(Lod{0}, 0, 0, 0);
    std::vector<BlockKey> misses{center, corner};

    // before resolve: both miss
    MCPP_CHECK(cache.get(center) == nullptr);
    MCPP_CHECK(cache.get(corner) == nullptr);

    std::size_t filled = coord.resolve(misses);
    MCPP_CHECK(filled == 2);

    // after resolve: both present. center is material (nonzero), corner is air.
    const std::uint8_t* pc = cache.get(center);
    const std::uint8_t* pk = cache.get(corner);
    MCPP_CHECK(pc != nullptr);
    MCPP_CHECK(pk != nullptr);
    // center block has material voxels
    bool center_has_material = false;
    for (std::size_t i = 0; i < 4096; ++i) if (pc[i] != 0) { center_has_material = true; break; }
    MCPP_CHECK(center_has_material);
    // corner block is all air (canonical zero block)
    for (std::size_t i = 0; i < 4096; ++i) MCPP_CHECK(pk[i] == 0);
    ::unlink(path.c_str());
}

MCPP_TEST("coordinator: dedups repeated misses; canonical order") {
    const std::string path = tmp_path("dedup");
    const std::uint32_t n = 256;
    auto vol = make_volume(n);
    Geometry geom{{n,n,n}, 1};
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }
    Archive a = Archive::open(path, os::Access::read_only);
    cache::Cache<std::uint8_t, 64> cache;
    Coordinator<std::uint8_t, 64> coord(a, cache);

    BlockKey k = BlockKey::make(Lod{0}, 8, 8, 8);
    std::vector<BlockKey> misses{k, k, k, k};  // same key 4x (multiple consumers)
    std::size_t filled = coord.resolve(misses);
    MCPP_CHECK(filled == 1);                   // deduped to one
    MCPP_CHECK(cache.get(k) != nullptr);
    MCPP_CHECK(cache.live_count() == 1);
    ::unlink(path.c_str());
}

MCPP_TEST("coordinator: multi-tick refine (miss, resolve, hit next frame)") {
    const std::string path = tmp_path("tick");
    const std::uint32_t n = 256;
    auto vol = make_volume(n);
    Geometry geom{{n,n,n}, 1};
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n,n,n}, {std::uint64_t(n)*n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }
    Archive a = Archive::open(path, os::Access::read_only);
    cache::Cache<std::uint8_t, 256> cache;
    Coordinator<std::uint8_t, 256> coord(a, cache);

    // frame 1: consumer reads -> miss -> accumulate
    BlockKey k = BlockKey::make(Lod{0}, 7, 9, 6);
    std::vector<BlockKey> local_misses;
    if (cache.get(k) == nullptr) local_misses.push_back(k);
    MCPP_CHECK(local_misses.size() == 1);

    // tick boundary: coordinator resolves
    coord.resolve(local_misses);

    // frame 2: same read now hits
    MCPP_CHECK(cache.get(k) != nullptr);
    ::unlink(path.c_str());
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
