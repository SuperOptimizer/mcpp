#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/volume_io.hpp"
#include "mcpp/ingest/ingest.hpp"
#include "mcpp/test/test.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <unistd.h>
#include <vector>

using namespace mcpp;
using namespace mcpp::ingest;
using namespace mcpp::archive;

static std::string tmp_path(const char* tag) {
    return std::string("/tmp/mcpp_ingest_") + tag + "_" + std::to_string(::getpid()) + ".mcp";
}

MCPP_TEST("ingest: air_mask zeros sub-threshold voxels") {
    std::vector<std::uint8_t> v{0, 5, 10, 50, 100, 200};
    air_mask(v, 10);
    MCPP_CHECK(v[0] == 0 && v[1] == 0 && v[2] == 0);  // <=10 -> air
    MCPP_CHECK(v[3] == 50 && v[4] == 100 && v[5] == 200);  // kept
}

MCPP_TEST("ingest: downsample 2x averages NONZERO children (air propagation)") {
    // 2x2x2 source: 4 material (100) + 4 air (0). dims {2,2,2} -> {1,1,1}.
    std::array<std::uint32_t,3> sd{2,2,2}, dd{};
    std::vector<std::uint8_t> src{100,100,0,0, 100,100,0,0};  // half material
    auto out = downsample_2x(src, sd, dd);
    MCPP_CHECK(dd[0]==1 && dd[1]==1 && dd[2]==1);
    // mean over the 4 NONZERO children = 100 (not 50 — air excluded)
    MCPP_CHECK(out[0] == 100);

    // all-air block downsamples to air
    std::vector<std::uint8_t> air(8, 0);
    auto o2 = downsample_2x(air, sd, dd);
    MCPP_CHECK(o2[0] == 0);
}

MCPP_TEST("ingest: full multi-LOD ingest; each LOD reads back, air exact") {
    const std::uint32_t n = 256;  // single chunk/axis at LOD0
    std::vector<std::uint8_t> vol(std::size_t(n)*n*n, 0);
    // a material box in the middle, plus some low-value noise that masking removes
    for (std::uint32_t z = 96; z < 160; ++z)
        for (std::uint32_t y = 96; y < 160; ++y)
            for (std::uint32_t x = 96; x < 160; ++x)
                vol[(std::size_t(z)*n + y)*n + x] = 150;
    // sprinkle sub-threshold "noise" (value 3) that masking should zero
    vol[0] = 3; vol[100] = 5; vol[200] = 8;

    const std::string path = tmp_path("full");
    Archive a = ingest_volume(path, vol, {n,n,n}, /*num_lods*/3, /*thresh*/10, /*q*/2.0f);

    // LOD0: read the material box back; air (incl. masked noise) exact 0.
    std::vector<std::uint8_t> out(std::size_t(n)*n*n, 0xAA);
    read_region<std::uint8_t>(a, Lod{0}, 0,0,0, n,n,n, out.data(),
                              {std::uint64_t(n)*n, n, 1});
    MCPP_CHECK(out[0] == 0);          // masked noise -> air, exact
    MCPP_CHECK(out[100] == 0);
    // a voxel deep in the material box is nonzero
    std::size_t mid = (std::size_t(128)*n + 128)*n + 128;
    MCPP_CHECK(out[mid] != 0);

    // LOD1 exists and has its (downsampled) dims; read a small region.
    auto d1 = a.geometry().dims_at(1);
    MCPP_CHECK(d1[0] == 128);
    std::vector<std::uint8_t> out1(static_cast<std::size_t>(d1[0])*d1[1]*d1[2], 0);
    read_region<std::uint8_t>(a, Lod{1}, 0,0,0, d1[0],d1[1],d1[2], out1.data(),
                              {std::uint64_t(d1[1])*d1[2], d1[2], 1});
    // material box center at LOD1 (~64,64,64) is nonzero
    std::size_t mid1 = (std::size_t(64)*d1[1] + 64)*d1[2] + 64;
    MCPP_CHECK(out1[mid1] != 0);
    // a corner at LOD1 is still air
    MCPP_CHECK(out1[0] == 0);
    ::unlink(path.c_str());
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
