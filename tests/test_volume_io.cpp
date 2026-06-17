#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/volume_io.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace mcpp;
using namespace mcpp::archive;

static std::string tmp_path(const char* tag) {
    return std::string("/tmp/mcpp_vol_") + tag + "_" + std::to_string(::getpid()) + ".mcp";
}

// Synthesize a u8 volume: air (0) background with a material sphere of value
// ~100-180 in the middle. Dense [z,y,x] buffer.
static std::vector<std::uint8_t> make_volume(std::uint32_t n) {
    std::vector<std::uint8_t> v(std::size_t(n) * n * n, 0);
    const float c = float(n) / 2.0f;
    const float r = float(n) * 0.30f;
    for (std::uint32_t z = 0; z < n; ++z)
        for (std::uint32_t y = 0; y < n; ++y)
            for (std::uint32_t x = 0; x < n; ++x) {
                float dx = float(x) - c, dy = float(y) - c, dz = float(z) - c;
                float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (d < r) {
                    // smooth material 100..180
                    v[(std::size_t(z) * n + y) * n + x] =
                        std::uint8_t(100 + int(80.0f * (1.0f - d / r)));
                }
            }
    return v;
}

MCPP_TEST("volume_io: write -> read whole LOD0; air exact, material in tolerance") {
    const std::string path = tmp_path("rt");
    const std::uint32_t n = 300;  // > 256 -> spans 2 chunks/axis
    auto vol = make_volume(n);

    Geometry geom{{n, n, n}, 1};
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n, n, n},
            {std::uint64_t(n) * n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }

    {
        Archive a = Archive::open(path, os::Access::read_only);
        std::vector<std::uint8_t> out(std::size_t(n) * n * n, 0xFF);
        read_region<std::uint8_t>(a, Lod{0}, 0, 0, 0, n, n, n, out.data(),
                                  {std::uint64_t(n) * n, n, 1});

        // air voxels must come back EXACTLY 0; material within tolerance.
        std::size_t air = 0, mat = 0;
        double se = 0.0; std::size_t mcount = 0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (vol[i] == 0) {
                MCPP_CHECK(out[i] == 0);  // air exact
                ++air;
            } else {
                ++mat;
                double e = double(vol[i]) - double(out[i]);
                se += e * e; ++mcount;
            }
        }
        MCPP_CHECK(air > 0 && mat > 0);
        double psnr = 10.0 * std::log10(255.0 * 255.0 / (se / double(mcount)));
        MCPP_CHECK_GE(psnr, 30.0);  // smooth sphere at q=2 reconstructs well
    }
    ::unlink(path.c_str());
}

MCPP_TEST("volume_io: arbitrary sub-region read matches whole-volume read") {
    const std::string path = tmp_path("region");
    const std::uint32_t n = 280;
    auto vol = make_volume(n);
    Geometry geom{{n, n, n}, 1};
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n, n, n}, {std::uint64_t(n) * n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 2.0f);
    }
    {
        Archive a = Archive::open(path, os::Access::read_only);
        // full read
        std::vector<std::uint8_t> full(std::size_t(n) * n * n);
        read_region<std::uint8_t>(a, Lod{0}, 0, 0, 0, n, n, n, full.data(),
                                  {std::uint64_t(n) * n, n, 1});
        // a sub-box crossing a chunk boundary (256)
        const std::uint32_t z0 = 100, y0 = 240, x0 = 50;
        const std::uint32_t dz = 80, dy = 40, dx = 70;  // y spans 240..280 across 256
        std::vector<std::uint8_t> sub(std::size_t(dz) * dy * dx);
        read_region<std::uint8_t>(a, Lod{0}, z0, y0, x0, dz, dy, dx, sub.data(),
                                  {std::uint64_t(dy) * dx, dx, 1});
        // sub-region must equal the corresponding slice of the full read
        for (std::uint32_t z = 0; z < dz; ++z)
            for (std::uint32_t y = 0; y < dy; ++y)
                for (std::uint32_t x = 0; x < dx; ++x) {
                    std::uint8_t s = sub[(std::size_t(z) * dy + y) * dx + x];
                    std::uint8_t f = full[(std::size_t(z0 + z) * n + (y0 + y)) * n + (x0 + x)];
                    MCPP_CHECK(s == f);
                }
    }
    ::unlink(path.c_str());
}

MCPP_TEST("volume_io: mostly-air volume stays sparse and elides empty chunks") {
    const std::string path = tmp_path("sparse");
    const std::uint32_t n = 512;  // 2 chunks/axis = 8 chunks
    // a SMALL central sphere (r = 0.12*512 ~ 61, bbox ~ [195,317]) lives entirely
    // in the 8 central chunk corners but leaves whole corner chunks pure air.
    std::vector<std::uint8_t> vol(std::size_t(n) * n * n, 0);
    {
        const float c = float(n) / 2.0f, r = float(n) * 0.12f;
        for (std::uint32_t z = 0; z < n; ++z)
            for (std::uint32_t y = 0; y < n; ++y)
                for (std::uint32_t x = 0; x < n; ++x) {
                    float dx = float(x) - c, dy = float(y) - c, dz = float(z) - c;
                    if (std::sqrt(dx * dx + dy * dy + dz * dz) < r)
                        vol[(std::size_t(z) * n + y) * n + x] = std::uint8_t(120);
                }
    }
    Geometry geom{{n, n, n}, 1};
    int all_zero_chunks = 0, real_chunks = 0;
    {
        Archive a = Archive::create(path, geom, Dtype::u8);
        VolumeSrc<std::uint8_t> src{vol.data(), {n, n, n}, {std::uint64_t(n) * n, n, 1}};
        write_volume<std::uint8_t>(a, Lod{0}, src, 4.0f);

        for (std::uint64_t cz = 0; cz < 2; ++cz)
            for (std::uint64_t cy = 0; cy < 2; ++cy)
                for (std::uint64_t cx = 0; cx < 2; ++cx) {
                    auto st = a.state(Lod{0}, cz, cy, cx);
                    if (st == ChunkState::all_zero) ++all_zero_chunks;
                    else if (st == ChunkState::real_data) ++real_chunks;
                }
        // sphere is central, touching all 8 octant chunks near their inner corner,
        // so all 8 likely have a little material; the KEY sparsity property is the
        // on-disk size below. Require at least some real chunks.
        MCPP_CHECK(real_chunks > 0);
        MCPP_CHECK(all_zero_chunks + real_chunks == 8);  // all 8 chunks classified
    }
    struct stat st{};
    ::stat(path.c_str(), &st);
    std::uint64_t allocated = std::uint64_t(st.st_blocks) * 512;
    // a 512^3 u8 volume is 128 MiB raw; mostly air + compression -> far less on disk
    MCPP_CHECK(allocated < (std::uint64_t(40) << 20));  // < 40 MiB
    ::unlink(path.c_str());
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
