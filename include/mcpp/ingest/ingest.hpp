// Ingest — build a complete .mcp archive from a source volume.
//
// Rationale (design memory mcpp-everything-else): the zarr->.mcp tool. Pipeline:
//   source volume (LOD0) -> air-mask -> for each LOD: re-block + encode + write,
//   generating coarser LODs by 2x2x2 downsample with AIR PROPAGATION (mean over
//   NONZERO children only, so air doesn't bleed into material at coarse LODs).
//
// The actual S3/zarr fetch is behind the caller's choice of how it produces the
// LOD0 buffer (a ByteSource/zarr decoder layers on top, untested-without-network);
// this is the masking + downsample + multi-LOD write logic, which is the real
// work and is fully testable from an in-memory volume.
//
// "mcpp core never generates LODs" — true: this downsample lives in the INGEST
// TOOL, which is the external LOD provider handing all LODs to the archive writer.
#ifndef MCPP_INGEST_INGEST_HPP
#define MCPP_INGEST_INGEST_HPP

#include "mcpp/archive/archive.hpp"
#include "mcpp/archive/volume_io.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace mcpp::ingest {

// Histogram-valley air mask: values <= threshold become 0 (air). The original's
// mc_mask used a histogram-valley cut; here threshold is provided (a real tool
// computes it from the histogram). Operates in place on a u8 volume.
inline void air_mask(std::vector<std::uint8_t>& vol, std::uint8_t threshold) {
    for (auto& v : vol) if (v <= threshold) v = 0;
}

// 2x2x2 downsample with air propagation: each output voxel = mean of its up-to-8
// NONZERO children (0 if all children are air). dims halve (ceil), >=1.
inline std::vector<std::uint8_t> downsample_2x(const std::vector<std::uint8_t>& src,
                                               std::array<std::uint32_t,3> sd,
                                               std::array<std::uint32_t,3>& dd) {
    dd = { (sd[0]+1)/2 ? (sd[0]+1)/2 : 1u, (sd[1]+1)/2, (sd[2]+1)/2 };
    for (auto& v : dd) if (v == 0) v = 1;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(dd[0])*dd[1]*dd[2], 0);
    auto at = [&](std::uint32_t z, std::uint32_t y, std::uint32_t x) -> std::uint8_t {
        if (z >= sd[0] || y >= sd[1] || x >= sd[2]) return 0;
        return src[(std::size_t(z)*sd[1] + y)*sd[2] + x];
    };
    for (std::uint32_t z = 0; z < dd[0]; ++z)
        for (std::uint32_t y = 0; y < dd[1]; ++y)
            for (std::uint32_t x = 0; x < dd[2]; ++x) {
                std::uint32_t sum = 0, cnt = 0;
                for (int dz = 0; dz < 2; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            std::uint8_t v = at(2*z+std::uint32_t(dz),
                                                2*y+std::uint32_t(dy),
                                                2*x+std::uint32_t(dx));
                            if (v != 0) { sum += v; ++cnt; }  // NONZERO children only
                        }
                out[(std::size_t(z)*dd[1] + y)*dd[2] + x] =
                    cnt ? std::uint8_t(sum / cnt) : std::uint8_t(0);
            }
    return out;
}

// Ingest a u8 LOD0 volume into a fresh archive at `path`, generating `num_lods`
// LODs by repeated 2x downsample. Applies air masking with the given threshold.
inline archive::Archive ingest_volume(const std::string& path,
                                      std::vector<std::uint8_t> lod0,
                                      std::array<std::uint32_t,3> dims,
                                      std::uint8_t num_lods, std::uint8_t mask_threshold,
                                      float q) {
    air_mask(lod0, mask_threshold);
    archive::Geometry geom{dims, num_lods};
    archive::Archive a = archive::Archive::create(path, geom, Dtype::u8);

    // LOD0
    {
        archive::VolumeSrc<std::uint8_t> src{lod0.data(), dims,
            {std::uint64_t(dims[1])*dims[2], dims[2], 1}};
        archive::write_volume<std::uint8_t>(a, Lod{0}, src, q);
    }
    // coarser LODs by cascade
    std::vector<std::uint8_t> cur = std::move(lod0);
    std::array<std::uint32_t,3> cd = dims;
    for (std::uint8_t l = 1; l < num_lods; ++l) {
        std::array<std::uint32_t,3> nd;
        std::vector<std::uint8_t> down = downsample_2x(cur, cd, nd);
        archive::VolumeSrc<std::uint8_t> src{down.data(), nd,
            {std::uint64_t(nd[1])*nd[2], nd[2], 1}};
        archive::write_volume<std::uint8_t>(a, Lod{l}, src, q);
        cur = std::move(down); cd = nd;
    }
    return a;
}

}  // namespace mcpp::ingest

#endif  // MCPP_INGEST_INGEST_HPP
