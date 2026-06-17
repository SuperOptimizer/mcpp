#include "mcpp/codec/airfill.hpp"
#include "mcpp/codec/mask_block.hpp"
#include "mcpp/test/test.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace mcpp;
using namespace mcpp::codec;

// A half-air block: a material "blob" in part of the volume, air elsewhere.
// Returns (voxels, mask). Material values in [1,255].
static void make_masked_block(std::vector<float>& vox, std::vector<std::uint8_t>& mask) {
    constexpr std::size_t N = 16;
    vox.assign(N * N * N, 0.0f);
    mask.assign(N * N * N, 0);
    for (std::size_t z = 0; z < N; ++z)
        for (std::size_t y = 0; y < N; ++y)
            for (std::size_t x = 0; x < N; ++x) {
                std::size_t i = (x * N + y) * N + z;
                // material in a sphere-ish region around the centre
                int dx = int(x) - 8, dy = int(y) - 8, dz = int(z) - 8;
                if (dx * dx + dy * dy + dz * dz < 36) {
                    vox[i] = float(40 + (dx * dx + dy * dy + dz * dz));  // 40..~75
                    mask[i] = 1;
                }
            }
}

MCPP_TEST("air: air_fill leaves material fixed, smooths air") {
    std::vector<float> vox; std::vector<std::uint8_t> mask;
    make_masked_block(vox, mask);
    std::vector<float> filled = vox;
    air_fill<16, 3>(filled.data(), mask.data());
    for (std::size_t i = 0; i < vox.size(); ++i) {
        if (mask[i]) MCPP_CHECK(filled[i] == vox[i]);  // material untouched
        // air voxels adjacent to material should now be > 0 (smoothed toward it)
    }
    // at least some air voxel got a nonzero fill (energy shaping happened)
    bool any_filled = false;
    for (std::size_t i = 0; i < vox.size(); ++i)
        if (!mask[i] && filled[i] != 0.0f) { any_filled = true; break; }
    MCPP_CHECK(any_filled);
}

MCPP_TEST("air: AIR BIT-EXACTNESS — air decodes to EXACTLY 0 (survives fast-math)") {
    std::vector<float> vox; std::vector<std::uint8_t> mask;
    make_masked_block(vox, mask);

    std::vector<std::uint8_t> bytes;
    encode_mask_block_framed<16, 3>(vox.data(), mask.data(), 4.0f, bytes);

    std::vector<float> recon(vox.size());
    decode_mask_block_framed<16, 3>(bytes.data(), bytes.size(), 4.0f, recon.data());

    // EVERY air voxel must be exactly 0.0f — bit-exact, not just small.
    std::size_t air = 0;
    for (std::size_t i = 0; i < vox.size(); ++i) {
        if (mask[i] == 0) {
            MCPP_CHECK(recon[i] == 0.0f);  // exact
            ++air;
        } else {
            MCPP_CHECK(recon[i] != 0.0f);  // material never collapses to air
        }
    }
    MCPP_CHECK(air > 0);  // there really were air voxels
}

MCPP_TEST("air: material reconstructs within tolerance while air is exact") {
    std::vector<float> vox; std::vector<std::uint8_t> mask;
    make_masked_block(vox, mask);

    std::vector<std::uint8_t> bytes;
    encode_mask_block_framed<16, 3>(vox.data(), mask.data(), 2.0f, bytes);
    std::vector<float> recon(vox.size());
    decode_mask_block_framed<16, 3>(bytes.data(), bytes.size(), 2.0f, recon.data());

    // PSNR over MATERIAL voxels only (air is exact by construction).
    std::vector<float> mo, mr;
    for (std::size_t i = 0; i < vox.size(); ++i)
        if (mask[i]) { mo.push_back(vox[i]); mr.push_back(recon[i]); }
    double p = mcpp::test::psnr<float>(mo, mr, 255.0);
    MCPP_CHECK_GE(p, 28.0);
}

MCPP_TEST("air: all-air block round-trips to all zeros") {
    std::vector<float> vox(16 * 16 * 16, 0.0f);
    std::vector<std::uint8_t> mask(16 * 16 * 16, 0);
    std::vector<std::uint8_t> bytes;
    encode_mask_block_framed<16, 3>(vox.data(), mask.data(), 4.0f, bytes);
    std::vector<float> recon(vox.size());
    decode_mask_block_framed<16, 3>(bytes.data(), bytes.size(), 4.0f, recon.data());
    for (float v : recon) MCPP_CHECK(v == 0.0f);
}

MCPP_TEST("air: all-material block has no forced zeros") {
    std::vector<float> vox(16 * 16 * 16);
    std::vector<std::uint8_t> mask(16 * 16 * 16, 1);
    for (std::size_t i = 0; i < vox.size(); ++i) vox[i] = float(50 + (i % 100));
    std::vector<std::uint8_t> bytes;
    encode_mask_block_framed<16, 3>(vox.data(), mask.data(), 4.0f, bytes);
    std::vector<float> recon(vox.size());
    decode_mask_block_framed<16, 3>(bytes.data(), bytes.size(), 4.0f, recon.data());
    for (float v : recon) MCPP_CHECK(v >= kMaterialFloor);  // none collapsed to air
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
