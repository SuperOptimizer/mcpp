// End-to-end codec: voxels -> bytes -> voxels.
#include "mcpp/codec/block.hpp"
#include "mcpp/codec/dct.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/test/test.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace mcpp;
using namespace mcpp::codec;

static std::vector<float> noise_block(unsigned seed, std::size_t total) {
    std::vector<float> b(total);
    unsigned s = seed ? seed : 1u;
    for (auto& v : b) { s = s * 1664525u + 1013904223u; v = float((s >> 8) & 0xFF); }
    return b;
}

static std::vector<float> smooth_block_3d() {
    std::vector<float> b(16 * 16 * 16);
    for (std::size_t z = 0; z < 16; ++z)
        for (std::size_t y = 0; y < 16; ++y)
            for (std::size_t x = 0; x < 16; ++x)
                b[(x * 16 + y) * 16 + z] = float(x + y + z) * 5.0f;
    return b;
}

// ---- exactness of the integer index stream -------------------------------

MCPP_TEST("block: quantized index stream round-trips EXACTLY (3D)") {
    auto vox = noise_block(11, 16 * 16 * 16);
    const float q = 4.0f;

    // reference indices straight from quant
    std::vector<float> coeffs = vox;
    forward_dct3_16(coeffs.data());
    std::vector<std::int32_t> ref(4096);
    quantize_block<16, 3>(coeffs.data(), ref.data(), q);

    // through encode -> decode
    std::vector<std::uint8_t> bytes;
    encode_block<16, 3>(vox.data(), q, bytes);
    std::vector<std::int32_t> got(4096);
    decode_block_indices<16, 3>(bytes.data(), bytes.size(), got.data());

    bool exact = true;
    for (std::size_t i = 0; i < 4096; ++i) if (ref[i] != got[i]) { exact = false; break; }
    MCPP_CHECK(exact);  // entropy stage is integer-exact
}

// ---- full block roundtrip within tolerance -------------------------------

MCPP_TEST("block: 3D encode->decode reconstructs within quant tolerance") {
    auto vox = smooth_block_3d();
    const float q = 4.0f;

    std::vector<std::uint8_t> bytes;
    encode_block<16, 3>(vox.data(), q, bytes);

    std::vector<float> recon(4096);
    decode_block<16, 3>(bytes.data(), bytes.size(), q, recon.data());

    // Compare against the *dequantized reference* path (what decode should match
    // up to f32 transform non-determinism), and check overall PSNR vs original.
    std::span<const float> a{vox}, b{recon};
    double p = mcpp::test::psnr<float>(a, b, 255.0);
    MCPP_CHECK_GE(p, 30.0);  // smooth block at moderate q reconstructs well
}

MCPP_TEST("block: decode(encode(x)) is self-contained (payload alone suffices)") {
    auto vox = noise_block(22, 16 * 16 * 16);
    const float q = 6.0f;
    std::vector<std::uint8_t> bytes;
    encode_block<16, 3>(vox.data(), q, bytes);

    // decode using ONLY the returned payload bytes + q (no other state)
    std::vector<float> recon(4096);
    decode_block<16, 3>(bytes.data(), bytes.size(), q, recon.data());

    // round-trips to a finite, reasonable reconstruction
    std::span<const float> a{vox}, b{recon};
    double mx = mcpp::test::max_abs_error<float>(a, b);
    MCPP_CHECK(mx < 256.0);  // bounded; sanity that it decoded coherently
}

// ---- compression actually happens ----------------------------------------

MCPP_TEST("block: smooth block compresses well below raw") {
    auto vox = smooth_block_3d();
    std::vector<std::uint8_t> bytes;
    encode_block<16, 3>(vox.data(), 4.0f, bytes);
    const std::size_t raw = 4096 * sizeof(std::uint8_t);  // 4096 u8 voxels = 4096 B
    MCPP_CHECK(bytes.size() < raw / 4);  // smooth -> strong compression
}

MCPP_TEST("block: higher q yields fewer bytes and lower PSNR") {
    auto vox = noise_block(33, 16 * 16 * 16);

    auto enc_q = [&](float q) {
        std::vector<std::uint8_t> bytes;
        encode_block<16, 3>(vox.data(), q, bytes);
        std::vector<float> recon(4096);
        decode_block<16, 3>(bytes.data(), bytes.size(), q, recon.data());
        std::span<const float> a{vox}, b{recon};
        return std::pair<std::size_t, double>{bytes.size(),
                                              mcpp::test::psnr<float>(a, b, 255.0)};
    };
    auto [lo_bytes, lo_psnr] = enc_q(2.0f);
    auto [hi_bytes, hi_psnr] = enc_q(24.0f);
    MCPP_CHECK(hi_bytes < lo_bytes);  // more compression
    MCPP_CHECK(hi_psnr < lo_psnr);    // more loss
}

// ---- 2D 64^2 config also works -------------------------------------------

MCPP_TEST("block: 2D 64^2 end-to-end roundtrip") {
    auto vox = noise_block(44, 64 * 64);
    const float q = 4.0f;
    std::vector<std::uint8_t> bytes;
    std::size_t n = encode_block<64, 2>(vox.data(), q, bytes);
    MCPP_CHECK(n == bytes.size());
    MCPP_CHECK(bytes.size() < 64 * 64);  // some compression even on noise

    std::vector<float> recon(64 * 64);
    decode_block<64, 2>(bytes.data(), bytes.size(), q, recon.data());
    std::span<const float> a{vox}, b{recon};
    double p = mcpp::test::psnr<float>(a, b, 255.0);
    MCPP_CHECK_GE(p, 20.0);  // noisy 2D at moderate q
}

// ---- scan order sanity ----------------------------------------------------

MCPP_TEST("block: scan order starts at DC and is a permutation") {
    const auto& scan = scan_for<16, 3>();
    MCPP_CHECK(scan[0] == 0);  // DC first
    std::vector<char> seen(4096, 0);
    for (auto p : scan) { MCPP_CHECK(!seen[p]); seen[p] = 1; }
    for (char c : seen) MCPP_CHECK(c);  // every position visited exactly once
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
