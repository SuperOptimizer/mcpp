#include "mcpp/codec/codec.hpp"
#include "mcpp/codec/component_transform.hpp"
#include "mcpp/codec/convert.hpp"
#include "mcpp/test/test.hpp"

#include <bit>
#include <cstdint>
#include <span>
#include <vector>

using namespace mcpp;
using namespace mcpp::codec;

// ---- conversion edges ----------------------------------------------------

MCPP_TEST("convert: integer store rounds and saturates") {
    MCPP_CHECK(store_from_f32<std::uint8_t>(3.4f) == 3);
    MCPP_CHECK(store_from_f32<std::uint8_t>(3.6f) == 4);
    MCPP_CHECK(store_from_f32<std::uint8_t>(-5.0f) == 0);     // clamp low
    MCPP_CHECK(store_from_f32<std::uint8_t>(999.0f) == 255);  // clamp high
    MCPP_CHECK(store_from_f32<std::int16_t>(-40000.0f) == -32768);
    MCPP_CHECK(store_from_f32<std::int16_t>(40000.0f) == 32767);
}

MCPP_TEST("convert: float load scrubs NaN/Inf at ingest") {
    const float qnan = std::bit_cast<float>(std::uint32_t{0x7FC0'0000u});
    const float pinf = std::bit_cast<float>(std::uint32_t{0x7F80'0000u});
    // route through volatile so the optimizer can't pre-fold (fast-math)
    volatile float vn = qnan, vi = pinf;
    MCPP_CHECK(load_f32<float>(float(vn)) == 0.0f);
    MCPP_CHECK(load_f32<float>(float(vi)) == 0.0f);
    MCPP_CHECK(load_f32<float>(2.5f) == 2.5f);
}

// ---- component transform --------------------------------------------------

MCPP_TEST("component: YCoCg forward/inverse round-trips (f32)") {
    for (auto rgb : {std::array<float, 3>{255, 0, 0},
                     std::array<float, 3>{12, 200, 77},
                     std::array<float, 3>{0, 0, 0},
                     std::array<float, 3>{128, 128, 128}}) {
        std::array<float, 3> px = rgb;
        YCoCg::forward<3>(px);
        YCoCg::inverse<3>(px);
        for (std::size_t c = 0; c < 3; ++c) MCPP_CHECK(std::fabs(px[c] - rgb[c]) < 1e-3f);
    }
}

MCPP_TEST("component: Identity is a no-op") {
    std::array<float, 1> a{42.0f};
    Identity::forward<1>(a); Identity::inverse<1>(a);
    MCPP_CHECK(a[0] == 42.0f);
}

// ---- generic codec, the four real configs --------------------------------

static std::vector<std::uint8_t> noise_u8(unsigned seed, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    unsigned s = seed ? seed : 1u;
    for (auto& x : v) { s = s * 1664525u + 1013904223u; x = std::uint8_t((s >> 8) & 0xFF); }
    return v;
}

MCPP_TEST("codec: u8 3D volume (main case) end-to-end") {
    using Cdc = VolumeCodec<std::uint8_t>;
    auto vox = noise_u8(1, Cdc::count);
    std::vector<std::uint8_t> bytes;
    Cdc::encode(vox.data(), 4.0f, bytes);
    MCPP_CHECK(bytes.size() < vox.size());  // compresses

    std::vector<std::uint8_t> recon(Cdc::count);
    Cdc::decode(bytes.data(), bytes.size(), 4.0f, recon.data());

    std::vector<float> a(vox.begin(), vox.end()), b(recon.begin(), recon.end());
    double p = mcpp::test::psnr<float>(a, b, 255.0);
    MCPP_CHECK_GE(p, 25.0);
}

MCPP_TEST("codec: u16 volume works through the same generic path") {
    using Cdc = VolumeCodec<std::uint16_t>;
    std::vector<std::uint16_t> vox(Cdc::count);
    unsigned s = 5;
    for (auto& x : vox) { s = s * 1664525u + 1013904223u; x = std::uint16_t((s >> 8) & 0x0FFF); }
    std::vector<std::uint8_t> bytes;
    Cdc::encode(vox.data(), 8.0f, bytes);
    std::vector<std::uint16_t> recon(Cdc::count);
    Cdc::decode(bytes.data(), bytes.size(), 8.0f, recon.data());
    // bounded, coherent reconstruction
    double worst = 0;
    for (std::size_t i = 0; i < vox.size(); ++i)
        worst = std::max(worst, double(std::abs(int(vox[i]) - int(recon[i]))));
    MCPP_CHECK(worst < 4096.0);
}

MCPP_TEST("codec: RGB image (YCoCg) end-to-end, 3 interleaved channels") {
    using Cdc = RgbImgCodec<std::uint8_t>;
    auto img = noise_u8(2, Cdc::count);  // 64*64*3 interleaved
    std::vector<std::uint8_t> bytes;
    Cdc::encode(img.data(), 6.0f, bytes);
    std::vector<std::uint8_t> recon(Cdc::count);
    Cdc::decode(bytes.data(), bytes.size(), 6.0f, recon.data());

    std::vector<float> a(img.begin(), img.end()), b(recon.begin(), recon.end());
    double p = mcpp::test::psnr<float>(a, b, 255.0);
    MCPP_CHECK_GE(p, 18.0);  // noisy RGB at moderate q
}

MCPP_TEST("codec: f32 parametric surface (xyz, Identity) round-trips smoothly") {
    using Cdc = SurfaceCodec<float>;
    // smooth coordinate field: each (u,v) -> a slowly varying xyz
    std::vector<float> surf(Cdc::count);
    for (std::size_t v = 0; v < 64; ++v)
        for (std::size_t u = 0; u < 64; ++u) {
            std::size_t i = (v * 64 + u) * 3;
            surf[i + 0] = float(u) * 1.5f + 100.0f;     // x
            surf[i + 1] = float(v) * 1.5f + 200.0f;     // y
            surf[i + 2] = float(u + v) * 0.3f + 50.0f;  // z
        }
    std::vector<std::uint8_t> bytes;
    Cdc::encode(surf.data(), 0.25f, bytes);  // tight q for subvoxel coords
    std::vector<float> recon(Cdc::count);
    Cdc::decode(bytes.data(), bytes.size(), 0.25f, recon.data());

    std::span<const float> a{surf}, b{recon};
    double mx = mcpp::test::max_abs_error<float>(a, b);
    // smooth coordinate surface at tight q -> sub-voxel max error
    MCPP_CHECK_LE(mx, 1.0);
    MCPP_CHECK(bytes.size() < surf.size() * sizeof(float));  // compresses
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
