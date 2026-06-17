#include "mcpp/codec/dct.hpp"
#include "mcpp/codec/quant.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using namespace mcpp;
using namespace mcpp::codec;

// ---- scalar quantizer properties -----------------------------------------

MCPP_TEST("quant: dead-zone maps small coefficients to zero") {
    const float step = 10.0f;
    // |c| below DZ_FRAC*step (=8.0) must quantize to 0.
    MCPP_CHECK(quantize(0.0f, step) == 0);
    MCPP_CHECK(quantize(5.0f, step) == 0);
    MCPP_CHECK(quantize(-7.9f, step) == 0);
    // beyond the dead zone, nonzero
    MCPP_CHECK(quantize(9.0f, step) != 0);
    MCPP_CHECK(quantize(-50.0f, step) < 0);
}

MCPP_TEST("quant: quantize/dequantize round-trip error is bounded by ~step") {
    const float step = 4.0f;
    double worst = 0.0;
    for (float c = -100.0f; c <= 100.0f; c += 0.25f) {
        std::int32_t i = quantize(c, step);
        float r = dequantize(i, step);
        // coefficients that survive the dead zone reconstruct within ~one step;
        // coefficients killed by the dead zone have error up to the dz threshold.
        worst = std::max(worst, double(std::fabs(c - r)));
    }
    // dead-zone kill can cost up to ~ (DZ_FRAC + recon)*step in the worst case.
    MCPP_CHECK_LE(worst, double(step) * 1.5);
}

MCPP_TEST("quant: sign is preserved") {
    const float step = 3.0f;
    MCPP_CHECK(quantize(20.0f, step) > 0);
    MCPP_CHECK(quantize(-20.0f, step) < 0);
    MCPP_CHECK(dequantize(quantize(20.0f, step), step) > 0.0f);
    MCPP_CHECK(dequantize(quantize(-20.0f, step), step) < 0.0f);
}

MCPP_TEST("quant: band weighting gives DC a smaller step than high freq") {
    const float q = 2.0f;
    float dc_step = step_for<16, 3>(0, q);
    float hi_step = step_for<16, 3>(max_band<16, 3>(), q);
    MCPP_CHECK(hi_step > dc_step);              // HF coarser
    MCPP_CHECK(std::fabs(dc_step - kBaseStep * q) < 1e-5f);  // DC weight == 1
}

// ---- full pipeline: transform -> quant -> dequant -> inverse -------------

static std::vector<float> make_block_3d(unsigned seed) {
    std::vector<float> b(16 * 16 * 16);
    unsigned s = seed ? seed : 1u;
    for (auto& v : b) {
        s = s * 1664525u + 1013904223u;
        v = float((s >> 8) & 0xFF);  // [0,255]
    }
    return b;
}

struct RD {
    double psnr;
    double zero_frac;  // fraction of quantized indices that are 0 (proxy for rate)
};

static RD roundtrip_rd(const std::vector<float>& orig, float q) {
    std::vector<float> coeffs = orig;
    forward_dct3_16(coeffs.data());

    std::vector<std::int32_t> idx(orig.size());
    quantize_block<16, 3>(coeffs.data(), idx.data(), q);

    std::size_t zeros = 0;
    for (auto v : idx) if (v == 0) ++zeros;

    std::vector<float> deq(orig.size());
    dequantize_block<16, 3>(idx.data(), deq.data(), q);
    inverse_dct3_16(deq.data());

    std::span<const float> a{orig}, b{deq};
    return RD{ mcpp::test::psnr<float>(a, b, 255.0),
              double(zeros) / double(idx.size()) };
}

MCPP_TEST("quant: pipeline produces a real rate/quality tradeoff") {
    auto orig = make_block_3d(2024);
    RD lo = roundtrip_rd(orig, 1.0f);   // high quality
    RD hi = roundtrip_rd(orig, 16.0f);  // aggressive

    // Higher q => more zeros (more compression) AND lower PSNR (more loss).
    MCPP_CHECK(hi.zero_frac > lo.zero_frac);
    MCPP_CHECK(hi.psnr < lo.psnr);
    // Sanity: even aggressive q keeps some structure; gentle q is decent.
    MCPP_CHECK_GE(lo.psnr, 28.0);     // random data is hard; this is a floor
    MCPP_CHECK(lo.zero_frac < 0.1);   // gentle q keeps most coefficients
    MCPP_CHECK(hi.zero_frac > 0.3);   // aggressive q zeros a large share
    // and a very aggressive q on white noise should zero the majority.
    RD vhi = roundtrip_rd(orig, 64.0f);
    MCPP_CHECK(vhi.zero_frac > 0.5);
}

MCPP_TEST("quant: monotonic — increasing q never increases PSNR") {
    auto orig = make_block_3d(99);
    double prev = 1e9;
    for (float q : {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f}) {
        RD rd = roundtrip_rd(orig, q);
        // allow tiny non-monotonic wobble from f32, but trend must hold
        MCPP_CHECK(rd.psnr <= prev + 0.5);
        prev = rd.psnr;
    }
}

MCPP_TEST("quant: smooth block compresses far better than noise") {
    // a smooth gradient block
    std::vector<float> smooth(16 * 16 * 16);
    for (std::size_t z = 0; z < 16; ++z)
        for (std::size_t y = 0; y < 16; ++y)
            for (std::size_t x = 0; x < 16; ++x)
                smooth[(x * 16 + y) * 16 + z] = float(x + y + z) * 4.0f;
    auto noise = make_block_3d(7);

    RD s = roundtrip_rd(smooth, 4.0f);
    RD n = roundtrip_rd(noise, 4.0f);
    MCPP_CHECK(s.zero_frac > n.zero_frac);   // smooth -> more zeros
    MCPP_CHECK(s.psnr > n.psnr);             // smooth -> better quality at same q
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
