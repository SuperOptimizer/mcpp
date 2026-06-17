#include "mcpp/codec/block.hpp"
#include "mcpp/codec/max_error.hpp"
#include "mcpp/test/test.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace mcpp;
using namespace mcpp::codec;

static std::vector<float> noise_block(unsigned seed) {
    std::vector<float> b(block_total<16,3>());
    unsigned s = seed ? seed : 1u;
    for (auto& v : b) { s = s*1664525u+1013904223u; v = float((s>>8)&0xFF); }
    return b;
}

MCPP_TEST("max_error: corrections bring every voxel within tau of reference decode") {
    auto orig = noise_block(7);
    const float q = 8.0f, tau = 2.0f;  // aggressive q -> some voxels exceed tau

    // encode then reference-decode (the encoder's own decode)
    std::vector<std::uint8_t> bytes;
    encode_block<16,3>(orig.data(), q, bytes);
    std::vector<float> decoded(orig.size());
    decode_block<16,3>(bytes.data(), bytes.size(), q, decoded.data());

    // some voxels exceed tau before correction
    double pre_max = 0;
    for (std::size_t i = 0; i < orig.size(); ++i)
        pre_max = std::max(pre_max, double(std::fabs(orig[i]-decoded[i])));
    MCPP_CHECK(pre_max > tau);  // q=8 on noise definitely exceeds tau=2

    // build + apply corrections against the reference decode
    std::vector<std::uint8_t> corr;
    build_corrections(orig.data(), decoded.data(), orig.size(), tau, corr);
    apply_corrections(corr.data(), corr.size(), decoded.data(), orig.size());

    // now EVERY voxel is within tau (best-effort bound on the reference decode)
    for (std::size_t i = 0; i < orig.size(); ++i)
        MCPP_CHECK(std::fabs(orig[i] - decoded[i]) <= tau + 1e-4);
}

MCPP_TEST("max_error: smooth data needs few corrections (sparse)") {
    // smooth ramp: lossy codec already nearly exact -> few out-of-tolerance voxels
    std::vector<float> smooth(block_total<16,3>());
    for (std::size_t z=0; z<16; ++z) for (std::size_t y=0; y<16; ++y) for (std::size_t x=0; x<16; ++x)
        smooth[(x*16+y)*16+z] = float(x+y+z)*4.0f;
    const float q = 2.0f, tau = 4.0f;

    std::vector<std::uint8_t> bytes;
    encode_block<16,3>(smooth.data(), q, bytes);
    std::vector<float> decoded(smooth.size());
    decode_block<16,3>(bytes.data(), bytes.size(), q, decoded.data());

    std::vector<std::uint8_t> corr;
    build_corrections(smooth.data(), decoded.data(), smooth.size(), tau, corr);
    // correction blob = 4 (count) + 8*count; smooth -> small
    std::uint32_t count = std::uint32_t(corr[0]) | (std::uint32_t(corr[1])<<8) |
                          (std::uint32_t(corr[2])<<16) | (std::uint32_t(corr[3])<<24);
    MCPP_CHECK(count < smooth.size() / 4);  // far fewer than all voxels
}

MCPP_TEST("max_error: tighter tau yields more (or equal) corrections") {
    auto orig = noise_block(99);
    const float q = 6.0f;
    std::vector<std::uint8_t> bytes;
    encode_block<16,3>(orig.data(), q, bytes);
    std::vector<float> decoded(orig.size());
    decode_block<16,3>(bytes.data(), bytes.size(), q, decoded.data());

    auto count_for = [&](float tau) {
        std::vector<std::uint8_t> c;
        build_corrections(orig.data(), decoded.data(), orig.size(), tau, c);
        return std::uint32_t(c[0])|(std::uint32_t(c[1])<<8)|(std::uint32_t(c[2])<<16)|(std::uint32_t(c[3])<<24);
    };
    MCPP_CHECK(count_for(1.0f) >= count_for(8.0f));  // tighter bound -> more fixes
}

MCPP_TEST("max_error: malformed correction blob is ignored (hardened)") {
    std::vector<float> v(block_total<16,3>(), 50.0f);
    // a blob claiming 1000000 fixes but only 4 bytes -> apply must bail safely
    std::vector<std::uint8_t> bad{0x40, 0x42, 0x0F, 0x00};  // count = 1000000
    apply_corrections(bad.data(), bad.size(), v.data(), v.size());
    for (float x : v) MCPP_CHECK(x == 50.0f);  // unchanged, no OOB
    // truncated entry
    std::vector<std::uint8_t> bad2{1,0,0,0, 5,0,0,0};  // count=1, but no delta bytes
    apply_corrections(bad2.data(), bad2.size(), v.data(), v.size());
    for (float x : v) MCPP_CHECK(x == 50.0f);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
