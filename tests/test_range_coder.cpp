// The range coder is PURE INTEGER => exact. Encoder and decoder must be exact
// mirror inverses. These are == tests (the one exact part of the codec).
#include "mcpp/codec/range_coder.hpp"
#include "mcpp/test/test.hpp"

#include <cstdint>
#include <vector>

using namespace mcpp::codec;

static std::vector<int> prng_bits(unsigned seed, std::size_t n, int p0_percent) {
    std::vector<int> bits(n);
    unsigned s = seed ? seed : 1u;
    for (auto& b : bits) {
        s = s * 1664525u + 1013904223u;
        b = (int((s >> 8) % 100) < p0_percent) ? 0 : 1;
    }
    return bits;
}

MCPP_TEST("rc: bypass bits round-trip exactly") {
    auto bits = prng_bits(1, 5000, 50);
    std::vector<std::uint8_t> buf;
    {
        RangeEncoder enc(buf);
        for (int b : bits) enc.encode_bypass(b);
        enc.finish();
    }
    RangeDecoder dec(buf.data(), buf.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
        MCPP_CHECK(dec.decode_bypass() == bits[i]);
}

MCPP_TEST("rc: modeled bits round-trip exactly (balanced)") {
    auto bits = prng_bits(2, 8000, 50);
    std::vector<std::uint8_t> buf;
    {
        RangeEncoder enc(buf);
        BitContext ctx;
        for (int b : bits) enc.encode_bit(ctx, b);
        enc.finish();
    }
    RangeDecoder dec(buf.data(), buf.size());
    BitContext ctx;
    for (std::size_t i = 0; i < bits.size(); ++i)
        MCPP_CHECK(dec.decode_bit(ctx) == bits[i]);
}

MCPP_TEST("rc: skewed source compresses AND round-trips exactly") {
    // 90% zeros: the adaptive model should compress well below 1 bit/symbol.
    auto bits = prng_bits(3, 20000, 90);
    std::vector<std::uint8_t> buf;
    {
        RangeEncoder enc(buf);
        BitContext ctx;
        for (int b : bits) enc.encode_bit(ctx, b);
        enc.finish();
    }
    // exact decode
    RangeDecoder dec(buf.data(), buf.size());
    BitContext ctx;
    bool ok = true;
    for (std::size_t i = 0; i < bits.size(); ++i)
        if (dec.decode_bit(ctx) != bits[i]) { ok = false; break; }
    MCPP_CHECK(ok);
    // compression: 20000 skewed bits should be well under 20000/8 bytes.
    double bits_per_sym = double(buf.size()) * 8.0 / double(bits.size());
    MCPP_CHECK(bits_per_sym < 0.8);   // ~entropy of a 90/10 source is ~0.47 bits
}

MCPP_TEST("rc: mixed modeled + bypass interleaved round-trips") {
    auto mbits = prng_bits(4, 3000, 70);
    auto bbits = prng_bits(5, 3000, 50);
    std::vector<std::uint8_t> buf;
    {
        RangeEncoder enc(buf);
        BitContext ctx;
        for (std::size_t i = 0; i < mbits.size(); ++i) {
            enc.encode_bit(ctx, mbits[i]);
            enc.encode_bypass(bbits[i]);
        }
        enc.finish();
    }
    RangeDecoder dec(buf.data(), buf.size());
    BitContext ctx;
    for (std::size_t i = 0; i < mbits.size(); ++i) {
        MCPP_CHECK(dec.decode_bit(ctx) == mbits[i]);
        MCPP_CHECK(dec.decode_bypass() == bbits[i]);
    }
}

MCPP_TEST("rc: all-zeros and all-ones edge cases") {
    for (int val : {0, 1}) {
        std::vector<std::uint8_t> buf;
        {
            RangeEncoder enc(buf);
            BitContext ctx;
            for (int i = 0; i < 1000; ++i) enc.encode_bit(ctx, val);
            enc.finish();
        }
        RangeDecoder dec(buf.data(), buf.size());
        BitContext ctx;
        for (int i = 0; i < 1000; ++i) MCPP_CHECK(dec.decode_bit(ctx) == val);
    }
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
