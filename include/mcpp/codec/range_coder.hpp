// Adaptive binary range coder (LZMA-style).
//
// Rationale (design memory mcpp-original-codec-facts): the codec entropy stage is
// an adaptive binary range coder with trained context priors. This is the core
// arithmetic coder; the coefficient model (significance/sign/magnitude) layers on
// top in coef_coder.hpp.
//
// IMPORTANT determinism note: unlike the f32 transform, the entropy stage is
// PURE INTEGER and therefore EXACT and deterministic. Encoder and decoder are
// exact mirror inverses: decode(encode(symbols)) == symbols, byte-for-byte. This
// is the one part of the codec that DOES get exact (==) tests.
//
// Design: 32-bit range coder with carry handling, 11-bit probability contexts
// (range [1,2047], total 2048), adaptation shift 4 (matching the original).
#ifndef MCPP_CODEC_RANGE_CODER_HPP
#define MCPP_CODEC_RANGE_CODER_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpp::codec {

inline constexpr int   kProbBits  = 11;
inline constexpr std::uint32_t kProbOne = 1u << kProbBits;   // 2048
inline constexpr std::uint32_t kProbInit = kProbOne / 2;     // 1024 (p=0.5)
inline constexpr int   kAdaptShift = 4;                      // adaptation rate
inline constexpr std::uint32_t kTopValue = 1u << 24;         // renormalization

// An adaptive bit-probability context: P(bit==0), 11-bit fixed point.
struct BitContext {
    std::uint16_t p0 = std::uint16_t(kProbInit);

    void update(int bit) noexcept {
        if (bit == 0) p0 = std::uint16_t(p0 + ((kProbOne - p0) >> kAdaptShift));
        else          p0 = std::uint16_t(p0 - (p0 >> kAdaptShift));
    }
};

// ---- encoder -------------------------------------------------------------

class RangeEncoder {
public:
    explicit RangeEncoder(std::vector<std::uint8_t>& out) : out_(out) {}

    // Encode one bit under an adaptive context.
    void encode_bit(BitContext& ctx, int bit) {
        const std::uint32_t bound = (range_ >> kProbBits) * ctx.p0;
        if (bit == 0) {
            range_ = bound;
        } else {
            low_ += bound;
            range_ -= bound;
        }
        ctx.update(bit);
        normalize();
    }

    // Encode one bit with fixed p=0.5 (bypass — no model).
    void encode_bypass(int bit) {
        range_ >>= 1;
        if (bit) low_ += range_;
        normalize();
    }

    void finish() {
        for (int i = 0; i < 5; ++i) { shift_low(); }
    }

private:
    void normalize() {
        while (range_ < kTopValue) {
            shift_low();
            range_ <<= 8;
        }
    }
    // Standard LZMA shift_low. The very first call emits the initial cache byte
    // (0), which the decoder discards on priming. cache_size_ starts at 1 so the
    // leading byte is always produced exactly once.
    void shift_low() {
        if (std::uint32_t(low_ >> 32) != 0 || low_ < 0xFF000000u) {
            std::uint8_t carry = std::uint8_t(low_ >> 32);
            do {
                out_.push_back(std::uint8_t(cache_ + carry));
                cache_ = 0xFF;
            } while (--cache_size_ != 0);
            cache_ = std::uint8_t(low_ >> 24);
        }
        ++cache_size_;
        low_ = (low_ << 8) & 0xFFFFFFFFu;
    }

    std::vector<std::uint8_t>& out_;
    std::uint64_t low_ = 0;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::uint8_t  cache_ = 0;
    std::uint64_t cache_size_ = 1;
};

// ---- decoder -------------------------------------------------------------

class RangeDecoder {
public:
    RangeDecoder(const std::uint8_t* data, std::size_t size)
        : data_(data), size_(size) {
        // prime the code register with the first 5 bytes (mirrors finish()).
        next_byte();  // discard the first (the encoder's initial cache slot)
        for (int i = 0; i < 4; ++i) code_ = (code_ << 8) | next_byte();
    }

    int decode_bit(BitContext& ctx) {
        const std::uint32_t bound = (range_ >> kProbBits) * ctx.p0;
        int bit;
        if (code_ < bound) {
            range_ = bound;
            bit = 0;
        } else {
            code_ -= bound;
            range_ -= bound;
            bit = 1;
        }
        ctx.update(bit);
        normalize();
        return bit;
    }

    int decode_bypass() {
        range_ >>= 1;
        int bit = (code_ >= range_) ? 1 : 0;
        if (bit) code_ -= range_;
        normalize();
        return bit;
    }

private:
    void normalize() {
        while (range_ < kTopValue) {
            code_ = (code_ << 8) | next_byte();
            range_ <<= 8;
        }
    }
    std::uint8_t next_byte() {
        return (pos_ < size_) ? data_[pos_++] : std::uint8_t(0);
    }

    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
    std::uint32_t range_ = 0xFFFFFFFFu;
    std::uint32_t code_ = 0;
};

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_RANGE_CODER_HPP
