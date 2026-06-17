// Max-error correction pass — BEST-EFFORT voxel-error bound.
//
// Rationale (design memory mcpp-scope + review C1 ruling): the codec can target a
// max per-voxel error tau by storing sparse additive corrections for voxels the
// lossy path missed by more than tau. This is BEST-EFFORT, computed against the
// ENCODER'S OWN reference decode — NOT a hard cross-decoder guarantee (the f32
// transform is non-deterministic under -ffast-math, so a different decoder may
// diverge by some delta). Useful for parametric surfaces where subvoxel coord
// fidelity matters.
//
// Correction format: [u32 count][ {u32 index, f32 delta} * count ]. On decode,
// each delta is ADDED to the reconstructed voxel — an additive fix outside the
// quant path. Sparse: only out-of-tolerance voxels are stored.
#ifndef MCPP_CODEC_MAX_ERROR_HPP
#define MCPP_CODEC_MAX_ERROR_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace mcpp::codec {

// Build sparse corrections so that orig[i] - (decoded[i] + delta) == 0 for every
// voxel where |orig[i] - decoded[i]| > tau. Appends a framed correction blob.
inline void build_corrections(const float* orig, const float* decoded,
                              std::size_t n, float tau,
                              std::vector<std::uint8_t>& out) {
    std::vector<std::pair<std::uint32_t, float>> fixes;
    for (std::size_t i = 0; i < n; ++i) {
        const float e = orig[i] - decoded[i];
        if (e > tau || e < -tau) fixes.push_back({std::uint32_t(i), e});
    }
    auto put_u32 = [&](std::uint32_t v) {
        out.push_back(std::uint8_t(v)); out.push_back(std::uint8_t(v>>8));
        out.push_back(std::uint8_t(v>>16)); out.push_back(std::uint8_t(v>>24));
    };
    put_u32(std::uint32_t(fixes.size()));
    for (auto [idx, d] : fixes) {
        put_u32(idx);
        std::uint32_t db; std::memcpy(&db, &d, 4); put_u32(db);
    }
}

// Apply a framed correction blob to a decoded block in place. Bounds-checked
// against `len` and `n` (hardened: malformed corrections are ignored).
inline void apply_corrections(const std::uint8_t* blob, std::size_t len,
                              float* decoded, std::size_t n) {
    if (len < 4) return;
    auto u32 = [&](std::size_t at) {
        return std::uint32_t(blob[at]) | (std::uint32_t(blob[at+1])<<8) |
               (std::uint32_t(blob[at+2])<<16) | (std::uint32_t(blob[at+3])<<24);
    };
    std::size_t off = 0;
    const std::uint32_t count = u32(off); off += 4;
    for (std::uint32_t k = 0; k < count; ++k) {
        if (off + 8 > len) return;            // malformed -> stop
        const std::uint32_t idx = u32(off); off += 4;
        const std::uint32_t db = u32(off);  off += 4;
        float d; std::memcpy(&d, &db, 4);
        if (idx < n) decoded[idx] += d;       // additive fix, bounds-checked
    }
}

// Size of a correction blob's header (the count), to skip when absent.
inline constexpr std::size_t kCorrectionHeader = 4;

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_MAX_ERROR_HPP
