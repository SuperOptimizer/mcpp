// consteval scan order for coefficient coding.
//
// Rationale (design memory mcpp-original-codec-facts): coefficients are coded in
// ascending-frequency (ascending L1 band) order so the trailing high-frequency
// zeros cluster at the end and an EOB (end-of-block) symbol terminates cheaply.
// The original built this lazily under a mutex; we generate it at compile time.
//
// We sort the N^Rank linear positions by their L1 band (sum of per-axis frequency
// indices), DC first. Ties broken by linear index for determinism.
#ifndef MCPP_CODEC_SCAN_HPP
#define MCPP_CODEC_SCAN_HPP

#include "mcpp/codec/quant.hpp"  // coeff_band

#include <array>
#include <cstddef>
#include <cstdint>

namespace mcpp::codec {

template <std::size_t N, std::size_t Rank>
consteval std::size_t block_total() {
    std::size_t t = 1;
    for (std::size_t a = 0; a < Rank; ++a) t *= N;
    return t;
}

// Ascending-L1-band scan order: scan[i] = linear position visited i-th.
// Implemented as a STABLE COUNTING SORT by band — O(total + bands), which keeps
// it well within the constexpr step limit (insertion sort's O(n^2) on 4096
// elements blows the limit). Stability => ties broken by ascending linear index.
template <std::size_t N, std::size_t Rank>
consteval std::array<std::uint16_t, block_total<N, Rank>()> make_scan() {
    constexpr std::size_t total = block_total<N, Rank>();
    static_assert(total <= 0x1'0000, "scan index must fit u16");
    constexpr int bands = max_band<N, Rank>() + 1;  // band in [0, max_band]

    // histogram of band counts
    std::array<std::size_t, std::size_t(max_band<N, Rank>()) + 1> count{};
    for (std::size_t i = 0; i < total; ++i)
        ++count[std::size_t(coeff_band<N, Rank>(i))];

    // exclusive prefix sum -> start offset of each band
    std::array<std::size_t, std::size_t(max_band<N, Rank>()) + 1> start{};
    std::size_t acc = 0;
    for (int b = 0; b < bands; ++b) { start[std::size_t(b)] = acc; acc += count[std::size_t(b)]; }

    // place each position into its band's slot, in ascending linear order (stable)
    std::array<std::uint16_t, total> scan{};
    std::array<std::size_t, std::size_t(max_band<N, Rank>()) + 1> cursor = start;
    for (std::size_t i = 0; i < total; ++i) {
        const std::size_t b = std::size_t(coeff_band<N, Rank>(i));
        scan[cursor[b]++] = std::uint16_t(i);
    }
    return scan;
}

// The two configurations the codec uses.
inline constexpr auto kScan16x3 = make_scan<16, 3>();  // 4096 entries
inline constexpr auto kScan64x2 = make_scan<64, 2>();  // 4096 entries

template <std::size_t N, std::size_t Rank>
consteval const std::array<std::uint16_t, block_total<N, Rank>()>& scan_for() {
    if constexpr (N == 16 && Rank == 3) return kScan16x3;
    else if constexpr (N == 64 && Rank == 2) return kScan64x2;
    else static_assert(N == 16, "unsupported scan configuration");
}

}  // namespace mcpp::codec

#endif  // MCPP_CODEC_SCAN_HPP
