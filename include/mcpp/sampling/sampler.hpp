// Sampler<View> — trilinear/nearest point sampling over a VolumeView.
//
// Rationale (design memory mcpp-sampling): pure read-phase BSP consumer. Works in
// voxel coordinates (subvoxel for trilinear), decomposes each coordinate into
// (16^3 block, in-block offset), reads the block through the View. A tiny LOCAL
// last-block memo exploits ray/region coherence (consecutive samples hit the same
// block) without re-resolving the View. f32 compute; -ffast-math fine; NaN scrub
// via fpbits is the codec's edge (data is already clean by the time it's cached).
//
// A miss (View returns nullptr) yields 0 for that corner and is recorded by the
// View (CachedView) — the sampler stays dumb about IO.
#ifndef MCPP_SAMPLING_SAMPLER_HPP
#define MCPP_SAMPLING_SAMPLER_HPP

#include "mcpp/sampling/volume_view.hpp"

#include <array>
#include <cmath>
#include <cstdint>

namespace mcpp::sampling {

enum class Filter { nearest, trilinear };

template <VolumeView View>
class Sampler {
public:
    using T = typename View::sample_type;
    explicit Sampler(View& view, Lod lod = Lod{0}) : view_(view), lod_(lod) {}

    // Sample one voxel value at (z,y,x) (integer for nearest, may be fractional
    // for trilinear). Returns f32. Out-of-bounds and missing blocks read as 0.
    float sample(float z, float y, float x, Filter f) {
        const auto d = view_.dims();
        if (f == Filter::nearest) {
            long iz = std::lround(z), iy = std::lround(y), ix = std::lround(x);
            return voxel(iz, iy, ix, d);
        }
        // trilinear: 8 corners
        const long z0 = long(std::floor(z)), y0 = long(std::floor(y)), x0 = long(std::floor(x));
        const float fz = z - float(z0), fy = y - float(y0), fx = x - float(x0);
        const float c000 = voxel(z0,   y0,   x0,   d), c001 = voxel(z0,   y0,   x0+1, d);
        const float c010 = voxel(z0,   y0+1, x0,   d), c011 = voxel(z0,   y0+1, x0+1, d);
        const float c100 = voxel(z0+1, y0,   x0,   d), c101 = voxel(z0+1, y0,   x0+1, d);
        const float c110 = voxel(z0+1, y0+1, x0,   d), c111 = voxel(z0+1, y0+1, x0+1, d);
        const float c00 = c000 + (c001 - c000) * fx;
        const float c01 = c010 + (c011 - c010) * fx;
        const float c10 = c100 + (c101 - c100) * fx;
        const float c11 = c110 + (c111 - c110) * fx;
        const float c0 = c00 + (c01 - c00) * fy;
        const float c1 = c10 + (c11 - c10) * fy;
        return c0 + (c1 - c0) * fz;
    }

private:
    // Fetch one integer voxel (0 if out of bounds or block missing).
    float voxel(long z, long y, long x, const std::array<std::uint32_t,3>& d) {
        if (z < 0 || y < 0 || x < 0) return 0.0f;
        if (std::uint32_t(z) >= d[0] || std::uint32_t(y) >= d[1] || std::uint32_t(x) >= d[2])
            return 0.0f;
        const std::uint32_t bz = std::uint32_t(z) / kBlk, by = std::uint32_t(y) / kBlk,
                            bx = std::uint32_t(x) / kBlk;
        const T* blk = resolve(bz, by, bx);
        if (!blk) return 0.0f;  // miss -> 0 (recorded by the View)
        const std::uint32_t lz = std::uint32_t(z) % kBlk, ly = std::uint32_t(y) % kBlk,
                            lx = std::uint32_t(x) % kBlk;
        return float(blk[(lz*kBlk + ly)*kBlk + lx]);
    }

    // Last-block memo: if the block coords match the cached ones, reuse the
    // pointer without re-resolving the View (ray/region coherence).
    const T* resolve(std::uint32_t bz, std::uint32_t by, std::uint32_t bx) {
        if (have_memo_ && bz == mz_ && by == my_ && bx == mx_) return memo_;
        memo_ = view_.block(BlockKey::make(lod_, bz, by, bx));
        mz_ = bz; my_ = by; mx_ = bx; have_memo_ = true;
        return memo_;
    }

    View& view_;
    Lod lod_;
    const T* memo_ = nullptr;
    std::uint32_t mz_ = 0, my_ = 0, mx_ = 0;
    bool have_memo_ = false;
};

}  // namespace mcpp::sampling

#endif  // MCPP_SAMPLING_SAMPLER_HPP
