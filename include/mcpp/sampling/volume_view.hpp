// VolumeView — the source abstraction a sampler reads through.
//
// Rationale (design memory mcpp-sampling): the sampler is a pure read-phase BSP
// consumer. It reads blocks through a VolumeView concept; on a miss it accumulates
// the miss locally and produces partial output. The sampler does NO IO. Views:
//   - DenseView : a plain in-RAM volume (mdview); never misses. Tests / resident.
//   - CachedView: reads the frozen Cache<T>; records misses for the coordinator.
//
// A VolumeView answers two things: the block at (lod, bz,by,bx) as const T* (or
// nullptr=miss), and the volume dims at a LOD (for bounds). The sampler works in
// voxel coordinates and decomposes them into (block, in-block offset).
#ifndef MCPP_SAMPLING_VOLUME_VIEW_HPP
#define MCPP_SAMPLING_VOLUME_VIEW_HPP

#include "mcpp/cache/block_key.hpp"
#include "mcpp/cache/cache.hpp"
#include "mcpp/core/dtype.hpp"
#include "mcpp/core/mdview.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <vector>

namespace mcpp::sampling {

using cache::BlockKey;

inline constexpr std::uint32_t kBlk = 16;  // block side (cache blocks are 16^3)

template <class V>
concept VolumeView = requires(V& v, BlockKey k) {
    typename V::sample_type;
    { v.block(k) } -> std::convertible_to<const typename V::sample_type*>;
    { v.dims() } -> std::convertible_to<std::array<std::uint32_t, 3>>;
};

// Dense in-RAM view: a volume_view<T> over [z,y,x]; never misses (returns a
// pointer into a per-view scratch block copied from the dense data).
template <Sample T>
class DenseView {
public:
    using sample_type = T;
    DenseView(const T* data, std::array<std::uint32_t, 3> dims,
              std::array<std::uint64_t, 3> stride)
        : data_(data), dims_(dims), stride_(stride) { scratch_.resize(kBlk*kBlk*kBlk); }

    std::array<std::uint32_t, 3> dims() const { return dims_; }

    // Copy the 16^3 block at block coords into scratch (zero-padded at edges) and
    // return it. Always succeeds (never a miss).
    const T* block(BlockKey k) {
        const std::uint32_t bz = k.bz()*kBlk, by = k.by()*kBlk, bx = k.bx()*kBlk;
        for (std::uint32_t lz = 0; lz < kBlk; ++lz)
            for (std::uint32_t ly = 0; ly < kBlk; ++ly)
                for (std::uint32_t lx = 0; lx < kBlk; ++lx) {
                    const std::uint32_t z = bz+lz, y = by+ly, x = bx+lx;
                    T v = T(0);
                    if (z < dims_[0] && y < dims_[1] && x < dims_[2])
                        v = data_[z*stride_[0] + y*stride_[1] + x*stride_[2]];
                    scratch_[(lz*kBlk + ly)*kBlk + lx] = v;
                }
        return scratch_.data();
    }

private:
    const T* data_;
    std::array<std::uint32_t, 3> dims_;
    std::array<std::uint64_t, 3> stride_;
    std::vector<T> scratch_;
};

// Cache-backed view: reads the frozen cache; a miss returns nullptr and is
// recorded into a caller-provided miss buffer (per-consumer, BSP-clean).
template <Sample T, std::size_t Capacity>
class CachedView {
public:
    using sample_type = T;
    CachedView(const cache::Cache<T, Capacity>& cache,
               std::array<std::uint32_t, 3> dims, Lod lod,
               std::vector<BlockKey>* miss_out)
        : cache_(cache), dims_(dims), lod_(lod), miss_(miss_out) {}

    std::array<std::uint32_t, 3> dims() const { return dims_; }

    const T* block(BlockKey k) {
        const T* p = cache_.get(k);
        if (!p && miss_) miss_->push_back(k);   // record miss (no IO here)
        return p;
    }

private:
    const cache::Cache<T, Capacity>& cache_;
    std::array<std::uint32_t, 3> dims_;
    Lod lod_;
    std::vector<BlockKey>* miss_;
};

}  // namespace mcpp::sampling

#endif  // MCPP_SAMPLING_VOLUME_VIEW_HPP
