// mcpp::mdview — our own small non-owning multidimensional view.
//
// Rationale (design memory mcpp-review-rulings, review C2): GCC 15's libstdc++
// does not ship <mdspan>, and the both-compiler CI gate needs it. Rather than
// vendor Kokkos mdspan, we write our own — our usage is narrow (fixed rank 2D/3D,
// one custom z-fastest layout, f32/integer elements) so std::mdspan's full
// generality is unnecessary, and this matches the project's write-our-own posture
// (no Boost/OpenCV, own spatial trees, own test framework).
//
// Layout policies map (multi-index) -> linear offset:
//   zfast_layout : VC convention [z,y,x] with z fastest-varying:
//                  offset = lz + ly*ez + lx*ez*ey   (column-major, z innermost)
//   row_major    : last index fastest (standard C order)
#ifndef MCPP_CORE_MDVIEW_HPP
#define MCPP_CORE_MDVIEW_HPP

#include <array>
#include <cstddef>

namespace mcpp {

// ---- layout policies -----------------------------------------------------

struct zfast_layout {
    // extents given in [z,y,x] order (3D) or [v,u] (2D, v outer / u inner...).
    // We define z (or the LAST listed axis) as the fastest-varying to match the
    // VC chunk convention data[lz + ly*cz + lx*cz*cy].
    template <std::size_t Rank>
    static constexpr std::size_t offset(const std::array<std::size_t, Rank>& ext,
                                        const std::array<std::size_t, Rank>& idx) noexcept {
        // ext/idx are [axis0, axis1, ... axisN-1] with axis0 the OUTERMOST.
        // z-fastest => iterate innermost-first. For [z,y,x]: x outer, y, z inner.
        // We treat the LAST axis as fastest.
        std::size_t off = 0;
        std::size_t stride = 1;
        for (std::size_t a = Rank; a-- > 0;) {
            off += idx[a] * stride;
            stride *= ext[a];
        }
        return off;
    }
};

struct row_major {
    template <std::size_t Rank>
    static constexpr std::size_t offset(const std::array<std::size_t, Rank>& ext,
                                        const std::array<std::size_t, Rank>& idx) noexcept {
        std::size_t off = 0;
        std::size_t stride = 1;
        for (std::size_t a = Rank; a-- > 0;) {
            off += idx[a] * stride;
            stride *= ext[a];
        }
        return off;
    }
};

// ---- the view ------------------------------------------------------------
//
// Non-owning. Cheap to copy. `data()` is the backing pointer (caller owns
// lifetime). Element access via operator() with Rank indices.

template <class T, std::size_t Rank, class Layout = zfast_layout>
class mdview {
public:
    using element_type = T;
    using extents_type = std::array<std::size_t, Rank>;

    constexpr mdview() = default;
    constexpr mdview(T* data, extents_type ext) noexcept : data_(data), ext_(ext) {}

    static constexpr std::size_t rank() noexcept { return Rank; }
    constexpr const extents_type& extents() const noexcept { return ext_; }
    constexpr std::size_t extent(std::size_t a) const noexcept { return ext_[a]; }
    constexpr T* data() const noexcept { return data_; }

    constexpr std::size_t size() const noexcept {
        std::size_t n = 1;
        for (std::size_t a = 0; a < Rank; ++a) n *= ext_[a];
        return n;
    }

    // Variadic element access: view(i0, i1, ...) with exactly Rank indices.
    template <class... Idx>
        requires (sizeof...(Idx) == Rank)
    constexpr T& operator()(Idx... idx) const noexcept {
        const extents_type i{static_cast<std::size_t>(idx)...};
        return data_[Layout::template offset<Rank>(ext_, i)];
    }

    constexpr T& operator[](const extents_type& i) const noexcept {
        return data_[Layout::template offset<Rank>(ext_, i)];
    }

private:
    T* data_ = nullptr;
    extents_type ext_{};
};

// Convenience aliases for the cases the codebase actually uses.
template <class T> using volume_view  = mdview<T, 3, zfast_layout>;  // [z,y,x] z-fastest
template <class T> using plane_view   = mdview<T, 2, zfast_layout>;  // [v,u] u-fastest

}  // namespace mcpp

#endif  // MCPP_CORE_MDVIEW_HPP
