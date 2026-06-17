// Surface concept + PlaneSurface / QuadSurface — the geometry layer.
//
// Rationale (design memory mcpp-rendering-geometry): mcpp owns the full surface
// algebra with mcpp-native types. Surface is a C++26 CONCEPT (zero-overhead,
// monomorphized samplers) — coord(u,v) -> world point, normal(u,v) -> unit normal,
// extent() -> grid size. PlaneSurface (origin+basis) and QuadSurface (point grid)
// satisfy it statically. A type-erased AnySurface handle (for heterogeneous UI
// lists) layers on later; the hot paths use the concept directly.
#ifndef MCPP_GEOM_SURFACE_HPP
#define MCPP_GEOM_SURFACE_HPP

#include "mcpp/geom/vec.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <vector>

namespace mcpp::geom {

// A Surface maps grid coords (u,v) -> a world point and a unit normal, over a
// finite grid of extent (gw, gh).
template <class S>
concept Surface = requires(const S& s, float u, float v) {
    { s.coord(u, v) } -> std::convertible_to<Vec3f>;
    { s.normal(u, v) } -> std::convertible_to<Vec3f>;
    { s.extent() } -> std::convertible_to<std::array<std::uint32_t, 2>>;
};

// An oriented plane: origin + orthonormal basis (bu, bv) + a normal. coord(u,v)
// = origin + u*bu + v*bv; normal is constant. setFromNormalAndUp builds a stable
// basis from a normal + up hint (no sign-flip discontinuity).
class PlaneSurface {
public:
    PlaneSurface() = default;
    PlaneSurface(Vec3f origin, Vec3f bu, Vec3f bv, Vec3f normal,
                 std::uint32_t gw, std::uint32_t gh)
        : origin_(origin), bu_(bu), bv_(bv), normal_(normal), gw_(gw), gh_(gh) {}

    static PlaneSurface from_normal_and_up(Vec3f origin, Vec3f normal, Vec3f up_hint,
                                           std::uint32_t gw, std::uint32_t gh) {
        Vec3f n = normalize(normal);
        Vec3f bu = normalize(cross(up_hint, n));
        if (length(bu) < 1e-6f) bu = normalize(cross(Vec3f{1,0,0}, n));  // fallback
        Vec3f bv = normalize(cross(n, bu));
        return PlaneSurface(origin, bu, bv, n, gw, gh);
    }

    Vec3f coord(float u, float v) const { return origin_ + bu_ * u + bv_ * v; }
    Vec3f normal(float, float) const { return normal_; }
    std::array<std::uint32_t, 2> extent() const { return {gw_, gh_}; }

    Vec3f basis_u() const { return bu_; }
    Vec3f basis_v() const { return bv_; }
    Vec3f origin() const { return origin_; }

private:
    Vec3f origin_{}, bu_{0,0,1}, bv_{0,1,0}, normal_{1,0,0};
    std::uint32_t gw_ = 0, gh_ = 0;
};

// A quad-mesh surface: a gw x gh grid of world points (row-major [v*gw+u]).
// coord() bilinearly interpolates the grid; normal() is the cross of grid
// tangents (cached lazily as a const-correctness-friendly recompute).
class QuadSurface {
public:
    QuadSurface(std::vector<Vec3f> points, std::uint32_t gw, std::uint32_t gh)
        : pts_(std::move(points)), gw_(gw), gh_(gh) {}

    std::array<std::uint32_t, 2> extent() const { return {gw_, gh_}; }

    // bilinear sample of the point grid at continuous (u,v)
    Vec3f coord(float u, float v) const {
        const std::uint32_t u0 = clampu(std::uint32_t(u), gw_), v0 = clampu(std::uint32_t(v), gh_);
        const std::uint32_t u1 = clampu(u0 + 1, gw_), v1 = clampu(v0 + 1, gh_);
        const float fu = u - float(u0), fv = v - float(v0);
        const Vec3f a = grid(u0, v0), b = grid(u1, v0), c = grid(u0, v1), d = grid(u1, v1);
        const Vec3f top = a + (b - a) * fu;
        const Vec3f bot = c + (d - c) * fu;
        return top + (bot - top) * fv;
    }

    // normal from grid tangents (forward differences, cross product).
    Vec3f normal(float u, float v) const {
        const std::uint32_t u0 = clampu(std::uint32_t(u), gw_), v0 = clampu(std::uint32_t(v), gh_);
        const std::uint32_t u1 = clampu(u0 + 1, gw_), v1 = clampu(v0 + 1, gh_);
        const Vec3f du = grid(u1, v0) - grid(u0, v0);
        const Vec3f dv = grid(u0, v1) - grid(u0, v0);
        return normalize(cross(du, dv));
    }

    const std::vector<Vec3f>& points() const { return pts_; }

private:
    static std::uint32_t clampu(std::uint32_t i, std::uint32_t n) { return i < n ? i : (n - 1); }
    Vec3f grid(std::uint32_t u, std::uint32_t v) const { return pts_[std::size_t(v) * gw_ + u]; }

    std::vector<Vec3f> pts_;
    std::uint32_t gw_, gh_;
};

static_assert(Surface<PlaneSurface>);
static_assert(Surface<QuadSurface>);

}  // namespace mcpp::geom

#endif  // MCPP_GEOM_SURFACE_HPP
