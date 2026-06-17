// mcpp geometry vector types — mcpp-native, NO OpenCV.
//
// Rationale (design memory mcpp-rendering-geometry): mcpp owns the full geometry
// layer with its OWN types (no cv::Vec3f). Vec3f is a plain 3-float aggregate;
// consumers (VC3D) adapt to it. Coordinates are [z,y,x] world voxel space.
#ifndef MCPP_GEOM_VEC_HPP
#define MCPP_GEOM_VEC_HPP

#include <cmath>
#include <cstddef>

namespace mcpp::geom {

struct Vec3f {
    float z = 0, y = 0, x = 0;

    friend constexpr Vec3f operator+(Vec3f a, Vec3f b) { return {a.z+b.z, a.y+b.y, a.x+b.x}; }
    friend constexpr Vec3f operator-(Vec3f a, Vec3f b) { return {a.z-b.z, a.y-b.y, a.x-b.x}; }
    friend constexpr Vec3f operator*(Vec3f a, float s) { return {a.z*s, a.y*s, a.x*s}; }
    friend constexpr Vec3f operator*(float s, Vec3f a) { return a * s; }
    friend constexpr bool operator==(Vec3f, Vec3f) = default;
};

constexpr float dot(Vec3f a, Vec3f b) { return a.z*b.z + a.y*b.y + a.x*b.x; }

constexpr Vec3f cross(Vec3f a, Vec3f b) {
    // standard cross product in (z,y,x) component order
    return { a.y*b.x - a.x*b.y,
             a.x*b.z - a.z*b.x,
             a.z*b.y - a.y*b.z };
}

inline float length(Vec3f a) { return std::sqrt(dot(a, a)); }

inline Vec3f normalize(Vec3f a) {
    const float L = length(a);
    return (L > 0.0f) ? a * (1.0f / L) : Vec3f{0, 0, 0};
}

struct Vec2f { float v = 0, u = 0; };

}  // namespace mcpp::geom

#endif  // MCPP_GEOM_VEC_HPP
