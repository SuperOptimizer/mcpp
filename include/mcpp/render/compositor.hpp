// Compositor policies — concept-constrained reducers along the normal ray.
//
// Rationale (design memory mcpp-rendering-geometry): compositing modes are
// pluggable, concept-constrained reducer policies (min/mean/max/alpha at launch;
// shaded modes later as more policies). A Compositor accumulates samples along a
// ray and finalizes to one output value. New compositors drop in without touching
// the renderer.
#ifndef MCPP_RENDER_COMPOSITOR_HPP
#define MCPP_RENDER_COMPOSITOR_HPP

#include <algorithm>
#include <concepts>
#include <cstdint>

namespace mcpp::render {

// A Compositor is reset(), fed samples via add(value), and finalized to a value.
template <class C>
concept Compositor = requires(C& c, float v) {
    { c.reset() } -> std::same_as<void>;
    { c.add(v) } -> std::same_as<void>;
    { c.value() } -> std::convertible_to<float>;
};

struct MaxComposite {
    float m = 0.0f;
    void reset() { m = 0.0f; }
    void add(float v) { m = std::max(m, v); }
    float value() const { return m; }
};

struct MinComposite {
    float m = 0.0f; bool any = false;
    void reset() { m = 0.0f; any = false; }
    void add(float v) { if (!any || v < m) { m = v; any = true; } }
    float value() const { return any ? m : 0.0f; }
};

struct MeanComposite {
    double sum = 0.0; std::uint32_t n = 0;
    void reset() { sum = 0.0; n = 0; }
    void add(float v) { sum += v; ++n; }
    float value() const { return n ? float(sum / double(n)) : 0.0f; }
};

// Front-to-back alpha-over with a simple linear opacity from sample value.
struct AlphaComposite {
    float acc = 0.0f, trans = 1.0f;
    void reset() { acc = 0.0f; trans = 1.0f; }
    void add(float v) {
        const float a = std::clamp(v / 255.0f, 0.0f, 1.0f);   // opacity
        acc += trans * a * v;
        trans *= (1.0f - a);
    }
    float value() const { return acc; }
};

static_assert(Compositor<MaxComposite>);
static_assert(Compositor<MinComposite>);
static_assert(Compositor<MeanComposite>);
static_assert(Compositor<AlphaComposite>);

}  // namespace mcpp::render

#endif  // MCPP_RENDER_COMPOSITOR_HPP
