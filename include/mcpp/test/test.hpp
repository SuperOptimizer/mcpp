// mcpp's own minimal test framework.
//
// Rationale (see design memory mcpp-testing-ci): we write our own — consistent
// with the no-Boost / no-OpenCV / own-spatial-trees posture. It is deliberately
// tiny: test registration, assertions, and TOLERANCE comparators as first-class
// citizens (PSNR/MAE/max-error are the codec's correctness currency, so they are
// built in rather than bolted on).
//
// Usage:
//   #include "mcpp/test/test.hpp"
//   MCPP_TEST("name of test") { MCPP_CHECK(1 + 1 == 2); }
//   int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
#ifndef MCPP_TEST_TEST_HPP
#define MCPP_TEST_TEST_HPP

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace mcpp::test {

struct Case {
    std::string_view name;
    std::function<void()> fn;
    std::string_view file;
    int line;
};

// A failure thrown by an assertion to abort the current case (not the process).
struct Failure {
    std::string message;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}

struct Registrar {
    Registrar(std::string_view name, std::function<void()> fn,
              std::string_view file, int line) {
        registry().push_back(Case{name, std::move(fn), file, line});
    }
};

// ---- assertions ----------------------------------------------------------

[[noreturn]] inline void fail(std::string msg) { throw Failure{std::move(msg)}; }

inline void check(bool cond, std::string_view expr,
                  std::string_view file, int line) {
    if (!cond) {
        fail(std::string(file) + ":" + std::to_string(line) +
             ": MCPP_CHECK failed: " + std::string(expr));
    }
}

// ---- tolerance comparators (first-class) ---------------------------------
//
// These operate on spans of the SAME length. They are the backbone of codec
// correctness testing under the f32 + fast-math regime where byte-exact
// comparison is impossible by design.

// Mean absolute error.
template <class T>
double mae(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size() || a.empty()) fail("mae: size mismatch / empty");
    double acc = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        acc += std::fabs(double(a[i]) - double(b[i]));
    return acc / double(a.size());
}

// Maximum absolute error (L-infinity) — the metric for max-error-bounded mode.
template <class T>
double max_abs_error(std::span<const T> a, std::span<const T> b) {
    if (a.size() != b.size() || a.empty()) fail("max_abs_error: size mismatch / empty");
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(double(a[i]) - double(b[i]));
        if (d > m) m = d;
    }
    return m;
}

// Peak signal-to-noise ratio in dB, given the signal peak value.
template <class T>
double psnr(std::span<const T> ref, std::span<const T> test, double peak) {
    if (ref.size() != test.size() || ref.empty()) fail("psnr: size mismatch / empty");
    double mse = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        double d = double(ref[i]) - double(test[i]);
        mse += d * d;
    }
    mse /= double(ref.size());
    if (mse == 0.0) return 1.0e9;  // identical -> effectively infinite PSNR
    return 10.0 * std::log10((peak * peak) / mse);
}

inline void check_le(double value, double bound, std::string_view what,
                     std::string_view file, int line) {
    if (!(value <= bound)) {
        fail(std::string(file) + ":" + std::to_string(line) + ": " +
             std::string(what) + " = " + std::to_string(value) +
             " exceeds bound " + std::to_string(bound));
    }
}

inline void check_ge(double value, double bound, std::string_view what,
                     std::string_view file, int line) {
    if (!(value >= bound)) {
        fail(std::string(file) + ":" + std::to_string(line) + ": " +
             std::string(what) + " = " + std::to_string(value) +
             " below bound " + std::to_string(bound));
    }
}

// ---- runner --------------------------------------------------------------

inline int run_all(int /*argc*/ = 0, char** /*argv*/ = nullptr) {
    int passed = 0, failed = 0;
    for (const auto& c : registry()) {
        try {
            c.fn();
            ++passed;
            std::printf("[ PASS ] %.*s\n",
                        int(c.name.size()), c.name.data());
        } catch (const Failure& f) {
            ++failed;
            std::printf("[ FAIL ] %.*s\n         %s\n",
                        int(c.name.size()), c.name.data(), f.message.c_str());
        } catch (const std::exception& e) {
            ++failed;
            std::printf("[ FAIL ] %.*s\n         unexpected exception: %s\n",
                        int(c.name.size()), c.name.data(), e.what());
        }
    }
    std::printf("\n%d passed, %d failed (%zu total)\n",
                passed, failed, registry().size());
    return failed == 0 ? 0 : 1;
}

}  // namespace mcpp::test

// ---- macros --------------------------------------------------------------

#define MCPP_TEST_CAT_(a, b) a##b
#define MCPP_TEST_CAT(a, b) MCPP_TEST_CAT_(a, b)

#define MCPP_TEST(NAME)                                                       \
    static void MCPP_TEST_CAT(mcpp_test_fn_, __LINE__)();                     \
    static const ::mcpp::test::Registrar MCPP_TEST_CAT(mcpp_test_reg_,        \
                                                       __LINE__){             \
        NAME, &MCPP_TEST_CAT(mcpp_test_fn_, __LINE__), __FILE__, __LINE__};   \
    static void MCPP_TEST_CAT(mcpp_test_fn_, __LINE__)()

#define MCPP_CHECK(EXPR) ::mcpp::test::check((EXPR), #EXPR, __FILE__, __LINE__)
#define MCPP_CHECK_LE(VAL, BOUND) \
    ::mcpp::test::check_le((VAL), (BOUND), #VAL " <= " #BOUND, __FILE__, __LINE__)
#define MCPP_CHECK_GE(VAL, BOUND) \
    ::mcpp::test::check_ge((VAL), (BOUND), #VAL " >= " #BOUND, __FILE__, __LINE__)

#endif  // MCPP_TEST_TEST_HPP
