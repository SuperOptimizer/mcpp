// Exercises the test framework's own tolerance comparators — these are the
// backbone of codec correctness testing under f32+fast-math, so they themselves
// must be correct.
#include "mcpp/test/test.hpp"

#include <cmath>
#include <span>
#include <vector>

using namespace mcpp::test;

MCPP_TEST("toleration: mae of identical is zero, ge for shifted") {
    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{1, 2, 3, 4};
    MCPP_CHECK(mae<float>(a, b) == 0.0);

    std::vector<float> c{2, 3, 4, 5};  // shifted by +1 everywhere
    MCPP_CHECK(std::fabs(mae<float>(a, c) - 1.0) < 1e-9);
}

MCPP_TEST("toleration: max_abs_error finds the worst voxel") {
    std::vector<float> a{0, 0, 0, 0};
    std::vector<float> b{0, 0, 5, 1};
    MCPP_CHECK(std::fabs(max_abs_error<float>(a, b) - 5.0) < 1e-9);
}

MCPP_TEST("toleration: psnr is high for near-identical, finite for noisy") {
    std::vector<float> ref{100, 100, 100, 100};
    std::vector<float> same = ref;
    MCPP_CHECK(psnr<float>(ref, same, 255.0) > 1e8);  // identical -> ~infinite

    std::vector<float> noisy{100, 110, 90, 100};  // some error
    double p = psnr<float>(ref, noisy, 255.0);
    MCPP_CHECK(p > 0.0 && p < 1e8);
}

MCPP_TEST("toleration: check_le / check_ge gating helpers") {
    MCPP_CHECK_LE(0.5, 1.0);   // passes
    MCPP_CHECK_GE(42.0, 40.0); // passes
    // A failing tolerance check should throw Failure (caught by runner); we
    // verify the throwing behavior here directly.
    bool threw = false;
    try { check_le(2.0, 1.0, "x", "f", 1); }
    catch (const Failure&) { threw = true; }
    MCPP_CHECK(threw);
}

int main(int argc, char** argv) { return mcpp::test::run_all(argc, argv); }
