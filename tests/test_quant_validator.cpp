// test_quant_validator.cpp
//
// Catch2-based test suite (replaces the hand-rolled check()/pass-fail
// reporting that used to live directly in quant_validator.cpp with named
// test cases, Catch2's assertion macros, and Approx-based float comparison).
//
// Build and run (from repo root):
//   g++ -std=c++17 -Wall -I. -Ithird_party tests/test_quant_validator.cpp -o tests_bin && ./tests_bin
//
// Catch2 v2.13.10, vendored as a single header (third_party/catch.hpp) so
// this builds with a plain compiler invocation, no package manager needed.

#define CATCH_CONFIG_MAIN
#include "../third_party/catch.hpp"
#include "../quant_validator.h"

TEST_CASE("round trip stays within the expected rounding error", "[quantize][dequantize]") {
  QuantParams qp{/*scale=*/0.1f, /*zero_point=*/0};
  std::vector<float> original = {0.0f, 1.25f, -3.7f, 5.0f, -0.05f};

  auto quantized = quantize(original, qp);
  auto recovered = dequantize(quantized, qp);

  // Worst case: rounds to the nearest half-step, plus a tiny epsilon to
  // absorb float32 representation noise (0.1f isn't exactly 0.1 in binary).
  const float max_rounding_error = qp.scale / 2.0f + 1e-5f;

  for (size_t i = 0; i < original.size(); ++i) {
    INFO("element " << i << ": original=" << original[i] << " recovered=" << recovered[i]);
    REQUIRE(recovered[i] == Approx(original[i]).margin(max_rounding_error));
  }
}

TEST_CASE("out-of-range values clamp instead of silently wrapping", "[quantize][clamping][regression]") {
  // Regression test for the original bug: quantize_value used to cast an
  // out-of-range int straight to int8_t with no bounds check. 20.0f at
  // scale=0.1 produces q=200, which is outside int8's [-128, 127] range;
  // the unclamped version silently wrapped this to -56 instead of erroring
  // or saturating. These two cases are exactly what caught that fault.
  QuantParams qp{/*scale=*/0.1f, /*zero_point=*/0};

  SECTION("large positive value clamps to int8 max") {
    REQUIRE(quantize_value(20.0f, qp) == 127);
  }

  SECTION("large negative value clamps to int8 min") {
    REQUIRE(quantize_value(-20.0f, qp) == -128);
  }
}

TEST_CASE("quantized multiply matches the float32 reference", "[quantized_multiply]") {
  QuantParams qp{/*scale=*/0.1f, /*zero_point=*/0};
  QuantParams out_qp{/*scale=*/0.1f, /*zero_point=*/0};

  std::vector<float> a_vals = {2.0f, -1.5f, 3.0f};
  std::vector<float> b_vals = {1.0f, 2.0f, -2.0f};

  auto a_q = quantize(a_vals, qp);
  auto b_q = quantize(b_vals, qp);
  auto result_q = quantized_multiply(a_q, qp, b_q, qp, out_qp);
  auto result_f = dequantize(result_q, out_qp);

  for (size_t i = 0; i < a_vals.size(); ++i) {
    float reference = a_vals[i] * b_vals[i];
    INFO("element " << i << ": got=" << result_f[i] << " reference=" << reference);
    // Two rounding steps happen here (quantizing the inputs, then the
    // output), so allow a slightly bigger tolerance than plain round-trip.
    REQUIRE(result_f[i] == Approx(reference).margin(out_qp.scale));
  }
}

TEST_CASE("dequantize_value is the exact inverse of quantize_value's affine map", "[dequantize]") {
  QuantParams qp{/*scale=*/0.25f, /*zero_point=*/4};

  // Pick a value that lands exactly on a representable step, so this is an
  // exact equality check rather than an approximate one.
  int8_t q = quantize_value(1.0f, qp);  // round(1.0/0.25) + 4 = 8
  REQUIRE(static_cast<int>(q) == 8);
  REQUIRE(dequantize_value(q, qp) == Approx(1.0f));
}
