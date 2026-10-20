// quant_validator.cpp
//
// Runnable demo: prints the same checks as test_quant_validator.cpp in a
// plain pass/fail format, with no test framework dependency. The actual
// test suite (tests/test_quant_validator.cpp) uses Catch2 and is what CI
// runs; this file exists so the library can also be exercised with a
// single compiler command and no dependencies at all.
//
// Build and run:
//   g++ -std=c++17 -Wall quant_validator.cpp -o quant_validator && ./quant_validator
// (or, with MSVC: cl /EHsc /std:c++17 /W4 quant_validator.cpp)

#include <iostream>
#include <string>

#include "quant_validator.h"

namespace {

int tests_run = 0;
int tests_failed = 0;

void check(bool condition, const std::string &test_name) {
  ++tests_run;
  if (!condition) {
    ++tests_failed;
    std::cout << "FAIL: " << test_name << "\n";
  } else {
    std::cout << "PASS: " << test_name << "\n";
  }
}

bool close_enough(float a, float b, float tol) {
  return std::fabs(a - b) <= tol;
}

}  // namespace

int main() {
  QuantParams qp{/*scale=*/0.1f, /*zero_point=*/0};

  // --- Round-trip accuracy: quantize then dequantize should stay close ---
  std::vector<float> original = {0.0f, 1.25f, -3.7f, 5.0f, -0.05f};
  auto quantized = quantize(original, qp);
  auto recovered = dequantize(quantized, qp);

  float max_rounding_error = qp.scale / 2.0f + 1e-5f;
  for (size_t i = 0; i < original.size(); ++i) {
    check(close_enough(original[i], recovered[i], max_rounding_error),
          "round-trip value " + std::to_string(i));
  }

  // --- Out-of-range values must clamp to int8's representable range, not overflow ---
  int8_t clamped = quantize_value(20.0f, qp);
  check(clamped == 127, "extreme positive value clamps to int8 max (127)");

  int8_t clamped_neg = quantize_value(-20.0f, qp);
  check(clamped_neg == -128, "extreme negative value clamps to int8 min (-128)");

  // --- Quantized multiply op, checked against the float32 reference ---
  QuantParams out_qp{/*scale=*/0.1f, /*zero_point=*/0};
  std::vector<float> a_vals = {2.0f, -1.5f, 3.0f};
  std::vector<float> b_vals = {1.0f, 2.0f, -2.0f};

  auto a_q = quantize(a_vals, qp);
  auto b_q = quantize(b_vals, qp);
  auto result_q = quantized_multiply(a_q, qp, b_q, qp, out_qp);
  auto result_f = dequantize(result_q, out_qp);

  for (size_t i = 0; i < a_vals.size(); ++i) {
    float reference = a_vals[i] * b_vals[i];
    check(close_enough(result_f[i], reference, out_qp.scale),
          "quantized multiply close to float reference, element " + std::to_string(i));
  }

  std::cout << "\n" << tests_run << " checks run, " << tests_failed << " failed.\n";
  return tests_failed == 0 ? 0 : 1;
}
