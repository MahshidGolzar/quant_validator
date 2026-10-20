// quant_validator.cpp
//
// A small, from-scratch int8 quantization validator.
//
// Why this exists: many hardware accelerators only run fast (low power,
// high throughput) low-precision integer arithmetic, not full float32.
// So AI framework code needs to convert ("quantize") float32 values to
// int8 and back ("dequantize"), and needs to verify that round trip
// doesn't lose more precision than expected. This file implements that
// conversion plus a small quantized "multiply" op, and validates both
// against hand-reasoned expectations.
//
// Build and run:
//   g++ -std=c++17 -Wall quant_validator.cpp -o quant_validator && ./quant_validator
// (or, with MSVC: cl /EHsc /std:c++17 /W4 quant_validator.cpp)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// Affine quantization parameters: float value x maps to int8 value
// round(x / scale) + zero_point.
struct QuantParams {
  float scale;     // size of one int8 "step", in float units
  int zero_point;  // the int8 value representing float 0.0
};

int8_t quantize_value(float x, const QuantParams &qp) {
  int q = static_cast<int>(std::round(x / qp.scale)) + qp.zero_point;
  q = std::clamp(q, -128, 127);  // int8's representable range
  return static_cast<int8_t>(q);
}

float dequantize_value(int8_t q, const QuantParams &qp) {
  return (static_cast<int>(q) - qp.zero_point) * qp.scale;
}

std::vector<int8_t> quantize(const std::vector<float> &xs, const QuantParams &qp) {
  std::vector<int8_t> out;
  out.reserve(xs.size());
  for (float x : xs) out.push_back(quantize_value(x, qp));
  return out;
}

std::vector<float> dequantize(const std::vector<int8_t> &qs, const QuantParams &qp) {
  std::vector<float> out;
  out.reserve(qs.size());
  for (int8_t q : qs) out.push_back(dequantize_value(q, qp));
  return out;
}

// A quantized "multiply" op, the kind of thing a framework's accelerator
// backend implements for an elementwise layer. For readability this
// dequantizes, multiplies in float, then re-quantizes to the output's
// parameters; a real accelerator kernel does the mathematical equivalent
// in pure integer arithmetic for speed, but this computes the same result.
std::vector<int8_t> quantized_multiply(const std::vector<int8_t> &a, const QuantParams &a_qp,
                                        const std::vector<int8_t> &b, const QuantParams &b_qp,
                                        const QuantParams &out_qp) {
  std::vector<int8_t> result;
  result.reserve(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    float fa = dequantize_value(a[i], a_qp);
    float fb = dequantize_value(b[i], b_qp);
    result.push_back(quantize_value(fa * fb, out_qp));
  }
  return result;
}

// ---- a tiny test "framework": named checks, pass/fail reporting ----

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

int main() {
  QuantParams qp{/*scale=*/0.1f, /*zero_point=*/0};

  // --- Round-trip accuracy: quantize then dequantize should stay close ---
  std::vector<float> original = {0.0f, 1.25f, -3.7f, 5.0f, -0.05f};
  auto quantized = quantize(original, qp);
  auto recovered = dequantize(quantized, qp);

  // Worst case: rounds to the nearest half-step, plus a tiny epsilon to
  // absorb float32 representation noise (0.1f isn't exactly 0.1 in binary).
  float max_rounding_error = qp.scale / 2.0f + 1e-5f;
  for (size_t i = 0; i < original.size(); ++i) {
    check(close_enough(original[i], recovered[i], max_rounding_error),
          "round-trip value " + std::to_string(i));
  }

  // --- Out-of-range values must clamp to int8's representable range, not overflow ---
  // 20.0 / 0.1 = 200, which is far outside int8's [-128, 127] range.
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
    // Two rounding steps happen here (quantizing the inputs, then the
    // output), so allow a slightly bigger tolerance than plain round-trip.
    check(close_enough(result_f[i], reference, out_qp.scale),
          "quantized multiply close to float reference, element " + std::to_string(i));
  }

  std::cout << "\n" << tests_run << " checks run, " << tests_failed << " failed.\n";
  return tests_failed == 0 ? 0 : 1;
}
