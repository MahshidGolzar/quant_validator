// quant_validator.h
//
// A small, from-scratch int8 quantization validator.
//
// Why this exists: many hardware accelerators only run fast (low power,
// high throughput) low-precision integer arithmetic, not full float32.
// So AI framework code needs to convert ("quantize") float32 values to
// int8 and back ("dequantize"), and needs to verify that round trip
// doesn't lose more precision than expected. This header implements that
// conversion plus a small quantized "multiply" op.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Affine quantization parameters: float value x maps to int8 value
// round(x / scale) + zero_point.
struct QuantParams {
  float scale;     // size of one int8 "step", in float units
  int zero_point;  // the int8 value representing float 0.0
};

inline int8_t quantize_value(float x, const QuantParams &qp) {
  int q = static_cast<int>(std::round(x / qp.scale)) + qp.zero_point;
  q = std::clamp(q, -128, 127);  // int8's representable range
  return static_cast<int8_t>(q);
}

inline float dequantize_value(int8_t q, const QuantParams &qp) {
  return (static_cast<int>(q) - qp.zero_point) * qp.scale;
}

inline std::vector<int8_t> quantize(const std::vector<float> &xs, const QuantParams &qp) {
  std::vector<int8_t> out;
  out.reserve(xs.size());
  for (float x : xs) out.push_back(quantize_value(x, qp));
  return out;
}

inline std::vector<float> dequantize(const std::vector<int8_t> &qs, const QuantParams &qp) {
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
inline std::vector<int8_t> quantized_multiply(const std::vector<int8_t> &a, const QuantParams &a_qp,
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
