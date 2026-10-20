# quant_validator — int8 quantization, validated

A small, from-scratch int8 quantization validator in C++: converting float32 values to int8 and back (and a quantized "multiply" op), with a hand-written test suite checking the conversion is actually correct.

## Why this exists

Hardware accelerators are often fast specifically because they run low-precision integer arithmetic (int8) rather than full float32 — less data to move, simpler circuits, lower power. So AI framework code that targets an accelerator needs to **quantize** float32 values down to int8, and **dequantize** them back, while keeping track of how much precision that costs. This is a small, honest version of that conversion step, deliberately kept to plain arithmetic rather than anything accelerator-specific — the point is validating a conversion that real framework code performs constantly, not reproducing real accelerator kernels.

## What's in here

- `QuantParams` — a scale and zero-point, the two numbers that define an affine mapping between float and int8 (`int8_value = round(float_value / scale) + zero_point`).
- `quantize_value` / `dequantize_value` — the conversion in each direction, for one value.
- `quantize` / `dequantize` — the same, applied to a whole vector.
- `quantized_multiply` — a small quantized elementwise multiply "op", the kind of thing a framework's accelerator backend implements for a layer.
- A tiny test "framework" (`check()`, pass/fail reporting) with 10 checks.

## Building and running

```bash
g++ -std=c++17 -Wall quant_validator.cpp -o quant_validator
./quant_validator
```

(Built and tested with MSVC on Windows: `cl /EHsc /std:c++17 /W4 quant_validator.cpp`.)
