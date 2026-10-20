# quant_validator — int8 quantization with tests

[![CI](https://github.com/MahshidGolzar/quant_validator/actions/workflows/ci.yml/badge.svg)](https://github.com/MahshidGolzar/quant_validator/actions/workflows/ci.yml)

A small int8 quantization validator in C++: converting float32 values to int8 and back (and a quantized "multiply" op), with a Catch2 test suite checking the conversion is actually correct, built and sanitizer-checked on every push via GitHub Actions.

## Why this exists

Hardware accelerators are often fast specifically because they run low-precision integer arithmetic (int8) rather than full float32 — less data to move, simpler circuits, lower power. So AI framework code that targets an accelerator needs to **quantize** float32 values down to int8, and **dequantize** them back, while keeping track of how much precision that costs. This is a small, version of that conversion step, deliberately kept to plain arithmetic rather than anything accelerator-specific — the point is validating a conversion that real framework code performs constantly, not reproducing real accelerator kernels.

## What's in here

- `QuantParams` — a scale and zero-point, the two numbers that define an affine mapping between float and int8 (`int8_value = round(float_value / scale) + zero_point`).
- `quantize_value` / `dequantize_value` — the conversion in each direction, for one value.
- `quantize` / `dequantize` — the same, applied to a whole vector.
- `quantized_multiply` — a small quantized elementwise multiply "op", the kind of thing a framework's accelerator backend implements for a layer. For readability this dequantizes, multiplies in float, then re-quantizes the result; a real accelerator kernel computes the mathematical equivalent in pure integer arithmetic for speed.
- `tests/test_quant_validator.cpp` — the real test suite, using [Catch2](https://github.com/catchorg/Catch2) (v2.13.10, vendored as a single header in `third_party/catch.hpp` so the build needs no package manager): round-trip accuracy, out-of-range clamping (the regression test for the bug below), the quantized multiply op against the float32 reference, and an exact-value check of the affine mapping.
- `quant_validator.cpp` — a small standalone demo with the same checks in plain pass/fail output and no test-framework dependency, for anyone who wants to run it with a single compiler command.
- `.github/workflows/ci.yml` — builds and runs the test suite on every push: a plain GCC build, a GCC build under AddressSanitizer + UBSan, and a Clang build under `-fsanitize=implicit-conversion` (see below for why that specific sanitizer, not just UBSan).

## A bug the test suite caught on the first commit

The first commit's `quantize_value` deliberately omitted the range clamp, to demonstrate that the test suite in that same commit would catch it:

```cpp
int q = static_cast<int>(std::round(x / qp.scale)) + qp.zero_point;
return static_cast<int8_t>(q);   // q might be outside [-128, 127]
```

int8 can only represent -128 to 127. Converting an out-of-range `int` to `int8_t` via `static_cast` is **implementation-defined behaviour** in C++17 (not undefined behaviour — C++20 tightened this to mandatory two's-complement wraparound), and on this compiler it wraps silently rather than raising an error. Concretely: quantizing `20.0f` at `scale=0.1` produced `q = 200`, and `static_cast<int8_t>(200)` came out as **-56**, not some obviously-broken giant number. A large, clearly-positive input silently became a plausible-looking, completely wrong *negative* value — exactly the kind of bug that's dangerous precisely because it doesn't look broken.

That same first commit already included the test suite, and two of its checks (`"extreme positive value clamps to int8 max (127)"`, `"extreme ... min (-128)"`) failed immediately against it. A second commit added the missing bound:

```cpp
int q = static_cast<int>(std::round(x / qp.scale)) + qp.zero_point;
q = std::clamp(q, -128, 127);
return static_cast<int8_t>(q);
```

Commit history (`git log -p`): commit 1 has the bug and its own two clamping checks already failing; commit 2 adds one line and turns all checks green. The Catch2 suite in `tests/` was added later, as a follow-up, and includes `"out-of-range values clamp instead of silently wrapping"` as a dedicated, permanent regression test for this same case.


## Known limitations

- `quantized_multiply` dequantizes to float internally rather than doing the integer-only arithmetic a real accelerator kernel would use — chosen deliberately for readability over realism.
- Only elementwise multiply is implemented; no quantized add, matmul, or convolution.
- Single, fixed quantization parameters per tensor (no per-channel scales, which real frameworks often use for better accuracy).

## Building and running

```bash
# Standalone demo, no dependencies
g++ -std=c++17 -Wall quant_validator.cpp -o quant_validator
./quant_validator

# Catch2 test suite (Catch2 is vendored in third_party/, no install needed)
g++ -std=c++17 -Wall -I. tests/test_quant_validator.cpp -o tests_bin
./tests_bin

# Same tests under AddressSanitizer + UBSan (GCC) or implicit-conversion (Clang)
g++ -std=c++17 -g -fsanitize=address,undefined -I. tests/test_quant_validator.cpp -o tests_asan_ubsan && ./tests_asan_ubsan
clang++ -std=c++17 -g -fsanitize=implicit-conversion -I. tests/test_quant_validator.cpp -o tests_implicit_conv && ./tests_implicit_conv
```

(Built and tested with MSVC on Windows: `cl /EHsc /std:c++17 /W4 quant_validator.cpp` for the demo, and the same `/I.` include path for the test suite. CI runs the GCC/Clang builds above on Linux.)
