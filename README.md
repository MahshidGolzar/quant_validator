# quant_validator — int8 quantization, validated

A small, from-scratch int8 quantization validator in C++: converting float32 values to int8 and back (and a quantized "multiply" op), with a hand-written test suite checking the conversion is actually correct.

## Why this exists

Hardware accelerators are often fast specifically because they run low-precision integer arithmetic (int8) rather than full float32 — less data to move, simpler circuits, lower power. So AI framework code that targets an accelerator needs to **quantize** float32 values down to int8, and **dequantize** them back, while keeping track of how much precision that costs. This is a small, honest version of that conversion step, deliberately kept to plain arithmetic rather than anything accelerator-specific — the point is validating a conversion that real framework code performs constantly, not reproducing real accelerator kernels.

## What's in here

- `QuantParams` — a scale and zero-point, the two numbers that define an affine mapping between float and int8 (`int8_value = round(float_value / scale) + zero_point`).
- `quantize_value` / `dequantize_value` — the conversion in each direction, for one value.
- `quantize` / `dequantize` — the same, applied to a whole vector.
- `quantized_multiply` — a small quantized elementwise multiply "op", the kind of thing a framework's accelerator backend implements for a layer. For readability this dequantizes, multiplies in float, then re-quantizes the result; a real accelerator kernel computes the mathematical equivalent in pure integer arithmetic for speed.
- A tiny test "framework" (`check()`, pass/fail reporting) with 10 checks: round-trip accuracy within the expected rounding error, out-of-range clamping, and the quantized multiply op checked against the plain float32 reference calculation.

## Manual fault injection: proving the clamping tests actually catch an out-of-range value

After writing the test suite, I deliberately removed the range-clamping logic from `quantize_value` to check the tests would actually notice — the same manual fault-injection check used in `pytorch-parametric-swish` (not tool-driven mutation testing: one hand-written fault, no mutation score): introduce a known defect, confirm the tests fail, then restore the fix, as evidence the tests have real detection power rather than just happening to pass.

The faulty version computed the int8 value but never checked it was actually representable:

```cpp
int q = static_cast<int>(std::round(x / qp.scale)) + qp.zero_point;
return static_cast<int8_t>(q);   // q might be outside [-128, 127]
```

int8 can only represent -128 to 127. Converting an out-of-range `int` to `int8_t` via `static_cast` is **implementation-defined behaviour** in C++17 (not undefined behaviour — C++20 tightened this to mandatory two's-complement wraparound), and on this compiler it wraps silently rather than raising an error. Concretely: quantizing `20.0f` with this scale produced `q = 200`, and `static_cast<int8_t>(200)` came out as **-56**, not some obviously-broken giant number. A large, clearly-positive input silently became a plausible-looking, completely wrong *negative* value — exactly the kind of bug that's dangerous precisely because it doesn't look broken, and exactly why a test that checks the actual value is more reliable than reading the code.

The two clamping tests (`"extreme positive value clamps to int8 max (127)"`, `"extreme ... min (-128)"`) failed against the faulty version, which is what caught it. The fix adds the missing bound:

```cpp
int q = static_cast<int>(std::round(x / qp.scale)) + qp.zero_point;
q = std::clamp(q, -128, 127);
return static_cast<int8_t>(q);
```

Commit history (`git log -p`): a first commit with the fault and the two clamping tests already failing, then a second commit that adds one line and turns all 10 checks green.

## Known limitations

- `quantized_multiply` dequantizes to float internally rather than doing the integer-only arithmetic a real accelerator kernel would use — chosen deliberately for readability over realism.
- Only elementwise multiply is implemented; no quantized add, matmul, or convolution.
- Not yet built or run under a sanitizer (Clang/GCC's `-fsanitize=implicit-conversion` or `-Wconversion` would be the natural tool to flag the narrowing cast this project's bug involved) — not done here because only MSVC, which lacks these specific checks, was available while building this.
- Single, fixed quantization parameters per tensor (no per-channel scales, which real frameworks often use for better accuracy).

## Building and running

```bash
g++ -std=c++17 -Wall quant_validator.cpp -o quant_validator
./quant_validator
```

(Built and tested with MSVC on Windows: `cl /EHsc /std:c++17 /W4 quant_validator.cpp`.)
