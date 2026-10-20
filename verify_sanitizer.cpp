// Scratch file, not part of the library: checks whether various sanitizers
// actually fire on the exact cast pattern this project's original bug
// involved (int q=200; static_cast<int8_t>(q)). Deleted after verification.
#include <cstdint>
#include <cstdio>

int main() {
  int q = 200;  // same value quantize_value(20.0f, {0.1f, 0}) produces
  int8_t r = static_cast<int8_t>(q);  // the exact cast from the original bug
  std::printf("r = %d\n", static_cast<int>(r));
  return 0;
}
