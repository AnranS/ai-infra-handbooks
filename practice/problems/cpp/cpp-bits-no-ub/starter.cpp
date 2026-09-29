#include <cstdint>
#include <cstring>

std::uint32_t float_bits(float f) { return *reinterpret_cast<std::uint32_t*>(&f); }
float bits_float(std::uint32_t u) { return *reinterpret_cast<float*>(&u); }

std::uint16_t f32_to_bf16(float f) { return static_cast<std::uint16_t>(float_bits(f) >> 16); }   // 直接截断
float bf16_to_f32(std::uint16_t h) { return bits_float(static_cast<std::uint32_t>(h) << 16); }

bool checked_add(std::int64_t a, std::int64_t b, std::int64_t* out) {
  std::int64_t s = a + b;                                    // 先加再检查：溢出已经发生了
  if ((a > 0 && b > 0 && s < 0) || (a < 0 && b < 0 && s >= 0)) return false;
  *out = s;
  return true;
}

std::int32_t midpoint(std::int32_t a, std::int32_t b) { return (a + b) / 2; }

std::uint32_t load_u32(const unsigned char* p) { return *reinterpret_cast<const std::uint32_t*>(p); }
