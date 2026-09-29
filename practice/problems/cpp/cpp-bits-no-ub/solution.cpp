#include <bit>
#include <cstdint>
#include <cstring>

// 按位查看 float：std::bit_cast（C++20）或 memcpy，不要用 reinterpret_cast（违反严格别名规则）
std::uint32_t float_bits(float f) { return std::bit_cast<std::uint32_t>(f); }
float bits_float(std::uint32_t u) { return std::bit_cast<float>(u); }

// fp32 → bf16：保留高 16 位，按"就近、平局取偶"舍入；NaN 保持是 NaN（置上静默位）
std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u = float_bits(f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  std::uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);   // 低 16 位超过一半进位；正好一半时看保留部分的最低位
  return static_cast<std::uint16_t>((u + rounding) >> 16);
}
float bf16_to_f32(std::uint16_t h) { return bits_float(static_cast<std::uint32_t>(h) << 16); }

// 溢出时返回 false 且不写 out；先判断再相加，永远不执行会溢出的加法
bool checked_add(std::int64_t a, std::int64_t b, std::int64_t* out) {
  if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) return false;
  *out = a + b;
  return true;
}

// ⌊(a + b) / 2⌋，不溢出：换成更宽的类型再算
std::int32_t midpoint(std::int32_t a, std::int32_t b) {
  std::int64_t s = static_cast<std::int64_t>(a) + b;
  return static_cast<std::int32_t>(s >= 0 ? s / 2 : -((-s + 1) / 2));
}

// 从任意地址读 4 个字节（机器字节序）：memcpy 对齐与否都合法，编译器会优化成一条 load 指令
std::uint32_t load_u32(const unsigned char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);
  return v;
}
