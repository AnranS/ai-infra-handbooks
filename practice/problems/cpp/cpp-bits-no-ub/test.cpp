#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

#include "judge.hpp"
#include "user.cpp"

// 用 double 独立地算"就近、平局取偶"的 bf16，作为参照
static std::uint16_t ref_bf16(std::uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  std::uint32_t lo = u & 0xFFFF0000u, hi = lo + 0x10000u;
  float flo, fhi;
  std::memcpy(&flo, &lo, 4);
  std::memcpy(&fhi, &hi, 4);
  double dl = std::fabs(double(f) - double(flo)), dh = std::fabs(double(fhi) - double(f));
  if (std::isinf(fhi) && !std::isinf(f)) dh = std::fabs(std::ldexp(1.0, 128) - std::fabs(double(f)));
  std::uint16_t a = lo >> 16, b = hi >> 16;
  if (dl < dh) return a;
  if (dh < dl) return b;
  return (a & 1) ? b : a;
}

int main() {
  pj::run("example", [] {
    pj::require_eq(float_bits(1.0f), 0x3F800000u, "1.0f 的位模式");
    pj::require(bits_float(0x40490FDBu) == 3.14159265f, "0x40490FDB 是 π");
    pj::require_eq(f32_to_bf16(1.0f), 0x3F80, "1.0f → bf16");
    std::int64_t out = 0;
    volatile std::int64_t big = INT64_MAX;
    pj::require(!checked_add(big, 1, &out), "INT64_MAX + 1 溢出");
    pj::require(checked_add(40, 2, &out) && out == 42, "40 + 2");
  });

  pj::run("bf16_rounding", [] {
    pj::require_eq(f32_to_bf16(bits_float(0x3F808000u)), 0x3F80, "正好一半、保留部分是偶数：舍去");
    pj::require_eq(f32_to_bf16(bits_float(0x3F818000u)), 0x3F82, "正好一半、保留部分是奇数：进位到偶数");
    pj::require_eq(f32_to_bf16(bits_float(0x3F808001u)), 0x3F81, "超过一半：进位");
    pj::require_eq(f32_to_bf16(-2.5f), 0xC020, "负数");
    pj::require_eq(f32_to_bf16(std::numeric_limits<float>::infinity()), 0x7F80, "+inf");
    pj::require_eq(f32_to_bf16(-std::numeric_limits<float>::infinity()), 0xFF80, "-inf");
    pj::require_eq(f32_to_bf16(bits_float(0x7F7FFFFFu)), 0x7F80, "float 的最大值舍入后溢出成 inf");
    std::uint16_t n = f32_to_bf16(std::numeric_limits<float>::quiet_NaN());
    pj::require((n & 0x7F80) == 0x7F80 && (n & 0x007F) != 0, "NaN 还是 NaN");
    std::uint16_t s = f32_to_bf16(bits_float(0x7F800001u));   // 尾数只有最低位是 1 的 signaling NaN：直接截断会变成 inf
    pj::require((s & 0x7F80) == 0x7F80 && (s & 0x007F) != 0, "尾数只在低 16 位的 NaN 不能变成 inf");
    std::mt19937 rng(0);
    for (int i = 0; i < 200000; ++i) {
      std::uint32_t u = rng();
      if ((u & 0x7F800000u) == 0x7F800000u) continue;
      if (f32_to_bf16(bits_float(u)) != ref_bf16(u)) pj::require(false, "随机的 float 舍入结果与参照不同");
    }
    pj::require(bf16_to_f32(0x3F82) == 1.015625f, "bf16 → f32");
  });

  pj::run("checked_add", [] {
    volatile std::int64_t mx = INT64_MAX, mn = INT64_MIN;
    std::int64_t out = 7;
    pj::require(!checked_add(mn, -1, &out) && out == 7, "INT64_MIN - 1 溢出，且不写 out");
    pj::require(!checked_add(mx, mx, &out), "INT64_MAX + INT64_MAX 溢出");
    pj::require(checked_add(mx, mn, &out) && out == -1, "INT64_MAX + INT64_MIN = -1");
    pj::require(checked_add(mx, 0, &out) && out == INT64_MAX, "加 0 不溢出");
    pj::require(checked_add(mn + 5, -5, &out) && out == INT64_MIN, "正好到 INT64_MIN");
  });

  pj::run("midpoint", [] {
    volatile std::int32_t mx = INT32_MAX, mn = INT32_MIN;
    pj::require_eq(midpoint(mx, mx), INT32_MAX, "两个最大值");
    pj::require_eq(midpoint(mn, mn), INT32_MIN, "两个最小值");
    pj::require_eq(midpoint(mx, mn), -1, "最大值和最小值：⌊-0.5⌋ = -1");
    pj::require_eq(midpoint(-3, 2), -1, "⌊-0.5⌋ = -1（不是向 0 取整）");
    pj::require_eq(midpoint(3, 4), 3, "⌊3.5⌋ = 3");
    pj::require_eq(midpoint(-7, -8), -8, "⌊-7.5⌋ = -8");
  });

  pj::run("unaligned_load", [] {
    alignas(8) unsigned char buf[16];
    for (int i = 0; i < 16; ++i) buf[i] = static_cast<unsigned char>(i * 17 + 3);
    for (int off = 0; off < 12; ++off) {
      std::uint32_t want;
      std::memcpy(&want, buf + off, 4);
      volatile int o = off;
      pj::require_eq(load_u32(buf + o), want, "从偏移 " + std::to_string(off) + " 读 4 个字节");
    }
  });
}
