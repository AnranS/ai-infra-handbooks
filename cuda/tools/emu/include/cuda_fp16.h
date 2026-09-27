// Emulator half type: real 16-bit IEEE storage (so uint4 copies move 8 halves), converted on use.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>

struct __half {
  uint16_t bits;
};
using half = __half;

inline __half __float2half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = int32_t((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  uint16_t h;
  if (((x >> 23) & 0xff) == 0xff) {                      // inf / nan
    h = uint16_t(sign | 0x7c00u | (mant ? 0x200u : 0));
  } else if (exp >= 31) {                                 // overflow -> inf
    h = uint16_t(sign | 0x7c00u);
  } else if (exp <= 0) {                                  // subnormal or zero
    if (exp < -10) {
      h = uint16_t(sign);
    } else {
      mant |= 0x800000u;
      uint32_t shift = uint32_t(14 - exp);
      uint32_t hm = mant >> shift;
      uint32_t rem = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
      if (rem > halfway || (rem == halfway && (hm & 1))) ++hm;
      h = uint16_t(sign | hm);
    }
  } else {
    uint32_t hm = mant >> 13, rem = mant & 0x1fffu;
    uint32_t v = (uint32_t(exp) << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) ++v;   // round to nearest even (may carry into exp)
    h = uint16_t(sign | v);
  }
  return __half{h};
}

inline float __half2float(__half hv) {
  uint32_t h = hv.bits, sign = (h & 0x8000u) << 16, exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      int e = -1;
      do { ++e; mant <<= 1; } while (!(mant & 0x400u));
      x = sign | (uint32_t(127 - 15 - e) << 23) | ((mant & 0x3ffu) << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
