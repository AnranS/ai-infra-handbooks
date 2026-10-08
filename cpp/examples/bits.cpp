#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>

// bf16 就是 float 的高 16 位（这里直接截断；正确的做法是按"舍入到最近的偶数"进位）
std::uint16_t to_bf16_truncate(float x) { return std::bit_cast<std::uint32_t>(x) >> 16; }

int main() {
  float x = 1.0f;
  std::printf("1.0f 的位模式：0x%08x\n", std::bit_cast<std::uint32_t>(x));
  std::uint32_t u;
  std::memcpy(&u, &x, sizeof u);   // C++20 之前的写法，同样正确
  std::printf("memcpy 得到：0x%08x\n", u);
  std::printf("3.14159f 截断成 bf16：0x%04x\n", to_bf16_truncate(3.14159f));
}
