#include <cstdint>
#include <cstdio>
#include <cstring>

std::uint32_t read_u32(const unsigned char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);   // 对任意地址都正确，编译器会生成一条普通的读指令
  return v;
}

int main() {
  unsigned char buf[8] = {0, 1, 0, 0, 0, 2, 0, 0};
  std::printf("%u %u\n", read_u32(buf + 1), read_u32(buf + 4));
}
