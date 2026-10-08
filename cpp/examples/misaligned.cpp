#include <cstdint>
#include <cstdio>

int main() {
  alignas(4) unsigned char buf[8] = {0, 1, 0, 0, 0, 2, 0, 0};
  // 从偏移 1 读一个 uint32：地址没有按 4 字节对齐，同时也违反了严格别名
  auto* p = reinterpret_cast<const std::uint32_t*>(buf + 1);
  std::printf("%u\n", *p);
}
