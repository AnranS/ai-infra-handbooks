#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

static bool aligned(const void* p, std::size_t a) { return reinterpret_cast<std::uintptr_t>(p) % a == 0; }

int main() {
  pj::run("example_alignment", [] {
    Arena arena(1 << 16);
    for (std::size_t a : {1, 2, 4, 8, 16, 64, 256, 4096}) {
      void* p = arena.allocate(3, a);
      pj::require(aligned(p, a), "地址应该按 " + std::to_string(a) + " 字节对齐");
    }
    pj::require(aligned(arena.allocate(8), alignof(std::max_align_t)), "默认按 max_align_t 对齐");
  });
  pj::run("no_overlap", [] {
    Arena arena(1 << 16);
    std::vector<std::pair<unsigned char*, std::size_t>> blocks;
    for (int i = 0; i < 50; ++i) {
      std::size_t n = 17 + i * 5;
      auto* p = static_cast<unsigned char*>(arena.allocate(n, std::size_t(1) << (i % 7)));
      std::memset(p, i, n);   // 写满整块：越界会被 ASan 发现，重叠会覆盖别的块
      blocks.push_back({p, n});
    }
    for (int i = 0; i < 50; ++i)
      for (std::size_t j = 0; j < blocks[i].second; ++j)
        pj::require(blocks[i].first[j] == (unsigned char)i, "第 " + std::to_string(i) + " 块的内容被别的分配覆盖了");
  });
  pj::run("capacity_and_used", [] {
    Arena arena(4096);
    arena.allocate(4000, 1);
    pj::require_eq(arena.used(), std::size_t(4000), "used");
    pj::require_throws<std::bad_alloc>([&] { arena.allocate(200, 1); }, "超出容量应该抛 bad_alloc");
    pj::require_eq(arena.used(), std::size_t(4000), "失败的分配不应该改变 used");
    arena.allocate(96, 1);
    pj::require_eq(arena.used(), std::size_t(4096), "正好用满");
    pj::require_eq(arena.capacity(), std::size_t(4096), "capacity");
  });
  pj::run("reset_reuses", [] {
    Arena arena(1 << 16);
    void* a1 = arena.allocate(100, 64);
    void* b1 = arena.allocate(10, 8);
    arena.reset();
    pj::require_eq(arena.used(), std::size_t(0), "reset 之后 used 为 0");
    void* a2 = arena.allocate(100, 64);
    void* b2 = arena.allocate(10, 8);
    pj::require(a1 == a2 && b1 == b2, "同样的请求序列在 reset 之后应该得到同样的地址");
    void* z = arena.allocate(0, 32);
    pj::require(z != nullptr && aligned(z, 32), "n=0 也返回对齐的有效地址");
  });
}
