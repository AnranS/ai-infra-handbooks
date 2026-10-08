#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory_resource>
#include <new>
#include <vector>

static int g_allocs = 0;
void* operator new(std::size_t n) {
  ++g_allocs;
  if (void* p = std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

std::size_t step_std() {
  std::vector<int> ids;
  for (int i = 0; i < 1000; ++i) ids.push_back(i);
  std::vector<float> w(512);
  return ids.size() + w.size();
}

std::size_t step_pmr(std::pmr::memory_resource* mr) {
  std::pmr::vector<int> ids(mr);
  for (int i = 0; i < 1000; ++i) ids.push_back(i);
  std::pmr::vector<float> w(512, mr);
  return ids.size() + w.size();
}

int main() {
  int before = g_allocs;
  for (int s = 0; s < 3; ++s) step_std();
  std::printf("std::vector：3 步共 %d 次堆分配\n", g_allocs - before);

  alignas(64) static std::byte buf[64 * 1024];
  before = g_allocs;
  for (int s = 0; s < 3; ++s) {
    // 每一步在同一块缓冲区上建一个单调资源：只分配不释放，析构时整体丢弃
    std::pmr::monotonic_buffer_resource step_mem(buf, sizeof buf, std::pmr::null_memory_resource());
    step_pmr(&step_mem);
  }
  std::printf("pmr::vector + 预分配缓冲区：3 步共 %d 次堆分配\n", g_allocs - before);
}
