#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <vector>

template <class T, std::size_t Align>
struct AlignedAllocator {
  using value_type = T;
  template <class U>
  struct rebind {
    using other = AlignedAllocator<U, Align>;
  };
  AlignedAllocator() = default;
  template <class U>
  AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Align})); }
  void deallocate(T* p, std::size_t n) noexcept { ::operator delete(p, n * sizeof(T), std::align_val_t{Align}); }
  template <class U>
  bool operator==(const AlignedAllocator<U, Align>&) const noexcept { return true; }
};

template <class V>
bool aligned(const V& v, std::size_t a) { return reinterpret_cast<std::uintptr_t>(v.data()) % a == 0; }

int main() {
  std::vector<float, AlignedAllocator<float, 64>> v(1000);
  std::vector<float, AlignedAllocator<float, 4096>> page(1024);
  v.resize(5000);   // 扩容后仍然对齐
  std::printf("64 字节对齐：%s，页对齐：%s\n", aligned(v, 64) ? "是" : "否", aligned(page, 4096) ? "是" : "否");
}
