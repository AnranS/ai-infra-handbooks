#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>

class Arena {
 public:
  explicit Arena(std::size_t capacity)   // capacity 必须是 4096 的倍数
      : buf_(static_cast<std::byte*>(std::aligned_alloc(4096, capacity))), cap_(capacity) {
    if (!buf_) throw std::bad_alloc();
  }
  void* allocate(std::size_t n, std::size_t align = alignof(std::max_align_t)) {
    auto base = reinterpret_cast<std::uintptr_t>(buf_.get());
    std::uintptr_t p = (base + off_ + align - 1) & ~(std::uintptr_t(align) - 1);   // 向上对齐（align 是 2 的幂）
    std::size_t end = p - base + n;
    if (end > cap_) throw std::bad_alloc();
    off_ = end;
    peak_ = std::max(peak_, off_);
    return reinterpret_cast<void*>(p);
  }
  template <class T>
  T* alloc_array(std::size_t n) { return static_cast<T*>(allocate(n * sizeof(T), alignof(T))); }
  void reset() { off_ = 0; }   // 一次性"释放"全部
  std::size_t used() const { return off_; }
  std::size_t peak() const { return peak_; }

 private:
  struct Free {
    void operator()(std::byte* p) const { std::free(p); }
  };
  std::unique_ptr<std::byte, Free> buf_;
  std::size_t cap_, off_ = 0, peak_ = 0;
};

int main() {
  Arena arena(1 << 20);
  const void* first = nullptr;
  for (int step = 0; step < 3; ++step) {
    int batch = 8 + step * 8;   // 每一步的 batch 大小不同
    auto* positions = arena.alloc_array<std::int32_t>(batch);
    auto* slot_mapping = arena.alloc_array<std::int64_t>(batch);
    auto* logits = static_cast<float*>(arena.allocate(batch * 1024 * sizeof(float), 64));   // 按缓存行对齐
    for (int i = 0; i < batch; ++i) {
      positions[i] = i;
      slot_mapping[i] = i;
      logits[i] = 0;
    }
    if (step == 0) first = positions;
    std::printf("step %d：batch=%d 用了 %zu 字节，positions 与第一步同地址：%s\n", step, batch, arena.used(),
                positions == first ? "是" : "否");
    arena.reset();
  }
  std::printf("峰值 %zu 字节\n", arena.peak());
}
