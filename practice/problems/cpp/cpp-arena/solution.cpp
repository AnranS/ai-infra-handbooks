#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

class Arena {
 public:
  explicit Arena(std::size_t capacity)
      : buf_(static_cast<std::byte*>(std::aligned_alloc(4096, capacity))), cap_(capacity) {
    if (!buf_) throw std::bad_alloc();
  }
  ~Arena() { std::free(buf_); }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void* allocate(std::size_t n, std::size_t align = alignof(std::max_align_t)) {
    auto base = reinterpret_cast<std::uintptr_t>(buf_);
    std::uintptr_t p = (base + off_ + align - 1) & ~(std::uintptr_t(align) - 1);   // 对齐绝对地址
    std::size_t end = p - base + n;
    if (end > cap_) throw std::bad_alloc();
    off_ = end;
    return reinterpret_cast<void*>(p);
  }
  void reset() { off_ = 0; }
  std::size_t used() const { return off_; }
  std::size_t capacity() const { return cap_; }

 private:
  std::byte* buf_;
  std::size_t cap_, off_ = 0;
};
