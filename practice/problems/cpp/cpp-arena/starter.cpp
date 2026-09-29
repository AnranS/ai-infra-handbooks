#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

class Arena {
 public:
  explicit Arena(std::size_t capacity) : buf_(static_cast<std::byte*>(std::malloc(capacity))), cap_(capacity) {}
  ~Arena() { std::free(buf_); }
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void* allocate(std::size_t n, std::size_t align = alignof(std::max_align_t)) {
    (void)align;               // TODO：对齐、越界检查
    void* p = buf_ + off_;
    off_ += n;
    return p;
  }
  void reset() {}              // TODO
  std::size_t used() const { return off_; }
  std::size_t capacity() const { return cap_; }

 private:
  std::byte* buf_;
  std::size_t cap_, off_ = 0;
};
