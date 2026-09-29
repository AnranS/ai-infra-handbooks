#include <optional>
#include <stdexcept>
#include <vector>

class BlockPool {
 public:
  explicit BlockPool(int num_blocks) : ref_(num_blocks, 0) {
    for (int i = num_blocks - 1; i >= 0; --i) free_.push_back(i);   // 栈顶是 0
  }
  std::optional<std::vector<int>> allocate(int n) {
    if (n < 0 || n > num_free()) return std::nullopt;
    std::vector<int> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
      out.push_back(free_.back());
      free_.pop_back();
      ref_[out.back()] = 1;
    }
    return out;
  }
  void retain(int b) {
    if (ref_.at(b) == 0) throw std::logic_error("retain 了一个空闲块");
    ++ref_[b];
  }
  void release(int b) {
    if (ref_.at(b) == 0) throw std::logic_error("重复释放");
    if (--ref_[b] == 0) free_.push_back(b);
  }
  int make_writable(int b) {
    if (ref_.at(b) == 0) throw std::logic_error("写一个空闲块");
    if (ref_[b] == 1) return b;
    auto fresh = allocate(1);
    if (!fresh) return -1;     // 没有空闲块：什么都不改
    release(b);
    return (*fresh)[0];
  }
  int num_free() const { return static_cast<int>(free_.size()); }
  int refcount(int b) const { return ref_.at(b); }

 private:
  std::vector<int> free_;
  std::vector<int> ref_;
};
