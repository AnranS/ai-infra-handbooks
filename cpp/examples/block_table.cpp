#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

class BlockTable {
 public:
  explicit BlockTable(std::size_t n) : n_(n), ids_(new int[n]) { std::fill(ids_, ids_ + n_, -1); }
  ~BlockTable() { delete[] ids_; }

  BlockTable(const BlockTable& o) : n_(o.n_), ids_(new int[o.n_]) { std::copy(o.ids_, o.ids_ + n_, ids_); }
  BlockTable& operator=(const BlockTable& o) {
    BlockTable tmp(o);   // 可能抛异常，但 *this 还没动
    swap(tmp);           // 不会失败
    return *this;        // tmp 带着旧数据析构
  }
  BlockTable(BlockTable&& o) noexcept : n_(std::exchange(o.n_, 0)), ids_(std::exchange(o.ids_, nullptr)) {}
  BlockTable& operator=(BlockTable&& o) noexcept {
    if (this != &o) {
      delete[] ids_;
      n_ = std::exchange(o.n_, 0);
      ids_ = std::exchange(o.ids_, nullptr);
    }
    return *this;
  }
  void swap(BlockTable& o) noexcept {
    std::swap(n_, o.n_);
    std::swap(ids_, o.ids_);
  }

  int& operator[](std::size_t i) { return ids_[i]; }
  std::size_t size() const { return n_; }

 private:
  std::size_t n_;
  int* ids_;
};

int main() {
  BlockTable a(4);
  a[0] = 7;
  BlockTable b = a;                // 深拷贝
  b[0] = 9;
  std::printf("拷贝：a[0]=%d b[0]=%d\n", a[0], b[0]);

  BlockTable& same = b;
  b = same;                        // 自我赋值
  std::printf("自我赋值后：b[0]=%d\n", b[0]);

  BlockTable c = std::move(a);     // 移动
  std::printf("移动：a.size=%zu c.size=%zu c[0]=%d\n", a.size(), c.size(), c[0]);

  std::vector<BlockTable> tables;
  for (int i = 0; i < 5; ++i) tables.emplace_back(2);   // 扩容时走 noexcept 的移动
  std::printf("tables=%zu\n", tables.size());
}
