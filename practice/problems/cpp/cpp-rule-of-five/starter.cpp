#include <algorithm>
#include <cstddef>
#include <utility>

class Tensor1D {
 public:
  Tensor1D() = default;
  Tensor1D(std::size_t n, float value) : n_(n), data_(new float[n]) { std::fill(data_, data_ + n, value); }
  // TODO：析构函数、拷贝构造、拷贝赋值、移动构造、移动赋值

  std::size_t size() const { return n_; }
  float* data() { return data_; }
  const float* data() const { return data_; }
  float& operator[](std::size_t i) { return data_[i]; }
  float operator[](std::size_t i) const { return data_[i]; }

 private:
  std::size_t n_ = 0;
  float* data_ = nullptr;
};
