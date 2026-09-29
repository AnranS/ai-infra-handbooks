#include <algorithm>
#include <cstddef>
#include <utility>

class Tensor1D {
 public:
  Tensor1D() = default;
  Tensor1D(std::size_t n, float value) : n_(n), data_(new float[n]) { std::fill(data_, data_ + n, value); }
  ~Tensor1D() { delete[] data_; }

  Tensor1D(const Tensor1D& o) : n_(o.n_), data_(o.n_ ? new float[o.n_] : nullptr) {
    std::copy(o.data_, o.data_ + n_, data_);
  }
  Tensor1D& operator=(const Tensor1D& o) {
    Tensor1D tmp(o);   // 可能抛出 bad_alloc，此时 *this 没有被改动
    swap(tmp);
    return *this;
  }
  Tensor1D(Tensor1D&& o) noexcept : n_(std::exchange(o.n_, 0)), data_(std::exchange(o.data_, nullptr)) {}
  Tensor1D& operator=(Tensor1D&& o) noexcept {
    if (this != &o) {
      delete[] data_;
      n_ = std::exchange(o.n_, 0);
      data_ = std::exchange(o.data_, nullptr);
    }
    return *this;
  }
  void swap(Tensor1D& o) noexcept {
    std::swap(n_, o.n_);
    std::swap(data_, o.data_);
  }

  std::size_t size() const { return n_; }
  float* data() { return data_; }
  const float* data() const { return data_; }
  float& operator[](std::size_t i) { return data_[i]; }
  float operator[](std::size_t i) const { return data_[i]; }

 private:
  std::size_t n_ = 0;
  float* data_ = nullptr;
};
