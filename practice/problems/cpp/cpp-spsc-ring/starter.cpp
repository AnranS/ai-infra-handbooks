#include <array>
#include <atomic>
#include <cstddef>

template <class T, std::size_t Cap>
class SpscQueue {
 public:
  bool try_push(const T& v) {   // TODO
    buf_[tail_ % Cap] = v;
    ++tail_;
    return true;
  }
  bool try_pop(T& out) {        // TODO
    if (head_ == tail_) return false;
    out = buf_[head_ % Cap];
    ++head_;
    return true;
  }

 private:
  std::size_t head_ = 0, tail_ = 0;
  std::array<T, Cap> buf_{};
};
