#include <array>
#include <atomic>
#include <cstddef>

template <class T, std::size_t Cap>
class SpscQueue {
  static_assert(Cap > 0 && (Cap & (Cap - 1)) == 0, "容量必须是 2 的幂");

 public:
  bool try_push(const T& v) {
    const std::size_t t = tail_.load(std::memory_order_relaxed);        // 只有生产者写 tail
    if (t - head_.load(std::memory_order_acquire) == Cap) return false;  // 满了
    buf_[t & (Cap - 1)] = v;
    tail_.store(t + 1, std::memory_order_release);                       // 先写元素，再发布
    return true;
  }
  bool try_pop(T& out) {
    const std::size_t h = head_.load(std::memory_order_relaxed);        // 只有消费者写 head
    if (h == tail_.load(std::memory_order_acquire)) return false;        // 空了
    out = buf_[h & (Cap - 1)];
    head_.store(h + 1, std::memory_order_release);                       // 槽位可以复用了
    return true;
  }

 private:
  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
  std::array<T, Cap> buf_{};
};
