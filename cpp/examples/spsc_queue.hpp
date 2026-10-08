#pragma once
#include <array>
#include <atomic>
#include <cstddef>

template <class T, std::size_t Cap>
class SpscQueue {
  static_assert(Cap > 0 && (Cap & (Cap - 1)) == 0, "容量必须是 2 的幂");

 public:
  bool try_push(const T& v) {   // 只能由生产者线程调用
    const std::size_t t = prod_.tail.load(std::memory_order_relaxed);
    if (t - prod_.head_cache == Cap) {                                 // 看起来满了：刷新消费者的进度
      prod_.head_cache = cons_.head.load(std::memory_order_acquire);
      if (t - prod_.head_cache == Cap) return false;
    }
    buf_[t & (Cap - 1)] = v;
    prod_.tail.store(t + 1, std::memory_order_release);                // 先写数据，再发布
    return true;
  }
  bool try_pop(T& out) {        // 只能由消费者线程调用
    const std::size_t h = cons_.head.load(std::memory_order_relaxed);
    if (h == cons_.tail_cache) {                                       // 看起来空了：刷新生产者的进度
      cons_.tail_cache = prod_.tail.load(std::memory_order_acquire);
      if (h == cons_.tail_cache) return false;
    }
    out = buf_[h & (Cap - 1)];
    cons_.head.store(h + 1, std::memory_order_release);                // 读完了，这个槽位可以复用
    return true;
  }

 private:
  struct alignas(64) Producer {       // 生产者写的东西放在一个缓存行
    std::atomic<std::size_t> tail{0};
    std::size_t head_cache = 0;
  };
  struct alignas(64) Consumer {       // 消费者写的东西放在另一个缓存行
    std::atomic<std::size_t> head{0};
    std::size_t tail_cache = 0;
  };
  Producer prod_;
  Consumer cons_;
  std::array<T, Cap> buf_{};
};
