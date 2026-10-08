#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

template <class T>
class BlockingQueue {
 public:
  bool push(T v) {                 // 已关闭时返回 false
    {
      std::lock_guard lk(m_);
      if (closed_) return false;
      q_.push_back(std::move(v));
    }
    cv_.notify_one();              // 在锁外通知：被唤醒的线程不用马上又等锁
    return true;
  }
  std::optional<T> pop() {         // 队列为空时阻塞；关闭且取空后返回 nullopt
    std::unique_lock lk(m_);
    cv_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }
  void close() {
    {
      std::lock_guard lk(m_);
      closed_ = true;
    }
    cv_.notify_all();              // 叫醒所有在等的消费者，让它们看到"已关闭"
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;                // 由 m_ 保护
  bool closed_ = false;            // 由 m_ 保护
};
