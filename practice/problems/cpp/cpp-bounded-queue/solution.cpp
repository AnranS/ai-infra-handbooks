#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

template <class T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : cap_(capacity) {}
  bool push(T v) {
    std::unique_lock lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    lk.unlock();
    not_empty_.notify_one();
    return true;
  }
  std::optional<T> pop() {
    std::unique_lock lk(m_);
    not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    lk.unlock();
    not_full_.notify_one();
    return v;
  }
  void close() {
    {
      std::lock_guard lk(m_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }
  std::size_t size() {
    std::lock_guard lk(m_);
    return q_.size();
  }

 private:
  std::mutex m_;
  std::condition_variable not_empty_, not_full_;
  std::deque<T> q_;
  std::size_t cap_;
  bool closed_ = false;
};
