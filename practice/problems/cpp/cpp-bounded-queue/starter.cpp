#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

template <class T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : cap_(capacity) {}
  bool push(T v) {                 // TODO：满时阻塞、关闭时返回 false
    std::lock_guard lk(m_);
    q_.push_back(std::move(v));
    return true;
  }
  std::optional<T> pop() {         // TODO：空时阻塞
    std::lock_guard lk(m_);
    if (q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }
  void close() {}                  // TODO
  std::size_t size() {
    std::lock_guard lk(m_);
    return q_.size();
  }

 private:
  std::mutex m_;
  std::deque<T> q_;
  std::size_t cap_;
};
