#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>

template <class T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t cap) : cap_(cap) {}
  bool push(T v) {
    std::unique_lock lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    max_seen_ = std::max(max_seen_, q_.size());
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
    not_full_.notify_one();   // 空出了一个位置
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
  std::size_t max_seen() {
    std::lock_guard lk(m_);
    return max_seen_;
  }

 private:
  std::mutex m_;
  std::condition_variable not_empty_, not_full_;
  std::deque<T> q_;
  std::size_t cap_, max_seen_ = 0;
  bool closed_ = false;
};

int main() {
  BoundedQueue<int> q(4);
  int handled = 0;
  std::thread consumer([&] {
    while (auto v = q.pop()) {
      ++handled;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));   // 慢消费者
    }
  });
  for (int i = 0; i < 20; ++i) q.push(i);
  q.close();
  consumer.join();
  std::printf("处理 %d 个，队列长度从未超过 4：%s\n", handled, q.max_seen() <= 4 ? "是" : "否");
}
