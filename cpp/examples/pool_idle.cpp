#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class Pool {
 public:
  explicit Pool(unsigned n) {
    for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { run(); });
  }
  ~Pool() {
    {
      std::lock_guard lk(m_);
      stopping_ = true;
    }
    cv_.notify_all();
  }
  void post(std::function<void()> f) {
    {
      std::lock_guard lk(m_);
      tasks_.push_back(std::move(f));
      ++pending_;
    }
    cv_.notify_one();
  }
  void wait_idle() {
    std::unique_lock lk(m_);
    idle_cv_.wait(lk, [&] { return pending_ == 0; });
  }

 private:
  void run() {
    for (;;) {
      std::function<void()> job;
      {
        std::unique_lock lk(m_);
        cv_.wait(lk, [&] { return stopping_ || !tasks_.empty(); });
        if (tasks_.empty()) return;
        job = std::move(tasks_.front());
        tasks_.pop_front();
      }
      job();
      {
        std::lock_guard lk(m_);
        if (--pending_ == 0) idle_cv_.notify_all();
      }
    }
  }
  std::mutex m_;
  std::condition_variable cv_, idle_cv_;
  std::deque<std::function<void()>> tasks_;
  int pending_ = 0;   // 已提交、还没执行完的任务数
  bool stopping_ = false;
  std::vector<std::jthread> workers_;
};

int main() {
  Pool pool(4);
  std::atomic<int> done{0};
  for (int i = 0; i < 1000; ++i) pool.post([&] { done.fetch_add(1, std::memory_order_relaxed); });
  pool.wait_idle();
  std::printf("wait_idle 返回时完成了 %d 个任务\n", done.load());
}
