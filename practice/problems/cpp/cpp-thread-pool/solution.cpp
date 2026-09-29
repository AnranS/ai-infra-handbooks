#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(unsigned n) {
    for (unsigned i = 0; i < n; ++i) workers_.emplace_back([this] { run(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard lk(m_);
      stopping_ = true;
    }
    cv_.notify_all();
  }   // workers_ 最先析构：jthread 自动 join

  template <class F>
  auto submit(F f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto fut = task->get_future();
    {
      std::lock_guard lk(m_);
      tasks_.emplace_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return fut;
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
    }
  }
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;
  bool stopping_ = false;
  std::vector<std::jthread> workers_;   // 最后声明
};
