#pragma once
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
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
  }   // 接着 workers_ 析构（jthread 自动 join）：工作线程做完剩下的任务后退出

  template <class F>
  auto submit(F f) -> std::future<std::invoke_result_t<F>> {
    using R = std::invoke_result_t<F>;
    // packaged_task 只能移动，而 std::function 要求可拷贝，所以放进 shared_ptr
    auto task = std::make_shared<std::packaged_task<R()>>(std::move(f));
    auto fut = task->get_future();
    {
      std::lock_guard lk(m_);
      if (stopping_) throw std::runtime_error("线程池已经关闭");
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
        if (tasks_.empty()) return;   // 已经停止，且队列取空了
        job = std::move(tasks_.front());
        tasks_.pop_front();
      }
      job();                          // 在锁外执行
    }
  }

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> tasks_;   // 由 m_ 保护
  bool stopping_ = false;                     // 由 m_ 保护
  std::vector<std::jthread> workers_;         // 最后声明、最先析构：join 时上面的成员都还活着
};
