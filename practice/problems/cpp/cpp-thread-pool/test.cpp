#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

using namespace std::chrono_literals;

int main() {
  pj::run("example_results", [] {
    ThreadPool pool(4);
    std::vector<std::future<long>> fs;
    for (long i = 0; i < 100; ++i) fs.push_back(pool.submit([i] { return i * i; }));
    long sum = 0;
    for (auto& f : fs) sum += f.get();
    pj::require_eq(sum, 328350L, "0 到 99 的平方和");
  });
  pj::run("exception_propagates", [] {
    ThreadPool pool(2);
    auto f = pool.submit([]() -> int { throw std::runtime_error("分词失败"); });
    std::string msg;
    try {
      f.get();
    } catch (const std::runtime_error& e) {
      msg = e.what();
    }
    pj::require_eq(msg, std::string("分词失败"), "任务里的异常应该在 get() 时抛出");
  });
  pj::run("runs_in_parallel", [] {
    const int n = 4;
    ThreadPool pool(n);
    std::atomic<int> started{0};
    std::vector<std::future<bool>> fs;
    for (int i = 0; i < n; ++i) {
      fs.push_back(pool.submit([&] {
        started.fetch_add(1);
        auto deadline = std::chrono::steady_clock::now() + 3s;
        while (started.load() < n && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
        return started.load() == n;   // 只有 n 个任务同时在运行，才会都看到 n
      }));
    }
    bool all = true;
    for (auto& f : fs) all = f.get() && all;
    pj::require(all, "n 个任务应该能同时运行（线程池里要有 n 个工作线程）");
  });
  pj::run("destructor_drains", [] {
    std::atomic<int> done{0};
    {
      ThreadPool pool(2);
      for (int i = 0; i < 200; ++i) {
        pool.submit([&] {
          std::this_thread::sleep_for(100us);
          done.fetch_add(1);
        });
      }
    }   // 析构：执行完所有已提交的任务
    pj::require_eq(done.load(), 200, "析构之前提交的任务都应该执行完");
  });
}
