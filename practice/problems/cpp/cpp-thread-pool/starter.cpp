#include <functional>
#include <future>
#include <thread>
#include <type_traits>
#include <vector>

class ThreadPool {
 public:
  explicit ThreadPool(unsigned n) { (void)n; }   // TODO：启动工作线程

  template <class F>
  auto submit(F f) -> std::future<std::invoke_result_t<F>> {
    // TODO：放进任务队列，由工作线程执行。下面是"在调用者线程里直接执行"的错误写法
    std::packaged_task<std::invoke_result_t<F>()> task(std::move(f));
    auto fut = task.get_future();
    task();
    return fut;
  }
};
