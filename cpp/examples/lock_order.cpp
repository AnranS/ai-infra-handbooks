#include <mutex>
#include <thread>

std::mutex cache_mu, sched_mu;

void evict() {        // 先拿缓存锁，再拿调度器锁
  std::lock_guard a(cache_mu);
  std::lock_guard b(sched_mu);
}
void schedule() {     // 顺序反过来
  std::lock_guard a(sched_mu);
  std::lock_guard b(cache_mu);
}

int main() {
  std::thread t1(evict);
  t1.join();
  std::thread t2(schedule);
  t2.join();
}
