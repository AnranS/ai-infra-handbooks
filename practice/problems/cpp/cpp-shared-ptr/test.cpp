#include <atomic>
#include <thread>
#include <utility>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

static std::atomic<int> g_alive{0};
struct Adapter {
  int rank;
  explicit Adapter(int r) : rank(r) { g_alive.fetch_add(1); }
  ~Adapter() { g_alive.fetch_sub(1); }
};

int main() {
  pj::run("example_counts", [] {
    SharedPtr<Adapter> a(new Adapter(8));
    pj::require_eq(a.use_count(), 1L, "新建时的引用计数");
    {
      SharedPtr<Adapter> b = a;
      pj::require_eq(a.use_count(), 2L, "拷贝之后的引用计数");
      pj::require(b.get() == a.get() && b->rank == 8, "拷贝指向同一个对象");
    }
    pj::require_eq(a.use_count(), 1L, "副本销毁之后的引用计数");
    a.reset();
    pj::require(!a && a.use_count() == 0, "reset 之后为空");
    pj::require_eq(g_alive.load(), 0, "最后一个持有者释放后对象应该被删除");
  });
  pj::run("assign_and_move", [] {
    SharedPtr<Adapter> a(new Adapter(1)), b(new Adapter(2));
    b = a;
    pj::require_eq(g_alive.load(), 1, "拷贝赋值应该释放 b 原来的对象");
    pj::require_eq(a.use_count(), 2L, "拷贝赋值之后的计数");
    SharedPtr<Adapter>& alias = a;
    a = alias;   // 自我赋值
    pj::require_eq(a.use_count(), 2L, "自我赋值不改变计数");
    SharedPtr<Adapter> c = std::move(a);
    pj::require(!a && c.use_count() == 2, "移动不改变计数，源对象变空");
    b = std::move(c);
    pj::require(!c && b.use_count() == 1 && b->rank == 1, "移动赋值");
    SharedPtr<Adapter> empty;
    SharedPtr<Adapter> empty2 = empty;
    pj::require(empty2.use_count() == 0, "空指针的拷贝");
  });
  pj::run("concurrent_copies", [] {
    SharedPtr<Adapter> root(new Adapter(42));
    std::atomic<long> sum{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
      ts.emplace_back([copy = root, &sum]() mutable {   // 每个线程各拿一个副本
        for (int i = 0; i < 2000; ++i) {
          SharedPtr<Adapter> local = copy;              // 拷贝、读取、销毁
          sum.fetch_add(local->rank, std::memory_order_relaxed);
        }
        copy.reset();
      });
    }
    root.reset();                                       // 主线程先放手：最后一个线程负责删除
    for (auto& th : ts) th.join();
    pj::require_eq(sum.load(), 8L * 2000 * 42, "所有线程都读到了对象");
    pj::require_eq(g_alive.load(), 0, "所有副本销毁后对象恰好被删除一次");
  });
}
