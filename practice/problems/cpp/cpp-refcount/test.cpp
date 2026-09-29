#include <atomic>
#include <thread>
#include <utility>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

struct Tracked : RefCounted {
  static inline std::atomic<int> destroyed{0};
  static inline std::atomic<long> checksum{0};
  int slots[8] = {};
  int value;
  explicit Tracked(int v) : value(v) {}
  ~Tracked() override {
    long s = 0;
    for (int x : slots) s += x;                        // 读其他线程在释放引用之前写下的数据
    checksum.store(s, std::memory_order_relaxed);
    destroyed.fetch_add(1, std::memory_order_relaxed);
  }
};

int main() {
  pj::run("example", [] {
    Tracked::destroyed = 0;
    {
      RefPtr<Tracked> a = make_ref<Tracked>(7);
      pj::require_eq(a->use_count(), 1, "新对象的计数");
      {
        RefPtr<Tracked> b = a;
        pj::require_eq(a->use_count(), 2, "拷贝之后的计数");
        pj::require_eq(b->value, 7, "拷贝指向同一个对象");
      }
      pj::require_eq(a->use_count(), 1, "拷贝析构之后的计数");
      pj::require_eq(Tracked::destroyed.load(), 0, "还有引用时不能销毁");
    }
    pj::require_eq(Tracked::destroyed.load(), 1, "最后一个引用消失时销毁，而且只销毁一次");
  });

  pj::run("move_and_assign", [] {
    Tracked::destroyed = 0;
    RefPtr<Tracked> a = make_ref<Tracked>(1);
    RefPtr<Tracked> m = std::move(a);
    pj::require(!a && m, "移动构造之后源对象为空");
    pj::require_eq(m->use_count(), 1, "移动不改变计数");
    RefPtr<Tracked> c = make_ref<Tracked>(2);
    c = m;
    pj::require_eq(Tracked::destroyed.load(), 1, "拷贝赋值释放了 c 原来指向的对象");
    pj::require_eq(m->use_count(), 2, "拷贝赋值之后的计数");
    c = c;
    pj::require_eq(m->use_count(), 2, "自赋值不改变计数");
    RefPtr<Tracked> d = make_ref<Tracked>(3);
    d = std::move(c);
    pj::require(!c && d && d.get() == m.get(), "移动赋值之后源对象为空");
    pj::require_eq(Tracked::destroyed.load(), 2, "移动赋值释放了 d 原来指向的对象");
    pj::require_eq(m->use_count(), 2, "移动赋值不改变计数");
    d.reset();
    m.reset();
    pj::require_eq(Tracked::destroyed.load(), 3, "全部释放");
    pj::require(!m && m.get() == nullptr, "reset 之后为空");
  });

  pj::run("concurrent_release", [] {
    for (int round = 0; round < 40; ++round) {
      Tracked::destroyed = 0;
      Tracked::checksum = 0;
      RefPtr<Tracked> obj = make_ref<Tracked>(round);
      std::vector<std::thread> ts;
      for (int i = 0; i < 8; ++i) {
        ts.emplace_back([p = obj, i]() mutable {
          p->slots[i] = i + 1;                         // 每个线程写自己的槽位
          RefPtr<Tracked> local = p;
          p.reset();
          local.reset();                               // 可能是最后一个引用：析构在这个线程上发生
        });
      }
      obj.reset();
      for (auto& t : ts) t.join();
      pj::require_eq(Tracked::destroyed.load(), 1, "并发释放之后恰好销毁一次");
      pj::require_eq(Tracked::checksum.load(), 36L, "析构时看到了所有线程写入的数据");
    }
  });
}
