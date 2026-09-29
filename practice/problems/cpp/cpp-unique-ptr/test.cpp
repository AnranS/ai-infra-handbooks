#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

static int g_alive = 0;
struct Req {
  int id;
  explicit Req(int i) : id(i) { ++g_alive; }
  ~Req() { --g_alive; }
};

struct CountingDeleter {   // 有状态的删除器：记录调用次数
  int* calls;
  void operator()(Req* p) const {
    ++*calls;
    delete p;
  }
};

int main() {
  pj::run("example_basic", [] {
    {
      UniquePtr<Req> p(new Req(7));
      pj::require(bool(p) && p->id == 7 && (*p).id == 7, "解引用和 -> 应该能访问对象");
      pj::require_eq(g_alive, 1, "存活对象数");
    }
    pj::require_eq(g_alive, 0, "离开作用域后对象应该被删除");
  });
  pj::run("move_only", [] {
    pj::require(!std::is_copy_constructible_v<UniquePtr<Req>> && !std::is_copy_assignable_v<UniquePtr<Req>>, "不能拷贝");
    pj::require(std::is_nothrow_move_constructible_v<UniquePtr<Req>> && std::is_nothrow_move_assignable_v<UniquePtr<Req>>,
                "移动要标 noexcept");
    UniquePtr<Req> a(new Req(1));
    Req* raw = a.get();
    UniquePtr<Req> b = std::move(a);
    pj::require(!a && b.get() == raw, "移动构造转移所有权");
    UniquePtr<Req> c(new Req(2));
    c = std::move(b);
    pj::require_eq(g_alive, 1, "移动赋值应该先删除自己原来的对象");
    pj::require(c.get() == raw && !b, "移动赋值之后");
    std::vector<UniquePtr<Req>> v;
    for (int i = 0; i < 10; ++i) v.emplace_back(new Req(i));
    pj::require_eq(g_alive, 11, "放进 vector 的对象都活着");
  });
  pj::run("release_reset", [] {
    UniquePtr<Req> a(new Req(3));
    Req* raw = a.release();
    pj::require(!a && raw->id == 3, "release 放弃所有权");
    pj::require_eq(g_alive, 1, "release 不删除对象");
    a.reset(raw);
    a.reset(new Req(4));
    pj::require(g_alive == 1 && a->id == 4, "reset 删除旧对象、接管新对象");
    a.reset();
    pj::require(!a && g_alive == 0, "reset() 删除并置空");
    UniquePtr<Req> n(nullptr);
    pj::require(!n, "nullptr 构造");
  });
  pj::run("custom_deleter", [] {
    int calls = 0;
    {
      UniquePtr<Req, CountingDeleter> a(new Req(5), CountingDeleter{&calls});
      UniquePtr<Req, CountingDeleter> b = std::move(a);   // 删除器跟着移动
      pj::require(b.get_deleter().calls == &calls, "移动时删除器也应该移动");
    }
    pj::require_eq(calls, 1, "删除器恰好被调用一次");
    pj::require_eq(g_alive, 0, "对象已被删除");
  });
  pj::run("zero_overhead", [] {
    pj::require_eq(sizeof(UniquePtr<Req>), sizeof(Req*), "默认删除器下 sizeof 应该等于一个指针");
  });
}
