#include <cstddef>
#include <set>
#include <type_traits>
#include <utility>

#include "judge.hpp"

static std::set<int> g_open;
static int g_next = 3, g_bad_close = 0;
int fake_open() {
  int fd = g_next++;
  g_open.insert(fd);
  return fd;
}
void fake_close(int fd) {
  if (!g_open.erase(fd)) ++g_bad_close;
}

#include "user.cpp"

int main() {
  pj::run("example_scope", [] {
    {
      UniqueFd f(fake_open());
      pj::require(bool(f), "持有描述符时 bool 应为真");
      pj::require_eq(g_open.size(), std::size_t(1), "打开的描述符数");
    }
    pj::require_eq(g_open.size(), std::size_t(0), "离开作用域后应该已经关闭");
  });
  pj::run("not_copyable_but_movable", [] {
    pj::require(!std::is_copy_constructible_v<UniqueFd> && !std::is_copy_assignable_v<UniqueFd>, "UniqueFd 不能拷贝");
    pj::require(std::is_nothrow_move_constructible_v<UniqueFd> && std::is_nothrow_move_assignable_v<UniqueFd>,
                "移动构造和移动赋值应该是 noexcept");
    UniqueFd a(fake_open());
    int fd = a.get();
    UniqueFd b = std::move(a);
    pj::require(!a && a.get() == -1, "被移动之后应该为空（get() == -1）");
    pj::require_eq(b.get(), fd, "移动之后 b 持有原来的描述符");
  });
  pj::run("move_assign_closes_old", [] {
    UniqueFd a(fake_open()), b(fake_open());
    int fa = a.get(), fb = b.get();
    a = std::move(b);
    pj::require(!g_open.count(fa), "移动赋值应该先关闭自己原来的描述符");
    pj::require_eq(a.get(), fb, "移动赋值之后持有对方的描述符");
    UniqueFd& alias = a;
    a = std::move(alias);   // 自我移动赋值
    pj::require_eq(a.get(), fb, "自我移动赋值之后仍然持有原来的描述符");
    pj::require(g_open.count(fb) == 1, "自我移动赋值不能关闭描述符");
  });
  pj::run("release_and_reset", [] {
    UniqueFd a(fake_open());
    int fd = a.release();
    pj::require(!a, "release 之后应该为空");
    pj::require(g_open.count(fd) == 1, "release 不应该关闭描述符");
    fake_close(fd);
    UniqueFd b(fake_open());
    int old = b.get();
    int fresh = fake_open();
    b.reset(fresh);
    pj::require(!g_open.count(old), "reset 应该关闭原来的描述符");
    pj::require_eq(b.get(), fresh, "reset 之后持有新的描述符");
    b.reset();
    pj::require(!b && !g_open.count(fresh), "reset() 关闭并置空");
    UniqueFd empty;
    empty.reset();   // 空对象 reset 不能去关闭 -1
  });
  pj::run("no_leak_no_bad_close", [] {
    pj::require_eq(g_open.size(), std::size_t(0), "结束时还开着的描述符（泄漏）");
    pj::require_eq(g_bad_close, 0, "重复关闭或关闭无效描述符的次数");
  });
}
