#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

int main() {
  pj::run("example_deep_copy", [] {
    Tensor1D a(4, 1.0f);
    Tensor1D b = a;
    b[0] = 9.0f;
    pj::require_eq(a[0], 1.0f, "拷贝之后修改 b 不应该影响 a");
    pj::require(a.data() != b.data(), "深拷贝：两个对象不能共用同一块内存");
    pj::require_eq(b.size(), std::size_t(4), "拷贝后的大小");
  });
  pj::run("copy_assign", [] {
    Tensor1D a(3, 2.0f), b(10, 5.0f);
    b = a;
    pj::require_eq(b.size(), std::size_t(3), "拷贝赋值之后的大小");
    pj::require_eq(b[2], 2.0f, "拷贝赋值之后的内容");
    Tensor1D& alias = b;
    b = alias;   // 自我赋值
    pj::require_eq(b.size(), std::size_t(3), "自我赋值之后大小不变");
    pj::require_eq(b[1], 2.0f, "自我赋值之后内容不变");
  });
  pj::run("move", [] {
    pj::require(std::is_nothrow_move_constructible_v<Tensor1D> && std::is_nothrow_move_assignable_v<Tensor1D>,
                "移动构造和移动赋值要标 noexcept");
    Tensor1D a(5, 3.0f);
    const float* p = a.data();
    Tensor1D b = std::move(a);
    pj::require(b.data() == p, "移动构造应该接管原来的内存，而不是拷贝");
    pj::require(a.size() == 0 && a.data() == nullptr, "被移动之后应该是空张量");
    Tensor1D c(2, 1.0f);
    c = std::move(b);
    pj::require(c.data() == p && c.size() == 5, "移动赋值应该接管对方的内存");
    pj::require(b.size() == 0 && b.data() == nullptr, "移动赋值之后源对象应该为空");
  });
  pj::run("vector_realloc", [] {
    std::vector<Tensor1D> v;
    for (int i = 0; i < 20; ++i) v.emplace_back(1000, float(i));
    for (int i = 0; i < 20; ++i) pj::require_eq(v[i][999], float(i), "扩容之后元素内容不变");
    std::vector<Tensor1D> w = v;          // 整个 vector 拷贝：每个元素深拷贝
    w[0][0] = -1.0f;
    pj::require_eq(v[0][0], 0.0f, "拷贝整个 vector 之后修改副本不应影响原件");
  });
  pj::run("empty", [] {
    Tensor1D e;
    Tensor1D f = e;
    Tensor1D g = std::move(f);
    pj::require(e.size() == 0 && f.size() == 0 && g.size() == 0, "空张量的拷贝和移动");
  });
}
