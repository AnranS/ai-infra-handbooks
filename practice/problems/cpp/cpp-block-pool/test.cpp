#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "judge.hpp"
#include "user.cpp"

static std::string show(const std::optional<std::vector<int>>& v) {
  if (!v) return "nullopt";
  std::string s = "{";
  for (std::size_t i = 0; i < v->size(); ++i) s += (i ? ", " : "") + std::to_string((*v)[i]);
  return s + "}";
}

int main() {
  pj::run("example_order", [] {
    BlockPool pool(4);
    auto a = pool.allocate(2);
    pj::require_eq(show(a), std::string("{0, 1}"), "初始按编号从小到大分配");
    pool.release((*a)[1]);
    pj::require_eq(show(pool.allocate(1)), std::string("{1}"), "最近释放的最先分配");
    pj::require_eq(pool.num_free(), 2, "剩余空闲块");
  });
  pj::run("all_or_nothing", [] {
    BlockPool pool(5);
    pool.allocate(3);
    pj::require_eq(show(pool.allocate(3)), std::string("nullopt"), "不够时一个都不分配");
    pj::require_eq(pool.num_free(), 2, "失败的分配不改变空闲数");
    pj::require_eq(show(pool.allocate(2)), std::string("{3, 4}"), "剩下的正好够");
    pj::require_eq(show(pool.allocate(0)), std::string("{}"), "分配 0 个");
  });
  pj::run("refcount_and_errors", [] {
    BlockPool pool(3);
    auto a = *pool.allocate(1);
    pool.retain(a[0]);
    pj::require_eq(pool.refcount(a[0]), 2, "retain 之后的引用计数");
    pool.release(a[0]);
    pj::require_eq(pool.num_free(), 2, "还有一份引用：不回到空闲栈");
    pool.release(a[0]);
    pj::require_eq(pool.num_free(), 3, "引用计数到 0：回到空闲栈");
    pj::require_throws<std::logic_error>([&] { pool.release(a[0]); }, "重复释放");
    pj::require_throws<std::logic_error>([&] { pool.retain(2); }, "retain 空闲块");
  });
  pj::run("copy_on_write", [] {
    BlockPool pool(4);
    auto seq = *pool.allocate(2);          // {0, 1}
    for (int b : seq) pool.retain(b);      // 另一个序列分叉，共享两个块
    int w = pool.make_writable(seq[1]);
    pj::require_eq(w, 2, "共享块写之前应该复制到新块");
    pj::require_eq(pool.refcount(1), 1, "旧块少了一份引用");
    pj::require_eq(pool.make_writable(w), w, "独占的块直接写");
    pool.allocate(1);                      // 用掉最后一个空闲块
    int before = pool.num_free();
    pj::require_eq(pool.make_writable(seq[0]), -1, "没有空闲块时返回 -1");
    pj::require_eq(pool.refcount(0), 2, "失败时不改变引用计数");
    pj::require_eq(pool.num_free(), before, "失败时不改变空闲数");
  });
}
