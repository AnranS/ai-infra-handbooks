#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "kvpool/block_pool.hpp"

static int g_checks = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    ++g_checks;                                                         \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "%s:%d: 检查失败：%s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

int main() {
  kvpool::BlockPool pool(8);
  auto a = pool.allocate(3);
  CHECK(a && a->size() == 3 && pool.num_free() == 5);
  CHECK(!pool.allocate(6));                     // 不够：一个都不分配
  CHECK(pool.num_free() == 5);

  for (auto b : *a) pool.retain(b);             // 另一个序列共享这 3 个块
  for (auto b : *a) pool.release(b);
  CHECK(pool.num_free() == 5 && pool.refcount((*a)[0]) == 1);
  for (auto b : *a) pool.release(b);
  CHECK(pool.num_free() == 8);

  bool threw = false;
  try {
    pool.release((*a)[0]);                      // 重复释放要被发现
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(pool.allocate(8) && pool.num_free() == 0);
  std::printf("block_pool：%d 个检查全部通过\n", g_checks);
}
