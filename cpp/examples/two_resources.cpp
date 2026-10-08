#include <cstdio>
#include <cstdlib>
#include <new>

static int live = 0;

struct Block {
  void* p = nullptr;
  explicit Block(std::size_t n, bool fail = false) {
    if (fail) throw std::bad_alloc();
    p = std::malloc(n);
    ++live;
  }
  ~Block() {
    std::free(p);
    --live;
  }
  Block(const Block&) = delete;
  Block& operator=(const Block&) = delete;
};

struct PinnedPair {        // 每个资源都是一个 RAII 成员，类本身遵守零法则
  Block host;
  Block device;
  PinnedPair(std::size_t n, bool fail) : host(n), device(n, fail) {}
};

int main() {
  try {
    PinnedPair p(1024, true);
  } catch (const std::bad_alloc&) {
    std::printf("构造失败，未释放 %d 块\n", live);
  }
  {
    PinnedPair p(1024, false);
    std::printf("构造成功，%d 块\n", live);
  }
  std::printf("离开作用域，%d 块\n", live);
}
