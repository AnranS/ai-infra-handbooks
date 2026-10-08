#include <cstdio>
#include <thread>

int main() {
  long tokens = 0;
  auto work = [&] {
    for (int i = 0; i < 100000; ++i) ++tokens;   // 两个线程无同步地读写同一个变量
  };
  std::thread a(work), b(work);
  a.join();
  b.join();
  std::printf("%ld\n", tokens);
}
