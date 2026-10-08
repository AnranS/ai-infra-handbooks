#include <cstdio>
#include <vector>

int main() {
  std::vector<int> blocks = {1, 2, 3};
  int& first = blocks[0];                       // 引用指向 vector 内部的缓冲区
  for (int i = 0; i < 100; ++i) blocks.push_back(i);   // 扩容：缓冲区搬家，旧的被释放
  std::printf("%d\n", first);                   // 读已经释放的内存
}
