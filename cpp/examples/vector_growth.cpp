#include <cstdio>
#include <vector>

int count_reallocs(bool reserve) {
  std::vector<int> v;
  if (reserve) v.reserve(1000);
  int reallocs = 0;
  const int* last = v.data();
  for (int i = 0; i < 1000; ++i) {
    v.push_back(i);
    if (v.data() != last) {   // 缓冲区地址变了：发生了一次重新分配
      ++reallocs;
      last = v.data();
    }
  }
  return reallocs;
}

int main() {
  std::printf("不预留：%d 次重新分配\n", count_reallocs(false));
  std::printf("预留 1000：%d 次\n", count_reallocs(true));
  std::vector<int> v(10);
  v.clear();
  std::printf("clear 之后 size=%zu capacity=%zu\n", v.size(), v.capacity());
}
