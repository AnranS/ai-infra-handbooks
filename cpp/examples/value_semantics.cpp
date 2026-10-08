#include <cstdio>
#include <vector>

int main() {
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = a;    // 拷贝：b 有自己的一份元素
  b.push_back(4);
  std::vector<int>& r = a;   // 引用：r 是 a 的另一个名字，不是新对象
  r[0] = 100;
  std::printf("a.size=%zu a[0]=%d b.size=%zu b[0]=%d\n", a.size(), a[0], b.size(), b[0]);
}
