#include <climits>
#include <cstdio>

int main(int argc, char**) {
  int tokens = INT_MAX - 1 + argc;   // 运行时才知道的值：argc 为 1 时就是 INT_MAX
  int next = tokens + 1;             // 有符号溢出：未定义行为
  std::printf("%d\n", next);
}
