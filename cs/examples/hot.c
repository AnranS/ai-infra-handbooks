#include <stdio.h>

/* 两个函数：一个占大部分时间，一个占小部分。用 perf record 采样，看它能不能找出热点 */
__attribute__((noinline)) static double hot_loop(long n) {
  double x = 0;
  for (long i = 1; i < n; i++) x += 1.0 / i;
  return x;
}

__attribute__((noinline)) static double cold_loop(long n) {
  double x = 0;
  for (long i = 1; i < n; i++) x += 1.0 / (i + 1);
  return x;
}

int main(void) {
  double s = 0;
  for (int r = 0; r < 20; r++) s += hot_loop(40000000 + r) + cold_loop(10000000 + r);   /* 参数每轮不同，编译器没法把调用提到循环外 */
  printf("%.3f\n", s);
  return 0;
}
