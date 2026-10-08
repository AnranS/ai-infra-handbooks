#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
  /* 带宽：顺序读 1 GiB，读 4 遍 */
  size_t n = (1UL << 30) / sizeof(uint64_t);
  uint64_t *a = malloc(n * sizeof *a);
  if (!a) return 1;
  memset(a, 1, n * sizeof *a);                   /* 先写一遍，让物理页按当前的内存策略分配好 */
  uint64_t sum = 0;
  double t0 = now();
  for (int r = 0; r < 4; r++)
    for (size_t i = 0; i < n; i++) sum += a[i];
  double bw = 4.0 * n * sizeof *a / (now() - t0) / 1e9;

  /* 延迟：在 256 MiB 里随机跳着读（指针追逐） */
  size_t m = (256UL << 20) / 64;
  uint32_t *next = (uint32_t *)a;                /* 复用同一块内存，每 64 字节一个节点 */
  uint32_t *perm = malloc(m * sizeof *perm);
  if (!perm) return 1;
  for (size_t i = 0; i < m; i++) perm[i] = i;
  uint64_t x = 88172645463325252ULL;
  for (size_t i = m - 1; i > 0; i--) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    size_t j = x % (i + 1);
    uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  for (size_t i = 0; i < m; i++) next[(size_t)perm[i] * 16] = perm[(i + 1) % m];
  uint32_t cur = perm[0];
  const int steps = 10000000;
  t0 = now();
  for (int s = 0; s < steps; s++) cur = next[(size_t)cur * 16];
  double lat = (now() - t0) / steps * 1e9;
  printf("单线程顺序读带宽 %.1f GB/s，随机访问延迟 %.0f ns%s\n", bw, lat, (sum == 0 && cur == 0) ? " " : "");
  return 0;
}
