#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#define SIZE (1UL << 30)            /* 1 GiB，远大于缓存，也远大于 4 KiB 页时 TLB 能覆盖的范围 */
#define LINE 64                     /* 每个缓存行放一个"指向下一个缓存行"的下标 */
#define STEPS 20000000

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t rng = 88172645463325252ULL;
static uint64_t next_rand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

/* 随机的指针追逐：每次读到的值决定下一次读哪里，CPU 没法提前预取，每一步都是一次缓存缺失（和可能的 TLB 缺失） */
static double chase(uint32_t *perm, size_t n, int huge) {
  char *buf = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED) return -1;
  madvise(buf, SIZE, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
  for (size_t i = 0; i < n; i++) *(uint32_t *)(buf + (size_t)perm[i] * LINE) = perm[(i + 1) % n];
  uint32_t cur = perm[0];
  double t0 = now();
  for (int s = 0; s < STEPS; s++) cur = *(uint32_t *)(buf + (size_t)cur * LINE);
  double t = now() - t0;
  if (cur == 0xFFFFFFFF) printf("%u\n", cur);   /* 让编译器不能把循环优化掉 */
  munmap(buf, SIZE);
  return t / STEPS * 1e9;
}

int main(void) {
  size_t n = SIZE / LINE;
  uint32_t *perm = malloc(n * sizeof *perm);
  if (!perm) return 1;
  for (size_t i = 0; i < n; i++) perm[i] = i;
  for (size_t i = n - 1; i > 0; i--) {         /* 随机打乱缓存行的访问顺序 */
    size_t j = next_rand() % (i + 1);
    uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  double small = chase(perm, n, 0), huge = chase(perm, n, 1);
  printf("4 KiB 页：每次随机访问 %.0f ns\n", small);
  printf("2 MiB 透明大页：每次随机访问 %.0f ns（快了 %.0f%%）\n", huge, (small - huge) / small * 100);
  free(perm);
  return 0;
}
