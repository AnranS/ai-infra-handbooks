// 指针追逐：每次读出的值就是下一次要读的地址，CPU 没法提前发出下一次访存，测到的就是纯延迟
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

typedef struct node { struct node *next; char pad[56]; } node;   // 一个节点正好占一条 64 字节的缓存行

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double chase(size_t bytes, int huge) {
    size_t n = bytes / sizeof(node), len = (bytes + (2u << 20) - 1) & ~((2ul << 20) - 1);
    node *a = aligned_alloc(2u << 20, len);
    if (huge) madvise(a, len, MADV_HUGEPAGE);            // 用 2 MB 大页：排除 TLB 缺失，只看缓存
    size_t *perm = malloc(n * sizeof(size_t));
    uint64_t seed = 42;
    for (size_t i = 0; i < n; i++) perm[i] = i;
    for (size_t i = n - 1; i > 0; i--) {                 // Sattolo 算法：随机排成一个大环，走完所有节点才回到起点
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = (seed >> 33) % i, t = perm[i];
        perm[i] = perm[j], perm[j] = t;
    }
    for (size_t i = 0; i < n; i++) a[perm[i]].next = &a[perm[(i + 1) % n]];
    node *p = &a[perm[0]];
    for (size_t i = 0; i < n; i++) p = p->next;           // 预热：能装进缓存的部分先装进去
    size_t steps = 10 * 1000 * 1000;
    double t0 = now();
    for (size_t i = 0; i < steps; i++) p = p->next;
    double ns = (now() - t0) / steps * 1e9;
    if (p == NULL) puts("");                               // 用一下 p，防止循环被优化掉
    free(perm);
    free(a);
    return ns;
}

int main(void) {
    size_t kb[] = {16, 32, 256, 1024, 8192, 32768, 131072, 524288};
    printf("工作集        4 KB 页    2 MB 大页\n");
    for (size_t i = 0; i < sizeof kb / sizeof *kb; i++)
        printf("%7zu KB  %7.1f ns  %7.1f ns\n", kb[i], chase(kb[i] << 10, 0), chase(kb[i] << 10, 1));
    return 0;
}
