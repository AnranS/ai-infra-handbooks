// 同一个循环、同一批数据，只是先排不排序：差别全在分支预测
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (1 << 20)

static int cmp(const void *a, const void *b) { return *(const unsigned char *)a - *(const unsigned char *)b; }

static double sum_big(const unsigned char *d, long *out) {
    struct timespec t0, t1;
    long s = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 50; r++)
        for (int i = 0; i < N; i++)
            if (d[i] >= 128) s += d[i];               // 数据随机时，这个分支一半跳一半不跳，猜不准
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *out = s;
    return ((t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9) / (50.0 * N) * 1e9;
}

int main(void) {
    unsigned char *d = malloc(N);
    srand(1);
    for (int i = 0; i < N; i++) d[i] = rand() & 255;
    long s1, s2;
    double random_ns = sum_big(d, &s1);
    qsort(d, N, 1, cmp);                               // 排序后：前一半全不跳，后一半全跳
    double sorted_ns = sum_big(d, &s2);
    printf("随机顺序：每个元素 %.2f ns\n排好序后：每个元素 %.2f ns\n结果相同：%s\n",
           random_ns, sorted_ns, s1 == s2 ? "是" : "否");
    return 0;
}
