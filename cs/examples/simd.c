// 同一个循环（对每个元素算一个 8 次多项式，8 次乘加），让编译器分别生成标量、AVX2（8 路）、AVX-512（16 路）版本
#include <stdio.h>
#include <time.h>

#define N 1024                       // 数组放得进 L1，每个元素又要算 8 次乘加：测的是计算，不是内存
#define REPS 200000

float x[N] __attribute__((aligned(64))), y[N] __attribute__((aligned(64)));

#define BODY                                                                    \
    for (int i = 0; i < N; i++) {                                               \
        float v = x[i], r = v;                                                  \
        _Pragma("GCC unroll 8")                                                 \
        for (int k = 0; k < 8; k++) r = r * v + 0.5f;  /* 编译器会生成 FMA */   \
        y[i] = r;                                                               \
    }

__attribute__((target("fma"), optimize("no-tree-vectorize"))) void poly_scalar(void) { BODY }
__attribute__((target("avx2,fma"), optimize("tree-vectorize"))) void poly_avx2(void) { BODY }
__attribute__((target("avx512f,prefer-vector-width=512"), optimize("tree-vectorize"))) void poly_avx512(void) { BODY }

static double gflops(void (*f)(void)) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) f();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    return 2.0 * 8 * N * REPS / s / 1e9;                        // 每次乘加 = 2 FLOP
}

int main(void) {
    for (int i = 0; i < N; i++) x[i] = (i % 7) * 0.1f;
    printf("标量：   %6.1f GFLOPS\n", gflops(poly_scalar));
    printf("AVX2：   %6.1f GFLOPS\n", gflops(poly_avx2));
    printf("AVX-512：%6.1f GFLOPS\n", gflops(poly_avx512));
    return 0;
}
