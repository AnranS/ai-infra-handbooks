// 同样是把 4096 个 float 加起来，只是用了几个互相独立的累加器
#include <stdio.h>
#include <time.h>

#define N 4096                           // 16 KB，放得进 L1
#define REPS 100000
float a[N];

static float sum1(void) {                // 每次加法都要等上一次的结果：一条依赖链
    float s = 0;
    for (int i = 0; i < N; i++) s += a[i];
    return s;
}
static float sum2(void) {
    float s0 = 0, s1 = 0;
    for (int i = 0; i < N; i += 2) s0 += a[i], s1 += a[i + 1];
    return s0 + s1;
}
static float sum4(void) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < N; i += 4) s0 += a[i], s1 += a[i + 1], s2 += a[i + 2], s3 += a[i + 3];
    return (s0 + s1) + (s2 + s3);
}
static float sum8(void) {                // 8 条互不依赖的链，可以同时在流水线里
    float s[8] = {0};
    for (int i = 0; i < N; i += 8) {
        s[0] += a[i], s[1] += a[i + 1], s[2] += a[i + 2], s[3] += a[i + 3];
        s[4] += a[i + 4], s[5] += a[i + 5], s[6] += a[i + 6], s[7] += a[i + 7];
    }
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

static void bench(const char *name, float (*f)(void)) {
    struct timespec t0, t1;
    volatile float sink = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < REPS; r++) sink += f();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ((double)REPS * N);
    printf("%s：每次加法 %.2f ns\n", name, ns);
}

int main(void) {
    for (int i = 0; i < N; i++) a[i] = 1.0f / (i + 1);
    bench("1 个累加器", sum1);
    bench("2 个累加器", sum2);
    bench("4 个累加器", sum4);
    bench("8 个累加器", sum8);
    return 0;
}
