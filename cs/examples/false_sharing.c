// 两个线程各自累加自己的计数器，互不相干；只是计数器放的位置不同
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <time.h>

#define N 20000000L

struct { volatile long a, b; } same;                                // a、b 挨着，在同一条缓存行里
struct { _Alignas(64) volatile long a; _Alignas(64) volatile long b; } padded;   // 各占一条缓存行

typedef struct { volatile long *p; int cpu; } arg_t;

static void *worker(void *v) {
    arg_t *a = v;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(a->cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);      // 绑核，保证两个线程在不同的核上
    for (long i = 0; i < N; i++) __atomic_fetch_add(a->p, 1, __ATOMIC_RELAXED);   // 原子加：每次都要独占这条缓存行
    return NULL;
}

static double run(volatile long *x, volatile long *y, int cpu_x, int cpu_y) {
    struct timespec t0, t1;
    pthread_t t[2];
    arg_t args[2] = {{x, cpu_x}, {y, cpu_y}};
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 2; i++) pthread_create(&t[i], NULL, worker, &args[i]);
    for (int i = 0; i < 2; i++) pthread_join(t[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
}

int main(void) {
    printf("两个计数器在同一个缓存行：%.2f s\n", run(&same.a, &same.b, 0, 2));
    printf("各占一个缓存行：          %.2f s\n", run(&padded.a, &padded.b, 0, 2));
    return 0;
}
