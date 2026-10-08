// 存储缓冲区实验（store buffering）：两个线程各写一个变量、再读对方的变量
// 按"程序顺序"执行的话，r1 和 r2 不可能同时为 0
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>

#define ROUNDS 1000000

static atomic_int x, y, r1, r2, go, done;
static int use_fence;

static void pin(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void *t1(void *arg) {
    (void)arg;
    pin(2);
    for (int i = 1; i <= ROUNDS; i++) {
        while (atomic_load_explicit(&go, memory_order_acquire) != i) {}    // 等主线程发令
        atomic_store_explicit(&x, 1, memory_order_relaxed);
        if (use_fence) atomic_thread_fence(memory_order_seq_cst);          // x86 上编译成 mfence
        atomic_store_explicit(&r1, atomic_load_explicit(&y, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(&done, 1, memory_order_release);
    }
    return NULL;
}

static void *t2(void *arg) {
    (void)arg;
    pin(4);
    for (int i = 1; i <= ROUNDS; i++) {
        while (atomic_load_explicit(&go, memory_order_acquire) != i) {}
        atomic_store_explicit(&y, 1, memory_order_relaxed);
        if (use_fence) atomic_thread_fence(memory_order_seq_cst);
        atomic_store_explicit(&r2, atomic_load_explicit(&x, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(&done, 1, memory_order_release);
    }
    return NULL;
}

static int trial(int fence) {
    pthread_t a, b;
    int both_zero = 0;
    use_fence = fence;
    atomic_store(&go, 0);
    pthread_create(&a, NULL, t1, NULL);
    pthread_create(&b, NULL, t2, NULL);
    pin(0);
    for (int i = 1; i <= ROUNDS; i++) {
        atomic_store(&x, 0), atomic_store(&y, 0), atomic_store(&done, 0);
        atomic_store_explicit(&go, i, memory_order_release);               // 发令：两个线程同时开跑
        while (atomic_load_explicit(&done, memory_order_acquire) != 2) {}
        both_zero += atomic_load(&r1) == 0 && atomic_load(&r2) == 0;
    }
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    return both_zero;
}

int main(void) {
    printf("不加屏障：%d 轮里 r1 == r2 == 0 出现了 %d 次\n", ROUNDS, trial(0));
    printf("加 mfence：%d 轮里 r1 == r2 == 0 出现了 %d 次\n", ROUNDS, trial(1));
    return 0;
}
