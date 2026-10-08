#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void pin(int cpu) {
  cpu_set_t s;
  CPU_ZERO(&s);
  CPU_SET(cpu, &s);
  sched_setaffinity(0, sizeof s, &s);
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

/* 两个进程通过两根管道来回传 1 个字节：每一轮 A 写、B 读后回写、A 读，各自阻塞等待对方 */
static double pingpong(int cpu_a, int cpu_b, int n) {
  int ab[2], ba[2];
  char c = 0;
  if (pipe(ab) || pipe(ba)) return -1;
  pid_t pid = fork();
  if (pid == 0) {
    pin(cpu_b);
    for (int i = 0; i < n; i++)
      if (read(ab[0], &c, 1) != 1 || write(ba[1], &c, 1) != 1) _exit(1);
    _exit(0);
  }
  pin(cpu_a);
  double t0 = now();
  for (int i = 0; i < n; i++)
    if (write(ab[1], &c, 1) != 1 || read(ba[0], &c, 1) != 1) return -1;
  double t = now() - t0;
  waitpid(pid, NULL, 0);
  return t / n;
}

int main(void) {
  const int n = 200000;
  double same = pingpong(0, 0, n), cross = pingpong(0, 1, n);
  /* 同一个 CPU 上：每一轮是两次上下文切换（A→B、B→A） */
  printf("两个进程在同一个 CPU 上来回：每次切换约 %.2f µs\n", same / 2 * 1e6);
  /* 不同 CPU 上：没有切换，但每次都要唤醒另一个核上睡着的进程 */
  printf("两个进程在不同 CPU 上来回：每一轮约 %.2f µs\n", cross * 1e6);
  return 0;
}
