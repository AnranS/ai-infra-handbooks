#include <fcntl.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

/* 不用 liburing，直接调系统调用：看清楚 io_uring 就是两个和内核共享的环形队列 */
#define BLOCK 4096
#define NREQ 4

int main(void) {
  /* 准备一个 16 KiB 的文件：第 i 个 4 KiB 块全是字母 'A' + i */
  int wfd = open("data.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  char blk[BLOCK];
  for (int i = 0; i < NREQ; i++) {
    memset(blk, 'A' + i, BLOCK);
    if (write(wfd, blk, BLOCK) != BLOCK) return 1;
  }
  close(wfd);
  int fd = open("data.bin", O_RDONLY);

  struct io_uring_params p;
  memset(&p, 0, sizeof p);
  int ring = syscall(__NR_io_uring_setup, 8, &p);
  if (ring < 0) { perror("io_uring_setup（容器可能禁用了 io_uring）"); return 1; }

  /* 把提交队列（SQ）、完成队列（CQ）和提交项数组映射进来 */
  size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
  size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
  if (cq_sz > sq_sz) sq_sz = cq_sz;                 /* 新内核的两个队列在同一块映射里（IORING_FEAT_SINGLE_MMAP） */
  char *sq = mmap(NULL, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQ_RING);
  char *cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq
             : mmap(NULL, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_CQ_RING);
  struct io_uring_sqe *sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
                                   MAP_SHARED | MAP_POPULATE, ring, IORING_OFF_SQES);
  if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) return 1;
  unsigned *sq_tail = (unsigned *)(sq + p.sq_off.tail), *sq_mask = (unsigned *)(sq + p.sq_off.ring_mask);
  unsigned *sq_array = (unsigned *)(sq + p.sq_off.array);
  unsigned *cq_head = (unsigned *)(cq + p.cq_off.head), *cq_tail = (unsigned *)(cq + p.cq_off.tail);
  unsigned *cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
  struct io_uring_cqe *cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);

  /* 一次填好 4 个读请求：倒着读，看完成的顺序和数据是否对得上 */
  static char bufs[NREQ][BLOCK];
  unsigned tail = *sq_tail;
  for (int i = 0; i < NREQ; i++) {
    unsigned idx = tail & *sq_mask;
    struct io_uring_sqe *e = &sqes[idx];
    memset(e, 0, sizeof *e);
    e->opcode = IORING_OP_READ;
    e->fd = fd;
    e->addr = (uint64_t)(uintptr_t)bufs[i];
    e->len = BLOCK;
    e->off = (uint64_t)(NREQ - 1 - i) * BLOCK;
    e->user_data = i;                                /* 完成事件会原样带回这个值 */
    sq_array[idx] = idx;
    tail++;
  }
  __atomic_store_n(sq_tail, tail, __ATOMIC_RELEASE); /* 先写好提交项，再发布新的队尾 */

  /* 一次系统调用：提交 4 个请求，并等待 4 个完成 */
  int n = syscall(__NR_io_uring_enter, ring, NREQ, NREQ, IORING_ENTER_GETEVENTS, NULL, 0);
  printf("一次 io_uring_enter 提交了 %d 个读请求\n", n);

  int ok = 0;
  unsigned head = *cq_head, ctail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
  for (; head != ctail; head++) {
    struct io_uring_cqe *c = &cqes[head & *cq_mask];
    int i = (int)c->user_data;
    ok += c->res == BLOCK && bufs[i][0] == 'A' + (NREQ - 1 - i) && bufs[i][BLOCK - 1] == 'A' + (NREQ - 1 - i);
  }
  __atomic_store_n(cq_head, head, __ATOMIC_RELEASE); /* 告诉内核这些完成事件已经取走 */
  printf("收到 %d 个完成事件，数据全部正确：%s\n", ok, ok == NREQ ? "是" : "否");
  return 0;
}
