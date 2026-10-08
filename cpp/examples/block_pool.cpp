#include <cstdio>
#include <vector>

class BlockPool {
 public:
  explicit BlockPool(int n) : ref_(n, 0) {
    for (int i = n - 1; i >= 0; --i) free_.push_back(i);   // 栈顶是 0：小编号先分配
  }
  int allocate() {
    if (free_.empty()) return -1;   // 真实系统在这里触发抢占或排队
    int b = free_.back();
    free_.pop_back();
    ref_[b] = 1;
    return b;
  }
  void retain(int b) { ++ref_[b]; }   // 又一个序列共享这个块
  void release(int b) {
    if (--ref_[b] == 0) free_.push_back(b);
  }
  // 写之前调用：块被共享时复制一份给自己（写时复制），返回可以写的块号
  int make_writable(int b) {
    if (ref_[b] == 1) return b;
    int nb = allocate();
    if (nb < 0) return -1;
    // copy_kv(b, nb);   真实系统在这里用一个小 kernel 复制块的 KV 数据
    release(b);
    return nb;
  }
  int num_free() const { return int(free_.size()); }
  int refcount(int b) const { return ref_[b]; }

 private:
  std::vector<int> free_;
  std::vector<int> ref_;
};

int main() {
  BlockPool pool(8);
  std::vector<int> a = {pool.allocate(), pool.allocate(), pool.allocate()};   // 请求 A：3 个块
  std::printf("A=[%d %d %d] 剩余 %d\n", a[0], a[1], a[2], pool.num_free());

  std::vector<int> b = a;   // B 从 A 分叉（并行采样 / beam search）：共享全部块
  for (int blk : b) pool.retain(blk);
  std::printf("分叉后块 2 的引用计数=%d 剩余 %d\n", pool.refcount(2), pool.num_free());

  b.back() = pool.make_writable(b.back());   // B 要往最后一个块里写新 token
  std::printf("B 写之前复制：B=[%d %d %d] 块 2 引用计数=%d 剩余 %d\n", b[0], b[1], b[2], pool.refcount(2),
              pool.num_free());

  for (int blk : a) pool.release(blk);   // A 结束
  std::printf("A 结束：块 0 引用计数=%d 剩余 %d\n", pool.refcount(0), pool.num_free());
  for (int blk : b) pool.release(blk);
  std::printf("B 结束：剩余 %d\n", pool.num_free());
}
