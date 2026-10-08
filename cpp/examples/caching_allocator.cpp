#include <cstdio>
#include <cstdlib>
#include <new>
#include <unordered_map>
#include <vector>

class CachingAllocator {
 public:
  void* allocate(std::size_t n) {
    std::size_t sz = round_up(n);
    auto& bin = free_[sz];
    void* p;
    if (!bin.empty()) {
      p = bin.back();
      bin.pop_back();
      ++hits_;
    } else {
      p = std::malloc(sz);   // 真实系统里这里是 cudaMalloc：慢，而且会同步
      if (!p) throw std::bad_alloc();
      ++raw_allocs_;
      reserved_ += sz;
    }
    sizes_[p] = sz;
    return p;
  }
  void deallocate(void* p) {   // 不还给系统，按大小放回空闲池
    std::size_t sz = sizes_.at(p);
    sizes_.erase(p);
    free_[sz].push_back(p);
  }
  void empty_cache() {         // 对应 torch.cuda.empty_cache()
    for (auto& [sz, bin] : free_) {
      for (void* p : bin) {
        std::free(p);
        reserved_ -= sz;
      }
      bin.clear();
    }
  }
  ~CachingAllocator() { empty_cache(); }

  static std::size_t round_up(std::size_t n) {
    const std::size_t k = n < (1 << 20) ? 512 : (2 << 20);
    return (n + k - 1) / k * k;
  }
  int raw_allocs() const { return raw_allocs_; }
  int hits() const { return hits_; }
  double reserved_mib() const { return reserved_ / 1048576.0; }

 private:
  std::unordered_map<std::size_t, std::vector<void*>> free_;
  std::unordered_map<void*, std::size_t> sizes_;
  std::size_t reserved_ = 0;
  int raw_allocs_ = 0, hits_ = 0;
};

int main() {
  CachingAllocator alloc;
  const int batches[] = {32, 48, 32, 48};
  for (int step = 0; step < 4; ++step) {
    std::size_t b = batches[step];
    void* hidden = alloc.allocate(b * 4096 * 2);    // [batch, 4096] 的 bf16 激活
    void* logits = alloc.allocate(b * 32000 * 4);   // [batch, 32000] 的 fp32 logits
    void* meta = alloc.allocate(b * 8);             // 每个请求一个 int64
    alloc.deallocate(meta);
    alloc.deallocate(logits);
    alloc.deallocate(hidden);
    std::printf("step %d：batch=%zu，底层分配累计 %d 次，复用累计 %d 次，reserved %.2f MiB\n", step, b,
                alloc.raw_allocs(), alloc.hits(), alloc.reserved_mib());
  }
  alloc.empty_cache();
  std::printf("empty_cache 之后 reserved %.2f MiB\n", alloc.reserved_mib());
}
