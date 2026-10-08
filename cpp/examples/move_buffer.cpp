#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static int live = 0;

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : ptr_(std::malloc(bytes)), bytes_(bytes) { ++live; }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();                                 // 先释放自己原来的
      ptr_ = std::exchange(o.ptr_, nullptr);
      bytes_ = std::exchange(o.bytes_, 0);
    }
    return *this;
  }

  std::size_t size() const { return bytes_; }

 private:
  void reset() {
    if (ptr_) {
      std::free(ptr_);
      --live;
      ptr_ = nullptr;
    }
  }
  void* ptr_;
  std::size_t bytes_;
};

DeviceBuffer allocate_kv(std::size_t blocks) { return DeviceBuffer(blocks * 4096); }

int main() {
  DeviceBuffer a = allocate_kv(16);   // 返回值直接构造到 a，既不拷贝也不移动
  DeviceBuffer b = std::move(a);      // 移动：b 接管内存，a 变成空壳
  std::printf("a.size=%zu b.size=%zu live=%d\n", a.size(), b.size(), live);

  std::vector<DeviceBuffer> pool;
  pool.push_back(std::move(b));       // 移动进 vector
  pool.emplace_back(8192);            // 直接在 vector 的内存里构造
  std::printf("pool=%zu live=%d\n", pool.size(), live);

  a = DeviceBuffer(100);              // 移动赋值：a 重新持有一块资源
  std::printf("a.size=%zu live=%d\n", a.size(), live);
}
