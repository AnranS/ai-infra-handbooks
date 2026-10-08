#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static int live = 0;   // 模拟"还没释放的显存块"数量
void* fake_malloc(std::size_t n) { ++live; return std::malloc(n); }
void fake_free(void* p) { --live; std::free(p); }

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : ptr_(fake_malloc(bytes)), bytes_(bytes) {}
  ~DeviceBuffer() { fake_free(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;              // 独占的资源：禁止拷贝
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  std::size_t size() const { return bytes_; }

 private:
  void* ptr_;
  std::size_t bytes_;
};

void forward_step(bool fail) {
  DeviceBuffer hidden(1 << 20);
  DeviceBuffer logits(1 << 16);
  if (fail) throw std::runtime_error("kernel 启动失败");
  std::printf("  前向完成，当前未释放 %d 块\n", live);
}

int main() {
  forward_step(false);
  std::printf("正常返回后：%d 块\n", live);
  try {
    forward_step(true);
  } catch (const std::exception& e) {
    std::printf("捕获异常：%s，%d 块\n", e.what(), live);
  }
}
