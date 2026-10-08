#include <cstdlib>

struct Buffer {
  float* data;
  explicit Buffer(int n) : data(static_cast<float*>(std::malloc(n * sizeof(float)))) {}
  ~Buffer() { std::free(data); }
};

int main() {
  Buffer a(1024);
  Buffer b = a;   // 默认拷贝：只拷贝了指针，a 和 b 指向同一块内存
}                 // b 先析构释放一次，a 再析构释放同一个指针
