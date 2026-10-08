#include <cstdio>
#include <string>
#include <utility>

void consume(const std::string&) { std::printf("  拷贝进来（左值）\n"); }
void consume(std::string&&) { std::printf("  移动进来（右值）\n"); }

template <class T>
void wrong(T&& x) { consume(x); }                    // x 有名字，是左值：永远走拷贝

template <class T>
void right(T&& x) { consume(std::forward<T>(x)); }   // 保持调用者的值类别

int main() {
  std::string s = "prompt";
  std::printf("wrong(s)：\n");
  wrong(s);
  std::printf("wrong(临时对象)：\n");
  wrong(std::string("tmp"));
  std::printf("right(s)：\n");
  right(s);
  std::printf("right(临时对象)：\n");
  right(std::string("tmp"));
}
