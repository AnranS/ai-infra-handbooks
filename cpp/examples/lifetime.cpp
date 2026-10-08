#include <cstdio>
#include <string>
#include <utility>

struct Tracer {
  std::string name;
  explicit Tracer(std::string n) : name(std::move(n)) { std::printf("构造 %s\n", name.c_str()); }
  ~Tracer() { std::printf("析构 %s\n", name.c_str()); }
};

Tracer make(const char* n) { return Tracer(n); }

int main() {
  Tracer a("a");
  {
    Tracer b("b");
    Tracer c("c");
  }                                      // 离开作用域：先析构 c，再析构 b
  make("临时对象");                       // 返回值没人接：这条语句结束时就析构
  const Tracer& r = make("被引用延长");    // 绑定到 const 引用：活到 r 离开作用域
  std::printf("main 结束 %s\n", r.name.c_str());
}
