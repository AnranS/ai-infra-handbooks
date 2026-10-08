#include <cstdio>

struct Log {
  const char* n;
  explicit Log(const char* s) : n(s) { std::printf("构造 %s\n", n); }
  ~Log() { std::printf("析构 %s\n", n); }
};

struct Engine {
  Log scheduler{"scheduler"};
  Log cache;
  Log model;
  Engine() : model("model"), cache("cache") {}   // 这里的顺序不决定构造顺序
};

int main() { Engine e; }
