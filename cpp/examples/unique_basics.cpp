#include <cstdio>
#include <memory>
#include <vector>

struct Stream {   // 假装是一个 CUDA stream
  int id;
  explicit Stream(int i) : id(i) { std::printf("创建 stream %d\n", id); }
  ~Stream() { std::printf("销毁 stream %d\n", id); }
};

std::unique_ptr<Stream> make_stream(int id) { return std::make_unique<Stream>(id); }

void launch_on(const Stream& s) { std::printf("在 stream %d 上启动 kernel\n", s.id); }   // 只借用

int main() {
  auto s1 = make_stream(1);
  launch_on(*s1);
  std::vector<std::unique_ptr<Stream>> pool;
  pool.push_back(std::move(s1));   // 所有权交给 pool，s1 变成空指针
  pool.push_back(make_stream(2));
  std::printf("s1 %s\n", s1 ? "非空" : "为空");
  pool.erase(pool.begin());        // 从 pool 里删掉，stream 1 立刻销毁
  std::printf("pool 剩 %zu 个\n", pool.size());
}
