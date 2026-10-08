#include <cstdio>
#include <memory>

struct Session;
struct Callback {
  std::shared_ptr<Session> owner;
};
struct Session {
  std::shared_ptr<Callback> on_done;
};

void open_session() {
  auto s = std::make_shared<Session>();
  s->on_done = std::make_shared<Callback>();
  s->on_done->owner = s;   // 环：Session → Callback → Session
}                          // s 离开作用域后两个对象的引用计数都还是 1：泄漏

int main() {
  for (int i = 0; i < 4; ++i) open_session();   // 每个会话都泄漏
  std::printf("4 个会话都结束了\n");
}
