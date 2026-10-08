#include <cstdio>
#include <functional>
#include <memory>

struct Session : std::enable_shared_from_this<Session> {
  int id;
  std::function<void()> on_done;
  explicit Session(int i) : id(i) {}
  ~Session() { std::printf("session %d 析构\n", id); }

  void arm() {
    std::weak_ptr<Session> self = weak_from_this();   // 不要捕获 shared_from_this()：那会形成环
    on_done = [self] {
      if (auto s = self.lock()) {
        std::printf("session %d 完成\n", s->id);
      } else {
        std::printf("session 已经不在了\n");
      }
    };
  }
};

int main() {
  auto s = std::make_shared<Session>(7);
  s->arm();
  auto cb = s->on_done;   // 模拟：回调被交给了异步完成队列
  cb();
  s.reset();              // 客户端断开，会话被销毁
  cb();                   // 迟到的完成事件：安全地什么都不做
}
