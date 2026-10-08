#include <cstdio>
#include <utility>

template <class F>
class ScopeExit {
 public:
  explicit ScopeExit(F f) : f_(std::move(f)) {}
  ~ScopeExit() {
    if (active_) f_();
  }
  void dismiss() { active_ = false; }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  F f_;
  bool active_ = true;
};

bool schedule(bool prefill_ok) {
  std::printf("预留 KV 块\n");
  ScopeExit rollback([] { std::printf("回滚：释放预留的 KV 块\n"); });
  if (!prefill_ok) return false;       // 提前返回：守卫自动回滚
  std::printf("prefill 成功，提交\n");
  rollback.dismiss();                  // 成功：取消回滚
  return true;
}

int main() {
  schedule(false);
  schedule(true);
}
