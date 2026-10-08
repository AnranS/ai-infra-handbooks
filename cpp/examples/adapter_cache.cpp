#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <utility>

struct Adapter {
  std::string name;
  explicit Adapter(std::string n) : name(std::move(n)) { std::printf("加载 %s\n", name.c_str()); }
  ~Adapter() { std::printf("卸载 %s\n", name.c_str()); }
};

class AdapterCache {
 public:
  std::shared_ptr<Adapter> get(const std::string& name) {
    if (auto it = cache_.find(name); it != cache_.end()) {
      if (auto a = it->second.lock()) return a;   // 还有请求在用：直接复用
    }
    auto a = std::make_shared<Adapter>(name);
    cache_[name] = a;
    return a;
  }
  std::size_t purge() {
    std::size_t n = 0;
    for (auto it = cache_.begin(); it != cache_.end();) {
      if (it->second.expired()) {
        it = cache_.erase(it);
        ++n;
      } else {
        ++it;
      }
    }
    return n;
  }

 private:
  std::map<std::string, std::weak_ptr<Adapter>> cache_;
};

int main() {
  AdapterCache cache;
  auto a1 = cache.get("lora-a");
  auto a2 = cache.get("lora-a");
  std::printf("同一份：%s，引用 %ld\n", a1 == a2 ? "是" : "否", a1.use_count());
  { auto b = cache.get("lora-b"); }   // 用完即卸载
  std::printf("清理了 %zu 个失效条目\n", cache.purge());
  a1.reset();
  a2.reset();
  auto a3 = cache.get("lora-a");      // 已经卸载过，重新加载
}
