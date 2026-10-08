#include <cstdio>
#include <memory>
#include <vector>

struct KvBlock {
  int id;
  explicit KvBlock(int i) : id(i) {}
  ~KvBlock() { std::printf("  释放块 %d\n", id); }
};

struct Request {
  const char* name;
  std::vector<std::shared_ptr<KvBlock>> blocks;
};

int main() {
  auto system_prompt = std::make_shared<KvBlock>(0);    // 公共前缀的 KV 块
  std::weak_ptr<KvBlock> cache_entry = system_prompt;   // 前缀缓存只观察
  {
    Request a{"a", {system_prompt}};
    {
      Request b{"b", {system_prompt, std::make_shared<KvBlock>(2)}};
      system_prompt.reset();                            // 创建者不再持有
      std::printf("块 0 的引用计数：%ld\n", b.blocks[0].use_count());
      std::printf("请求 b 结束：\n");
    }
    std::printf("请求 a 结束：\n");
  }
  std::printf("缓存里的块 0 %s\n", cache_entry.expired() ? "已经失效" : "还活着");
  if (auto blk = cache_entry.lock()) {
    std::printf("命中 %d\n", blk->id);
  } else {
    std::printf("未命中：需要重新计算前缀\n");
  }
}
