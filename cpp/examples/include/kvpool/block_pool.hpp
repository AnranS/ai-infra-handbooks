#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace kvpool {

// 固定数量的 KV 块：空闲栈 + 引用计数。只能在一个线程里使用（调度器线程）。
class BlockPool {
 public:
  explicit BlockPool(std::int32_t num_blocks);

  // 分配 n 个块，要么全部成功，要么一个都不分配（返回 nullopt）
  std::optional<std::vector<std::int32_t>> allocate(std::int32_t n);
  void retain(std::int32_t block);    // 共享：引用计数加一
  void release(std::int32_t block);   // 引用计数减一，到 0 时回到空闲栈

  std::int32_t num_free() const { return static_cast<std::int32_t>(free_.size()); }
  std::int32_t refcount(std::int32_t block) const { return ref_.at(block); }

 private:
  std::vector<std::int32_t> free_;
  std::vector<std::int32_t> ref_;
};

}  // namespace kvpool
