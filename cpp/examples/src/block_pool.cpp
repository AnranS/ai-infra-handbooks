#include "kvpool/block_pool.hpp"

#include <stdexcept>

namespace kvpool {

BlockPool::BlockPool(std::int32_t num_blocks) : ref_(num_blocks, 0) {
  free_.reserve(num_blocks);
  for (std::int32_t i = num_blocks - 1; i >= 0; --i) free_.push_back(i);
}

std::optional<std::vector<std::int32_t>> BlockPool::allocate(std::int32_t n) {
  if (n < 0 || n > num_free()) return std::nullopt;
  std::vector<std::int32_t> out(free_.end() - n, free_.end());
  free_.resize(free_.size() - n);
  for (auto b : out) ref_[b] = 1;
  return out;
}

void BlockPool::retain(std::int32_t block) {
  if (ref_.at(block) == 0) throw std::logic_error("retain 了一个空闲块");
  ++ref_[block];
}

void BlockPool::release(std::int32_t block) {
  if (ref_.at(block) == 0) throw std::logic_error("重复释放");
  if (--ref_[block] == 0) free_.push_back(block);
}

}  // namespace kvpool
