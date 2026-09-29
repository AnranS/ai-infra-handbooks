#include <optional>
#include <stdexcept>
#include <vector>

class BlockPool {
 public:
  explicit BlockPool(int num_blocks) : ref_(num_blocks, 0) {}
  std::optional<std::vector<int>> allocate(int n) { (void)n; return std::nullopt; }   // TODO
  void retain(int b) { ++ref_.at(b); }                                               // TODO：检查
  void release(int b) { --ref_.at(b); }                                              // TODO
  int make_writable(int b) { return b; }                                             // TODO
  int num_free() const { return 0; }                                                 // TODO
  int refcount(int b) const { return ref_.at(b); }

 private:
  std::vector<int> ref_;
};
