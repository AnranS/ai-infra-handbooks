// 测试程序已经提供：int fake_open(); void fake_close(int fd);
#include <utility>

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  ~UniqueFd() { reset(); }

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& o) noexcept : fd_(o.release()) {}
  UniqueFd& operator=(UniqueFd&& o) noexcept {
    reset(o.release());   // 自我赋值时：release 先置空再返回原值，reset 接管回来，不会关闭
    return *this;
  }

  int get() const { return fd_; }
  explicit operator bool() const { return fd_ >= 0; }
  int release() { return std::exchange(fd_, -1); }
  void reset(int fd = -1) {
    int old = std::exchange(fd_, fd);
    if (old >= 0 && old != fd) fake_close(old);
  }

 private:
  int fd_ = -1;
};
