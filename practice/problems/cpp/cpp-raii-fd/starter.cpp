// 测试程序已经提供：int fake_open(); void fake_close(int fd);
#include <utility>

class UniqueFd {
 public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  // TODO：析构、禁止拷贝、移动构造和移动赋值

  int get() const { return fd_; }
  explicit operator bool() const { return fd_ >= 0; }
  int release() { return fd_; }        // TODO
  void reset(int fd = -1) { fd_ = fd; } // TODO

 private:
  int fd_ = -1;
};
