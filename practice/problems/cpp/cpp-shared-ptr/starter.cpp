#include <atomic>
#include <utility>

template <class T>
class SharedPtr {
 public:
  SharedPtr() = default;
  explicit SharedPtr(T* p) : ptr_(p) {}   // TODO：分配控制块
  // TODO：析构、拷贝、移动、reset

  T* get() const { return ptr_; }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  long use_count() const { return 0; }    // TODO
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
};
