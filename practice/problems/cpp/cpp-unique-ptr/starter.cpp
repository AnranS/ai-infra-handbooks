#include <cstddef>
#include <utility>

template <class T>
struct DefaultDelete {
  void operator()(T* p) const { delete p; }
};

template <class T, class D = DefaultDelete<T>>
class UniquePtr {
 public:
  UniquePtr() = default;
  UniquePtr(std::nullptr_t) {}
  explicit UniquePtr(T* p, D d = D()) : ptr_(p), del_(std::move(d)) {}
  // TODO：析构、禁止拷贝、移动构造、移动赋值

  T* get() const { return ptr_; }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }
  T* release() { return ptr_; }            // TODO
  void reset(T* p = nullptr) { ptr_ = p; } // TODO
  D& get_deleter() { return del_; }

 private:
  T* ptr_ = nullptr;
  D del_{};
};
