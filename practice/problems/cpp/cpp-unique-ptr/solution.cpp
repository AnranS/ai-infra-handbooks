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
  ~UniquePtr() { reset(); }

  UniquePtr(const UniquePtr&) = delete;
  UniquePtr& operator=(const UniquePtr&) = delete;
  UniquePtr(UniquePtr&& o) noexcept : ptr_(o.release()), del_(std::move(o.del_)) {}
  UniquePtr& operator=(UniquePtr&& o) noexcept {
    if (this != &o) {
      reset(o.release());
      del_ = std::move(o.del_);
    }
    return *this;
  }

  T* get() const { return ptr_; }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  explicit operator bool() const { return ptr_ != nullptr; }
  T* release() { return std::exchange(ptr_, nullptr); }
  void reset(T* p = nullptr) {
    T* old = std::exchange(ptr_, p);   // 先更新状态，再删除旧对象
    if (old) del_(old);
  }
  D& get_deleter() { return del_; }

 private:
  T* ptr_ = nullptr;
  [[no_unique_address]] D del_{};
};
