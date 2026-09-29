#include <atomic>
#include <utility>

// 侵入式引用计数：计数放在对象里面（和 std::shared_ptr 的控制块不同）
class RefCounted {
 public:
  RefCounted() = default;
  RefCounted(const RefCounted&) = delete;
  RefCounted& operator=(const RefCounted&) = delete;

  void retain() const noexcept {
    count_.fetch_add(1, std::memory_order_relaxed);
  }

  // 返回 true 表示这次调用把计数减到 0 并销毁了对象
  bool release() const noexcept {
    if (count_.fetch_sub(1, std::memory_order_relaxed) == 1) {   // TODO：内存序对吗？
      delete this;
      return true;
    }
    return false;
  }

  int use_count() const noexcept { return count_.load(std::memory_order_relaxed); }

 protected:
  virtual ~RefCounted() = default;

 private:
  mutable std::atomic<int> count_{1};                 // 新建的对象由创建者持有一个引用
};

template <class T>
class RefPtr {
 public:
  RefPtr() = default;
  explicit RefPtr(T* p) noexcept : p_(p) {}            // 接管一个引用（不加计数）
  RefPtr(const RefPtr& o) noexcept : p_(o.p_) {
    if (p_) p_->retain();
  }
  RefPtr(RefPtr&& o) noexcept : p_(o.p_) {}             // TODO：源对象还指着同一个对象
  RefPtr& operator=(const RefPtr& o) noexcept {
    // TODO：拷贝赋值（注意自赋值）
    return *this;
  }
  RefPtr& operator=(RefPtr&& o) noexcept {
    // TODO：移动赋值
    return *this;
  }
  ~RefPtr() { reset(); }

  void reset() noexcept {
    if (T* old = std::exchange(p_, nullptr)) old->release();
  }
  T* get() const noexcept { return p_; }
  T* operator->() const noexcept { return p_; }
  T& operator*() const noexcept { return *p_; }
  explicit operator bool() const noexcept { return p_ != nullptr; }

 private:
  T* p_ = nullptr;
};

template <class T, class... Args>
RefPtr<T> make_ref(Args&&... args) {
  return RefPtr<T>(new T(std::forward<Args>(args)...));
}
