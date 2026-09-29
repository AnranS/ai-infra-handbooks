#include <atomic>
#include <utility>

// 侵入式引用计数：计数放在对象里面（和 std::shared_ptr 的控制块不同）
class RefCounted {
 public:
  RefCounted() = default;
  RefCounted(const RefCounted&) = delete;
  RefCounted& operator=(const RefCounted&) = delete;

  void retain() const noexcept {
    count_.fetch_add(1, std::memory_order_relaxed);   // 能拷贝引用的线程已经持有一个引用，不需要同步任何数据
  }

  // 返回 true 表示这次调用把计数减到 0 并销毁了对象
  bool release() const noexcept {
    // release：本线程之前对对象的写入，要在"计数变小"之前完成并对最后一个线程可见；
    // acquire：最后一个线程要看到其他线程的所有写入，才能安全地析构
    if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
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
  RefPtr(RefPtr&& o) noexcept : p_(std::exchange(o.p_, nullptr)) {}
  RefPtr& operator=(const RefPtr& o) noexcept {
    if (o.p_) o.p_->retain();                          // 先加后减：自赋值时也不会提前销毁
    T* old = std::exchange(p_, o.p_);
    if (old) old->release();
    return *this;
  }
  RefPtr& operator=(RefPtr&& o) noexcept {
    if (this != &o) {
      T* old = std::exchange(p_, std::exchange(o.p_, nullptr));
      if (old) old->release();
    }
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
