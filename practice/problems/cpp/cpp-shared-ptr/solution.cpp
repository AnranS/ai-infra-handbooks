#include <atomic>
#include <utility>

template <class T>
class SharedPtr {
  struct Control {
    std::atomic<long> count{1};
  };

 public:
  SharedPtr() = default;
  explicit SharedPtr(T* p) : ptr_(p), ctl_(p ? new Control : nullptr) {}
  ~SharedPtr() { release(); }

  SharedPtr(const SharedPtr& o) : ptr_(o.ptr_), ctl_(o.ctl_) {
    if (ctl_) ctl_->count.fetch_add(1, std::memory_order_relaxed);
  }
  SharedPtr& operator=(const SharedPtr& o) {
    SharedPtr tmp(o);   // 先加一再减一：自我赋值也安全
    swap(tmp);
    return *this;
  }
  SharedPtr(SharedPtr&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), ctl_(std::exchange(o.ctl_, nullptr)) {}
  SharedPtr& operator=(SharedPtr&& o) noexcept {
    SharedPtr tmp(std::move(o));
    swap(tmp);
    return *this;
  }
  void swap(SharedPtr& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(ctl_, o.ctl_);
  }
  void reset() { SharedPtr().swap(*this); }

  T* get() const { return ptr_; }
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  long use_count() const { return ctl_ ? ctl_->count.load(std::memory_order_relaxed) : 0; }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  void release() {
    // acq_rel：release 发布本线程对对象的访问，acquire 让删除者看到所有线程的访问
    if (ctl_ && ctl_->count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete ptr_;
      delete ctl_;
    }
    ptr_ = nullptr;
    ctl_ = nullptr;
  }
  T* ptr_ = nullptr;
  Control* ctl_ = nullptr;
};
