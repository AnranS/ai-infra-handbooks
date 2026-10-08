# Value semantics, object lifetimes and RAII

<p class="lead">C++'s biggest difference from Python and Java is not the syntax but the object model: a variable is the object itself, assignment is a copy, and an object is destroyed at a definite moment. Binding "releasing a resource" to "destroying an object" is RAII, C++'s one way of managing device memory, files, locks and communication handles.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. After `std::vector<int> b = a;`, does changing `b` change `a`? What about `b = a` in Python?
    2. A scope constructs `x`, `y` and `z` in that order. In what order are they destroyed on leaving it?
    3. When is a temporary returned by a function destroyed? What happens when it is bound to a `const T&`?
    4. When a constructor throws, is the destructor called? What about the members already constructed?
    5. What are the "rule of zero" and the "rule of five"? What does the default copy constructor do when a class has a raw pointer member?

??? success "Answers (try it yourself first, then expand)"
    1. No: `b` is an independent copy of `a`. Python's `b = a` merely points two names at the same list, so changing `b` is changing `a`.
    2. In reverse order of construction: `z` first, then `y`, then `x`.
    3. At the end of the full expression (usually just that statement). Bound to a local `const T&` (or `T&&`) reference, the temporary's lifetime is extended to the end of that reference's scope (one level only).
    4. No: the object was never fully constructed, so its destructor is not called. But the members and base classes already constructed are destroyed in reverse order, which is why every resource should be held by an RAII member rather than acquired by hand in the constructor's body.
    5. The rule of zero: the class manages no resource directly, leaving it all to RAII members, and all five special member functions are the compiler's. The rule of five: a class that manages a resource directly has to define (or delete) the destructor, the copy constructor, copy assignment, the move constructor and move assignment. With a raw pointer member, the default copy constructor copies only the pointer (a shallow copy), and the two objects free the same memory twice when destroyed.

<!-- comic ../assets/comics/raii.webp is in Chinese; put it back once the English version exists -->

## Value semantics: a variable is the object {#值语义变量就是对象}

A variable in Python is a label stuck on an object, and `b = a` sticks two labels on the same list. A variable in C++ **is the object itself**, and `b = a` copies `a`'s contents into `b`:

```cpp title="value_semantics.cpp"
#include <cstdio>
#include <vector>

int main() {
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = a;    // a copy: b has its own elements
  b.push_back(4);
  std::vector<int>& r = a;   // a reference: r is another name for a, not a new object
  r[0] = 100;
  std::printf("a.size=%zu a[0]=%d b.size=%zu b[0]=%d\n", a.size(), a[0], b.size(), b[0]);
}
```

```text title="output"
a.size=3 a[0]=100 b.size=4 b[0]=1
```

Value semantics buy **local reasoning**: given a value, you need not worry that somebody elsewhere quietly changes it. The cost is that a copy can be expensive: copying a `vector` holding a million tokens is 4 MB of memory traffic. Hence the conventions for parameters:

| How the parameter is written | When to use it |
| --- | --- |
| `T x` (by value) | a small object (`int`, a pointer, `std::string_view`, `std::span`), or when the function keeps a copy anyway (together with a move, see the next chapter) |
| `const T& x` | a large object, read only |
| `T& x` | to modify the caller's object (an output parameter); use sparingly, since a return value is usually clearer |
| `T* x` | an optional parameter that may be null, or a C-style interface |

## An object's lifetime {#对象的生命周期}

An object exists from the moment its constructor finishes to the moment its destructor starts. When it is destroyed is decided by its **storage duration**:

- **automatic storage** (a local variable): destroyed on leaving the scope, **in reverse order of construction**;
- **dynamic storage** (`new`ed): destroyed at the `delete`, which is exactly where mistakes happen, and the next section makes it automatic with RAII;
- **static storage** (a global, a `static` local): destroyed when the program ends;
- **a temporary**: destroyed at the end of its **full expression** (usually that one statement); bound to a local `const T&` or `T&&` reference, it lives until that reference leaves scope.

A class that prints on construction and destruction makes it clearest:

```cpp title="lifetime.cpp"
#include <cstdio>
#include <string>
#include <utility>

struct Tracer {
  std::string name;
  explicit Tracer(std::string n) : name(std::move(n)) { std::printf("构造 %s\n", name.c_str()); }
  ~Tracer() { std::printf("析构 %s\n", name.c_str()); }
};

Tracer make(const char* n) { return Tracer(n); }

int main() {
  Tracer a("a");
  {
    Tracer b("b");
    Tracer c("c");
  }                                      // leaving the scope: c is destroyed first, then b
  make("临时对象");                       // nobody takes the return value: destroyed at the end of this statement
  const Tracer& r = make("被引用延长");    // bound to a const reference: lives until r leaves the scope
  std::printf("main 结束 %s\n", r.name.c_str());
}
```

```text title="output"
构造 a
构造 b
构造 c
析构 c
析构 b
构造 临时对象
析构 临时对象
构造 被引用延长
main 结束 被引用延长
析构 被引用延长
析构 a
```

Note that `return Tracer(n);` in `make` constructs only once: from C++17, initializing an object from a temporary of the same type **guarantees the copy is elided**, and the return value is constructed directly where the caller wants it.

!!! warning "Lifetime extension extends one level only"
    `const T& r = f();` does extend the temporary `f()` returned, but "a member of a temporary" like `const std::string& name = make("x").name;`,
    or a function that returns its reference parameter unchanged (`const T& id(const T& x) { return x; }`, then `const T& r = id(T{});`), extends nothing and `r` dangles immediately.
    The commonest trap is the range-for loop: in `for (auto& tok : get_request().tokens)` before C++23, the temporary `get_request()` returned is already destroyed before the loop starts.

## RAII: binding a resource to an object {#raii把资源绑在对象上}

![Figure: RAII - a resource's lifetime is bound to a stack object and destroyed in reverse order on leaving the scope](../assets/figures/object-lifetime.svg){.aig-svg}

**RAII** (Resource Acquisition Is Initialization): acquire the resource in the constructor and release it in the destructor. Because the moment of destruction is definite, triggered by leaving the scope normally, by a `return` and by an exception alike, the resource cannot leak.

Nearly every resource in an inference system is managed this way: device memory (`cudaMalloc` / `cudaFree`), CUDA streams and events, NCCL communicators, file descriptors, an `mmap`ed weight file, mutexes, memory registered with an RDMA network card.
Below, a pair of fake `fake_malloc` / `fake_free` stands in for `cudaMalloc` / `cudaFree`, counting how many blocks are still unfreed:

```cpp title="raii_buffer.cpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

static int live = 0;   // stands in for the number of device memory blocks not yet freed
void* fake_malloc(std::size_t n) { ++live; return std::malloc(n); }
void fake_free(void* p) { --live; std::free(p); }

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : ptr_(fake_malloc(bytes)), bytes_(bytes) {}
  ~DeviceBuffer() { fake_free(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;              // an exclusively owned resource: copying forbidden
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return ptr_; }
  std::size_t size() const { return bytes_; }

 private:
  void* ptr_;
  std::size_t bytes_;
};

void forward_step(bool fail) {
  DeviceBuffer hidden(1 << 20);
  DeviceBuffer logits(1 << 16);
  if (fail) throw std::runtime_error("kernel 启动失败");
  std::printf("  前向完成，当前未释放 %d 块\n", live);
}

int main() {
  forward_step(false);
  std::printf("正常返回后：%d 块\n", live);
  try {
    forward_step(true);
  } catch (const std::exception& e) {
    std::printf("捕获异常：%s，%d 块\n", e.what(), live);
  }
}
```

```text title="output"
  前向完成，当前未释放 2 块
正常返回后：0 块
捕获异常：kernel 启动失败，0 块
```

There is not one line of release code in `forward_step`, yet both blocks of "device memory" are freed whether it returns normally or throws partway. Compare the C style: every `return` has to remember to free and every error branch needs a `goto cleanup`, and one missed path leaks.

Those two `= delete` lines matter: a `DeviceBuffer` owns one resource exclusively, and copying it would give "two objects freeing the same memory". With copying forbidden, transferring ownership means a move (the next chapter).

### The trap of the default copy {#默认拷贝的陷阱}

Forget to forbid copying and the compiler's default copy constructor copies **member by member**, which for a raw pointer means copying the address. **This program has a bug:**

```cpp title="double_free.cpp" expect="fail"
#include <cstdlib>

struct Buffer {
  float* data;
  explicit Buffer(int n) : data(static_cast<float*>(std::malloc(n * sizeof(float)))) {}
  ~Buffer() { std::free(data); }
};

int main() {
  Buffer a(1024);
  Buffer b = a;   // the default copy: only the pointer is copied, so a and b point at the same memory
}                 // b is destroyed and frees it once, then a is destroyed and frees the same pointer
```

```text title="the ASan report (excerpt)"
==12345==ERROR: AddressSanitizer: attempting double-free on 0x625000002100 in thread T0:
    #1 in Buffer::~Buffer() double_free.cpp:6
```

### The rule of zero and the rule of five {#零法则与五法则}

Which brings up C++'s two rules for managing resources:

- **the rule of five**: a class that needs a custom destructor (which says it manages a resource directly) almost certainly also needs to define (or delete) the other four, the copy constructor, copy assignment, the move constructor and move assignment;
- **the rule of zero**: better still is **not to manage the resource in the class at all**, leaving it to RAII members already written (`std::vector`, `std::unique_ptr`, `std::string`, your own `DeviceBuffer`), and then writing none of the special member functions, because what the compiler generates is right.

```cpp
struct Buffer {                      // the rule of zero: nothing to write
  std::vector<float> data;
  explicit Buffer(int n) : data(n) {}
};
```

In real work, only a few "lowest-level handle classes" (wrapping a `cudaStream_t`, a file descriptor, a NCCL communicator) need the rule of five written by hand; every other class should follow the rule of zero.

### When a constructor fails {#构造函数失败时}

When a constructor throws, **this object's destructor is not called** (the object never existed completely), but **the members and base classes already constructed are destroyed**.
So when a class holds several resources, make each one its own RAII member rather than `malloc`ing two raw pointers in turn in the constructor: when the second fails, the first leaks. This chapter's exercise 2 is that scenario.

## A scope guard: "what to do on the way out", written once {#作用域守卫自定义的离开时做什么}

Some cleanup is not worth a class of its own, say "give the KV blocks just reserved back if a later step fails". A general **scope guard** runs a function on destruction, cancelled by `dismiss()` on success:

```cpp title="scope_exit.cpp"
#include <cstdio>
#include <utility>

template <class F>
class ScopeExit {
 public:
  explicit ScopeExit(F f) : f_(std::move(f)) {}
  ~ScopeExit() {
    if (active_) f_();
  }
  void dismiss() { active_ = false; }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  F f_;
  bool active_ = true;
};

bool schedule(bool prefill_ok) {
  std::printf("预留 KV 块\n");
  ScopeExit rollback([] { std::printf("回滚：释放预留的 KV 块\n"); });
  if (!prefill_ok) return false;       // an early return: the guard rolls back automatically
  std::printf("prefill 成功，提交\n");
  rollback.dismiss();                  // success: cancel the rollback
  return true;
}

int main() {
  schedule(false);
  schedule(true);
}
```

```text title="output"
预留 KV 块
回滚：释放预留的 KV 块
预留 KV 块
prefill 成功，提交
```

This is the same idea as Go's `defer` and Rust's `Drop`. `std::lock_guard` / `std::scoped_lock` (unlocking on leaving the scope) and `std::unique_ptr` (releasing on leaving the scope) are RAII classes the standard library already provides.

!!! interview "How to explain it"
    On value semantics and RAII: a C++ variable is the object itself and `b = a` is a copy (Python's `b = a` only adds a name); local objects are destroyed in reverse order of construction on leaving the scope, a temporary is destroyed at the end of the full expression, and binding it to a `const T&` extends its life one level. RAII binds a resource to an object: acquired on construction and released on destruction, so a normal return, an early return and an exception all release it, which is how locks, files, CUDA streams and events are managed. When a constructor throws, the destructor is not called but the members already constructed are destroyed, so several resources go to several RAII members. A class that manages a resource directly handles copy and move by the rule of five (usually forbidding copying), and every other class follows the rule of zero.

## Exercises {#练习}

1. In what order does the program below print? Write your answer down before looking.

    ```cpp
    struct Engine {
      Log scheduler{"scheduler"};
      Log cache;
      Log model;
      Engine() : model("model"), cache("cache") {}
    };
    int main() { Engine e; }
    ```

??? success "Answer"
    Members are constructed **in the order they are declared**, regardless of the order in the initializer list; destruction is the reverse. `g++ -Wall` warns about this with `-Wreorder`, because if `model`'s initialization depended on `cache` it would use a member that is not constructed yet.

    ```cpp title="member_order.cpp" flags="-Wno-reorder"
    #include <cstdio>

    struct Log {
      const char* n;
      explicit Log(const char* s) : n(s) { std::printf("构造 %s\n", n); }
      ~Log() { std::printf("析构 %s\n", n); }
    };

    struct Engine {
      Log scheduler{"scheduler"};
      Log cache;
      Log model;
      Engine() : model("model"), cache("cache") {}   // the order here does not decide the construction order
    };

    int main() { Engine e; }
    ```

    ```text title="output"
    构造 scheduler
    构造 cache
    构造 model
    析构 model
    析构 cache
    析构 scheduler
    ```

2. A class has to hold a block of "pinned memory" and a block of "device memory" at once, and the second allocation may fail with `std::bad_alloc`. Write a version that cannot leak, and prove it with the counter: no unfreed blocks when construction fails, 2 blocks when it succeeds, and back to 0 after leaving the scope.

??? success "Answer"
    Make each block its own RAII member. When the second member's construction fails, the first is already constructed and is destroyed automatically:

    ```cpp title="two_resources.cpp"
    #include <cstdio>
    #include <cstdlib>
    #include <new>

    static int live = 0;

    struct Block {
      void* p = nullptr;
      explicit Block(std::size_t n, bool fail = false) {
        if (fail) throw std::bad_alloc();
        p = std::malloc(n);
        ++live;
      }
      ~Block() {
        std::free(p);
        --live;
      }
      Block(const Block&) = delete;
      Block& operator=(const Block&) = delete;
    };

    struct PinnedPair {        // each resource is its own RAII member and the class itself follows the rule of zero
      Block host;
      Block device;
      PinnedPair(std::size_t n, bool fail) : host(n), device(n, fail) {}
    };

    int main() {
      try {
        PinnedPair p(1024, true);
      } catch (const std::bad_alloc&) {
        std::printf("构造失败，未释放 %d 块\n", live);
      }
      {
        PinnedPair p(1024, false);
        std::printf("构造成功，%d 块\n", live);
      }
      std::printf("离开作用域，%d 块\n", live);
    }
    ```

    ```text title="output"
    构造失败，未释放 0 块
    构造成功，2 块
    离开作用域，0 块
    ```

    Written as `PinnedPair(n) { host = malloc(n); device = alloc_or_throw(n); }`, allocating raw pointers in turn in the constructor's body, the throw in the second step means `PinnedPair`'s destructor never runs and `host` leaks.

## Summary {#小结}

- [x] A C++ variable is the object itself and assignment is a copy; pass a large object as `const T&`, and by value plus a move when a copy is kept.
- [x] Local objects are destroyed in reverse order of construction on leaving the scope; a temporary is destroyed at the end of the statement, and binding it to a `const T&` extends it one level.
- [x] RAII: acquire the resource on construction and release it on destruction. A normal return, an early return and an exception all release it, which is how C++ manages every resource.
- [x] A class that manages a resource directly handles copy and move by the rule of five (usually forbidding copying); every other class follows the rule of zero and leaves the resource to RAII members.
- [x] When a constructor fails, the members already constructed are destroyed, so several resources are each held by their own RAII member.
