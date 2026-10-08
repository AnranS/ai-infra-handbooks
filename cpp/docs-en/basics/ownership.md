# Smart pointers and ownership

<p class="lead">In C++ every resource should have a definite owner. Smart pointers write that ownership into the type: <code>unique_ptr</code> is exclusive, <code>shared_ptr</code> is shared, and <code>weak_ptr</code> and raw pointers only observe. To understand an unfamiliar code base, look at how its ownership is designed first; to write a new component, draw the ownership first.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Can a `unique_ptr` be copied? How do you put one in a `vector`? How much more memory does it take than a raw pointer?
    2. How do you manage a C-style handle (a `cudaStream_t`, a `FILE*`) with a `unique_ptr`?
    3. How do `make_shared` and `shared_ptr<T>(new T)` differ? Is a `shared_ptr`'s reference count thread-safe?
    4. What happens when two objects hold each other by `shared_ptr`? How do you fix it?
    5. For a function that only reads an object, should the parameter be `const shared_ptr<T>&`, `T*` or `const T&`?

??? success "Answers (try it yourself first, then expand)"
    1. It cannot be copied, only moved; put it in a `vector` with `push_back(std::move(p))` or `emplace_back(std::make_unique<T>(...))`. With the default deleter it is the same size as a raw pointer, at no cost.
    2. `std::unique_ptr<std::remove_pointer_t<cudaStream_t>, a deleter>`: the deleter is a function object calling `cudaStreamDestroy` (and one calling `fclose` for a `FILE*`), which releases it on leaving the scope.
    3. `make_shared` allocates once, putting the object and the control block (the reference count) together, which is faster and friendlier to the cache; `shared_ptr<T>(new T)` allocates twice. Incrementing and decrementing the count is atomic and thread-safe, but access to the object it points at is not.
    4. The count never reaches 0, neither object is ever freed, and both leak. Make one side (usually the one where a child points at a parent, or a callback points at the object) a `weak_ptr` and `lock()` it when using it.
    5. `const T&` (or `const T*` when it may be null): a function that only uses the object and takes no part in ownership should not require the caller to hold a `shared_ptr`; pass a smart pointer only when keeping or sharing ownership.

## Three ownership relationships {#三种所有权关系}

![Figure: three kinds of ownership - unique_ptr owns exclusively, shared_ptr shares, weak_ptr observes](../assets/figures/ownership-kinds.svg){.aig-svg}

| Relationship | How it is written | What it means |
| --- | --- | --- |
| exclusive | `std::unique_ptr<T>`, a member held by value | I alone release it; I can **hand ownership over** to someone else |
| shared | `std::shared_ptr<T>` | the last holder releases it; the reference count decides the lifetime |
| observing | `T&`, `T*`, `std::weak_ptr<T>` | I only use it and do not release it; it has to be alive while I use it |

Exclusive is the default. Use shared only when "who finishes last is not knowable" (several requests sharing the KV blocks of the same prefix, a session object still needed in an asynchronous callback). **A raw pointer or reference means "not owning" in modern C++**: a `T*` parameter should be read as "borrowed".

## `unique_ptr`: exclusive ownership {#unique_ptr独占所有权}

A `unique_ptr` is a pointer that can be moved and not copied, releasing what it points at on destruction. With the default deleter it is the same size as a raw pointer and costs nothing at run time:

```cpp title="unique_basics.cpp"
#include <cstdio>
#include <memory>
#include <vector>

struct Stream {   // pretending to be a CUDA stream
  int id;
  explicit Stream(int i) : id(i) { std::printf("创建 stream %d\n", id); }
  ~Stream() { std::printf("销毁 stream %d\n", id); }
};

std::unique_ptr<Stream> make_stream(int id) { return std::make_unique<Stream>(id); }

void launch_on(const Stream& s) { std::printf("在 stream %d 上启动 kernel\n", s.id); }   // borrowed only

int main() {
  auto s1 = make_stream(1);
  launch_on(*s1);
  std::vector<std::unique_ptr<Stream>> pool;
  pool.push_back(std::move(s1));   // ownership goes to pool and s1 becomes null
  pool.push_back(make_stream(2));
  std::printf("s1 %s\n", s1 ? "非空" : "为空");
  pool.erase(pool.begin());        // erased from pool, so stream 1 is destroyed at once
  std::printf("pool 剩 %zu 个\n", pool.size());
}
```

```text title="output"
创建 stream 1
在 stream 1 上启动 kernel
创建 stream 2
s1 为空
销毁 stream 1
pool 剩 1 个
销毁 stream 2
```

Always create one with `std::make_unique<T>(args...)` and never write a bare `new`.

### A custom deleter: wrapping a C-style handle {#自定义删除器包装-c-风格的句柄}

The CUDA, NCCL, RDMA (`ibverbs`) and filesystem APIs are all C-style interfaces where "a create function returns a handle and a destroy function releases it". Giving a `unique_ptr` a deleter produces an RAII handle class:

```cpp title="custom_deleter.cpp"
#include <cstdio>
#include <memory>

// a C-style handle API (shaped like cudaStreamCreate / cudaStreamDestroy and ncclCommInitRank / ncclCommDestroy)
struct handle_t {
  int id;
};
handle_t* handle_create(int id) {
  std::printf("create %d\n", id);
  return new handle_t{id};
}
void handle_destroy(handle_t* h) {
  std::printf("destroy %d\n", h->id);
  delete h;
}

struct HandleDeleter {
  void operator()(handle_t* h) const { handle_destroy(h); }
};
using Handle = std::unique_ptr<handle_t, HandleDeleter>;

int main() {
  Handle a(handle_create(1));
  std::unique_ptr<handle_t, void (*)(handle_t*)> b(handle_create(2), handle_destroy);
  std::printf("sizeof(unique_ptr<int>)=%zu\n", sizeof(std::unique_ptr<int>));
  std::printf("sizeof(Handle)=%zu\n", sizeof(Handle));
  std::printf("sizeof(函数指针当删除器)=%zu\n", sizeof(b));
}
```

```text title="output"
create 1
create 2
sizeof(unique_ptr<int>)=8
sizeof(Handle)=8
sizeof(函数指针当删除器)=16
destroy 2
destroy 1
```

With a deleter type that has no members (or the type of a capture-less lambda), the `unique_ptr` is still one pointer in size; with a function pointer as the deleter, that function pointer is stored too.
A `cudaStream_t` is itself a pointer type (`CUstream_st*`), so `std::unique_ptr<CUstream_st, StreamDeleter>` wraps it directly; when the handle is an integer (a file descriptor, say), write a small handle class like the previous chapter's.

## `shared_ptr`: shared ownership {#shared_ptr共享所有权}

A `shared_ptr` adds a **control block** holding the reference count. Copying a `shared_ptr` increments it, destroying one decrements it, and reaching 0 releases the object.

- the increment and decrement are **atomic**, so several threads copying and destroying their own `shared_ptr`s is safe; but **several threads reading and writing the same `shared_ptr` variable** (one assigning to it while another copies it) is not, and needs a lock or `std::atomic<std::shared_ptr<T>>` <span class="since">C++20</span>; nor does any of this make the object it points at thread-safe;
- `std::make_shared<T>(args...)` puts the object and the control block in **one** allocation, one fewer than `shared_ptr<T>(new T)` and friendlier to the cache;
- the atomic reference count is not free: on a hot path (copying once per token per block, say) it becomes the bottleneck, which is why an inference system's KV block reference counts usually live in the block allocator as ordinary integers (see [allocators and memory pools](../memory/allocators.md)).

Below, a `shared_ptr` stands for "several requests sharing the system prompt's KV blocks" and a `weak_ptr` for "the prefix cache only observes and does not extend the lifetime":

```cpp title="shared_prefix.cpp"
#include <cstdio>
#include <memory>
#include <vector>

struct KvBlock {
  int id;
  explicit KvBlock(int i) : id(i) {}
  ~KvBlock() { std::printf("  释放块 %d\n", id); }
};

struct Request {
  const char* name;
  std::vector<std::shared_ptr<KvBlock>> blocks;
};

int main() {
  auto system_prompt = std::make_shared<KvBlock>(0);    // the shared prefix's KV blocks
  std::weak_ptr<KvBlock> cache_entry = system_prompt;   // the prefix cache only observes
  {
    Request a{"a", {system_prompt}};
    {
      Request b{"b", {system_prompt, std::make_shared<KvBlock>(2)}};
      system_prompt.reset();                            // the creator no longer holds it
      std::printf("块 0 的引用计数：%ld\n", b.blocks[0].use_count());
      std::printf("请求 b 结束：\n");
    }
    std::printf("请求 a 结束：\n");
  }
  std::printf("缓存里的块 0 %s\n", cache_entry.expired() ? "已经失效" : "还活着");
  if (auto blk = cache_entry.lock()) {
    std::printf("命中 %d\n", blk->id);
  } else {
    std::printf("未命中：需要重新计算前缀\n");
  }
}
```

```text title="output"
块 0 的引用计数：2
请求 b 结束：
  释放块 2
请求 a 结束：
  释放块 0
缓存里的块 0 已经失效
未命中：需要重新计算前缀
```

A `weak_ptr` does not increment the count; to use what it points at, call `lock()` for a `shared_ptr` (empty if the object has been released). "Check `expired()` and then use it" races under several threads, so always use `lock()`'s return value.

### Reference cycles {#循环引用}

Two objects holding each other by `shared_ptr` never let the count reach 0 and both leak. The commonest form is **a callback capturing itself**: a session object holds a callback and the callback captures a `shared_ptr` to the session. **This program has a bug:**

```cpp title="cycle_leak.cpp" expect="fail"
#include <cstdio>
#include <memory>

struct Session;
struct Callback {
  std::shared_ptr<Session> owner;
};
struct Session {
  std::shared_ptr<Callback> on_done;
};

void open_session() {
  auto s = std::make_shared<Session>();
  s->on_done = std::make_shared<Callback>();
  s->on_done->owner = s;   // the cycle: Session -> Callback -> Session
}                          // after s leaves the scope both objects' counts are still 1: a leak

int main() {
  for (int i = 0; i < 4; ++i) open_session();   // every session leaks
  std::printf("4 个会话都结束了\n");
}
```

```text title="the LeakSanitizer report (excerpt)"
==12345==ERROR: LeakSanitizer: detected memory leaks
Indirect leak of 32 byte(s) in 1 object(s) allocated from:
    #8 in std::make_shared<Session>() ...
    #9 in open_session() cycle_leak.cpp:14
SUMMARY: AddressSanitizer: 256 byte(s) leaked in 8 allocation(s).
```

The two objects point at each other and neither is reachable from outside, so LeakSanitizer reports them as indirect leaks.

The fix is to make one edge of the cycle a `weak_ptr`. The standard form for an asynchronous callback: capture a `weak_ptr`, `lock()` when it runs, and do nothing if the object is gone:

```cpp title="weak_callback.cpp"
#include <cstdio>
#include <functional>
#include <memory>

struct Session : std::enable_shared_from_this<Session> {
  int id;
  std::function<void()> on_done;
  explicit Session(int i) : id(i) {}
  ~Session() { std::printf("session %d 析构\n", id); }

  void arm() {
    std::weak_ptr<Session> self = weak_from_this();   // do not capture shared_from_this(): that forms the cycle
    on_done = [self] {
      if (auto s = self.lock()) {
        std::printf("session %d 完成\n", s->id);
      } else {
        std::printf("session 已经不在了\n");
      }
    };
  }
};

int main() {
  auto s = std::make_shared<Session>(7);
  s->arm();
  auto cb = s->on_done;   // pretending the callback went to an asynchronous completion queue
  cb();
  s.reset();              // the client disconnects and the session is destroyed
  cb();                   // a late completion event: safely does nothing
}
```

```text title="output"
session 7 完成
session 7 析构
session 已经不在了
```

`std::enable_shared_from_this` lets an object obtain a `shared_ptr` / `weak_ptr` to itself from inside (provided the object is managed by a `shared_ptr`). Connection objects in networking libraries and RPC frameworks are written this way almost without exception.

## How to write a parameter {#函数参数怎么写}

| What the function does | How the parameter is written |
| --- | --- |
| reads / modifies the object, not caring who owns it | `const T&` / `T&` (or `T*` when it may be null) |
| takes ownership over | `std::unique_ptr<T>` (by value; the caller hands it over with `std::move`) |
| becomes one of the shared owners too (storing it, say) | `std::shared_ptr<T>` (by value, then `std::move`d into a member) |
| may or may not keep a copy | `const std::shared_ptr<T>&` |

The commonest mistake is passing a `shared_ptr` by value all the way down: every level does an atomic increment and decrement while the function in fact only reads the object. To use it, pass `const T&`.

!!! interview "How to explain it"
    On ownership: distinguish the three relationships first, exclusive (`unique_ptr` or a member by value), shared (`shared_ptr`) and observing (a reference, a raw pointer, `weak_ptr`), with exclusive as the default. A `unique_ptr` can only be moved, is the size of a raw pointer with the default deleter and costs nothing, and with a custom deleter wraps a C handle like `cudaStream_t` or `FILE*`; `make_shared` allocates the object and the control block at once, and the reference count's increments are atomic (the count is thread-safe, the object is not), so copying a `shared_ptr` often on a hot path has a cost; two objects holding each other by `shared_ptr` leak through the cycle, so an asynchronous callback captures a `weak_ptr` and `lock()`s it when it runs. A function that only uses the object takes `const T&` or `T*`, and only transferring or sharing ownership calls for a smart pointer.

## Exercises {#练习}

1. Choose the parameter type for each of these:
    1. `void record(const Request& r)` or `void record(std::shared_ptr<Request> r)`: logging one request;
    2. the scheduler hands a newly created request to the batch queue, which owns its lifetime from then on;
    3. LoRA weights shared by several requests, held by an asynchronous loading task in the background and not to be released before the load finishes;
    4. an optional "draft model" parameter that may be absent.

??? success "Answer"
    1. `const Request&`: read only, no ownership involved.
    2. `void enqueue(std::unique_ptr<Request> r)`, called as `queue.enqueue(std::move(req))`.
    3. `std::shared_ptr<LoraWeights>` by value to the background task (which holds one, guaranteeing it is not released during the load).
    4. `const DraftModel*` (which may be null); or `std::optional<std::reference_wrapper<const DraftModel>>`, though the raw pointer is both more common and clearer.

2. In a multi-LoRA service, the same adapter may be in use by many requests at once and should be unloaded once no request uses it. Write an `AdapterCache`: `get(name)` returns a `shared_ptr`; the cache itself keeps only `weak_ptr`s, returning the same instance when it is already in memory and reloading otherwise; `purge()` removes the expired entries and returns how many it removed.

??? success "Answer"
    ```cpp title="adapter_cache.cpp"
    #include <cstdio>
    #include <map>
    #include <memory>
    #include <string>
    #include <utility>

    struct Adapter {
      std::string name;
      explicit Adapter(std::string n) : name(std::move(n)) { std::printf("加载 %s\n", name.c_str()); }
      ~Adapter() { std::printf("卸载 %s\n", name.c_str()); }
    };

    class AdapterCache {
     public:
      std::shared_ptr<Adapter> get(const std::string& name) {
        if (auto it = cache_.find(name); it != cache_.end()) {
          if (auto a = it->second.lock()) return a;   // requests are still using it: reuse it
        }
        auto a = std::make_shared<Adapter>(name);
        cache_[name] = a;
        return a;
      }
      std::size_t purge() {
        std::size_t n = 0;
        for (auto it = cache_.begin(); it != cache_.end();) {
          if (it->second.expired()) {
            it = cache_.erase(it);
            ++n;
          } else {
            ++it;
          }
        }
        return n;
      }

     private:
      std::map<std::string, std::weak_ptr<Adapter>> cache_;
    };

    int main() {
      AdapterCache cache;
      auto a1 = cache.get("lora-a");
      auto a2 = cache.get("lora-a");
      std::printf("同一份：%s，引用 %ld\n", a1 == a2 ? "是" : "否", a1.use_count());
      { auto b = cache.get("lora-b"); }   // unloaded as soon as nobody uses it
      std::printf("清理了 %zu 个失效条目\n", cache.purge());
      a1.reset();
      a2.reset();
      auto a3 = cache.get("lora-a");      // already unloaded, so load it again
    }
    ```

    ```text title="output"
    加载 lora-a
    同一份：是，引用 2
    加载 lora-b
    卸载 lora-b
    清理了 1 个失效条目
    卸载 lora-a
    加载 lora-a
    卸载 lora-a
    ```

    In a real system, "unload the moment nobody uses it" is often too aggressive (the next request loads it again immediately), so there is usually an LRU layer on top: the cache holds `shared_ptr`s to the few most recently used adapters and only demotes them to `weak_ptr` beyond that capacity.

## Summary {#小结}

- [x] Ownership comes in three kinds: exclusive (`unique_ptr`, a member by value), shared (`shared_ptr`) and observing (a reference, a raw pointer, `weak_ptr`). Exclusive by default.
- [x] A `unique_ptr` can only be moved and costs nothing with the default deleter; with a custom deleter it wraps any C-style handle.
- [x] A `shared_ptr` uses an atomic reference count and `make_shared` allocates once; copying it often on a hot path has a cost.
- [x] A reference cycle leaks; an asynchronous callback captures a `weak_ptr` and `lock()`s it when it runs.
- [x] A function that only uses the object takes a reference, and only transferring or sharing ownership calls for a smart pointer.
