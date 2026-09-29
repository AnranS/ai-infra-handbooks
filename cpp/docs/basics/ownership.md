# 智能指针与所有权

<p class="lead">在 C++ 里，每一块资源都应该有明确的"主人"。智能指针把所有权写进了类型：<code>unique_ptr</code> 是独占，<code>shared_ptr</code> 是共享，<code>weak_ptr</code> 和裸指针只是观察。读懂一个陌生代码库，先看它的所有权怎么设计；写一个新组件，也先把所有权画清楚。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `unique_ptr` 能拷贝吗？怎么把它存进 `vector`？它比裸指针多占多少内存？
    2. 用 `unique_ptr` 管理一个 C 风格的句柄（比如 `cudaStream_t`、`FILE*`），要怎么写？
    3. `make_shared` 和 `shared_ptr<T>(new T)` 有什么区别？`shared_ptr` 的引用计数是线程安全的吗？
    4. 两个对象用 `shared_ptr` 互相持有，会发生什么？怎么解决？
    5. 一个函数只是读一下对象，参数应该写成 `const shared_ptr<T>&`、`T*` 还是 `const T&`？

## 三种所有权关系

| 关系 | 写法 | 含义 |
| --- | --- | --- |
| 独占 | `std::unique_ptr<T>`、按值持有的成员 | 只有我负责释放它；可以把所有权**移交**给别人 |
| 共享 | `std::shared_ptr<T>` | 最后一个持有者负责释放；引用计数决定寿命 |
| 观察 | `T&`、`T*`、`std::weak_ptr<T>` | 我只是用一下，不负责释放；要保证用的时候它还活着 |

默认选独占。只有"谁最后用完说不准"的情况（多个请求共享同一段前缀的 KV 块、异步回调里还要用到会话对象）才用共享。**裸指针和引用在现代 C++ 里表示"不拥有"**：看到 `T*` 参数，就应该理解为"借用"。

## `unique_ptr`：独占所有权

`unique_ptr` 是一个只能移动、不能拷贝的指针，析构时释放所指的对象。用默认删除器时它和裸指针一样大，没有任何运行时开销：

```cpp title="unique_basics.cpp"
#include <cstdio>
#include <memory>
#include <vector>

struct Stream {   // 假装是一个 CUDA stream
  int id;
  explicit Stream(int i) : id(i) { std::printf("创建 stream %d\n", id); }
  ~Stream() { std::printf("销毁 stream %d\n", id); }
};

std::unique_ptr<Stream> make_stream(int id) { return std::make_unique<Stream>(id); }

void launch_on(const Stream& s) { std::printf("在 stream %d 上启动 kernel\n", s.id); }   // 只借用

int main() {
  auto s1 = make_stream(1);
  launch_on(*s1);
  std::vector<std::unique_ptr<Stream>> pool;
  pool.push_back(std::move(s1));   // 所有权交给 pool，s1 变成空指针
  pool.push_back(make_stream(2));
  std::printf("s1 %s\n", s1 ? "非空" : "为空");
  pool.erase(pool.begin());        // 从 pool 里删掉，stream 1 立刻销毁
  std::printf("pool 剩 %zu 个\n", pool.size());
}
```

```text title="输出"
创建 stream 1
在 stream 1 上启动 kernel
创建 stream 2
s1 为空
销毁 stream 1
pool 剩 1 个
销毁 stream 2
```

总是用 `std::make_unique<T>(args...)` 创建，不写裸的 `new`。

### 自定义删除器：包装 C 风格的句柄

CUDA、NCCL、RDMA（`ibverbs`）、文件系统的 API 都是"创建函数返回句柄，销毁函数释放句柄"的 C 风格接口。给 `unique_ptr` 一个删除器，就得到一个 RAII 句柄类：

```cpp title="custom_deleter.cpp"
#include <cstdio>
#include <memory>

// C 风格的句柄 API（形状和 cudaStreamCreate / cudaStreamDestroy、ncclCommInitRank / ncclCommDestroy 一样）
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

```text title="输出"
create 1
create 2
sizeof(unique_ptr<int>)=8
sizeof(Handle)=8
sizeof(函数指针当删除器)=16
destroy 2
destroy 1
```

用一个没有成员的删除器类型（或者无捕获的 lambda 的类型），`unique_ptr` 仍然只有一个指针大；用函数指针当删除器，就要额外存一个函数指针。
`cudaStream_t` 本身就是一个指针类型（`CUstream_st*`），所以 `std::unique_ptr<CUstream_st, StreamDeleter>` 可以直接包装它；如果句柄是整数（比如文件描述符），就写一个上一章那样的小句柄类。

## `shared_ptr`：共享所有权

`shared_ptr` 多了一个**控制块**，里面有引用计数。拷贝一个 `shared_ptr` 计数加一，析构一个计数减一，减到 0 时释放对象。

- 计数的增减是**原子操作**，所以多个线程各自拷贝、销毁自己的 `shared_ptr` 是安全的；但**多个线程读写同一个 `shared_ptr` 变量**（一个线程给它赋值、另一个线程拷贝它）不安全，要加锁或用 `std::atomic<std::shared_ptr<T>>` <span class="since">C++20</span>；它所指的对象本身也不因此变得线程安全；
- `std::make_shared<T>(args...)` 把对象和控制块放在**一次**内存分配里，比 `shared_ptr<T>(new T)` 少一次分配、缓存也更友好；
- 原子的引用计数不是免费的：在热路径上（比如每个 token 每个块都拷贝一次）会成为瓶颈，这也是推理系统里 KV 块的引用计数通常放在块分配器里、用普通整数管理的原因（见[分配器与内存池](../memory/allocators.md)）。

下面用 `shared_ptr` 表示"多个请求共享系统提示词的 KV 块"，用 `weak_ptr` 表示"前缀缓存只观察、不延长寿命"：

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
  auto system_prompt = std::make_shared<KvBlock>(0);    // 公共前缀的 KV 块
  std::weak_ptr<KvBlock> cache_entry = system_prompt;   // 前缀缓存只观察
  {
    Request a{"a", {system_prompt}};
    {
      Request b{"b", {system_prompt, std::make_shared<KvBlock>(2)}};
      system_prompt.reset();                            // 创建者不再持有
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

```text title="输出"
块 0 的引用计数：2
请求 b 结束：
  释放块 2
请求 a 结束：
  释放块 0
缓存里的块 0 已经失效
未命中：需要重新计算前缀
```

`weak_ptr` 不增加引用计数；要用它所指的对象时调用 `lock()`，得到一个 `shared_ptr`（对象已经释放时是空的）。"检查 `expired()` 再使用"在多线程下有竞态，永远用 `lock()` 的返回值。

### 循环引用

两个对象用 `shared_ptr` 互相持有，引用计数永远不会降到 0，两个对象都泄漏。最常见的形式是**回调捕获了自己**：会话对象持有一个回调，回调里捕获了指向会话的 `shared_ptr`。**这个程序有 bug：**

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
  s->on_done->owner = s;   // 环：Session → Callback → Session
}                          // s 离开作用域后两个对象的引用计数都还是 1：泄漏

int main() {
  for (int i = 0; i < 4; ++i) open_session();   // 每个会话都泄漏
  std::printf("4 个会话都结束了\n");
}
```

```text title="LeakSanitizer 的报告（节选）"
==12345==ERROR: LeakSanitizer: detected memory leaks
Indirect leak of 32 byte(s) in 1 object(s) allocated from:
    #8 in std::make_shared<Session>() ...
    #9 in open_session() cycle_leak.cpp:14
SUMMARY: AddressSanitizer: 256 byte(s) leaked in 8 allocation(s).
```

两个对象互相指着，谁也不是"从外面可达"的，所以 LeakSanitizer 把它们报成间接泄漏（indirect leak）。

解法是让环上的某一条边变成 `weak_ptr`。异步回调的标准写法：回调里捕获 `weak_ptr`，执行时 `lock()`，对象已经不在了就什么都不做：

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
    std::weak_ptr<Session> self = weak_from_this();   // 不要捕获 shared_from_this()：那会形成环
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
  auto cb = s->on_done;   // 模拟：回调被交给了异步完成队列
  cb();
  s.reset();              // 客户端断开，会话被销毁
  cb();                   // 迟到的完成事件：安全地什么都不做
}
```

```text title="输出"
session 7 完成
session 7 析构
session 已经不在了
```

`std::enable_shared_from_this` 让对象能从自己内部拿到指向自己的 `shared_ptr` / `weak_ptr`（前提是对象本身由 `shared_ptr` 管理）。网络库、RPC 框架里的连接对象几乎都这么写。

## 函数参数怎么写

| 函数要做什么 | 参数写法 |
| --- | --- |
| 只读 / 修改对象，不关心它归谁 | `const T&` / `T&`（可能为空时用 `T*`） |
| 接管所有权 | `std::unique_ptr<T>`（按值，调用者用 `std::move` 交出来） |
| 也要成为共享所有者之一（比如存起来） | `std::shared_ptr<T>`（按值，再 `std::move` 进成员） |
| 可能会、也可能不会保存一份 | `const std::shared_ptr<T>&` |

最常见的错误是把 `shared_ptr` 一路按值传下去：每一层都做一次原子的加一和减一，而函数其实只是读一下对象。只是使用，就传 `const T&`。

## 练习

1. 给下面的场景选参数类型：
    1. `void record(const Request& r)` 还是 `void record(std::shared_ptr<Request> r)`：记录一条请求的日志；
    2. 调度器把一个新建的请求交给批处理队列，从此由队列负责它的寿命；
    3. 多个请求共享的 LoRA 权重，被一个异步加载任务在后台持有，加载完成前不能释放；
    4. 一个可选的"草稿模型"参数，可以没有。

??? success "参考答案"
    1. `const Request&`：只读，不涉及所有权。
    2. `void enqueue(std::unique_ptr<Request> r)`，调用处 `queue.enqueue(std::move(req))`。
    3. `std::shared_ptr<LoraWeights>` 按值传给后台任务（任务持有一份，保证加载期间不被释放）。
    4. `const DraftModel*`（可以为空）；或者 `std::optional<std::reference_wrapper<const DraftModel>>`，但裸指针更常见也更清楚。

2. 多 LoRA 服务里，同一个适配器可能被很多请求同时使用，没有请求用它时就应该卸载。写一个 `AdapterCache`：`get(name)` 返回 `shared_ptr`；缓存自己只保存 `weak_ptr`，已经在内存里就直接返回同一份，否则重新加载；`purge()` 删除已经失效的条目并返回删除的数量。

??? success "参考答案"
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
          if (auto a = it->second.lock()) return a;   // 还有请求在用：直接复用
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
      { auto b = cache.get("lora-b"); }   // 用完即卸载
      std::printf("清理了 %zu 个失效条目\n", cache.purge());
      a1.reset();
      a2.reset();
      auto a3 = cache.get("lora-a");      // 已经卸载过，重新加载
    }
    ```

    ```text title="输出"
    加载 lora-a
    同一份：是，引用 2
    加载 lora-b
    卸载 lora-b
    清理了 1 个失效条目
    卸载 lora-a
    加载 lora-a
    卸载 lora-a
    ```

    真实系统里"没人用就立刻卸载"往往太激进（下一个请求马上又要加载），通常会再加一层 LRU：最近用过的若干个适配器由缓存持有 `shared_ptr`，超出容量才降级成 `weak_ptr`。

## 小结

- [x] 所有权分三种：独占（`unique_ptr`、按值成员）、共享（`shared_ptr`）、观察（引用、裸指针、`weak_ptr`）。默认独占。
- [x] `unique_ptr` 只能移动，默认删除器下零开销；配上自定义删除器就能包装任何 C 风格句柄。
- [x] `shared_ptr` 用原子引用计数，`make_shared` 一次分配；热路径上的频繁拷贝有成本。
- [x] 循环引用会泄漏；异步回调捕获 `weak_ptr`，执行时 `lock()`。
- [x] 函数只是使用对象就传引用，只有转移或共享所有权时才传智能指针。
