# CMake、测试、sanitizer 与 perf

<p class="lead">单个文件用一行 <code>g++</code> 就能编译，真正的组件却有头文件、源文件、测试、依赖和不同的构建配置。这一章把前面写过的块分配器做成一个像样的 CMake 工程：带测试、带 sanitizer 构建、能生成给 IDE 用的编译数据库；再讲出了问题怎么查——崩溃用 gdb，卡住用 gdb 或 py-spy，慢用 perf。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `target_include_directories(... PUBLIC ...)` 和 `PRIVATE` 有什么区别？
    2. 怎样让同一份 CMake 工程既能出 Release 构建，又能出 ASan 构建和 TSan 构建？
    3. `compile_commands.json` 是做什么用的？
    4. 一个 C++ 写的推理服务进程卡住了、CPU 占用为 0，你第一步做什么？
    5. `perf stat` 和 `perf record` 分别回答什么问题？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `PUBLIC` 的头文件目录既用于编译这个目标本身，也传递给链接了它的其他目标；`PRIVATE` 只用于这个目标自己，不传递。
    2. 把 sanitizer 做成一个 CMake 选项（比如 `-DSANITIZE=address,undefined`），每种配置用一个独立的构建目录（`build-release`、`build-asan`、`build-tsan`），分别配置、构建、跑测试。
    3. 记录每个源文件的完整编译命令（包含路径、宏定义、标准），供 clangd、IDE、clang-tidy 等工具准确地解析代码，在大型 C++ 仓库里跳转和补全。
    4. 先看所有线程在等什么：`gdb -p PID` 后 `thread apply all bt`（Python 进程用 `py-spy dump`），通常能直接看到死锁的锁、等待的条件变量、阻塞的 I/O 或者 GPU 同步。
    5. `perf stat` 回答"慢在哪一类"：数出总的周期、指令、缓存未命中、分支预测失败等计数；`perf record` 回答"慢在哪个函数"：采样调用栈，配合 `perf report` 或火焰图定位热点。

## 一个 C++ 组件的工程结构

把[分配器与内存池](../memory/allocators.md)里的块分配器做成一个库 `kvpool`：

```text
kvpool/
├── CMakeLists.txt
├── include/kvpool/block_pool.hpp   公开的头文件：库的使用者 #include <kvpool/block_pool.hpp>
├── src/block_pool.cpp              实现
└── tests/test_block_pool.cpp       测试
```

头文件只放声明和必须内联的小函数，实现放进 `.cpp`：改实现时只需要重新编译一个文件，使用者也不会被实现细节（以及它 `#include` 的一大堆头文件）拖慢编译。

```cpp title="include/kvpool/block_pool.hpp" project="kvpool"
#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace kvpool {

// 固定数量的 KV 块：空闲栈 + 引用计数。只能在一个线程里使用（调度器线程）。
class BlockPool {
 public:
  explicit BlockPool(std::int32_t num_blocks);

  // 分配 n 个块，要么全部成功，要么一个都不分配（返回 nullopt）
  std::optional<std::vector<std::int32_t>> allocate(std::int32_t n);
  void retain(std::int32_t block);    // 共享：引用计数加一
  void release(std::int32_t block);   // 引用计数减一，到 0 时回到空闲栈

  std::int32_t num_free() const { return static_cast<std::int32_t>(free_.size()); }
  std::int32_t refcount(std::int32_t block) const { return ref_.at(block); }

 private:
  std::vector<std::int32_t> free_;
  std::vector<std::int32_t> ref_;
};

}  // namespace kvpool
```

```cpp title="src/block_pool.cpp" project="kvpool"
#include "kvpool/block_pool.hpp"

#include <stdexcept>

namespace kvpool {

BlockPool::BlockPool(std::int32_t num_blocks) : ref_(num_blocks, 0) {
  free_.reserve(num_blocks);
  for (std::int32_t i = num_blocks - 1; i >= 0; --i) free_.push_back(i);
}

std::optional<std::vector<std::int32_t>> BlockPool::allocate(std::int32_t n) {
  if (n < 0 || n > num_free()) return std::nullopt;
  std::vector<std::int32_t> out(free_.end() - n, free_.end());
  free_.resize(free_.size() - n);
  for (auto b : out) ref_[b] = 1;
  return out;
}

void BlockPool::retain(std::int32_t block) {
  if (ref_.at(block) == 0) throw std::logic_error("retain 了一个空闲块");
  ++ref_[block];
}

void BlockPool::release(std::int32_t block) {
  if (ref_.at(block) == 0) throw std::logic_error("重复释放");
  if (--ref_[block] == 0) free_.push_back(block);
}

}  // namespace kvpool
```

测试不依赖任何框架，一个 `CHECK` 宏就够用；真实项目里通常用 GoogleTest 或 Catch2（用 CMake 的 `FetchContent` 引入）。

```cpp title="tests/test_block_pool.cpp" project="kvpool"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "kvpool/block_pool.hpp"

static int g_checks = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    ++g_checks;                                                         \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "%s:%d: 检查失败：%s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                     \
    }                                                                   \
  } while (0)

int main() {
  kvpool::BlockPool pool(8);
  auto a = pool.allocate(3);
  CHECK(a && a->size() == 3 && pool.num_free() == 5);
  CHECK(!pool.allocate(6));                     // 不够：一个都不分配
  CHECK(pool.num_free() == 5);

  for (auto b : *a) pool.retain(b);             // 另一个序列共享这 3 个块
  for (auto b : *a) pool.release(b);
  CHECK(pool.num_free() == 5 && pool.refcount((*a)[0]) == 1);
  for (auto b : *a) pool.release(b);
  CHECK(pool.num_free() == 8);

  bool threw = false;
  try {
    pool.release((*a)[0]);                      // 重复释放要被发现
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(pool.allocate(8) && pool.num_free() == 0);
  std::printf("block_pool：%d 个检查全部通过\n", g_checks);
}
```

## CMakeLists.txt

现代 CMake 的核心概念是**目标**（target）：库、可执行文件各是一个目标，编译选项、头文件路径、依赖都挂在目标上，并用 `PUBLIC` / `PRIVATE` 说明它们是否要传递给依赖这个目标的人：

- `PRIVATE`：只有自己编译时用（比如 `-Werror`、实现用到的内部头文件）；
- `PUBLIC`：自己用，链接我的目标也会自动得到（比如公开头文件的目录、要求 C++20）；
- `INTERFACE`：自己不用，只传给使用者（纯头文件库）。

```cmake title="CMakeLists.txt" project="kvpool"
cmake_minimum_required(VERSION 3.20)
project(kvpool LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # 生成 build/compile_commands.json，给 clangd、IDE 用

# 用 -DKVPOOL_SANITIZE=address,undefined 或 thread 打开 sanitizer；要放在创建目标之前
set(KVPOOL_SANITIZE "" CACHE STRING "要打开的 sanitizer，例如 address,undefined 或 thread")
if(KVPOOL_SANITIZE)
  add_compile_options(-fsanitize=${KVPOOL_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${KVPOOL_SANITIZE})
endif()

add_library(kvpool src/block_pool.cpp)
target_include_directories(kvpool PUBLIC include)        # 使用者也能 #include <kvpool/...>
target_compile_features(kvpool PUBLIC cxx_std_20)        # 使用者也必须用 C++20
target_compile_options(kvpool PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_executable(test_block_pool tests/test_block_pool.cpp)
target_link_libraries(test_block_pool PRIVATE kvpool)    # 自动得到 include 目录和 C++20
add_test(NAME block_pool COMMAND test_block_pool)
```

配置、构建、测试是三条命令。构建目录和源码目录分开（`-B build`），想要几种配置就建几个构建目录：

```bash title="build.sh" project="kvpool" run="yes"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DKVPOOL_SANITIZE=address,undefined > /dev/null
cmake --build build -j 8 > /dev/null
./build/test_block_pool
ctest --test-dir build -Q && echo "ctest：全部通过"
```

```text title="输出"
block_pool：7 个检查全部通过
ctest：全部通过
```

常用的几种构建目录：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release          # -O3 -DNDEBUG：测性能、发布
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DKVPOOL_SANITIZE=address,undefined
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKVPOOL_SANITIZE=thread
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache  # Ninja 比 make 快；ccache 缓存编译结果
```

`compile_commands.json` 记录了每个文件的完整编译命令，clangd（VS Code、Neovim 的 C++ 插件）靠它才能正确地跳转定义、补全和报错。读一个大型 C++ 仓库之前，先把它配置一遍、生成这个文件，体验会好很多。

## 编译时间

模板重的推理库，编译一次几十分钟很常见。常用的手段：

- **ccache**：同样的输入直接复用上次的编译结果；
- **Ninja** + 足够的并行度（`-j`）；
- **头文件瘦身**：头文件里用前向声明代替 `#include`，重的实现放进 `.cpp`；
- **控制模板实例化**：只实例化会用到的组合，把实例化分散到多个 `.cpp` 并行编译，或者改成 JIT（见[模板、concepts 与 constexpr](../basics/templates.md)）；
- `-ftime-trace`（clang）能输出每个文件、每个模板实例化花了多少时间，找出最慢的那几个。

## 出了问题怎么查

### 崩溃：sanitizer 和 gdb

大部分崩溃，用 ASan + UBSan 构建重新跑一遍，报告里就有出错的行和内存的来历。仍然需要调试器时：

```bash
gdb --args ./build/test_block_pool      # 启动
(gdb) run
(gdb) bt                                # 崩溃后看调用栈
(gdb) frame 2                           # 切到第 2 层
(gdb) print pool.free_.size()           # 查看变量
(gdb) info locals

# 不进交互界面，直接打印崩溃时的调用栈（适合放进脚本和 CI）
gdb -batch -ex run -ex bt --args ./build/test_block_pool
```

线上的崩溃靠 **core dump**：`ulimit -c unlimited` 打开，崩溃后用 `gdb ./server core.12345` 打开，同样 `bt`。发布构建要带上调试信息（`RelWithDebInfo`，或者用 `-g` 编译后把符号剥离单独保存），否则调用栈里只有地址。

### 卡住：看每个线程在等什么

服务卡住、CPU 占用为 0，几乎一定是死锁或者在等一个永远不会到来的事件（没人 `notify`、对端没发数据、NCCL 集合通信里某张卡没到）。第一步是**看每个线程的调用栈**：

```bash
gdb -p <pid> -batch -ex "thread apply all bt"   # C++ 进程：每个线程停在哪里
py-spy dump --pid <pid> --native                 # Python + C++ 扩展的进程（推理服务的常见形态）：同时显示 Python 和 C++ 栈
```

找到"停在 `pthread_cond_wait` / `futex` 里"的线程，看它等的是哪把锁、哪个条件，再找本该唤醒它的那一方。多卡程序卡在集合通信里时，设置 `NCCL_DEBUG=INFO` 能看到每张卡进行到了哪一步；PyTorch 分布式还可以设置 `TORCH_NCCL_ASYNC_ERROR_HANDLING` 和超时，让卡住的集合通信超时报错而不是永远挂着。

### 慢：perf

`perf` 是 Linux 自带的性能分析工具，基于硬件性能计数器和采样，开销很小，可以直接对线上进程用：

```bash
perf stat -e task-clock,cycles,instructions,cache-misses,branch-misses ./bench   # 总体统计：IPC、缓存未命中率
perf record -g ./bench                   # 采样调用栈（-g），生成 perf.data
perf report                              # 按函数看时间花在哪里
perf top -p <pid>                        # 实时看一个运行中的进程的热点
```

`perf stat` 回答"慢在哪一类"：IPC（每周期指令数）很低、缓存未命中很多，说明是访存问题（回到[对象布局、对齐与缓存](../memory/layout.md)）；分支预测失败多，说明有难以预测的分支。
`perf record` 回答"慢在哪个函数"。用 Brendan Gregg 的 FlameGraph 脚本把 `perf script` 的输出画成火焰图，一眼就能看出最宽的调用栈。

几点注意：

- 性能分析用 **Release + 调试信息**（`RelWithDebInfo` 或 `-O2 -g`），并加 `-fno-omit-frame-pointer`，否则调用栈不完整；Debug 构建的热点和 Release 完全不同；
- 在容器或虚拟机里，硬件计数器（`cycles`、`cache-misses`）可能不可用，`perf` 会显示 `<not supported>`，这时退回到基于时钟的采样（`-e task-clock`）；
- GPU 程序要看 CPU 和 GPU 的时间线怎么交织，用 Nsight Systems（见 CUDA 手册的[性能分析](cuda://tools/profiling/)）。

## 练习

1. 给 `kvpool` 加一个 TSan 构建的 CI 步骤，并说明为什么这个库目前在 TSan 下测不出任何问题。

??? success "参考答案"
    ```bash
    cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKVPOOL_SANITIZE=thread
    cmake --build build-tsan -j
    ctest --test-dir build-tsan --output-on-failure
    ```

    现在的测试是单线程的，TSan 只能发现**实际发生了**的并发访问。`BlockPool` 的设计前提是"只在调度器线程里使用"，要让 TSan 有意义，需要一个多线程的测试——比如故意从两个线程调用 `allocate`，确认 TSan 能报告出来（这是一个"反向测试"：证明违反使用约定时会被发现）；如果以后要支持多线程使用，就在类里加锁，再写正向的并发测试。

2. 一个用 pybind11 暴露给 Python 的 C++ 调度器，线上偶尔卡住。你会按什么顺序排查？

??? success "参考答案"
    1. `py-spy dump --pid <pid> --native`，看 Python 线程和 C++ 线程分别停在哪里；
    2. 如果停在锁或条件变量上，找到持有这把锁、或者本该 `notify` 的线程；特别注意 **GIL**：C++ 代码持有自己的锁时去获取 GIL，而另一个持有 GIL 的 Python 线程又在等这把锁——这是 Python 扩展里最典型的死锁（见 [pybind11 与 PyTorch C++ 扩展](python-binding.md)）；
    3. 用 TSan 构建的扩展在测试环境里复现（需要用 TSan 构建的 Python，或者把 C++ 部分单独写成多线程测试）；
    4. 修复后加一个压力测试，并考虑给等待加超时和日志，下次卡住时能直接看到在等什么。

## 小结

- [x] 用目标组织 CMake 工程：头文件目录和语言标准 `PUBLIC`，警告选项 `PRIVATE`；源码目录和构建目录分开，每种配置一个构建目录。
- [x] sanitizer 做成一个 CMake 选项，ASan + UBSan、TSan 各有一个构建，都跑测试。
- [x] `compile_commands.json` 让 IDE 读懂大型 C++ 仓库；ccache、Ninja、控制模板实例化来缩短编译时间。
- [x] 崩溃看 sanitizer 报告和 gdb 的 `bt`；卡住看所有线程的调用栈（gdb、py-spy）；慢先 `perf stat` 分类，再 `perf record` 定位函数。
