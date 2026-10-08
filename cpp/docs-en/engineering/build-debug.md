# CMake, tests, the sanitizers and perf

<p class="lead">One file compiles with a single line of <code>g++</code>, while a real component has headers, sources, tests, dependencies and several build configurations. This chapter turns the block allocator written earlier into a proper CMake project: with tests, with sanitizer builds and producing a compilation database for an IDE; then how to investigate when something goes wrong, gdb for a crash, gdb or py-spy for a hang, perf for slowness.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the difference between `PUBLIC` and `PRIVATE` in `target_include_directories(...)`?
    2. How do you get a Release build, an ASan build and a TSan build out of the same CMake project?
    3. What is `compile_commands.json` for?
    4. An inference service process written in C++ has hung with 0% CPU. What is your first step?
    5. What question does `perf stat` answer, and what does `perf record` answer?

??? success "Answers (try it yourself first, then expand)"
    1. A `PUBLIC` include directory is used both to compile this target and by any target that links it; a `PRIVATE` one is used only by this target and is not propagated.
    2. Make the sanitizers a CMake option (`-DSANITIZE=address,undefined`, say) and give each configuration its own build directory (`build-release`, `build-asan`, `build-tsan`), configured, built and tested separately.
    3. It records the full compile command for every source file (include paths, macro definitions, the standard) so that clangd, an IDE and clang-tidy can parse the code accurately and give navigation and completion in a large C++ repository.
    4. Look at what every thread is waiting for first: `gdb -p PID` then `thread apply all bt` (`py-spy dump` for a Python process), which usually shows the deadlocked lock, the condition variable being waited on, the blocked I/O or the GPU synchronization directly.
    5. `perf stat` answers "which class of slowness": counting total cycles, instructions, cache misses, branch mispredictions and the rest; `perf record` answers "which function": sampling call stacks, located with `perf report` or a flame graph.

## A C++ component's project layout {#一个-c-组件的工程结构}

![Figure: a C++ component's layout - headers, implementation, tests, bindings](../assets/figures/project-layout.svg){.aig-svg}

Turn the block allocator from [allocators and memory pools](../memory/allocators.md) into a library called `kvpool`:

<!-- i18n:diagram 692ab62c7f -->
```text
kvpool/
├── CMakeLists.txt
├── include/kvpool/block_pool.hpp   the public header: users #include <kvpool/block_pool.hpp>
├── src/block_pool.cpp              the implementation
└── tests/test_block_pool.cpp       the tests
```

The header holds only declarations and the small functions that have to be inline, with the implementation in a `.cpp`: changing the implementation then recompiles one file, and users are not slowed down by the implementation details (and the pile of headers they `#include`).

```cpp title="include/kvpool/block_pool.hpp" project="kvpool"
#pragma once
#include <cstdint>
#include <optional>
#include <vector>

namespace kvpool {

// a fixed number of KV blocks: a free stack plus reference counts. For use in one thread only (the scheduler's).
class BlockPool {
 public:
  explicit BlockPool(std::int32_t num_blocks);

  // allocate n blocks: either all of them or none at all (returning nullopt)
  std::optional<std::vector<std::int32_t>> allocate(std::int32_t n);
  void retain(std::int32_t block);    // sharing: increment the reference count
  void release(std::int32_t block);   // decrement the reference count, returning the block to the free stack at 0

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

The test depends on no framework and one `CHECK` macro is enough; a real project usually uses GoogleTest or Catch2 (brought in with CMake's `FetchContent`).

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
  CHECK(!pool.allocate(6));                     // not enough: allocate none
  CHECK(pool.num_free() == 5);

  for (auto b : *a) pool.retain(b);             // another sequence shares these 3 blocks
  for (auto b : *a) pool.release(b);
  CHECK(pool.num_free() == 5 && pool.refcount((*a)[0]) == 1);
  for (auto b : *a) pool.release(b);
  CHECK(pool.num_free() == 8);

  bool threw = false;
  try {
    pool.release((*a)[0]);                      // a double free has to be caught
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  CHECK(pool.allocate(8) && pool.num_free() == 0);
  std::printf("block_pool：%d 个检查全部通过\n", g_checks);
}
```

## CMakeLists.txt {#cmakeliststxt}

Modern CMake's central concept is the **target**: a library and an executable are each a target, and the compile options, include paths and dependencies hang on the target, with `PUBLIC` / `PRIVATE` saying whether they propagate to whoever depends on it:

- `PRIVATE`: used only when building this target (`-Werror`, the internal headers the implementation uses);
- `PUBLIC`: used by it and obtained automatically by anything that links it (the public headers' directory, requiring C++20);
- `INTERFACE`: not used by it, only passed to users (a header-only library).

```cmake title="CMakeLists.txt" project="kvpool"
cmake_minimum_required(VERSION 3.20)
project(kvpool LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # generates build/compile_commands.json for clangd and IDEs

# turn the sanitizers on with -DKVPOOL_SANITIZE=address,undefined or thread; it has to come before the targets are created
set(KVPOOL_SANITIZE "" CACHE STRING "要打开的 sanitizer，例如 address,undefined 或 thread")
if(KVPOOL_SANITIZE)
  add_compile_options(-fsanitize=${KVPOOL_SANITIZE} -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${KVPOOL_SANITIZE})
endif()

add_library(kvpool src/block_pool.cpp)
target_include_directories(kvpool PUBLIC include)        # users can #include <kvpool/...> too
target_compile_features(kvpool PUBLIC cxx_std_20)        # users have to use C++20 as well
target_compile_options(kvpool PRIVATE -Wall -Wextra -Werror)

enable_testing()
add_executable(test_block_pool tests/test_block_pool.cpp)
target_link_libraries(test_block_pool PRIVATE kvpool)    # gets the include directory and C++20 automatically
add_test(NAME block_pool COMMAND test_block_pool)
```

Configuring, building and testing are three commands. The build directory is kept apart from the source (`-B build`), and as many build directories are created as there are configurations:

```bash title="build.sh" project="kvpool" run="yes"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DKVPOOL_SANITIZE=address,undefined > /dev/null
cmake --build build -j 8 > /dev/null
./build/test_block_pool
ctest --test-dir build -Q && echo "ctest：全部通过"
```

```text title="output"
block_pool：7 个检查全部通过
ctest：全部通过
```

The build directories commonly used:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release          # -O3 -DNDEBUG: for measuring performance and for release
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DKVPOOL_SANITIZE=address,undefined
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKVPOOL_SANITIZE=thread
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache  # Ninja is faster than make; ccache caches the compilation results
```

`compile_commands.json` records the full compile command for every file, and clangd (the C++ plugin for VS Code and Neovim) needs it to get definitions, completion and diagnostics right. Configuring a large C++ repository once to generate this file before reading it makes the experience much better.

## Compilation time {#编译时间}

A template-heavy inference library routinely takes tens of minutes to build. The usual measures:

- **ccache**: the same input reuses the last build's result;
- **Ninja** plus enough parallelism (`-j`);
- **slimmer headers**: forward declarations instead of `#include` in headers, with the heavy implementation in a `.cpp`;
- **controlling template instantiation**: instantiate only the combinations used, spread the instantiations over several `.cpp` files to compile in parallel, or move to JIT (see [templates, concepts and constexpr](../basics/templates.md));
- `-ftime-trace` (clang) reports how long each file and each template instantiation took, which finds the slowest few.

## Investigating a problem {#出了问题怎么查}

### A crash: the sanitizers and gdb {#崩溃sanitizer-和-gdb}

Most crashes, rerun under an ASan + UBSan build, come with the failing line and the memory's history in the report. When a debugger is still needed:

```bash
gdb --args ./build/test_block_pool      # start it
(gdb) run
(gdb) bt                                # the call stack after the crash
(gdb) frame 2                           # move to frame 2
(gdb) print pool.free_.size()           # inspect a variable
(gdb) info locals

# print the crash's call stack without an interactive session (good in a script or CI)
gdb -batch -ex run -ex bt --args ./build/test_block_pool
```

A crash in production needs a **core dump**: enable it with `ulimit -c unlimited`, open it afterwards with `gdb ./server core.12345` and `bt` as before. A release build has to carry debug information (`RelWithDebInfo`, or compile with `-g` and strip the symbols into a separate file), or the stack holds nothing but addresses.

### A hang: what each thread is waiting for {#卡住看每个线程在等什么}

A service that hangs with 0% CPU is almost certainly deadlocked or waiting for an event that will never come (nobody `notify`s, the peer sent nothing, one card never reached a NCCL collective). The first step is **the call stack of every thread**:

```bash
gdb -p <pid> -batch -ex "thread apply all bt"   # a C++ process: where each thread is stopped
py-spy dump --pid <pid> --native                 # a process of Python plus a C++ extension (the usual shape of an inference service): both the Python and the C++ stacks
```

Find the threads stopped in `pthread_cond_wait` / `futex`, see which lock or condition they wait on, and then look for whoever should have woken them. When a multi-GPU program hangs in a collective, `NCCL_DEBUG=INFO` shows how far each card got; PyTorch distributed also has `TORCH_NCCL_ASYNC_ERROR_HANDLING` and a timeout, so a stuck collective fails with a timeout rather than hanging forever.

### Slowness: perf {#慢perf}

`perf` is Linux's own profiler, based on the hardware performance counters and sampling, with a very small overhead, and it can be pointed straight at a production process:

```bash
perf stat -e task-clock,cycles,instructions,cache-misses,branch-misses ./bench   # the overall counts: IPC and the cache miss rate
perf record -g ./bench                   # sample call stacks (-g), producing perf.data
perf report                              # where the time goes, by function
perf top -p <pid>                        # watch a running process's hot spots live
```

`perf stat` answers "which class of slowness": a very low IPC (instructions per cycle) with many cache misses says it is a memory problem (back to [object layout, alignment and the cache](../memory/layout.md)); many branch mispredictions say there is an unpredictable branch.
`perf record` answers "which function". Brendan Gregg's FlameGraph scripts turn `perf script`'s output into a flame graph, where the widest stack is visible at a glance.

A few notes:

- profile a **Release build with debug information** (`RelWithDebInfo` or `-O2 -g`) plus `-fno-omit-frame-pointer`, or the stacks are incomplete; a Debug build's hot spots are nothing like a Release build's;
- inside a container or a virtual machine the hardware counters (`cycles`, `cache-misses`) may be unavailable and `perf` shows `<not supported>`, in which case fall back to clock-based sampling (`-e task-clock`);
- for a GPU program, seeing how the CPU's and the GPU's timelines interleave calls for Nsight Systems (see the CUDA handbook's [profiling](cuda://tools/profiling/)).

!!! interview "How to explain it"
    On engineering: CMake centres on targets, with the include directories and the language standard `PUBLIC` (propagated to users) and the warning options `PRIVATE`; the sanitizers become an option, with one build directory each for Release, ASan + UBSan and TSan, all running the tests; `compile_commands.json` lets clangd and other tools understand a large repository. Investigating: a crash goes to the sanitizer report and gdb's `bt`; a process hung at 0% CPU starts with printing every thread's call stack (`gdb -p` then `thread apply all bt`, or `py-spy dump` for a service mixing Python), to see who waits on a lock, a condition variable or a collective; slowness starts with `perf stat` to tell computing slowly from waiting on memory, then `perf record` to find the function.

## Exercises {#练习}

1. Add a TSan build step to `kvpool`'s CI, and explain why this library currently shows nothing under TSan.

??? success "Answer"
    ```bash
    cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKVPOOL_SANITIZE=thread
    cmake --build build-tsan -j
    ctest --test-dir build-tsan --output-on-failure
    ```

    The tests are single-threaded today, and TSan can only find concurrent accesses that **actually happened**. `BlockPool` is designed on the premise of "used only in the scheduler thread", so for TSan to mean anything there has to be a multithreaded test, calling `allocate` from two threads on purpose and confirming TSan reports it (a "reverse test": proving that violating the contract is caught); and if multithreaded use is to be supported later, add a lock inside the class and write the positive concurrency tests.

2. A C++ scheduler exposed to Python with pybind11 hangs occasionally in production. In what order would you investigate?

??? success "Answer"
    1. `py-spy dump --pid <pid> --native`, to see where the Python threads and the C++ threads are stopped;
    2. if it is stopped on a lock or a condition variable, find the thread holding that lock or the one that should have `notify`ed; watch the **GIL** especially: C++ code holding its own lock then acquiring the GIL while another Python thread holding the GIL waits for that lock is the most typical deadlock in a Python extension (see [pybind11 and PyTorch C++ extensions](python-binding.md));
    3. reproduce it in a test environment with a TSan build of the extension (which needs a TSan build of Python, or the C++ part written as a separate multithreaded test);
    4. after fixing it, add a stress test, and consider giving the waits a timeout and a log line, so the next hang says outright what it is waiting for.

## Summary {#小结}

- [x] Organize a CMake project around targets: the include directories and the language standard `PUBLIC`, the warning options `PRIVATE`; keep the source and build directories apart, one build directory per configuration.
- [x] Make the sanitizers a CMake option, with a build each for ASan + UBSan and TSan, both running the tests.
- [x] `compile_commands.json` lets an IDE understand a large C++ repository; ccache, Ninja and controlling template instantiation shorten the build.
- [x] A crash goes to the sanitizer report and gdb's `bt`; a hang goes to every thread's call stack (gdb, py-spy); slowness goes to `perf stat` to classify and `perf record` to find the function.
