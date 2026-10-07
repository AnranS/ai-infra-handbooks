# C++ 进阶手册

<p class="lead">推理系统的"地基"大多是 C++：通信库、KV 传输引擎、算子的 host 端、Python 扩展、调度器的热路径。这本手册面向写过其他语言、C++ 只学过基础的工程师，目标是读得懂、改得动这些代码，并且写出正确、高效、经得起 sanitizer 检查的系统代码。</p>

## 这份手册适合谁

你大概处在这个阶段：

- 能用 Python（或 Rust、Go、Java）写出像样的程序，也写过一些 C / C++ 的作业或小工具；
- 打开 SGLang 的 `sgl-kernel`、FlashInfer、DeepEP、Mooncake 这类仓库时，能看懂大意，但对模板、移动语义、`std::atomic` 的内存序、各种 RAII 包装心里没底；
- 知道 C++ "容易出内存错误"，但说不清哪些写法是未定义行为，也不知道怎么系统地把它们找出来。

读完并练完这份手册，你应该能做到：

- 讲清楚对象的生命周期和所有权，写出不泄漏、不悬垂、异常安全的资源管理代码；
- 读懂推理基础库里常见的模板技巧（按数据类型、头维度的编译期派发）、RAII 句柄和错误处理宏；
- 设计对缓存友好的数据布局，写出内存池、块分配器这类"推理系统的内存管理"组件；
- 写出正确的并发代码：线程池、阻塞队列、无锁的单生产者单消费者队列，并能用 ThreadSanitizer 证明没有数据竞争；
- 用 CMake 组织工程，用 sanitizer 和 `perf` 定位问题，用 pybind11 / PyTorch C++ 扩展把实现暴露给 Python。

## 怎么用

1. **先做自测。** 每章开头有一个自测框，全部能答上来就跳到练习。
2. **代码一定要编译运行。** 每个标了文件名的程序都是完整的，复制下来按页面上的命令编译即可。用 sanitizer 编译是默认动作，不是可选项。
3. **先做练习再看答案。** 每章末尾有练习，答案默认折叠；配套的 C++ 练习题可以在本地一键判题（见下文）。
4. **对照真实代码读。** [读懂推理基础库的 C++](engineering/reading-code.md) 一章把前面的知识点映射到开源推理库里的真实写法，建议每学完一部分就回去读一段。

## 学习路线

各本手册合在一起的逐章路线见[学习路线图](root://roadmap/)，求职冲刺的逐周安排见[冲刺计划](root://plan/)。本书内部的顺序：

<div class="roadmap" markdown>

| 阶段 | 章节 | 学完能做什么 | 建议用时 |
| --- | --- | --- | --- |
| 一、现代 C++ 核心 | [编译模型与未定义行为](basics/compile-ub.md) · [值语义与 RAII](basics/value-raii.md) · [移动语义](basics/move.md) · [智能指针与所有权](basics/ownership.md) · [模板与 constexpr](basics/templates.md) · [标准库的性能视角](basics/stl-perf.md) | 读懂现代 C++ 代码，写出不泄漏、不悬垂的资源管理 | 1 周 |
| 二、内存 | [对象布局、对齐与缓存](memory/layout.md) · [分配器与内存池](memory/allocators.md) | 设计缓存友好的数据结构，写出块分配器 | 3～4 天 |
| 三、并发 | [线程、锁与条件变量](concurrency/threads.md) · [atomic 与内存序](concurrency/atomics.md) · [无锁队列与线程池](concurrency/lockfree-pool.md) | 写出能通过 TSan 的并发组件 | 1 周 |
| 四、工程与互操作 | [CMake、sanitizer 与 perf](engineering/build-debug.md) · [pybind11 与 PyTorch 扩展](engineering/python-binding.md) · [读懂推理基础库](engineering/reading-code.md) · [面试高频题](engineering/interview.md) | 独立搭建、调试一个 C++ 组件并接入 Python | 3～4 天 |

</div>

## 版本约定

- 以 **C++20** 为基线（`-std=c++20`），只在 C++23 才有的特性会标出来，比如 <span class="since">C++23</span>。
- 所有标了文件名的程序都用 **g++ 12** 编译（`-Wall -Wextra -Werror`），默认在 **AddressSanitizer + UndefinedBehaviorSanitizer** 下运行，并发章节的程序在 **ThreadSanitizer** 下运行；页面上的"输出"与实际运行结果逐行一致。
- 故意演示错误的程序会标明"这个程序有 bug"，页面上给出的是 sanitizer 报告的节选。

## 开始之前：准备环境

=== "Linux / WSL2"

    ```bash
    sudo apt install -y g++ cmake gdb linux-tools-generic   # g++ 12 以上
    g++ --version
    ```

=== "macOS"

    ```bash
    xcode-select --install          # Apple clang，支持 ASan、UBSan、TSan
    brew install cmake
    ```

    macOS 上的 Apple clang 不支持 LeakSanitizer，内存泄漏可以用 `leaks --atExit -- ./a.out` 检查。`std::jthread` 需要较新的 libc++：建议 `brew install llvm`，用 `$(brew --prefix llvm)/bin/clang++` 编译并加上 `-fexperimental-library`（`tools/check_code.py` 会自动这样做）。libc++ 与 Linux 上的 libstdc++ 在实现细节上不同，`sizeof(std::string)`、容量、哈希表桶数这类输出会与页面不一样——页面上的输出以 Linux + GCC 为准。完整的 Mac 环境说明见[学习环境](root://setup/)。

建议给自己定义一个编译别名，让 sanitizer 成为默认：

```bash
# 加到 ~/.bashrc 或 ~/.zshrc
alias cxx='g++ -std=c++20 -g -O1 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer'
alias cxxt='g++ -std=c++20 -g -O1 -Wall -Wextra -fsanitize=thread'
cxx hello.cpp -o hello && ./hello
```

## 配套练习

练习题网站上有一组 C++ 题（[练习题 · C++ 进阶](root://practice/)），因为浏览器里没有 C++ 编译器，这些题在本地判题：

```bash
python practice/judge.py start cpp-raii-fd     # 复制模板到 practice/workspace/
python practice/judge.py test cpp-raii-fd      # g++ 编译，在 sanitizer 下跑测试
```
