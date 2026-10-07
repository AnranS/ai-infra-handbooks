# Advanced C++

<p class="lead">An inference system's foundations are mostly C++: the communication libraries, the KV transfer engine, an operator's host side, the Python extensions, the scheduler's hot path. This handbook is for engineers who have written other languages and only the basics of C++, and aims at being able to read and change that code, and to write systems code that is correct, efficient and survives the sanitizers.</p>

## Who this handbook is for {#这份手册适合谁}

You are probably around here:

- you can write a decent program in Python (or Rust, Go, Java) and have written some C / C++ exercises or small tools;
- opening a repository like SGLang's `sgl-kernel`, FlashInfer, DeepEP or Mooncake, you follow the gist, but templates, move semantics, `std::atomic`'s memory orders and the various RAII wrappers leave you unsure;
- you know C++ "makes memory errors easy", but cannot say which constructs are undefined behaviour, nor how to find them systematically.

Having read this handbook and done the exercises, you should be able to:

- explain object lifetimes and ownership, and write resource management that neither leaks nor dangles and is exception-safe;
- read the template tricks common in inference libraries (compile-time dispatch by data type and head dimension), the RAII handles and the error-handling macros;
- design cache-friendly data layouts and write the memory pools and block allocators that are "an inference system's memory management";
- write correct concurrent code: a thread pool, a blocking queue, a lock-free single-producer single-consumer queue, and prove there is no data race with ThreadSanitizer;
- organize a project with CMake, locate problems with the sanitizers and `perf`, and expose an implementation to Python with pybind11 / a PyTorch C++ extension.

## How to use it {#怎么用}

1. **Take the self-test first.** Every chapter opens with one, and if you can answer all of it, skip to the exercises.
2. **Always compile and run the code.** Every program with a file name on it is complete, so copy it and compile it with the command on the page. Compiling with the sanitizers is the default, not an option.
3. **Do the exercises before reading the answers.** Every chapter ends with exercises whose answers are collapsed; the companion C++ problems are graded locally with one command (below).
4. **Read it against real code.** [Reading an inference library's C++](engineering/reading-code.md) maps what came before onto the real constructs in open-source inference libraries, and is worth going back to after each part.

## The learning path {#学习路线}

The chapter-by-chapter path across all the handbooks is in the [roadmap](root://roadmap/), and the week-by-week arrangement for a job-hunting sprint is in the [sprint plan](root://plan/). The order within this book:

<div class="roadmap" markdown>

| Stage | Chapters | What you can do afterwards | Suggested time |
| --- | --- | --- | --- |
| 1. Modern C++ essentials | [the compilation model and undefined behaviour](basics/compile-ub.md) · [value semantics and RAII](basics/value-raii.md) · [move semantics](basics/move.md) · [smart pointers and ownership](basics/ownership.md) · [templates and constexpr](basics/templates.md) · [the standard library from a performance angle](basics/stl-perf.md) | read modern C++ code, write resource management that neither leaks nor dangles | 1 week |
| 2. Memory | [object layout, alignment and the cache](memory/layout.md) · [allocators and memory pools](memory/allocators.md) | design cache-friendly data structures, write a block allocator | 3-4 days |
| 3. Concurrency | [threads, locks and condition variables](concurrency/threads.md) · [atomics and memory order](concurrency/atomics.md) · [a lock-free queue and a thread pool](concurrency/lockfree-pool.md) | write concurrent components that pass TSan | 1 week |
| 4. Engineering and interop | [CMake, the sanitizers and perf](engineering/build-debug.md) · [pybind11 and PyTorch extensions](engineering/python-binding.md) · [reading an inference library](engineering/reading-code.md) · [frequent interview questions](engineering/interview.md) | build, debug and ship a C++ component on your own and wire it into Python | 3-4 days |

</div>

## Version conventions {#版本约定}

- **C++20** is the baseline (`-std=c++20`), and anything that needs C++23 is marked, like <span class="since">C++23</span>.
- Every program with a file name on it is compiled with **g++ 12** (`-Wall -Wextra -Werror`) and run under **AddressSanitizer + UndefinedBehaviorSanitizer** by default, with the concurrency chapters' programs run under **ThreadSanitizer**; the "output" on the page matches the real run line by line.
- A program that demonstrates a mistake on purpose says "this program has a bug", and what the page shows is an excerpt of the sanitizer's report.

## Before you start: setting up {#开始之前准备环境}

=== "Linux / WSL2"

    ```bash
    sudo apt install -y g++ cmake gdb linux-tools-generic   # g++ 12 or newer
    g++ --version
    ```

=== "macOS"

    ```bash
    xcode-select --install          # Apple clang, which supports ASan, UBSan and TSan
    brew install cmake
    ```

    Apple clang on macOS has no LeakSanitizer, so check for leaks with `leaks --atExit -- ./a.out`. `std::jthread` needs a recent libc++: `brew install llvm` and compiling with `$(brew --prefix llvm)/bin/clang++` plus `-fexperimental-library` is the suggestion (`tools/check_code.py` does this automatically). libc++ differs from Linux's libstdc++ in its implementation details, so output like `sizeof(std::string)`, capacities and hash table bucket counts will not match the page: the output on the page is Linux + GCC. The full Mac setup is in [the environment](root://setup/).

Defining a compiler alias of your own is worth it, so that the sanitizers become the default:

```bash
# add to ~/.bashrc or ~/.zshrc
alias cxx='g++ -std=c++20 -g -O1 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer'
alias cxxt='g++ -std=c++20 -g -O1 -Wall -Wextra -fsanitize=thread'
cxx hello.cpp -o hello && ./hello
```

## The companion exercises {#配套练习}

The exercise site has a set of C++ problems ([exercises · Advanced C++](root://practice/)), graded locally because there is no C++ compiler in the browser:

```bash
python practice/judge.py start cpp-raii-fd     # copy the template into practice/workspace/
python practice/judge.py test cpp-raii-fd      # compile with g++, run the tests under the sanitizers
```
