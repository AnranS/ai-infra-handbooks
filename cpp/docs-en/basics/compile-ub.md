# The compilation model and undefined behaviour

<p class="lead">To read a large C++ repository, you first have to know how the code is compiled and linked: why a function in a header has to be <code>inline</code>, why changing one header rebuilds half the repository. To write C++, you first have to know which constructs are "undefined behaviour": they do not necessarily crash on the spot, but they break suddenly after optimization, a different compiler or different data. This chapter makes both clear and turns the sanitizers into everyday tools.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What happens when an ordinary (non-`inline`) function defined in a header is included by two `.cpp` files?
    2. With `static int counter = 0;` in a header, how many `counter`s does the program have?
    3. Which overflows is undefined behaviour, `int` addition or `unsigned` addition?
    4. What is wrong with reading a `float`'s bit pattern through `reinterpret_cast<uint32_t*>(&f)`? How should it be written?
    5. After a `std::vector`'s `push_back`, is a reference to an element taken earlier still usable?

??? success "Answers (try it yourself first, then expand)"
    1. Each translation unit has its own definition of that function, and the link reports a "multiple definition" (a violation of the one-definition rule). It needs `inline`, or a declaration in the header and a definition in one `.cpp`.
    2. Every `.cpp` that includes the header has its own independent `counter` (`static` means internal linkage), unrelated to the others, which is usually not what was wanted. One shared by the whole program needs `inline int counter = 0;`.
    3. `int` (signed) overflow is undefined behaviour; `unsigned` wraps modulo $2^n$, which is well defined.
    4. It violates the strict aliasing rule: reading a `float` object through a `uint32_t*` is undefined behaviour, and the compiler may optimize on the assumption that "pointers of two types never point at the same memory". Use `std::bit_cast<uint32_t>(f)` (C++20) or `std::memcpy` into a `uint32_t` variable.
    5. Not necessarily: if the `push_back` caused a reallocation, the elements moved to new memory and every earlier reference, pointer and iterator dangles. `reserve` enough capacity beforehand, or use an index instead of a reference.

## From source to an executable {#从源码到可执行文件}

![Figure: from source to an executable - preprocessing, compiling, assembling, linking](../assets/figures/build-pipeline.svg){.aig-svg}

One `.cpp` file becomes a program in four steps:

1. **preprocessing**: expanding `#include` (pasting the header's text in verbatim), the macros and the conditional compilation;
2. **compiling**: compiling the whole preprocessed text, one **translation unit**, into assembly;
3. **assembling**: producing an object file (`.o`) holding machine code and a symbol table: "these are the functions and variables I define, and these are the ones I reference";
4. **linking**: putting all the `.o` files and libraries together and matching every "reference" to some "definition".

A few immediate consequences:

- every `.cpp` is **compiled independently** and the compiler cannot see inside another `.cpp`. For a function to be called from several `.cpp` files, put the **declaration** in a header and the **definition** in one `.cpp`;
- a header is compiled once for every `.cpp` that includes it. A template has to have its definition in the header (every translation unit that uses it needs the complete definition to instantiate it), which is why a template-heavy library compiles slowly: an attention kernel expanded over data type × head dimension × page size is routinely instantiated hundreds of times;
- the linker only knows symbol names. Two `.cpp` files defining an ordinary function of the same name give a "duplicate definition" at link time; a declaration with no definition gives an "undefined reference".

### What can go in a header: `inline` and the one-definition rule {#头文件里能放什么inline-与单一定义规则}

**The one-definition rule** (ODR): an ordinary function or global variable may have exactly one definition in the whole program. But a definition in a header appears in every translation unit that includes it, so C++ grants an exception: a function or variable marked `inline` (variables from C++17) may have one **identical** definition in each of several translation units, and the linker keeps one.

Member functions defined in a class, templates and `constexpr` functions are implicitly `inline`, so they go in a header safely.

`static` means something else entirely: it makes the name visible only within this translation unit (internal linkage), so **every `.cpp` that includes it has its own copy**. The example below puts the two side by side:

```cpp title="counter.hpp"
#pragma once

static int static_counter = 0;   // internal linkage: one per translation unit
inline int inline_counter = 0;   // a C++17 inline variable: one for the whole program
```

```cpp title="a.cpp" lib="yes"
#include "counter.hpp"

void bump_in_a() {
  ++static_counter;
  ++inline_counter;
}
```

```cpp title="b.cpp" lib="yes"
#include "counter.hpp"

void bump_in_b() {
  ++static_counter;
  ++inline_counter;
}
```

```cpp title="counter_main.cpp" with="a.cpp b.cpp"
// g++ -std=c++20 counter_main.cpp a.cpp b.cpp && ./a.out
#include <cstdio>

#include "counter.hpp"

void bump_in_a();
void bump_in_b();

int main() {
  bump_in_a();
  bump_in_a();
  bump_in_b();
  std::printf("main 看到的 static_counter = %d\n", static_counter);
  std::printf("inline_counter = %d\n", inline_counter);
}
```

```text title="output"
main 看到的 static_counter = 0
inline_counter = 3
```

`a.cpp`, `b.cpp` and `main` each change their own `static_counter`, so `main` still sees 0; there is one `inline_counter` in the whole program and it was incremented 3 times.
For a globally shared variable in a header (a global configuration, a log level), use `inline`; for something "private to each file", put it in an anonymous namespace in a `.cpp` rather than writing `static` in a header.

!!! warning "Violating the ODR usually raises nothing"
    If two `.cpp` files each have an `inline int block_size() { return 16; }` and an `inline int block_size() { return 32; }` (because two files expanded the same header with different macros, say),
    the linker raises nothing and simply **keeps one arbitrarily**, and the call in the other file quietly gets the wrong value. A fair number of large repositories' "only breaks under this combination of build options" bugs come from exactly this.
    The lesson: a header's content should not depend on what macros the including file defined beforehand.

### Name mangling and `extern "C"` {#名字修饰与-extern-c}

C++ has overloading, so functions of the same name need their parameter types in the symbol table to be distinguished, which is **name mangling**: `void launch(float*, int)` is `_Z6launchPfi` in the symbol table (`c++filt _Z6launchPfi` decodes it).
C has neither overloading nor mangling. To let C code, Python's `ctypes` or another language find a C++ function by name, turn mangling off with `extern "C"`:

```cpp
extern "C" int kv_store_put(const char* key, const void* data, size_t len);   // the symbol name is exactly kv_store_put
```

A cross-language boundary in an inference system (a C API, a plugin loaded by `dlopen`, Rust's FFI into C++) looks like this almost without exception.

## Undefined behaviour {#未定义行为}

The C++ standard says of certain operations: "if a program does this, the result is **undefined**" (undefined behaviour, UB). That is not "the result is unpredictable" but **the standard no longer guarantees anything about the whole program's behaviour**. The compiler may assume UB never happens and optimize accordingly, which is one reason C++ is fast and the reason it is dangerous.

A classic example:

```cpp
bool will_overflow(int x) { return x + 1 < x; }   // meant to check whether x + 1 overflows
```

Signed integer overflow is UB, so the compiler may take it that `x + 1` never overflows, which makes `x + 1 < x` always false. `g++ -O2` compiles the whole function into `return false;`.
Check for overflow **before** doing the arithmetic: `x == INT_MAX`, or `__builtin_add_overflow`.

The kinds of UB that come up most in inference systems code:

| Kind | Example | How to find it |
| --- | --- | --- |
| out-of-bounds access | an index past the end, a miscomputed `memcpy` length | ASan |
| dangling pointers and references | using an old reference after a container reallocated, returning a reference to a local, using after `delete` | ASan |
| signed integer overflow | a token count or byte count computed in `int` passing $2^{31}$ | UBSan |
| violating strict aliasing, unaligned access | `reinterpret_cast`ing a byte buffer and reading it as `float*` / `uint32_t*` | UBSan (partly) |
| reading uninitialized memory | using a local or a `new`ed array before initializing it | MSan (clang), Valgrind |
| data races | two threads reading and writing one variable without synchronization | TSan |

Below, the sanitizers catch each in turn.

### A reallocation leaves a reference dangling {#容器扩容让引用悬垂}

This is the easiest mistake to make when writing a scheduler or KV management code. **This program has a bug:**

```cpp title="dangling.cpp" expect="fail"
#include <cstdio>
#include <vector>

int main() {
  std::vector<int> blocks = {1, 2, 3};
  int& first = blocks[0];                       // the reference points into the vector's buffer
  for (int i = 0; i < 100; ++i) blocks.push_back(i);   // a reallocation: the buffer moves and the old one is freed
  std::printf("%d\n", first);                   // reading memory that has been freed
}
```

```text title="the ASan report (excerpt)"
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000010
READ of size 4 at 0x502000000010 thread T0
    #0 in main dangling.cpp:8
freed by thread T0 here:
    #0 in operator delete(void*, unsigned long)
    #1 in std::vector<int>::_M_realloc_insert ...
```

Without the sanitizers this program will most likely print a number "normally", sometimes 1 and sometimes garbage, depending on whether the freed memory was reused. That is exactly what makes UB frightening.
The fixes: keep the **index** rather than the reference; or `reserve` enough capacity beforehand; or use a container that does not move its elements (a `std::deque`'s element addresses stay put when inserting at either end).
A `std::unordered_map`'s rehash and a `std::string`'s growth have the same problem.

### Signed integer overflow {#有符号整数溢出}

**This program has a bug:**

```cpp title="overflow.cpp" expect="fail"
#include <climits>
#include <cstdio>

int main(int argc, char**) {
  int tokens = INT_MAX - 1 + argc;   // a value known only at run time: INT_MAX when argc is 1
  int next = tokens + 1;             // signed overflow: undefined behaviour
  std::printf("%d\n", next);
}
```

```text title="the UBSan report"
overflow.cpp:6:7: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
```

In an inference system a product like "token count × hidden dimension × bytes" passes $2^{31}$ easily: 128K tokens × 8192 dimensions × 2 bytes is exactly $2^{31}$. Use `size_t` or `int64_t` for byte counts and offsets without exception.
Unsigned overflow is **defined** (wrapping modulo $2^{n}$), but wrapping is usually a bug too: `a - b` on `size_t` becomes an enormous positive number when `a < b`.

### Strict aliasing and `std::bit_cast` {#严格别名与-stdbit_cast}

C++ says that an object may generally only be accessed through its own type (or `char` / `unsigned char` / `std::byte`). Reading a `float` through `reinterpret_cast<uint32_t*>(&x)` violates the **strict aliasing rule**, is UB, and may read a stale value after optimization.
To look at a floating-point number's bit pattern, for a bf16 conversion say, use C++20's `std::bit_cast` or `memcpy` (which the compiler turns into a single register move):

```cpp title="bits.cpp"
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>

// bf16 is a float's top 16 bits (truncated here; the correct way rounds to nearest even)
std::uint16_t to_bf16_truncate(float x) { return std::bit_cast<std::uint32_t>(x) >> 16; }

int main() {
  float x = 1.0f;
  std::printf("1.0f 的位模式：0x%08x\n", std::bit_cast<std::uint32_t>(x));
  std::uint32_t u;
  std::memcpy(&u, &x, sizeof u);   // the pre-C++20 way, equally correct
  std::printf("memcpy 得到：0x%08x\n", u);
  std::printf("3.14159f 截断成 bf16：0x%04x\n", to_bf16_truncate(3.14159f));
}
```

```text title="output"
1.0f 的位模式：0x3f800000
memcpy 得到：0x3f800000
3.14159f 截断成 bf16：0x4049
```

### Unaligned access {#未对齐的访问}

What comes out of a socket, a file or shared memory is a byte buffer whose fields are not necessarily aligned to their types. **This program has a bug:**

```cpp title="misaligned.cpp" expect="fail"
#include <cstdint>
#include <cstdio>

int main() {
  alignas(4) unsigned char buf[8] = {0, 1, 0, 0, 0, 2, 0, 0};
  // reading a uint32 at offset 1: the address is not 4-byte aligned, and this violates strict aliasing too
  auto* p = reinterpret_cast<const std::uint32_t*>(buf + 1);
  std::printf("%u\n", *p);
}
```

```text title="the UBSan report"
misaligned.cpp:8:15: runtime error: load of misaligned address 0x7ffd... for type 'const uint32_t', which requires 4 byte alignment
```

An unaligned read usually "happens to work" on x86, may fail outright on ARM or a GPU, and may fail when the compiler generates vector instructions. The correct way is `memcpy` again:

```cpp title="aligned_read.cpp"
#include <cstdint>
#include <cstdio>
#include <cstring>

std::uint32_t read_u32(const unsigned char* p) {
  std::uint32_t v;
  std::memcpy(&v, p, sizeof v);   // correct at any address, and the compiler generates an ordinary load
  return v;
}

int main() {
  unsigned char buf[8] = {0, 1, 0, 0, 0, 2, 0, 0};
  std::printf("%u %u\n", read_u32(buf + 1), read_u32(buf + 4));
}
```

```text title="output"
1 512
```

(x86 is little-endian: `buf[1..4]` is `01 00 00 00`, which reads as 1; `buf[4..7]` is `00 02 00 00`, which is $2 \times 256 = 512$.)

## Making the sanitizers the default {#把-sanitizer-变成默认}

| Tool | Compile flag | What it catches | Cost |
| --- | --- | --- | --- |
| AddressSanitizer | `-fsanitize=address` | out-of-bounds, use after free, double free, leaks | about 2x slower, 2-3x the memory |
| UndefinedBehaviorSanitizer | `-fsanitize=undefined` | signed overflow, misalignment, null pointers, invalid shifts and more | very small |
| ThreadSanitizer | `-fsanitize=thread` | data races, lock order inversions | 5-15x slower, cannot be combined with ASan |
| libstdc++ assertions | `-D_GLIBCXX_ASSERTIONS` | an out-of-range index into a standard container, `front()` on an empty one | very small, worth leaving on in test builds |

A few rules of thumb:

- have development and test builds carry `-fsanitize=address,undefined -g -fno-omit-frame-pointer` by default, with `-g` so the report has line numbers;
- UBSan only prints by default and does not exit, so tests should add `UBSAN_OPTIONS=halt_on_error=1` (or `-fno-sanitize-recover=all` at compile time) to turn it into a failure;
- run concurrent code once in a separate TSan build (see [threads, locks and condition variables](../concurrency/threads.md));
- the counterpart for GPU code is `compute-sanitizer` (`memcheck`, `racecheck`, `initcheck`), with exactly the same reasoning; see the CUDA handbook's [profiling](cuda://tools/profiling/).

An example of the libstdc++ assertions. **This program has a bug** (the index equals the length):

```cpp title="assertions.cpp" expect="fail" sanitize="none" flags="-D_GLIBCXX_ASSERTIONS"
#include <vector>

int main() {
  std::vector<float> logits(128);
  int token = 128;                 // the vocabulary is 128, so the valid indices are 0 to 127
  return logits[token] > 0;
}
```

```text title="what it prints"
/usr/include/c++/12/bits/stl_vector.h:1123: ... Assertion '__n < this->size()' failed.
Aborted
```

!!! interview "Answering in an interview"
    On the compilation model: every `.cpp` is one translation unit, a header is just text that gets pasted in, and the linker joins them by symbol name; an ordinary function defined in a header and included by two `.cpp` files is a duplicate definition, and it has to be `inline` (one per program) or a template; a `static` variable in a header gets one copy per translation unit. On undefined behaviour, the point to make is that "the compiler assumes it does not happen and optimizes accordingly", so the error can surface far from the bug; the common ones are out-of-bounds access, a dangling reference after a `vector` reallocates, signed overflow (unsigned overflow is defined wrapping), violating strict aliasing (read a bit pattern with `std::bit_cast` or `memcpy`) and data races. Development builds turn ASan + UBSan on by default, and concurrent code goes through TSan.

## Exercises {#练习}

1. Which of these are undefined behaviour?
    1. `uint32_t a = 0; a - 1;`
    2. `int32_t n = 1 << 31;`
    3. `std::vector<int> v; v.reserve(10); v[3] = 1;`
    4. calling `push_back` on `requests` inside a `for (auto& r : requests)` loop
    5. `float f; std::memcpy(&f, bytes, 4);` (`bytes` being an `unsigned char*` at an arbitrary address)

??? success "Answer"
    1. No. Unsigned arithmetic is modulo $2^{32}$ and the result is 4294967295 (though this is often a bug).
    2. Yes (before C++20). `1 << 31` exceeds what an `int` can represent; from C++20 a signed left shift's result is defined by two's complement and this is no longer UB, but `1u << 31` or `INT32_MIN` says it more clearly.
    3. Yes. `reserve` only allocates capacity and does not change `size()`, so `v[3]` touches an element that does not exist (ASan cannot catch it because the memory really is allocated; `-D_GLIBCXX_ASSERTIONS` can). Use `resize`.
    4. Yes. `push_back` may reallocate, and the iterator the range-for holds is invalidated with it.
    5. No. `memcpy` is legal for any address and any type, which is exactly why it is the recommended way.

2. The scheduler code below means to add arriving requests to a queue and count the steps of "the request currently being processed", but it has a bug. Find it and fix it so that the fixed version runs under ASan and prints `processed=3 queue=6`.

    ```cpp
    #include <cstdio>
    #include <vector>

    struct Request { int id; int steps = 0; };

    int main() {
      std::vector<Request> queue = {{1}, {2}, {3}};
      Request& current = queue[0];
      int processed = 0;
      for (int i = 0; i < 3; ++i) {
        queue.push_back({10 + i});      // a new request arrives
        current.steps++;                // count a step for the current request
        processed++;
      }
      std::printf("processed=%d queue=%zu steps=%d\n", processed, queue.size(), current.steps);
    }
    ```

??? success "Answer"
    `current` is a reference to an element inside `queue`, and the `push_back`'s reallocation leaves it dangling. Keep the index and take the element each time it is used:

    ```cpp title="scheduler_fixed.cpp"
    #include <cstdio>
    #include <vector>

    struct Request {
      int id;
      int steps = 0;
    };

    int main() {
      std::vector<Request> queue = {{1}, {2}, {3}};
      std::size_t current = 0;         // keep the index, not the reference
      int processed = 0;
      for (int i = 0; i < 3; ++i) {
        queue.push_back({10 + i});
        queue[current].steps++;
        processed++;
      }
      std::printf("processed=%d queue=%zu\n", processed, queue.size());
    }
    ```

    ```text title="output"
    processed=3 queue=6
    ```

    The other fix is `queue.reserve(...)` before the loop, but it rests on the unstated premise that the reserved capacity is never exceeded, and breaks again as soon as someone changes the loop count; keeping the index is sounder.

## Summary {#小结}

- [x] Every `.cpp` is compiled independently into a translation unit and the linker joins them by symbol name; a header is just text that gets pasted in.
- [x] A function or variable defined in a header has to be `inline` (one per program), a template or a definition inside a class; do not write `static` variables in a header.
- [x] Undefined behaviour is not "a random result": the compiler assumes it does not happen and optimizes accordingly, so the error can surface far from the real bug.
- [x] The common UB: out-of-bounds access, a dangling reference (a container reallocating), signed overflow, violating strict aliasing and alignment, reading uninitialized memory, data races. Read a bit pattern with `std::bit_cast` or `memcpy`.
- [x] Development builds turn ASan + UBSan on by default and concurrent code goes through TSan separately; they are basic tools for writing C++, not a debugger reached for after something goes wrong.
