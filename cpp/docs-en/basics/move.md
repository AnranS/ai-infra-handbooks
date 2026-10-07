# Move semantics and perfect forwarding

<p class="lead">Value semantics make code easy to reason about, but a copy can be expensive, and an exclusively owned resource (device memory, a file, a communication handle) cannot be copied at all. What move semantics solve is "handing something over without duplicating it": transferring ownership by moving a few pointers. This chapter explains rvalue references, what <code>std::move</code> actually does, why a move constructor has to be <code>noexcept</code>, and perfect forwarding in a template.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Does `std::move(x)` move anything?
    2. What can and cannot be done with an object after it has been moved from?
    3. Why does `std::vector` fall back to copying when it reallocates and the element type's move constructor is not `noexcept`?
    4. What is wrong with `return std::move(local);`?
    5. How does the `T&&` in `template <class T> void f(T&& x)` differ from the `&&` in `void g(std::string&& x)`? Why write `std::forward<T>(x)`?

??? success "Answers (try it yourself first, then expand)"
    1. No: it merely converts the argument to an rvalue reference type, and what actually transfers the resource is the move constructor or move assignment operator that gets called (if there is one).
    2. Only destroy it, or assign to it again; a standard library type is in a "valid but unspecified" state, so do not rely on its value (unless the type says otherwise, as a `unique_ptr` is empty after a move).
    3. A reallocation has to move the old elements to the new memory, and if a move throws halfway the elements already moved cannot be restored, which breaks `push_back`'s strong exception guarantee; copying does not damage the originals. So it moves only when the move constructor is `noexcept` (or the type cannot be copied).
    4. It prevents the return value optimization (NRVO): the object could have been constructed directly where the caller wants it with no move at all, and writing `std::move` forces one; besides, the compiler already treats a returned local as an rvalue.
    5. The `T&&` in a template is a forwarding reference: for an lvalue argument `T` is deduced as an lvalue reference and collapses to one, and only for an rvalue is it an rvalue reference. `std::string&&` binds only to rvalues. A named parameter is itself an lvalue, and `std::forward<T>(x)` restores its original value category from `T` and passes it on unchanged.

## Lvalues, rvalues and rvalue references {#左值右值与右值引用}

![Figure: a copy allocates a new buffer and duplicates it; a move only takes the pointer over](../assets/figures/move-vs-copy.svg){.aig-svg}

Roughly:

- an **lvalue** is something with a name and an address: a variable, `v[i]`, `*p`;
- an **rvalue** is a temporary about to disappear: `f()`'s return value, `std::string("tmp")`, `a + b`.

An rvalue is about to be destroyed and nobody will use the resource it holds again, so that resource can be "stolen" rather than duplicated. An **rvalue reference** `T&&` is a reference that binds only to rvalues, and overloading a constructor on `T&&` takes the stealing path whenever the source is a temporary:

```cpp
std::vector<int> a = make_tokens();   // make_tokens() is an rvalue: a move (elided in practice)
std::vector<int> b = a;               // a is an lvalue: a copy, since a is still needed
std::vector<int> c = std::move(a);    // saying outright "I am done with a": a move
```

**`std::move` itself moves nothing**: it is a cast, turning an lvalue into an rvalue reference (`static_cast<T&&>(x)`) to tell overload resolution "this one may be stolen". What does the work is the **move constructor** or **move assignment operator** that gets selected.

## Giving a resource class a move {#给资源类写上移动}

Continuing the previous chapter's `DeviceBuffer`, which forbids copying: now add a move. The move constructor takes the source's pointer and nulls the source, so its destructor does nothing:

```cpp title="move_buffer.cpp"
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static int live = 0;

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : ptr_(std::malloc(bytes)), bytes_(bytes) { ++live; }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept
      : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      reset();                                 // release what we already hold first
      ptr_ = std::exchange(o.ptr_, nullptr);
      bytes_ = std::exchange(o.bytes_, 0);
    }
    return *this;
  }

  std::size_t size() const { return bytes_; }

 private:
  void reset() {
    if (ptr_) {
      std::free(ptr_);
      --live;
      ptr_ = nullptr;
    }
  }
  void* ptr_;
  std::size_t bytes_;
};

DeviceBuffer allocate_kv(std::size_t blocks) { return DeviceBuffer(blocks * 4096); }

int main() {
  DeviceBuffer a = allocate_kv(16);   // the return value is constructed straight into a, neither copied nor moved
  DeviceBuffer b = std::move(a);      // a move: b takes the memory over and a becomes an empty shell
  std::printf("a.size=%zu b.size=%zu live=%d\n", a.size(), b.size(), live);

  std::vector<DeviceBuffer> pool;
  pool.push_back(std::move(b));       // moved into the vector
  pool.emplace_back(8192);            // constructed directly in the vector's memory
  std::printf("pool=%zu live=%d\n", pool.size(), live);

  a = DeviceBuffer(100);              // move assignment: a holds a resource again
  std::printf("a.size=%zu live=%d\n", a.size(), live);
}
```

```text title="output"
a.size=0 b.size=65536 live=1
pool=2 live=2
a.size=100 live=3
```

A few points:

- `std::exchange(x, v)` sets `x` to `v` and returns the old value, which is very handy when writing a move;
- the destructor has to cope with an object "already moved from" (the null check in `reset()`);
- move assignment has to release what it already holds first, and handle self-assignment like `a = std::move(a)`;
- a move only moves pointers and cannot fail, so mark it `noexcept`; the next section says why that matters.

### An object after it has been moved from {#被移动之后的对象}

What the standard library promises about "an object moved from" is: **it is in a valid but unspecified state**. Operations that do not depend on its value are fine, destroying it, assigning to it, `clear()`, but do not read its contents. A class of your own is best left in a definite "empty" state after a move (like `size() == 0` above).

## `noexcept` and a `vector`'s reallocation {#noexcept-与-vector-扩容}

Move the dials first: how many push_backs, how many reallocations, how many elements moved, and the difference `noexcept` on the move constructor makes:

<div class="aig-widget" data-widget="vector-realloc"></div>

A `std::vector` reallocating has to move the old buffer's elements into the new one, and it makes a strong promise: **if something throws during the move, the vector is left as it was** (the strong exception guarantee).
Moving by copy, a failure leaves the old buffer intact and can roll back; moving by move, a failure halfway leaves some elements already emptied with no way back. So the vector's rule is: **move only when the move constructor promises not to throw (`noexcept`), and otherwise fall back to copying**.

```cpp title="noexcept_realloc.cpp"
#include <cstdio>
#include <vector>

struct Counts {
  int copies = 0, moves = 0;
};

template <bool Noexcept>
struct Item {
  static inline Counts c;
  Item() = default;
  Item(const Item&) { ++c.copies; }
  Item(Item&&) noexcept(Noexcept) { ++c.moves; }
};

template <bool N>
void run(const char* name) {
  std::vector<Item<N>> v;
  for (int i = 0; i < 5; ++i) v.push_back(Item<N>{});   // capacity 1 -> 2 -> 4 -> 8, three reallocations
  std::printf("%s：copies=%d moves=%d\n", name, Item<N>::c.copies, Item<N>::c.moves);
}

int main() {
  run<false>("移动构造没有 noexcept");
  run<true>("移动构造标了 noexcept");
}
```

```text title="output"
移动构造没有 noexcept：copies=7 moves=5
移动构造标了 noexcept：copies=0 moves=12
```

The 5 `push_back`s each move once; the 3 reallocations moved $1 + 2 + 4 = 7$ elements in all, and without `noexcept` all 7 are copies. With elements holding a few MB each, the difference is tens of MB of memory traffic.
**The rule: mark a move constructor and move assignment `noexcept` whenever they cannot fail.** A compiler-generated move deduces `noexcept` on its own (it is `noexcept` when every member's move is).

## Taking by value and moving {#按值传参再移动}

For a function that **keeps** a copy of its argument (a constructor storing a parameter in a member being the commonest case), the simplest and most efficient form takes it by value and moves it in:

```cpp
class Request {
 public:
  Request(std::string prompt, std::vector<int> tokens)
      : prompt_(std::move(prompt)), tokens_(std::move(tokens)) {}
 private:
  std::string prompt_;
  std::vector<int> tokens_;
};

Request r1(prompt, tokens);                          // the caller still needs them: one copy each
Request r2(std::move(prompt), tokenize(text));       // the caller is done with them: moves all the way
```

One constructor then covers both "the caller passes an lvalue" and "passes an rvalue", with no need for separate `const T&` and `T&&` overloads.

!!! warning "Do not write `return std::move(local);`"
    When returning a local, the compiler first tries to **elide the copy** (NRVO, constructing the local directly in the return value's place), and where it cannot, it treats the local as an rvalue and moves it anyway.
    Writing `std::move` by hand **prevents the elision**, and `g++ -Wall` warns with `-Wpessimizing-move`. Just `return local;`.

## Perfect forwarding {#完美转发}

A general wrapper function, an object pool's `acquire(args...)`, `vector::emplace_back`, `std::make_unique`, has to pass its arguments to another function **unchanged**: an lvalue from the caller goes on as an lvalue, and an rvalue as an rvalue.

A `T&&` among a template's parameters is a special case called a **forwarding reference**: for an lvalue, `T` is deduced as `U&` and `T&&` collapses to `U&`; for an rvalue, `T` is `U` and `T&&` is `U&&`.
But inside the function body the parameter `x` has a name, so it **is always an lvalue**. To restore the caller's original value category, use `std::forward<T>(x)`:

```cpp title="forwarding.cpp"
#include <cstdio>
#include <string>
#include <utility>

void consume(const std::string&) { std::printf("  拷贝进来（左值）\n"); }
void consume(std::string&&) { std::printf("  移动进来（右值）\n"); }

template <class T>
void wrong(T&& x) { consume(x); }                    // x has a name and is an lvalue: always a copy

template <class T>
void right(T&& x) { consume(std::forward<T>(x)); }   // keep the caller's value category

int main() {
  std::string s = "prompt";
  std::printf("wrong(s)：\n");
  wrong(s);
  std::printf("wrong(临时对象)：\n");
  wrong(std::string("tmp"));
  std::printf("right(s)：\n");
  right(s);
  std::printf("right(临时对象)：\n");
  right(std::string("tmp"));
}
```

```text title="output"
wrong(s)：
  拷贝进来（左值）
wrong(临时对象)：
  拷贝进来（左值）
right(s)：
  拷贝进来（左值）
right(临时对象)：
  移动进来（右值）
```

The variadic form (which is how `emplace_back` is implemented):

```cpp
template <class T, class... Args>
T* construct_at_slot(void* slot, Args&&... args) {
  return new (slot) T(std::forward<Args>(args)...);   // construct in place in existing memory, forwarding the arguments unchanged
}
```

`emplace_back(8192)` saves one move over `push_back(DeviceBuffer(8192))`: it forwards the argument to the constructor and builds the element directly in the vector's memory.

!!! interview "Answering in an interview"
    On move semantics: `std::move` is only a cast to an rvalue reference, and what transfers the resource is the move constructor and move assignment (taking the pointer over and nulling the source); an object moved from is "valid but unspecified" and can only be destroyed or assigned to. When a `vector` reallocates and the element's move constructor is not `noexcept`, it falls back to copying to keep the strong exception guarantee (`move_if_noexcept`), so a move should be `noexcept`; `return std::move(local)` gets in the way of the return value optimization, so just `return local`. A `T&&` in a template is a forwarding reference that binds both lvalues and rvalues, and with `std::forward<T>` it keeps the argument's original value category, which is how `emplace_back` builds an element in place.

## Exercises {#练习}

1. The `BlockTable` below holds one request's KV block numbers in a raw array. Complete it by the rule of five: the copy has to be deep, the move has to transfer ownership and be `noexcept`, and assignment has to handle self-assignment correctly. It has to run under ASan and print what the answer prints.

    ```cpp
    class BlockTable {
     public:
      explicit BlockTable(std::size_t n);
      ~BlockTable();
      // to complete: the copy constructor, copy assignment, the move constructor, move assignment
      int& operator[](std::size_t i);
      std::size_t size() const;
     private:
      std::size_t n_;
      int* ids_;
    };
    ```

??? success "Answer"
    Write copy assignment as "copy and swap": copy `o` into a temporary first (which may throw, but `*this` is untouched at that point), then exchange with a `swap` that cannot fail, which handles self-assignment naturally and gives the strong exception guarantee.

    ```cpp title="block_table.cpp"
    #include <algorithm>
    #include <cstdio>
    #include <utility>
    #include <vector>

    class BlockTable {
     public:
      explicit BlockTable(std::size_t n) : n_(n), ids_(new int[n]) { std::fill(ids_, ids_ + n_, -1); }
      ~BlockTable() { delete[] ids_; }

      BlockTable(const BlockTable& o) : n_(o.n_), ids_(new int[o.n_]) { std::copy(o.ids_, o.ids_ + n_, ids_); }
      BlockTable& operator=(const BlockTable& o) {
        BlockTable tmp(o);   // may throw, but *this is untouched
        swap(tmp);           // cannot fail
        return *this;        // tmp is destroyed carrying the old data
      }
      BlockTable(BlockTable&& o) noexcept : n_(std::exchange(o.n_, 0)), ids_(std::exchange(o.ids_, nullptr)) {}
      BlockTable& operator=(BlockTable&& o) noexcept {
        if (this != &o) {
          delete[] ids_;
          n_ = std::exchange(o.n_, 0);
          ids_ = std::exchange(o.ids_, nullptr);
        }
        return *this;
      }
      void swap(BlockTable& o) noexcept {
        std::swap(n_, o.n_);
        std::swap(ids_, o.ids_);
      }

      int& operator[](std::size_t i) { return ids_[i]; }
      std::size_t size() const { return n_; }

     private:
      std::size_t n_;
      int* ids_;
    };

    int main() {
      BlockTable a(4);
      a[0] = 7;
      BlockTable b = a;                // a deep copy
      b[0] = 9;
      std::printf("拷贝：a[0]=%d b[0]=%d\n", a[0], b[0]);

      BlockTable& same = b;
      b = same;                        // self-assignment
      std::printf("自我赋值后：b[0]=%d\n", b[0]);

      BlockTable c = std::move(a);     // a move
      std::printf("移动：a.size=%zu c.size=%zu c[0]=%d\n", a.size(), c.size(), c[0]);

      std::vector<BlockTable> tables;
      for (int i = 0; i < 5; ++i) tables.emplace_back(2);   // the reallocation takes the noexcept move
      std::printf("tables=%zu\n", tables.size());
    }
    ```

    ```text title="output"
    拷贝：a[0]=7 b[0]=9
    自我赋值后：b[0]=9
    移动：a.size=0 c.size=4 c[0]=7
    tables=5
    ```

    In real code, of course, a `std::vector<int>` member (the rule of zero) is enough. The value of writing the rule of five by hand is understanding what the standard containers and smart pointers do inside.

2. The code below means to save a copy when adding a request to a batch, but the token count always comes out as 0. Why?

    ```cpp
    batch.tokens.insert(batch.tokens.end(),
                        std::make_move_iterator(req.tokens.begin()), std::make_move_iterator(req.tokens.end()));
    batch.requests.push_back(std::move(req));
    stats.total_tokens += req.tokens.size();
    ```

??? success "Answer"
    `req` was already moved into `batch.requests` on the second line, so the third reads an object that has been moved from, where `req.tokens` is usually empty already (the standard only promises "valid but unspecified").
    After a move only destruction and reassignment are allowed, not reading. Move the counting before the move, or read from `batch.requests.back()` instead. Besides, "moving" a fundamental type like `int` is the same as copying, so the `make_move_iterator` on the first line means nothing.

## Summary {#小结}

- [x] `std::move` is only a cast, and what transfers the resource is the move constructor and the move assignment operator.
- [x] A resource class's move: take the pointer over, null the source, and check for null in the destructor; move assignment releases what it holds first.
- [x] An object moved from is in a "valid but unspecified" state and can only be destroyed or assigned to.
- [x] Mark a move `noexcept`, or a `vector` falls back to copying when it reallocates.
- [x] To keep an argument, take it by value and move it; do not write `std::move` when returning a local.
- [x] A `T&&` in a template is a forwarding reference, and with `std::forward<T>` it passes the argument on unchanged; `emplace_back` builds the element directly in the container.
