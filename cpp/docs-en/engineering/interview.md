# Frequent C++ interview questions

<p class="lead">In an AI infrastructure interview, C++ usually appears in two forms: a chain of follow-up questions on the fundamentals (the object model, memory, concurrency), and writing a component live (a smart pointer, a thread-safe queue, a memory pool). Here are the frequent questions and the points to make, grouped by topic, each linked to where the body of the handbook covers it; the ones marked ★ come up most.</p>

!!! tip "How to use this"
    - read only the question first, answer out loud for 2-3 minutes, then expand the points and compare;
    - the shape of an answer: **what it is → why it is needed → how it is implemented or used → the costs and traps**. Being able to say "where an inference system uses this" is a bonus;
    - the hands-on problems are at the end, and most have a locally graded version in the [exercises](root://practice/).

## 1. The object model and language fundamentals {#一对象模型与语言基础}

**1. ★ What is the difference between a pointer and a reference?**

??? success "The points"
    A reference has to be initialized, cannot be rebound and has no "null reference"; semantically it is an alias for an object. A pointer is an object holding an address, which may be null, may be repointed and supports arithmetic.
    In modern C++: a parameter that must exist takes a reference and one that "may be absent" takes a pointer or a `std::optional`; a raw pointer and a reference both mean **not owning**, and owning means a smart pointer. See [smart pointers and ownership](../basics/ownership.md#三种所有权关系).

**2. ★ How are virtual functions implemented? Why must a base class's destructor be virtual?**

??? success "The points"
    Every class with virtual functions has a virtual function table (vtable) and every object holds a pointer to it (the vptr); a virtual call jumps indirectly through the vptr. The costs: one more pointer per object, one more indirect jump per call, and usually no inlining.
    When a derived object is `delete`d through a base pointer and the destructor is not virtual, only the base's destructor runs and the derived class's members are not destroyed (undefined behaviour, showing up in practice as a resource leak).
    Calling a virtual function in a constructor or destructor calls **the version of the layer being constructed or destroyed** and does not dispatch to the derived class.
    Polymorphism on a hot path goes to templates and coarse-grained polymorphism to virtual functions, see [a template or a virtual function](../basics/templates.md#模板还是虚函数).

**3. ★ What is RAII? What are the levels of exception safety?**

??? success "The points"
    Acquire the resource on construction and release it on destruction, with the destructor running on a normal return, an early return and an exception alike, so the resource cannot leak.
    The three levels of exception safety: **the basic guarantee** (the object is still valid and nothing leaks after an exception), **the strong guarantee** (the state returns to what it was before the call, like a transaction, for which "copy and swap" is the typical form) and **the nothrow guarantee** (`noexcept`, which destructors, moves and `swap` should meet). See [value semantics and RAII](../basics/value-raii.md).

**4. ★ Explain the rule of three / five / zero.**

??? success "The points"
    A class that manages a resource directly and needs a custom destructor almost certainly also needs to define or delete the copy constructor and copy assignment (the rule of three), plus the move constructor and move assignment from C++11 (the rule of five); otherwise the default member-by-member copy has two objects releasing the same resource.
    The rule of zero: do not manage the resource in the class, leave it to RAII members like `vector` and `unique_ptr`, and write none of the special member functions. See [the rule of zero and the rule of five](../basics/value-raii.md#零法则与五法则).

**5. ★ What does `std::move` do? What state is a moved-from object in?**

??? success "The points"
    `std::move` only casts an lvalue to an rvalue reference and moves nothing itself; what transfers the resource is the move constructor or move assignment that gets selected. A moved-from object is "valid but unspecified" and can only be destroyed or assigned to.
    The follow-up, why must a move constructor be `noexcept`: a `vector` reallocating moves only when the move cannot throw, and otherwise falls back to copying to keep the strong exception guarantee. See [move semantics](../basics/move.md).

**6. What are lvalues, rvalues and forwarding references? What problem does `std::forward` solve?**

??? success "The points"
    Something with a name and an address is an lvalue; a temporary is an rvalue. A `T&&` among a template's parameters is a forwarding reference, deduced as an lvalue reference for an lvalue (reference collapsing). A named parameter inside the function body is always an lvalue, and `std::forward<T>(x)` restores the caller's original value category, which is perfect forwarding and how `emplace_back` and `make_unique` are implemented. See [perfect forwarding](../basics/move.md#完美转发).

**7. What are the uses of `const`? And `mutable`?**

??? success "The points"
    `const T*` (a pointer to a constant), `T* const` (a constant pointer), a `const` member function (which does not change the object's observable state, with `this` a pointer to a constant), a `const T&` parameter (read only and able to bind a temporary).
    A `mutable` member can be modified in a `const` member function, which is for caches, mutexes and other members that do not affect the observable state (a `const` getter that has to take a lock makes it a `mutable std::mutex`).

**8. What does `static` mean in its various places? Is a local static variable's initialization thread-safe?**

??? success "The points"
    At global / namespace scope: internal linkage (visible only in this translation unit); as a class member: belonging to the class rather than an object; as a local in a function: static storage duration, initialized the first time control reaches it.
    From C++11 a local static's initialization is thread-safe ("magic statics"), which is the usual way to implement a thread-safe singleton.
    A `static` variable in a header gets one copy per translation unit, see [`inline` and the one-definition rule](../basics/compile-ub.md#头文件里能放什么inline-与单一定义规则).

**9. How do `new` / `delete` differ from `malloc` / `free`? What is placement new?**

??? success "The points"
    `new` allocates and calls the constructor, throws `std::bad_alloc` on failure and is type-safe; `malloc` only allocates raw memory and returns null on failure. The two cannot be mixed.
    Placement new (`new (ptr) T(args)`) constructs an object in memory that already exists, which memory pools, `vector`'s internals and arenas all use; correspondingly the destructor has to be called by hand as `p->~T()`. See [allocators and memory pools](../memory/allocators.md).

## 2. Memory and undefined behaviour {#二内存与未定义行为}

**10. ★ Name several common kinds of undefined behaviour and how to find them.**

??? success "The points"
    Out-of-bounds access, use after free, double free, using an old reference or iterator after a container reallocated, signed integer overflow, violating strict aliasing (reading another type through a `reinterpret_cast`), unaligned access, reading uninitialized memory, data races.
    What makes UB dangerous is that the compiler assumes it does not happen and optimizes accordingly. The ways to find it: ASan (memory errors), UBSan (overflow, alignment and so on), TSan (data races), `-D_GLIBCXX_ASSERTIONS`, `-Wall -Wextra`. See [the compilation model and undefined behaviour](../basics/compile-ub.md).

**11. ★ How is a struct's size computed? Why is there padding?**

??? success "The points"
    Every member goes at an address that is a multiple of its own alignment, with padding between members; the struct's alignment is the largest of its members' and the total size is rounded up to a multiple of it. Ordering the fields by alignment, largest first, cuts the padding.
    The follow-up, why alignment is needed: an unaligned access fails or runs slower on some platforms and cannot use vector instructions; a GPU's 128-bit load requires 16-byte alignment. See [object layout](../memory/layout.md#sizeofalignof-与填充).

**12. ★ How do `unique_ptr` and `shared_ptr` differ? Is `shared_ptr` thread-safe?**

??? success "The points"
    A `unique_ptr` owns exclusively, can only be moved and costs nothing with the default deleter; a `shared_ptr` shares, with a control block holding the reference count.
    A `shared_ptr`'s **reference count** is atomic, so several threads copying and destroying their own copies is safe; but several threads reading and writing **the same `shared_ptr` variable** is not, and the object it points at is not made thread-safe either.
    `make_shared` puts the object and the control block in one allocation, whose drawback is that the whole block (including the object's storage) cannot be freed while any `weak_ptr` remains. Break a reference cycle with a `weak_ptr`. See [smart pointers and ownership](../basics/ownership.md).

**13. How do you locate a memory leak?**

??? success "The points"
    During development: ASan's own LeakSanitizer, which reports the leaking allocation sites when the process exits; `leaks` on macOS. In production: heaptrack, or jemalloc's / tcmalloc's heap profiling, comparing two heap snapshots.
    In a C++ service, "the memory keeps growing" is more often not a leak but an unbounded cache, an allocator that does not return memory to the system (fragmentation), or a `shared_ptr` reference cycle.

## 3. Templates and the standard library {#三模板与标准库}

**14. Why is a template's implementation usually in a header? How do templates affect compilation time?**

??? success "The points"
    A template is instantiated where it is used and the compiler has to see the complete definition; every translation unit that uses it instantiates its own copy (deduplicated at link time). A combinatorial explosion grows both the compilation time and the binary, and the ways to control it are instantiating only the combinations needed, explicit instantiation, and JIT on demand. See [a template is a code generator](../basics/templates.md#模板就是代码生成器).

**15. What are SFINAE and concepts?**

??? success "The points"
    SFINAE: a substitution failure in a template argument is not an error but merely removes that candidate from the overload set, and older code uses `std::enable_if` to "enable this only for certain types". C++20's concepts state the requirements on template parameters directly through `requires`, with clearer errors and more readable code. See [concepts](../basics/templates.md#concepts给模板参数写接口).

**16. ★ How does a `vector` grow? Which operations invalidate iterators?**

??? success "The points"
    When the capacity runs out it allocates a new buffer by a factor (2x in libstdc++) and moves the elements (when the move constructor is `noexcept`) or copies them, making `push_back` amortized O(1). A reallocation invalidates **every** pointer, reference and iterator; `insert` / `erase` invalidate everything after the point of insertion or removal. `reserve` avoids a reallocation and `clear` does not release the capacity. See [vector](../basics/stl-perf.md#vector默认的容器).

**17. How is `unordered_map` implemented? How do you choose between it and `map`?**

??? success "The points"
    A `std::unordered_map` is a chained hash table with one node and one allocation per element; it rehashes when the load factor passes the threshold, invalidating iterators while element addresses stay put. A `map` is a red-black tree, ordered, with O(log n) lookup.
    Use a `map` when order or range queries are needed; use an `unordered_map` for lookup by key alone, and a flat open-addressing hash table (Abseil's `flat_hash_map`) when performance matters. See [associative containers](../basics/stl-perf.md#关联容器unordered_map-与-map).

**18. What is risky about `std::string_view`?**

??? success "The points"
    It does not own the data and is only a pointer plus a length. Bound to a temporary `std::string` it dangles immediately; stored in a member or a container, the referenced data has to be guaranteed to outlive it. It suits a function parameter and not storage. See [views](../basics/stl-perf.md#视图string_view-与-span).

## 4. Concurrency {#四并发}

**19. ★ What is the difference between a data race and a race condition?**

??? success "The points"
    A data race: two threads access the same memory location without synchronization and at least one writes, which is undefined behaviour and TSan detects. A race condition: the result depends on the timing, as when the state changes between "check" and "act", which can happen without a data race and can only be handled by design (putting the check and the act in the same critical section). See [threads and data races](../concurrency/threads.md#线程与数据竞争).

**20. ★ How is a condition variable used? Why does `wait` take a predicate?**

??? success "The points"
    Hold a `unique_lock`, check the condition, and if it does not hold, `cv.wait(lock, pred)`: atomically releasing the lock and sleeping, then retaking the lock and **checking the predicate again** on waking. The predicate is needed because of spurious wakeups and because another thread may have consumed the condition by the time it wakes. Whoever changes the data calls `notify_one` / `notify_all` afterwards; closing a queue calls `notify_all` so every waiter leaves. See [condition variables and a blocking queue](../concurrency/threads.md#条件变量与阻塞队列).

**21. ★ Explain memory order. When can `memory_order_relaxed` be used?**

??? success "The points"
    Atomicity only guarantees that a single variable is not torn, and the memory order decides the visible order of the other reads and writes. A release write pairs with the acquire read that sees it, and everything the writer wrote before the release is visible to the reader: "write the data, then publish the flag".
    `relaxed` guarantees atomicity alone, which suits counters and statistics where only the final value matters. When in doubt use the default `seq_cst`. See [atomics and memory order](../concurrency/atomics.md).

**22. What are the conditions for deadlock? How do you avoid it?**

??? success "The points"
    The four necessary conditions: mutual exclusion, hold and wait, no preemption, circular wait. In practice "circular wait" is the one broken: a fixed global lock order, `std::scoped_lock` to take several at once, and calling no external code (a callback, a virtual function) while holding a lock; more fundamentally, share less and pass messages. TSan detects lock order inversions. See [deadlock](../concurrency/threads.md#死锁).

**23. What is false sharing? How do you avoid it?**

??? success "The points"
    Different variables written often by different threads land in the same cache line, and the coherence protocol shuttles that line between cores, costing an order of magnitude. Separate them by a cache line (64 bytes, 128 on some ARM) with `alignas`. See [false sharing](../memory/layout.md#伪共享).

**24. ★ How do you implement a single-producer single-consumer lock-free queue?**

??? success "The points"
    A ring array with a power-of-two capacity; `tail` is written only by the producer and `head` only by the consumer. The producer writes the element and updates `tail` with a release; the consumer reads `tail` with an acquire, reads the element and updates `head` with a release. The two ends' pointers go in different cache lines and each caches the other's to cut cross-core reads. See [a single-producer single-consumer ring queue](../concurrency/lockfree-pool.md#单生产者单消费者环形队列).

**25. What is the ABA problem?**

??? success "The points"
    CAS only compares values: a thread reads A, it is changed to B and back to A in between, and the CAS still succeeds while the change is missed. In a lock-free stack this appears as a node at the same address being freed and reallocated. The fixes: a CAS carrying a version number, hazard pointers, epoch-based reclamation. See [a CAS loop](../concurrency/atomics.md#cas-循环).

**26. What do you have to consider in designing a thread pool?**

??? success "The points"
    The task queue (a lock plus a condition variable), `submit` returning a `future` (a `packaged_task` carrying the result and the exceptions), running tasks outside the lock, shutting down gracefully (finishing the remaining tasks or discarding them), and the members' destruction order (the workers declared last and joined first).
    The traps: waiting inside a task for a task of the same pool causing starvation deadlock, blocking I/O occupying a thread, nested parallelism causing oversubscription, and scheduling dominating when tasks are too small (chunk them or use work stealing). See [a thread pool](../concurrency/lockfree-pool.md#线程池).

## 5. Systems, performance and interop {#五系统性能与互操作}

**27. ★ What steps take source to an executable? How do a static and a shared library differ?**

??? success "The points"
    Preprocessing (expanding headers and macros) → compiling (each translation unit independently) → assembling (an object file and a symbol table) → linking (matching references to definitions).
    A static library is copied into the executable at link time; a shared library is loaded at run time, shared by several processes and upgradable on its own, at the cost of ABI compatibility and symbol collisions. A cross-language interface turns name mangling off with `extern "C"`. See [from source to an executable](../basics/compile-ub.md#从源码到可执行文件).

**28. ★ Why is a struct of arrays (SoA) sometimes faster than an array of structs (AoS)?**

??? success "The points"
    A CPU reads memory in 64-byte cache lines. When only a few fields are touched, most bytes of an AoS cache line are useless; everything an SoA brings in is useful, and it vectorizes besides. Conversely, when all of an object's fields are used every time, AoS is better. A GPU's coalesced access is the same reasoning. See [an array of structs or a struct of arrays](../memory/layout.md#数组的结构体还是结构体的数组).

**29. How do you locate a performance problem in a C++ service?**

??? success "The points"
    Establish first whether it is CPU, memory, I/O or lock waiting; `perf stat` classifies it by IPC and cache misses, and `perf record -g` plus a flame graph finds the hot functions; build Release with debug information and `-fno-omit-frame-pointer`. When it hangs (0% CPU), look at every thread's call stack (`gdb -p`, `py-spy dump --native`). See [CMake, the sanitizers and perf](build-debug.md).

**30. ★ Design a KV cache block allocator.**

??? success "The points"
    Allocate it once at startup by the memory budget and cut it into fixed-size blocks; keep the free block numbers on a stack, making allocation and freeing O(1) with no fragmentation; reference counts support prefix sharing and forking, with copy-on-write before writing to a shared block; an "all or nothing" bulk allocation interface gives the scheduler its admission control; and it is touched only by the scheduler thread, so the counts need no atomics. See [a block allocator](../memory/allocators.md#块分配器分页-kv-的核心).

**31. What are the GIL's traps when writing an extension with pybind11?**

??? success "The points"
    Long-running pure C++ computation releases the GIL (`py::gil_scoped_release`) and touches no Python object afterwards; a C++ thread takes the GIL before calling back into Python; taking the GIL while holding a C++ lock, while the thread holding the GIL waits for that lock, deadlocks. Besides, the automatic conversion of STL containers copies every time, so a large array goes through numpy's buffer. See [pybind11 and PyTorch C++ extensions](python-binding.md).

## 6. Writing code live {#六手写题}

C++ live-coding questions usually take 20-40 minutes and require **correct boundary handling and passing the sanitizers**:

| Problem | What it tests | Reference |
| --- | --- | --- |
| implement `unique_ptr` (with a custom deleter) | move semantics, RAII, the rule of five | [smart pointers](../basics/ownership.md) |
| implement a simplified `shared_ptr` | the control block, the atomic reference count, copy and move | [smart pointers](../basics/ownership.md) |
| implement a `String` class | the deep copy, copy and swap, moves | [move semantics, exercise 1](../basics/move.md#练习) |
| a thread-safe bounded blocking queue | locks, two condition variables, closing | [threads, exercise 1](../concurrency/threads.md#练习) |
| a single-producer single-consumer lock-free queue | acquire / release, false sharing | [a lock-free queue](../concurrency/lockfree-pool.md) |
| a thread pool | `packaged_task`, `future`, the shutdown order | [a thread pool](../concurrency/lockfree-pool.md#线程池) |
| a fixed-size memory pool / block allocator | the free list, alignment, reference counts | [allocators and memory pools](../memory/allocators.md) |
| an LRU cache | a hash table plus a list, O(1) | [the exercises](root://practice/) |

Once it is written, volunteer: the complexity, the boundary of its thread safety ("this class is single-threaded only" is an answer too), the level of exception safety, and how you would test it (the sanitizers + boundary cases + a concurrency stress test).

## Summary {#小结}

- [x] The object model: references and pointers, virtual functions, RAII and exception safety, the rule of five and the rule of zero, moves and forwarding.
- [x] Memory: the common UB and the sanitizers, alignment and layout, the boundary of a smart pointer's thread safety.
- [x] Templates and the standard library: instantiation and compilation time, concepts, a container's growth and iterator invalidation, a view's risks.
- [x] Concurrency: data races and race conditions, condition variables, memory order, deadlock, false sharing, a lock-free queue, a thread pool.
- [x] Systems: compiling and linking, SoA, profiling, the KV block allocator, the GIL. Live-coded answers have to pass the sanitizers and state their boundaries.
