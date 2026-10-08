# The standard library from a performance angle

<p class="lead">How to use the standard containers and algorithms is easy to look up, but what matters more in systems code is their <strong>cost model</strong>: how memory is laid out, when it allocates, which operations invalidate pointers, which forms copy quietly. This chapter starts from the situations an inference system meets, token sequences, request queues, vocabularies and sampling, and covers how to use the standard library well.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why is a `vector`'s `push_back` "amortized O(1)"? What does `reserve` save?
    2. What are `std::string_view` and `std::span`? When do they dangle?
    3. How many allocations does a `std::unordered_map` make per element inserted?
    4. Which algorithm takes the largest 50 of 32000 logits? At what complexity?
    5. When should you use a `std::list`?

??? success "Answers (try it yourself first, then expand)"
    1. When the capacity runs out it grows by a factor (1.5 or 2), so each element is moved a constant number of times on average and n `push_back`s cost O(n) in total. `reserve` saves the intermediate reallocations and moves, and keeps a reallocation from invalidating pointers and references.
    2. Views that do not own their data: a `string_view` points at a run of characters and a `span` at a run of contiguous elements, each holding only a pointer and a length. They dangle once what they point at has been destroyed or reallocated (a temporary string, a `vector` growing) and is still in use, which is why a view should not be stored.
    3. One per element (the node allocation); and when the element count passes the load factor, the bucket array is reallocated and everything is rehashed.
    4. `std::nth_element` moves the largest 50 to the front (O(n) on average), and a `sort` over those 50 orders them when order is needed (O(k log k)); or `std::partial_sort` (O(n log k)). Both beat a full sort's O(n log n).
    5. Almost never: only when elements are inserted and removed in the middle often and the other elements' iterators and references have to stay valid (an LRU list paired with a hash table, splicing with `splice`). Every node is allocated separately and traversal is very unfriendly to the cache.

## `vector`: the default container {#vector默认的容器}

![Figure: a vector's elements are contiguous while a list's nodes are scattered - traversal is an order of magnitude slower](../assets/figures/vector-vs-list.svg){.aig-svg}

A `std::vector` puts its elements contiguously in one block of heap memory. Contiguous means a high cache hit rate while traversing, effective hardware prefetching and vectorization by the compiler: **use `vector` by default unless there is a definite reason not to**.

Its cost model:

- when the capacity runs out it allocates a larger block (2x in libstdc++), moves the elements and frees the old one. So `push_back` is amortized O(1), but **that one reallocation** is O(n) and invalidates every pointer, reference and iterator into the elements (see [the compilation model and undefined behaviour](compile-ub.md#容器扩容让引用悬垂));
- when roughly how many elements there will be is known beforehand, `reserve`:

```cpp title="vector_growth.cpp"
#include <cstdio>
#include <vector>

int count_reallocs(bool reserve) {
  std::vector<int> v;
  if (reserve) v.reserve(1000);
  int reallocs = 0;
  const int* last = v.data();
  for (int i = 0; i < 1000; ++i) {
    v.push_back(i);
    if (v.data() != last) {   // the buffer's address changed: a reallocation happened
      ++reallocs;
      last = v.data();
    }
  }
  return reallocs;
}

int main() {
  std::printf("不预留：%d 次重新分配\n", count_reallocs(false));
  std::printf("预留 1000：%d 次\n", count_reallocs(true));
  std::vector<int> v(10);
  v.clear();
  std::printf("clear 之后 size=%zu capacity=%zu\n", v.size(), v.capacity());
}
```

```text title="output"
不预留：11 次重新分配
预留 1000：0 次
clear 之后 size=0 capacity=10
```

`clear()` does not free the memory, which suits exactly "a temporary buffer rebuilt every step": the scheduler reuses the same `vector` when forming a batch each step, `clear()`ing and refilling it, and stops allocating once it settles.

## Views: `string_view` and `span` {#视图string_view-与-span}

A `std::string_view` is "a pointer to a run of characters plus a length" and a `std::span<T>` (C++20) is "a pointer to a run of `T` plus a length". They **do not own** the data and copying one copies two words, which makes them the standard way to pass "a run of contiguous data":

```cpp title="views.cpp"
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

std::vector<std::string_view> split(std::string_view s, char sep) {
  std::vector<std::string_view> out;
  while (true) {
    auto pos = s.find(sep);
    out.push_back(s.substr(0, pos));   // just (pointer, length), copying no characters
    if (pos == std::string_view::npos) break;
    s.remove_prefix(pos + 1);
  }
  return out;
}

// takes any contiguous run of floats: a vector, an array, part of another span
float mean(std::span<const float> xs) {
  float s = 0;
  for (float x : xs) s += x;
  return xs.empty() ? 0 : s / xs.size();
}

int main() {
  std::string line = "model=qwen3 tp=8 max_len=32768";
  for (auto kv : split(line, ' ')) std::printf("[%.*s] ", int(kv.size()), kv.data());
  std::printf("\n");
  std::vector<float> latencies = {12.5f, 13.0f, 40.0f, 12.0f};
  float arr[3] = {1, 2, 3};
  std::printf("mean(vector)=%.2f mean(前两个)=%.2f mean(数组)=%.2f\n", mean(latencies),
              mean(std::span(latencies).first(2)), mean(arr));
}
```

```text title="output"
[model=qwen3] [tp=8] [max_len=32768]
mean(vector)=19.38 mean(前两个)=12.75 mean(数组)=2.00
```

The conventions for parameters: `std::string_view` for a read-only string and `std::span<const T>` for a read-only array, both more general than `const std::string&` and `const std::vector<T>&` (the caller can pass any contiguous storage) and neither causing a copy.

A view's cost is that **the lifetime is yours to guarantee**. A view dangles on a temporary most easily. **This program has a bug:**

```cpp title="dangling_view.cpp" expect="fail"
#include <cstdio>
#include <string>
#include <string_view>

std::string load_prompt() { return std::string(64, 'x'); }   // 64 characters: past the small-string optimization, so the contents are on the heap

int main() {
  std::string_view v = load_prompt();   // the temporary string is destroyed at the end of this line
  std::printf("%c\n", v[0]);            // reading memory that has been freed
}
```

```text title="the ASan report (excerpt)"
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x606000000020
```

The rule: **use a view only as a parameter or a local, and never store it in a member or a container**, unless you can guarantee the referenced data outlives it (the views `split` returns point into the `line` the caller holds).

## Associative containers: `unordered_map` and `map` {#关联容器unordered_map-与-map}

A `std::unordered_map` is a hash table, but it is **node-based**: every element gets its own node and the buckets hold list pointers. So:

- inserting an element is an allocation; traversal and lookup both chase pointers, which is unfriendly to the cache;
- an element's address **does not change** on a rehash (the nodes stay put and are only relinked into new buckets), which is one of its few advantages, though the iterators are invalidated;
- when the element count is known, `reserve` to avoid repeated rehashing:

```cpp title="rehash.cpp"
#include <cstdio>
#include <unordered_map>

int rehashes(bool reserve) {
  std::unordered_map<int, int> m;
  if (reserve) m.reserve(100000);
  int n = 0;
  auto buckets = m.bucket_count();
  for (int i = 0; i < 100000; ++i) {
    m[i] = i;
    if (m.bucket_count() != buckets) {
      ++n;
      buckets = m.bucket_count();
    }
  }
  return n;
}

int main() {
  std::printf("不预留：rehash %d 次\n", rehashes(false));
  std::printf("预留：rehash %d 次\n", rehashes(true));
}
```

```text title="output"
不预留：rehash 14 次
预留：rehash 0 次
```

A `std::map` is a red-black tree, ordered by key, with O(log n) lookup and a separate allocation per node too. Its use is when **order is needed**: taking the earliest request by arrival time, finding the free block adjacent to a block number, a range query.

A rough measurement (`bench_containers.cpp`, compiled with `-O2` on an x86 server, the fastest of 5 runs):

```cpp title="bench_containers.cpp" sanitize="none" flags="-O2"
// g++ -std=c++20 -O2 bench_containers.cpp && ./a.out
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <list>
#include <map>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

template <class F>
double ms(F&& f, int reps = 5) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

volatile long long sink;   // stops the compiler from deleting a computation whose result nobody uses

int main() {
  const int n = 10'000'000;
  std::vector<int> v(n, 1);
  std::list<int> l(v.begin(), v.end());
  std::printf("遍历求和 %d 个 int：vector %.1f ms，list %.1f ms\n", n,
              ms([&] { sink = std::accumulate(v.begin(), v.end(), 0LL); }),
              ms([&] { sink = std::accumulate(l.begin(), l.end(), 0LL); }));

  const int keys = 4096, queries = 1'000'000;
  std::mt19937 rng(0);
  std::vector<int> ks(keys);
  for (int i = 0; i < keys; ++i) ks[i] = i * 3;
  std::vector<int> qs(queries);
  for (auto& q : qs) q = ks[rng() % keys];
  std::map<int, int> m;
  std::unordered_map<int, int> um;
  std::vector<std::pair<int, int>> sorted;
  for (int k : ks) {
    m[k] = k;
    um[k] = k;
    sorted.push_back({k, k});
  }
  std::printf("%d 个键里查 %d 次：map %.1f ms，unordered_map %.1f ms，有序 vector + 二分 %.1f ms\n", keys, queries,
              ms([&] { long long s = 0; for (int q : qs) s += m.find(q)->second; sink = s; }),
              ms([&] { long long s = 0; for (int q : qs) s += um.find(q)->second; sink = s; }),
              ms([&] {
                long long s = 0;
                for (int q : qs) s += std::lower_bound(sorted.begin(), sorted.end(), std::pair{q, 0})->second;
                sink = s;
              }));
}
```

```text title="one run"
遍历求和 10000000 个 int：vector 3.6 ms，list 36.3 ms
4096 个键里查 1000000 次：map 84.3 ms，unordered_map 3.4 ms，有序 vector + 二分 72.9 ms
```

- traversal: the `list` is 10 times slower than the `vector`, and that is the ideal case where the nodes were **allocated in order** and happen to sit next to each other in memory; with the nodes scattered the gap is larger;
- lookup: the `unordered_map` with an integer key is the fastest (`std::hash<int>` is the identity function, leaving little more than a modulo and one memory access); the `map` and the binary search both take $\log_2 4096 = 12$ steps, each an unpredictable branch.

When a high-performance hash table is needed, the usual engineering choice is an "open addressing + flat storage" implementation (Abseil's `flat_hash_map`, Boost's `unordered_flat_map`), which allocates nothing per node and is usually a good deal faster than `std::unordered_map`.

## Algorithms: pick the right one {#算法用对的那一个}

`<algorithm>` has many algorithms that do only "part of the work", and picking the right one saves an order of magnitude. Taking the top k of a vocabulary's logits while sampling is the typical case:

| Need | Algorithm | Complexity |
| --- | --- | --- |
| the largest k, order not needed | `std::nth_element` | $O(n)$ |
| the largest k, ordered | `nth_element` then a `sort` over the first k, or `std::partial_sort` | $O(n + k \log k)$ / $O(n \log k)$ |
| sort everything | `std::sort` | $O(n \log n)$ |
| the pth percentile (P99 latency, say) | `std::nth_element` | $O(n)$ |
| a lookup in a sorted array | `std::lower_bound` | $O(\log n)$ |

```cpp title="topk.cpp"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

std::vector<int> topk(const std::vector<float>& logits, int k) {
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  auto cmp = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);   // O(n): the first k are the largest k, unordered
  idx.resize(k);
  std::sort(idx.begin(), idx.end(), cmp);                            // sort only those k
  return idx;
}

int main() {
  std::vector<float> logits(32000);
  for (int i = 0; i < 32000; ++i) logits[i] = float((i * 7919) % 32000) / 32000.0f;
  for (int id : topk(logits, 5)) std::printf("%d:%.5f ", id, logits[id]);
  std::printf("\n");
}
```

```text title="output"
14321:0.99997 28642:0.99994 10963:0.99991 25284:0.99988 7605:0.99984
```

The `a < b` in the comparator orders equal logits by token id, which makes the result deterministic: sorting algorithms promise nothing about the order of "equal" elements, so when sampling has to be reproducible the comparator has to be a strict total order.

## The other containers {#其他容器}

- **`std::deque`**: contiguous in segments, O(1) insertion and removal at either end, and **element addresses do not change when inserting at either end**. Good for a request queue.
- **`std::priority_queue`**: a heap, good for "take the highest-priority request each time". Note that it cannot change an existing element's priority, for which a `std::set` or a heap of your own is needed.
- **`std::list`**: almost never the right answer. Its only advantage is "O(1) insertion and removal at a known position with the other elements' addresses unchanged", and for something like an LRU cache the usual efficient form is a `vector` of nodes plus an index-based list, or a `list` whose nodes come from a preallocated pool.
- **`std::array`**: a size fixed at compile time, on the stack, allocating nothing.

!!! interview "How to explain it"
    On the standard library's performance: use `vector` by default, for contiguous memory and cache friendliness, with `push_back` growing geometrically so it is amortized O(1), `reserve` when the size is known, and a reallocation invalidating pointers and references; `string_view` / `span` read without copying but do not own the data, so storing one may dangle; an `unordered_map` costs one heap allocation per element, and a flat open-addressing hash table is the choice when performance matters; use a partial algorithm when only part of the result is needed, so the top 50 of 32000 logits goes to `nth_element` (O(n) on average) plus a sort over those 50, or to `partial_sort`, far faster than a full sort, and a deterministic result needs a comparator that is a strict total order (equal scores broken by index). A `list` is almost never used.

## Exercises {#练习}

1. Implement top-p (nucleus) filtering: given probabilities that have been through a softmax, take the fewest tokens from the largest probability downwards whose probabilities sum to at least `p`, and return their ids (largest probability first, smaller id first on a tie). Do it without sorting the whole vocabulary.

??? success "Answer"
    Take a large enough candidate set with `nth_element` first (k=64 here; a real implementation usually does top-k before top-p), then sort and accumulate. If the candidate set's probabilities still do not reach `p`, double the set and start over:

    ```cpp title="top_p.cpp"
    #include <algorithm>
    #include <cstdio>
    #include <numeric>
    #include <vector>

    std::vector<int> top_p(const std::vector<float>& probs, float p) {
      std::vector<int> idx(probs.size());
      std::iota(idx.begin(), idx.end(), 0);
      auto cmp = [&](int a, int b) { return probs[a] > probs[b] || (probs[a] == probs[b] && a < b); };
      for (std::size_t k = 64;; k *= 2) {
        k = std::min(k, idx.size());
        std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), cmp);
        std::sort(idx.begin(), idx.begin() + k, cmp);
        float acc = 0;
        for (std::size_t i = 0; i < k; ++i) {
          acc += probs[idx[i]];
          if (acc >= p) return {idx.begin(), idx.begin() + i + 1};
        }
        if (k == idx.size()) return idx;   // floating-point error leaves the total slightly below p: keep them all
      }
    }

    int main() {
      std::vector<float> probs(1000, 0.0005f);   // 997 small probabilities, about 0.4985 in total
      probs[42] = 0.3f;
      probs[7] = 0.15f;
      probs[900] = 0.0515f;
      auto keep = top_p(probs, 0.5f);
      std::printf("保留 %zu 个：", keep.size());
      for (std::size_t i = 0; i < std::min<std::size_t>(keep.size(), 4); ++i) std::printf("%d ", keep[i]);
      std::printf("\n");
      std::printf("p=0.9 保留 %zu 个\n", top_p(probs, 0.9f).size());
    }
    ```

    ```text title="output"
    保留 3 个：42 7 900
    p=0.9 保留 800 个
    ```

    At `p=0.9` it takes the 3 high-probability tokens plus 797 of the 0.0005 ones, so the candidate set grows from 64 to 128, 256, 512 and 1000, and the answer is found among the 1000. A real sampling kernel on a GPU uses a threshold bisection or a radix selection, with the same idea: avoid a full sort.

2. How many unnecessary costs does the batching code below have?

    ```cpp
    std::vector<int> build_batch(std::vector<Request> reqs, std::map<int, int> slot_of) {
      std::vector<int> input_ids;
      for (auto r : reqs) {
        if (slot_of.find(r.id) != slot_of.end()) {
          int slot = slot_of[r.id];
          for (int t : r.tokens) input_ids.push_back(t);
        }
      }
      return input_ids;
    }
    ```

??? success "Answer"
    1. `reqs` and `slot_of` are taken by value, so every call copies the whole vector and map: take `std::span<const Request>` (or `const std::vector<Request>&`) and `const std::map<int, int>&`;
    2. `for (auto r : reqs)` copies a `Request` (and its tokens) each iteration: use `const auto& r`;
    3. the `find` is followed by another lookup through `operator[]` (which inserts when the key is absent): use the iterator `find` returned;
    4. `input_ids` reserves no capacity: compute the total length first and `reserve`, or insert each run at once with `insert(end, begin, end)`;
    5. looking a slot up by request id: if the ids are small consecutive integers, indexing a `vector<int>` is an order of magnitude faster than a `map`.

## Summary {#小结}

- [x] Use `vector` by default: contiguous memory, friendly to the cache. `reserve` when the size is known; a reallocation invalidates pointers and references; `clear()` keeps the capacity, which suits reuse.
- [x] Use `string_view` / `span` for read-only string and array parameters: no copy and more general, but do not store a view.
- [x] `unordered_map` and `map` cost one allocation per element; an `unordered_map` with an integer key looks up fast, and a flat open-addressing hash table is faster still.
- [x] Use a partial algorithm when only part of the result is needed: `nth_element` for top-k and percentiles, far faster than a full sort; a deterministic result needs a comparator that is a strict total order.
- [x] A queue goes to `deque`, priorities to `priority_queue`, and `list` almost never.
