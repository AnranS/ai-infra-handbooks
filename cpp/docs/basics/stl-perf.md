# 标准库的性能视角

<p class="lead">标准库容器和算法的"用法"很容易查到，但写系统代码时更重要的是它们的<strong>成本模型</strong>：内存怎么排布、什么时候分配、哪些操作会让指针失效、哪种写法会悄悄地拷贝。这一章从推理系统常见的场景出发——token 序列、请求队列、词表、采样——讲怎样用对标准库。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `vector` 的 `push_back` 为什么是"均摊 O(1)"？`reserve` 省下了什么？
    2. `std::string_view` 和 `std::span` 是什么？什么时候会悬垂？
    3. `std::unordered_map` 每插入一个元素会分配几次内存？
    4. 从 32000 个 logits 里取最大的 50 个，用什么算法？复杂度多少？
    5. 什么情况下应该用 `std::list`？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 容量不够时按倍数（1.5 或 2 倍）扩容，每个元素被搬动的次数平均是常数，n 次 `push_back` 的总代价是 O(n)。`reserve` 省掉了中间的多次重新分配和搬运，也避免了扩容让指针和引用失效。
    2. 不拥有数据的视图：`string_view` 指向一段字符，`span` 指向一段连续的元素，只保存指针和长度。当它指向的数据被销毁或者重新分配（临时字符串、`vector` 扩容）之后还在使用，就悬垂了——所以不要把视图存下来。
    3. 每个元素一次（节点分配）；元素数超过负载因子时还要重新分配桶数组、整体再哈希。
    4. `std::nth_element` 把最大的 50 个换到前面（平均 O(n)），需要有序时再对这 50 个 `sort`（O(k log k)）；或者用 `std::partial_sort`（O(n log k)）。都比全排序的 O(n log n) 快。
    5. 几乎不需要：只有需要在中间频繁插入删除、并且要保持其他元素的迭代器和引用有效（比如 LRU 链表配哈希表、`splice` 拼接）时才用。它的每个节点单独分配、遍历对缓存很不友好。

## `vector`：默认的容器

![图：vector 的元素连续、list 的节点散落——遍历慢一个数量级](../assets/figures/vector-vs-list.svg){.aig-svg}

`std::vector` 把元素连续地放在一块堆内存里。连续意味着遍历时缓存命中率高、硬件预取有效、编译器能向量化——**除非有明确的理由，默认用 `vector`**。

它的成本模型：

- 容量不够时，分配一块更大的内存（libstdc++ 是 2 倍），把元素搬过去，释放旧的。所以 `push_back` 是均摊 O(1)，但**扩容那一次**是 O(n)，并且会让所有指向元素的指针、引用、迭代器失效（见[编译模型与未定义行为](compile-ub.md#容器扩容让引用悬垂)）；
- 事先知道大概有多少元素，就 `reserve`：

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
    if (v.data() != last) {   // 缓冲区地址变了：发生了一次重新分配
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

```text title="输出"
不预留：11 次重新分配
预留 1000：0 次
clear 之后 size=0 capacity=10
```

`clear()` 不释放内存，这正好适合"每一步都要重建一遍的临时缓冲区"：调度器每一步组 batch 时复用同一个 `vector`，`clear()` 之后再填，稳定之后就不再分配内存了。

## 视图：`string_view` 与 `span`

`std::string_view` 是"一段字符的指针 + 长度"，`std::span<T>`（C++20）是"一段 `T` 的指针 + 长度"。它们**不拥有**数据，拷贝它们只拷贝两个字，是传递"一段连续数据"的标准方式：

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
    out.push_back(s.substr(0, pos));   // 只是（指针, 长度），不拷贝字符
    if (pos == std::string_view::npos) break;
    s.remove_prefix(pos + 1);
  }
  return out;
}

// 接受任何连续的 float 序列：vector、数组、另一个 span 的一段
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

```text title="输出"
[model=qwen3] [tp=8] [max_len=32768]
mean(vector)=19.38 mean(前两个)=12.75 mean(数组)=2.00
```

函数参数的约定：只读一段字符串用 `std::string_view`，只读一段数组用 `std::span<const T>`，它们比 `const std::string&`、`const std::vector<T>&` 更通用（调用者可以传任何连续存储），也不会触发拷贝。

视图的代价是**生命周期要自己保证**。视图最容易悬垂在临时对象上。**这个程序有 bug：**

```cpp title="dangling_view.cpp" expect="fail"
#include <cstdio>
#include <string>
#include <string_view>

std::string load_prompt() { return std::string(64, 'x'); }   // 64 个字符：超过短字符串优化的长度，内容在堆上

int main() {
  std::string_view v = load_prompt();   // 临时的 string 在这一行结束时就析构了
  std::printf("%c\n", v[0]);            // 读已经释放的内存
}
```

```text title="ASan 的报告（节选）"
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x606000000020
```

规则：**视图只作为参数和局部变量用，不要存进成员变量或容器里**，除非你能保证被引用的数据活得更久（比如 `split` 返回的视图指向调用者持有的 `line`）。

## 关联容器：`unordered_map` 与 `map`

`std::unordered_map` 是哈希表，但它是**基于节点**的：每个元素单独分配一个节点，桶里存的是链表指针。于是：

- 插入一个元素就是一次内存分配；遍历、查找都要跟着指针跳，缓存不友好；
- 元素的地址在 rehash 时**不变**（节点不搬家，只是重新挂到新的桶上），这是它为数不多的优点——但迭代器会失效；
- 知道元素个数就 `reserve`，避免反复 rehash：

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

```text title="输出"
不预留：rehash 14 次
预留：rehash 0 次
```

`std::map` 是红黑树，按键有序，查找 O(log n)，每个节点也单独分配。它的用处是**需要有序**：按到达时间取最早的请求、按块号找相邻的空闲块、范围查询。

一个粗略的实测（`bench_containers.cpp`，在一台 x86 服务器上用 `-O2` 编译，取 5 次中最快的一次）：

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

volatile long long sink;   // 防止编译器把结果没人用的计算整个删掉

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

```text title="一次运行的结果"
遍历求和 10000000 个 int：vector 3.6 ms，list 36.3 ms
4096 个键里查 1000000 次：map 84.3 ms，unordered_map 3.4 ms，有序 vector + 二分 72.9 ms
```

- 遍历：`list` 比 `vector` 慢 10 倍，这还是节点**按顺序分配**、在内存里恰好挨着的理想情况；节点散落各处时差距更大；
- 查找：整数键的 `unordered_map` 最快（`std::hash<int>` 就是恒等函数，几乎只剩一次取模和一次访存）；`map` 和二分查找都要走 $\log_2 4096 = 12$ 步，每步是一次难以预测的分支。

需要高性能哈希表时，工程上常用"开放寻址 + 扁平存储"的实现（Abseil 的 `flat_hash_map`、Boost 的 `unordered_flat_map`），它们没有逐节点分配，通常比 `std::unordered_map` 快一截。

## 算法：用对的那一个

`<algorithm>` 里有很多只做"部分工作"的算法，用对了能省下一个数量级。采样时从词表大小的 logits 里取 top-k 就是典型：

| 需求 | 算法 | 复杂度 |
| --- | --- | --- |
| 最大的 k 个，不关心顺序 | `std::nth_element` | $O(n)$ |
| 最大的 k 个，要有序 | `nth_element` 后再对前 k 个 `sort`，或 `std::partial_sort` | $O(n + k \log k)$ / $O(n \log k)$ |
| 全部排序 | `std::sort` | $O(n \log n)$ |
| 第 p 百分位数（比如 P99 延迟） | `std::nth_element` | $O(n)$ |
| 有序数组里查找 | `std::lower_bound` | $O(\log n)$ |

```cpp title="topk.cpp"
#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

std::vector<int> topk(const std::vector<float>& logits, int k) {
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  auto cmp = [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); };
  std::nth_element(idx.begin(), idx.begin() + k, idx.end(), cmp);   // O(n)：前 k 个就是最大的 k 个，但无序
  idx.resize(k);
  std::sort(idx.begin(), idx.end(), cmp);                            // 只排这 k 个
  return idx;
}

int main() {
  std::vector<float> logits(32000);
  for (int i = 0; i < 32000; ++i) logits[i] = float((i * 7919) % 32000) / 32000.0f;
  for (int id : topk(logits, 5)) std::printf("%d:%.5f ", id, logits[id]);
  std::printf("\n");
}
```

```text title="输出"
14321:0.99997 28642:0.99994 10963:0.99991 25284:0.99988 7605:0.99984
```

比较函数里的 `a < b` 让相等的 logits 按 token id 排序，结果是确定的——排序类算法对"相等"元素的顺序不做保证，采样结果要可复现时，比较函数必须是一个严格的全序。

## 其他容器

- **`std::deque`**：分段连续，两端插入删除 O(1)，并且**元素地址在两端插入时不变**。适合做请求队列。
- **`std::priority_queue`**：堆，适合"每次取优先级最高的请求"。注意它不支持修改已有元素的优先级，需要的话用 `std::set` 或自己维护堆。
- **`std::list`**：几乎永远不是正确答案。唯一的优势是"在已知位置插入删除 O(1)、且其他元素的地址不变"，而 LRU 缓存这种场景，常见的高效写法是 `vector` 存节点 + 下标链表，或者直接用 `list` 但把节点从预分配的池里取。
- **`std::array`**：大小编译期固定，放在栈上，不分配内存。

!!! interview "面试怎么答"
    标准库性能题：默认用 `vector`——连续内存、对缓存友好，`push_back` 按几何级数扩容所以均摊 O(1)，知道大小就 `reserve`，扩容会让指针和引用失效；`string_view` / `span` 只读不拷贝，但不拥有数据，存下来就可能悬垂；`unordered_map` 每个元素一次堆分配，追求性能用扁平的开放寻址哈希表；只要部分结果就用部分算法——从 32000 个 logits 里取 top-50 用 `nth_element`（平均 O(n)）再排这 50 个，或 `partial_sort`，比全排序快得多，要确定性的结果时比较函数必须是严格全序（分数相同再比下标）。`list` 几乎不用。

## 练习

1. 实现 top-p（nucleus）过滤：给定已经 softmax 过的概率，按概率从大到小取最少的若干个 token，使它们的概率和不小于 `p`，返回这些 token 的 id（按概率从大到小，相等时 id 小的在前）。要求不对整个词表排序。

??? success "参考答案"
    先用 `nth_element` 取一个足够大的候选集（这里取 k=64，真实实现里通常先做 top-k 再做 top-p），排序后累加。如果候选集的概率和还不够 `p`，就把候选集扩大一倍重来：

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
        if (k == idx.size()) return idx;   // 浮点误差导致总和略小于 p：全部保留
      }
    }

    int main() {
      std::vector<float> probs(1000, 0.0005f);   // 997 个小概率，合计约 0.4985
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

    ```text title="输出"
    保留 3 个：42 7 900
    p=0.9 保留 800 个
    ```

    `p=0.9` 时要 3 个大概率 token 加上 797 个 0.0005，候选集从 64 扩到 128、256、512、1000，最后在 1000 个里找到。实际的采样 kernel 在 GPU 上用基于阈值二分或基数选择的方法，思路一样：避免全排序。

2. 下面这段组 batch 的代码有几处不必要的开销？

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

??? success "参考答案"
    1. `reqs` 和 `slot_of` 按值传入，每次调用都拷贝整个 vector 和 map：改成 `std::span<const Request>`（或 `const std::vector<Request>&`）和 `const std::map<int, int>&`；
    2. `for (auto r : reqs)` 每次循环拷贝一个 `Request`（连同它的 tokens）：改成 `const auto& r`；
    3. `find` 之后又用 `operator[]` 查了一次（而且 `operator[]` 在找不到时会插入）：用 `find` 返回的迭代器；
    4. `input_ids` 没有预留容量：先算出总长度再 `reserve`，或者用 `insert(end, begin, end)` 一次插入整段；
    5. 按请求 id 查槽位，如果 id 是连续的小整数，`vector<int>` 下标访问比 `map` 快一个数量级。

## 小结

- [x] 默认用 `vector`：连续内存、对缓存友好。知道大小就 `reserve`；扩容会让指针和引用失效；`clear()` 保留容量，适合复用。
- [x] 只读的字符串和数组参数用 `string_view` / `span`：不拷贝、更通用，但不要把视图存下来。
- [x] `unordered_map` 和 `map` 每个元素一次分配；整数键的 `unordered_map` 查找很快，要更快就用扁平的开放寻址哈希表。
- [x] 只要部分结果就用部分算法：top-k 和分位数用 `nth_element`，比全排序快得多；需要确定性结果时比较函数要是严格全序。
- [x] 队列用 `deque`，优先级用 `priority_queue`，`list` 几乎不用。
