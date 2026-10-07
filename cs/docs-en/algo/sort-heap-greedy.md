# Sorting, heaps and greedy: sort first, then one pass

<p class="lead">The solution to this family is often two lines: sort, then scan. The hard part is working out what to sort by and why that choice is correct. A heap is the standard tool for taking the extreme dynamically, used by top-k sampling, multi-way merging, and a scheduler picking the machine that frees up first. This chapter sets out the templates for top-k, multi-way merging, interval problems and task scheduling, alongside the sorting and bin-packing problems in an inference system.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. To find the largest k of n numbers, what is the complexity with sorting and with a heap? When should each be used?
    2. Python's `heapq` is a min-heap, so what do you do for a max-heap? What has to be watched when putting tuples in a heap?
    3. What do you sort by to merge overlapping intervals? And to pick as many non-overlapping intervals as possible?
    4. What does `sorted` being stable mean? How do you write a multi-key sort?
    5. Why does a greedy approach need a proof? Give an example where greedy is wrong.

??? success "Answers for the self-test (answer first, then open this)"
    1. Sorting is O(n log n) and keeping a min-heap of size k is O(n log k). Use the heap when k is far smaller than n (the top 100 out of a hundred million log lines); sort when k approaches n, or when the fully ordered result is needed afterwards. There is also quickselect, O(n) on average, which is the fastest when only the kth largest is wanted and not the order.
    2. Negate the values (`heappush(h, -x)`), or put in tuples of `(-priority, data)`. Tuples compare item by item, so the second item has to be **comparable**; when putting objects in, add a monotonically increasing sequence number to break ties, or the objects themselves get compared and raise a `TypeError`.
    3. Merging overlapping intervals sorts by **the start**; picking as many non-overlapping intervals as possible sorts by **the end** (finishing earlier leaves the most room for what follows). Picking the most by start is wrong.
    4. Stable means the sort does not change the existing order among equal elements. For multiple keys, write `key=lambda x: (-x[1], x[0])` (the second item descending, then the first ascending), or exploit the stability and sort twice (the secondary key first, then the primary).
    5. Because a greedy choice is locally optimal at each step, which does not necessarily lead to a global optimum. Either give an exchange argument ("any optimal solution can be reshaped to include my choice") or a counterexample. The classic counterexample: with coin denominations `[1, 3, 4]` making 6, greedy takes a 4 and then two 1s, three coins, while the optimum is two 3s, two coins. Which is why making change needs dynamic programming.

## The heap: taking the extreme dynamically {#堆动态取最值}

![Figure: a heap, a complete binary tree stored in an array, with sift-up and sift-down](../assets/figures/heap-ops.svg){.aig-svg}

```python title="heaps.py"
# three typical uses of a heap: top-k, multi-way merging, and scheduling by taking the minimum dynamically
import heapq
import random

rng = random.Random(0)
data = [rng.randrange(1000) for _ in range(20000)]

# 1. top-k: keep a min-heap of size k, whose top is the kth largest
def top_k(a, k):
    heap = []
    for x in a:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:                      # larger than the kth largest, so replace the top
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)


print("Top-5：", top_k(data, 5), "（和排序取前 5 一致：", sorted(data, reverse=True)[:5] == top_k(data, 5), "）")
print("复杂度：排序 O(n log n)，堆 O(n log k)——k 远小于 n 时堆明显更省")

# 2. multi-way merging: k sorted sequences into one
streams = [sorted(rng.randrange(100) for _ in range(5)) for _ in range(4)]
print("\n四条有序序列：", streams)
print("归并结果：", list(heapq.merge(*streams)))

# 3. scheduling: always taking the machine that frees up first, which is a min-heap
def schedule(tasks, machines):
    """tasks 是每个任务的耗时，返回全部完成的时刻"""
    heap = [0.0] * machines                    # each machine's free time
    heapq.heapify(heap)
    for cost in tasks:
        free_at = heapq.heappop(heap)          # the machine that frees up first
        heapq.heappush(heap, free_at + cost)
    return max(heap)


tasks = [5, 3, 8, 2, 7, 1, 6]
for m in (1, 2, 3):
    print(f"{len(tasks)} 个任务（共 {sum(tasks)} 单位）用 {m} 台机器：{schedule(tasks, m):.0f} 单位完成，"
          f"理论下界 {max(max(tasks), sum(tasks) / m):.1f}")
print("\n把任务按耗时从大到小排序再调度（最长优先，LPT）：")
for m in (2, 3):
    print(f"  {m} 台：顺序到达 {schedule(tasks, m):.0f}，先排序 {schedule(sorted(tasks, reverse=True), m):.0f}")
```

```text title="output"
Top-5： [999, 999, 999, 999, 999] （和排序取前 5 一致： True ）
复杂度：排序 O(n log n)，堆 O(n log k)——k 远小于 n 时堆明显更省

四条有序序列： [[28, 29, 46, 69, 71], [1, 6, 6, 56, 68], [33, 48, 51, 52, 66], [2, 26, 39, 81, 94]]
归并结果： [1, 2, 6, 6, 26, 28, 29, 33, 39, 46, 48, 51, 52, 56, 66, 68, 69, 71, 81, 94]
7 个任务（共 32 单位）用 1 台机器：32 单位完成，理论下界 32.0
7 个任务（共 32 单位）用 2 台机器：18 单位完成，理论下界 16.0
7 个任务（共 32 单位）用 3 台机器：12 单位完成，理论下界 10.7

把任务按耗时从大到小排序再调度（最长优先，LPT）：
  2 台：顺序到达 18，先排序 16
  3 台：顺序到达 12，先排序 11
```

Each of the three uses matches a family of problems:

- **Top-k**: keep a min-heap of size k, and its top is the current kth largest. Anything smaller is discarded straight away. The inference counterpart is top-k during sampling (the top 50 from a 150,000-word vocabulary), which uses a parallel reduction on the GPU but is the same idea.
- **Multi-way merging**: put one current element from each of k sorted sequences into the heap, then repeatedly take the smallest and replace it with its successor. External sorting, merging k sorted lists and merging the results of a distributed query are all this. Python's `heapq.merge` is ready to use.
- **Scheduling**: put each machine's free time in the heap and always take the one that frees up first. This is the embryo of an inference scheduler, although real scheduling also has to consider KV memory, prefix hits and priorities (see [The scheduler](serving://engine/scheduler/)).

Two traps in `heapq`: it is min-heap only (negate for a max-heap), and tuples compare item by item, so when putting in `(priority, object)` insert a sequence number in between, or equal priorities will compare the objects themselves.

## Sorting: stability and a custom key {#排序稳定性与自定义键}

Python's `sorted` is **Timsort**: O(n log n) at worst, close to O(n) on partly ordered data, and **stable**. Stability gives a multi-key sort two forms:

```python
items.sort(key=lambda x: (-x.score, x.name))        # one sort: the score descending, ties by name
items.sort(key=lambda x: x.name)                     # or two sorts, the secondary key before the primary
items.sort(key=lambda x: -x.score)
```

A few practical points:

- **Do not write a `cmp`-style comparison function**; Python 3 only accepts `key`. Use `functools.cmp_to_key` if you really must.
- **Use a tuple when sorting by several fields**, not a concatenated string.
- **With a lot of data and only the top k wanted**: `heapq.nlargest(k, data)` uses less memory than `sorted(data)[:k]`.
- **Quickselect** finds the kth largest in O(n) on average, O(n squared) at worst (a random pivot avoids it). This is what is meant when an interview asks for the kth largest in O(n).

## Greedy: sort plus one pass {#贪心排序--一遍扫描}

```python title="greedy.py"
# greedy: sort by some key, then one pass. The difficulty is not the code but proving why the choice is correct
def merge_intervals(intervals):
    """合并重叠区间：按起点排序，能接上就扩展右端点"""
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def max_non_overlapping(intervals):
    """最多能选几个互不重叠的区间：按**终点**排序，能选就选"""
    count, last_end = 0, float("-inf")
    for start, end in sorted(intervals, key=lambda t: t[1]):
        if start >= last_end:
            count += 1
            last_end = end
    return count


def min_rooms(meetings):
    """同时进行的最大数量（需要几个会议室 / 几个并发槽位）：把端点排序后扫描"""
    events = sorted([(s, 1) for s, _ in meetings] + [(e, -1) for _, e in meetings])
    cur = best = 0
    for _, delta in events:                    # at the same instant, handle the exits first (-1 sorts before +1)
        cur += delta
        best = max(best, cur)
    return best


data = [[1, 3], [2, 6], [8, 10], [15, 18]]
print("区间：", data)
print("合并后：", merge_intervals(data))
print("最多能选几个互不重叠：", max_non_overlapping(data))
print("需要几个并发槽位：", min_rooms(data))
print()
print("三道题排序的关键字不同：合并按起点，选最多按终点，算并发按端点扫描。")
print("按起点选最多是错的——先来的可能很长，挡住后面好几个短的。")
print()

# an example from inference systems: sorting by output length reduces the waste within a batch
reqs = [("A", 500), ("B", 20), ("C", 480), ("D", 30)]
for name, order in [("按到达顺序", reqs), ("按预估长度分组", sorted(reqs, key=lambda t: t[1]))]:
    batches = [order[:2], order[2:]]
    waste = sum(max(x[1] for x in b) * len(b) - sum(x[1] for x in b) for b in batches)
    print(f"{name}：分两批 {[[x[0] for x in b] for b in batches]}，浪费的 token 槽位 {waste}")
```

```text title="output"
区间： [[1, 3], [2, 6], [8, 10], [15, 18]]
合并后： [[1, 6], [8, 10], [15, 18]]
最多能选几个互不重叠： 3
需要几个并发槽位： 2

三道题排序的关键字不同：合并按起点，选最多按终点，算并发按端点扫描。
按起点选最多是错的——先来的可能很长，挡住后面好几个短的。

按到达顺序：分两批 [['A', 'B'], ['C', 'D']]，浪费的 token 槽位 930
按预估长度分组：分两批 [['B', 'D'], ['C', 'A']]，浪费的 token 槽位 30
```

A greedy problem's solution is usually "sort by some key, then one pass", and all of the difficulty is in **what to sort by**:

| Problem | Sort key | The intuition |
| --- | --- | --- |
| Merging overlapping intervals | the start | extend the right end whenever it connects |
| The most non-overlapping intervals / the fewest arrows | **the end** | finishing earlier leaves the most room for what follows |
| The number of meeting rooms / the maximum concurrency | scan the endpoints (a difference array) | +1 on an entry, -1 on an exit |
| The fewest platforms, task scheduling | a heap plus time order | always reuse the resource that frees up first |
| Handing out biscuits, assigning tasks | sort both arrays | pair with two pointers |
| Gas stations, the jump game | one pass keeping the furthest reachable point | local reachability |

That last example is a real problem in an inference system: **the wider the spread of output lengths in one batch, the more is wasted**. A batch runs until its longest request finishes and the short requests' slots sit empty (continuous batching replaces a finished request promptly, but within one step it is still aligned). So grouping by estimated length, or batching the long requests separately, raises throughput noticeably. This is the same family of greedy as bin packing. Expert load balancing in a large mixture-of-experts deployment is bin packing too: assign experts to cards so that the heaviest card is as light as possible (see [Large-scale expert-parallel deployment](serving://moe/ep-deploy/)).

**A greedy approach is either provable or wrong.** The usual proof technique is an exchange argument: assume an optimal solution differs from your choice, and show that substituting your choice does not make it worse. When you cannot prove it, use dynamic programming honestly. Making change (denominations `[1, 3, 4]` for 6) is the classic case of greedy coming off the rails.

## What this chapter's problems are testing {#这一章的题在考什么}

| When you see these words | Think of this |
| --- | --- |
| the kth largest, the top k, the most frequent | a heap (O(n log k)) or quickselect (O(n)) |
| merge k sorted, external sorting | multi-way merging (a heap) |
| intervals, overlap, merge, the fewest removals | sorting plus greedy (mind whether it is by start or end) |
| meeting rooms, the concurrency, the maximum overlap | sorting and scanning the endpoints / a difference array |
| assigning tasks, the earliest finish, the fewest machines | heap simulation plus greedy (longest processing time first) |
| bin packing, load balancing, grouping | a greedy approximation (most remaining, longest first) |
| custom sorting, several keys | `key=lambda x: (-a, b)` |

!!! interview "How to answer in an interview"
    For top-k, give three solutions and their complexity first: "sorting at O(n log n), a heap of size k at O(n log k), quickselect at O(n) on average", then the situations (a data stream allows only the heap, a full ordering means sorting). For a greedy problem you have to volunteer **what you sort by** and **why**: merging intervals by the start, picking the most non-overlapping by the end (the exchange argument that finishing earlier is better). The `heapq` details (min-heap only, a sequence number in the tuple to avoid comparing objects) count nicely in your favour. Mapping the problem to engineering is better still: top-k is sampling, multi-way merging is external sorting and merging distributed results, bin packing is expert load balancing, and grouping by output length reduces the waste within a batch. And do not forget to say that a greedy approach has to be provable, and dynamic programming is the fallback when it is not.

## Summary {#小结}

- [x] Top-k with a min-heap of size k is O(n log k); when only the kth largest is wanted, quickselect is O(n) on average.
- [x] `heapq` is min-heap only (negate for a max-heap), and tuple comparison needs a sequence number to avoid comparing the objects themselves.
- [x] Multi-way merging uses a heap and underlies external sorting and merging distributed results.
- [x] Greedy is sorting plus one pass: merge intervals by the start, pick the most non-overlapping by the end, and compute concurrency by scanning the endpoints.
- [x] Scheduling uses a heap to take the earliest free, and bin packing takes the longest first; in inference these are request grouping and expert load balancing.
- [x] A greedy approach has to be provable (an exchange argument), and dynamic programming is the fallback when it is not.
