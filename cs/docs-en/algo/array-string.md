# Arrays and strings: two pointers, the sliding window, prefix sums and binary search

<p class="lead">More than half of the problems in an interview land in this chapter: given an array or a string, find some optimal subarray, substring or pair of indices. They share one pattern. The brute force has two nested loops, and the problem hides some monotonicity that turns the inner loop into a pointer that only moves forward or into a binary search, bringing O(n squared) down to O(n) or O(n log n). This chapter writes the four templates (two pointers, the sliding window, prefix sums and difference arrays, binary search) in a form you can apply directly, and uses the chunking policy of chunked prefill as the example of binary searching the answer.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What framework do the sliding window's three forms (fixed length, longest satisfying, shortest satisfying) share? Where do they differ?
    2. Why is a sliding window O(n) rather than O(n squared)?
    3. What do `lower_bound` and `upper_bound` each return? How do you count a value's occurrences with them?
    4. When can you binary search the answer? Give an example from an inference system.
    5. Which operation does a prefix sum make O(1), and which does a difference array?

??? success "Answers for the self-test (answer first, then open this)"
    1. The framework is always "advance the right pointer one slot and add the new element to the window, then shrink the left boundary by a condition". They differ only in when the shrinking happens: a fixed-length window takes one in on the right and one out on the left; the longest form **shrinks when the condition fails**; the shortest form **shrinks as soon as the condition holds, recording the answer**.
    2. Because both pointers only move right, each at most n steps, for 2n moves in total. The code has a nested `while`, but the total number of times it runs is bounded by how far the left pointer can travel altogether, not by n steps for every position of the right pointer.
    3. `lower_bound(a, x)` is the first position >= x and `upper_bound(a, x)` is the first position > x (Python's `bisect_left` and `bisect_right`). The number of occurrences is `upper_bound - lower_bound`.
    4. When the answer lies in an interval and a feasible answer implies that every larger (or smaller) answer is feasible, you can binary search the answer space, turning "find the optimum" into "decide feasibility". Examples in an inference system: the per-chunk token cap of chunked prefill, working the largest batch back from a latency target, and working the concurrency back from the device memory.
    5. A prefix sum makes a range sum O(1) (after O(n) preprocessing); a difference array makes adding to a whole range O(1) (restored with one prefix sum at the end). The two are inverses. When you need both range queries and range updates, move to a Fenwick tree or a segment tree.

## Two pointers and the sliding window {#双指针与滑动窗口}

Watch the window move step by step first:

<div class="aig-widget" data-widget="window-step"></div>

Two pointers come in two forms: **converging** (the two ends walking toward the middle, for two-sum on a sorted array, the water container, palindrome checks) and **same-direction** (fast and slow pointers, for in-place removal, deduplication, moving zeroes). The sliding window is a special case of same-direction two pointers and the most important template in this chapter:

```python title="windows.py"
# the sliding window's three forms: fixed length, the longest satisfying a condition, the shortest satisfying one. All three share one framework
from collections import Counter


def fixed_window_max_sum(a, k):
    """定长窗口：长度为 k 的子数组里最大的和"""
    cur = sum(a[:k])
    best = cur
    for right in range(k, len(a)):
        cur += a[right] - a[right - k]          # one in, one out
        best = max(best, cur)
    return best


def longest_at_most_k_distinct(s, k):
    """最长窗口：最多含 k 种字符的最长子串长度"""
    cnt = Counter()
    left = best = 0
    for right, ch in enumerate(s):
        cnt[ch] += 1
        while len(cnt) > k:                     # shrink the left boundary while the condition fails
            cnt[s[left]] -= 1
            if cnt[s[left]] == 0:
                del cnt[s[left]]
            left += 1
        best = max(best, right - left + 1)
    return best


def shortest_sum_at_least(a, target):
    """最短窗口：和 >= target 的最短子数组长度（全为正数）"""
    left = cur = 0
    best = len(a) + 1
    for right, x in enumerate(a):
        cur += x
        while cur >= target:                    # once it holds, shrink as far as possible
            best = min(best, right - left + 1)
            cur -= a[left]
            left += 1
    return best if best <= len(a) else 0


a = [2, 1, 5, 1, 3, 2]
print("定长窗口 k=3 的最大和：", fixed_window_max_sum(a, 3))
print("最长窗口（最多 2 种字符）：", longest_at_most_k_distinct("eceba", 2))
print("最短窗口（和 >= 7）：", shortest_sum_at_least(a, 7))
print()
print("三者的共同框架：右指针每步进一个元素，左指针只会往右走，所以总共 O(n)。")
print("区别只在收缩的时机：定长是右进左出，最长是不满足时收缩，最短是满足时收缩。")
```

```text title="output"
定长窗口 k=3 的最大和： 9
最长窗口（最多 2 种字符）： 3
最短窗口（和 >= 7）： 3

三者的共同框架：右指针每步进一个元素，左指针只会往右走，所以总共 O(n)。
区别只在收缩的时机：定长是右进左出，最长是不满足时收缩，最短是满足时收缩。
```

When applying it, ask three questions: **what state the window holds** (a sum, counts, a maximum), **when it shrinks**, and **at which step the answer is recorded**. Settle those three and the code is a template.

The common variants:

| Problem | Window state | Shrink when |
| --- | --- | --- |
| The longest substring without repeats | character to its last position | a repeat appears |
| The minimum covering substring | how many characters are still missing | everything is covered |
| The shortest subarray with a sum >= target | the window's sum | the sum is already enough |
| The longest run of 1s with at most k 0s | the number of 0s in the window | the 0s exceed k |
| The maximum average of a fixed-length window | the window's sum | every step, one in and one out |

When the window has to yield a maximum or a minimum (the sliding window maximum, say), it needs a monotonic queue, which is in [Linked lists, stacks and hashing](linked-stack-hash.md).

## Prefix sums and difference arrays {#前缀和与差分}

```python title="prefix.py"
# prefix sums and difference arrays: recomputing a range sum every time becomes O(1), and adding to a whole range becomes O(1)
def prefix_sums(a):
    ps = [0] * (len(a) + 1)
    for i, x in enumerate(a):
        ps[i + 1] = ps[i] + x                   # ps[i] is the sum of the first i elements
    return ps


a = [3, 1, 4, 1, 5, 9, 2, 6]
ps = prefix_sums(a)
print("数组：", a)
print("前缀和：", ps)
print("区间 [2, 5) 的和 =", ps[5] - ps[2], "（直接算是", sum(a[2:5]), "）")
print()

# the difference array: adding v to every element of [l, r) becomes two point updates, restored with one prefix sum at the end
n = 8
diff = [0] * (n + 1)
for l, r, v in [(0, 3, 5), (2, 6, -2), (5, 8, 10)]:
    diff[l] += v
    diff[r] -= v
restored, cur = [], 0
for i in range(n):
    cur += diff[i]
    restored.append(cur)
print("三次区间加之后的数组：", restored)
print()

# the two-dimensional prefix sum: the sum over a rectangle, often used for KV block occupancy statistics and chunked attention-mask statistics
grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
rows, cols = len(grid), len(grid[0])
ps2 = [[0] * (cols + 1) for _ in range(rows + 1)]
for i in range(rows):
    for j in range(cols):
        ps2[i + 1][j + 1] = grid[i][j] + ps2[i][j + 1] + ps2[i + 1][j] - ps2[i][j]


def rect_sum(r1, c1, r2, c2):                   # closed on the left, open on the right
    return ps2[r2][c2] - ps2[r1][c2] - ps2[r2][c1] + ps2[r1][c1]


print("矩阵：", grid)
print("左上 2x2 的和 =", rect_sum(0, 0, 2, 2), "，右下 2x2 的和 =", rect_sum(1, 1, 3, 3))
print()
print("规律：前缀和把区间查询变成 O(1)，差分把区间修改变成 O(1)；两者互为逆运算。")
print("需要同时支持区间查询和单点/区间修改时，就该上树状数组或线段树了。")
```

```text title="output"
数组： [3, 1, 4, 1, 5, 9, 2, 6]
前缀和： [0, 3, 4, 8, 9, 14, 23, 25, 31]
区间 [2, 5) 的和 = 10 （直接算是 10 ）

三次区间加之后的数组： [5, 5, 3, -2, -2, 8, 10, 10]

矩阵： [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
左上 2x2 的和 = 12 ，右下 2x2 的和 = 28

规律：前缀和把区间查询变成 O(1)，差分把区间修改变成 O(1)；两者互为逆运算。
需要同时支持区间查询和单点/区间修改时，就该上树状数组或线段树了。
```

Prefix sums have two common variants as well:

- **A prefix sum plus a hash table**: count the subarrays whose sum is k. Keep the running prefix sum `cur` while iterating and look up how many times `cur - k` has occurred, which turns two loops into one. This is the standard answer for the subarray-sum family.
- **A prefix XOR, a prefix maximum, a prefix product**: the same idea with another operation. A product needs care around 0 (use two passes, left and right, instead of division).

The counterparts in an inference system: KV block occupancy statistics, the cumulative device memory of each layer's activations, and attention-mask statistics by token position are all range queries; adding an offset to a whole range of positions (updating a batch of requests' slot counts, say) is a difference array.

## Binary search: over indices, and over the answer {#二分在下标上也在答案上}

```python title="binary.py"
# three uses of binary search: finding an insertion point, searching the answer space, and cross-checking against the standard library
import bisect
import random


def lower_bound(a, x):
    """第一个 >= x 的位置（等价于 bisect.bisect_left）"""
    lo, hi = 0, len(a)                          # the half-open interval [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def upper_bound(a, x):
    """第一个 > x 的位置（等价于 bisect.bisect_right）"""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


rng = random.Random(0)
for _ in range(2000):                           # cross-check against the standard library on random inputs
    a = sorted(rng.randrange(10) for _ in range(rng.randrange(8)))
    x = rng.randrange(10)
    assert lower_bound(a, x) == bisect.bisect_left(a, x), (a, x)
    assert upper_bound(a, x) == bisect.bisect_right(a, x), (a, x)
print("和 bisect 对拍 2000 组：完全一致")
print("出现次数 = upper_bound - lower_bound，例如 [1,2,2,2,3] 里 2 出现",
      upper_bound([1, 2, 2, 2, 3], 2) - lower_bound([1, 2, 2, 2, 3], 2), "次")
print()


def min_chunk_size(lengths, budget, max_chunks):
    """二分答案：把一串请求按顺序切成若干块，每块的 token 数不超过 size，
    问 size 最小取多少才能不超过 max_chunks 块（这就是分块 prefill 的分块策略）"""
    def chunks_needed(size):
        used = cur = 0
        for x in lengths:
            if x > size:
                return float("inf")             # one request alone exceeds it, so this size is not feasible
            if cur + x > size:
                used, cur = used + 1, 0
            cur += x
        return used + (1 if cur else 0)

    lo, hi = max(lengths), sum(lengths)         # the answer is certainly within this range
    while lo < hi:
        mid = (lo + hi) // 2
        if chunks_needed(mid) <= max_chunks:
            hi = mid                            # feasible, so try smaller
        else:
            lo = mid + 1
    return lo if lo <= budget else -1


reqs = [512, 1024, 256, 2048, 128, 768]
for k in (2, 3, 4, 6):
    print(f"把 {len(reqs)} 个请求切成不超过 {k} 块时，每块至少要能装 {min_chunk_size(reqs, 99999, k)} 个 token")
```

```text title="output"
和 bisect 对拍 2000 组：完全一致
出现次数 = upper_bound - lower_bound，例如 [1,2,2,2,3] 里 2 出现 3 次

把 6 个请求切成不超过 2 块时，每块至少要能装 2944 个 token
把 6 个请求切成不超过 3 块时，每块至少要能装 2048 个 token
把 6 个请求切成不超过 4 块时，每块至少要能装 2048 个 token
把 6 个请求切成不超过 6 块时，每块至少要能装 2048 个 token
```

Two things:

- **Keep one template**. Use the half-open interval `[lo, hi)`, the condition `while lo < hi`, and `lo = mid + 1` / `hi = mid`, so that when the loop ends `lo == hi` is the answer. Written this way it cannot loop forever and there is no agonising over `+1` and `-1`. In Python `bisect` is easier still, but an interview often asks for it by hand.
- **Binary searching the answer comes up more often**. The condition is that the answer is monotone: if `size` is feasible then every larger `size` is too. Finding the optimum then becomes writing a `check(x)` predicate and binary searching the answer space. The example above is the chunking policy of chunked prefill: cut a run of requests into chunks in order with at most `size` tokens each, and ask how small `size` can be while staying within `max_chunks` chunks (see [Chunked prefill](minisgl://schedule/chunked-prefill/)).

Problems of the same kind: minimise the maximum (split an array, ship packages, jump between stones), maximise the minimum (Koko eating bananas, making bouquets), and, in engineering, working the maximum concurrency back from a service-level objective.

## What this chapter's problems are testing {#这一章的题在考什么}

Matching the common problems to their templates:

| When you see these words | Think of this template |
| --- | --- |
| contiguous subarray, substring, window | the sliding window |
| a sorted array, two-sum, both ends | converging two pointers |
| in place, O(1) space, keep the relative order | fast and slow pointers |
| a range sum, a range query | prefix sums (plus a hash table) |
| updating a whole range | a difference array |
| sorted, find a position, the first that satisfies | binary search |
| minimise the maximum, maximise the minimum, can it be done within X | binary search the answer |
| the kth largest, top-k | a heap or quickselect (see [Sorting, heaps and greedy](sort-heap-greedy.md)) |

!!! interview "How to explain it"
    Almost every problem in this chapter can open with the same sentence: "the brute force is two nested loops at O(n squared), but there is monotonicity here, so the inner loop can become a pointer move or a binary search." Then make three things clear: what state the window or the pointers hold, when they move or shrink, and at which step the answer is updated. For a binary search, volunteer that "I use the half-open template, so `lo` is the answer when the loop ends", and say why it cannot loop forever. On a minimise-the-maximum problem, point out directly that "this is binary searching the answer: write a `check(x)` predicate, and the answer is monotone so a binary search works", then analyse the predicate's complexity times the log. Finish by giving the time and space yourself, along with the edge cases: an empty array, all identical, a window longer than the array.

## Summary {#小结}

- [x] The sliding window's three forms share one framework and differ only in when they shrink; each pointer traverses once, so it is O(n).
- [x] A prefix sum makes a range query O(1) and a difference array makes a range update O(1); a prefix sum plus a hash table solves the subarray-sum-equals-k problem.
- [x] Use the half-open template for binary search, and the gap between `lower_bound` and `upper_bound` is the number of occurrences.
- [x] Binary searching the answer applies to optimisation problems whose answer is monotone, and chunked prefill's chunk size is one example.
- [x] Contiguous subarray means a window, in-place with O(1) means fast and slow pointers, and minimise the maximum means binary searching the answer.
