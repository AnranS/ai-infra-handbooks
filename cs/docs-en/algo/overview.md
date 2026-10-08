# How to prepare for the algorithm round: complexity, a map of the problem types, and Python's traps

<p class="lead">Inference roles still test algorithms: a round of writing code by hand is all but standard, and a competitive-programming background still counts in your favour. But preparing for it is not the same as a daily problem on a practice site. The goal is to read a problem, explain the approach, write code without an off-by-one error and analyse the complexity, all within 40 minutes. This chapter covers estimating complexity (with constants measured on this machine), what the map of problem types looks like, which traps Python has, and a procedure for solving a problem that you can reuse in the interview itself. The five chapters after it go type by type, each with exercises graded in the browser.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How far apart are `x in list` and `x in set`? How much slower is taking 100,000 elements out of a queue with `list.pop(0)` than with `deque.popleft`?
    2. A problem gives n <= 1e5. What complexity gets through? What about n <= 1e3? And n <= 20?
    3. What is wrong with `[[0] * 3] * 2`?
    4. Why is a linked-list problem best written iteratively in Python?
    5. In an interview, do you write the code first or explain the approach first?

??? success "Answers for the self-test (answer first, then open this)"
    1. Measured on this machine: looking for a value that is not there takes 1.1 milliseconds in a 100,000-element list and only 0.05 microseconds in a set, over twenty thousand times apart (one is O(n), the other O(1)). The queue question: `list.pop(0)` shifts every following element forward each time, which is O(n squared) and takes 1.2 seconds for 30,000 elements, while `deque.popleft` is O(1) and takes 1 millisecond.
    2. A rough rule of thumb (Python does about ten million simple operations per second): n <= 1e5 needs O(n) or O(n log n); n <= 1e3 allows O(n squared); n <= 20 allows enumerating subsets at O(2^n); n <= 10 allows all permutations at O(n!). Read it backwards and the data range tells you what complexity the problem setter wants.
    3. The outer `* 2` copies a reference to the same inner list, so changing `grid[0][0]` changes both rows. Use `[[0] * 3 for _ in range(2)]`.
    4. Python's default recursion limit is 1000, so a linked list of ten thousand gives a `RecursionError`; the call overhead is not small either. A tree problem has the same issue if the tree degenerates into a chain.
    5. The approach first. Restate the problem, confirm the edge cases and the data range, give a brute-force solution and its complexity, then say how you would optimise, and only start writing once the interviewer agrees. Writing code straight away risks getting halfway and finding the direction was wrong, with no time left.

## Complexity: growth and constants {#复杂度增长与常数}

Put a few curves on one chart first and drag n and the constant:

<div class="aig-widget" data-widget="complexity"></div>

Complexity analysis is about how fast something grows as n grows, but the real time also depends on the constant. A few numbers from this machine:

```python title="complexity.py"
# complexity is about growth, but keep the constants in mind too: the measured time of a few common operations on this machine
import random
import time
from collections import deque

random.seed(0)
N = 100_000
data = list(range(N))
random.shuffle(data)
s, d = set(data), {x: x for x in data}
dq = deque(data)


def per_call(fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


rows = [
    ("x in list（未命中）", "O(n)", per_call(lambda: -1 in data, 20)),
    ("x in set（未命中）", "O(1)", per_call(lambda: -1 in s, 200000)),
    ("d[x]", "O(1)", per_call(lambda: d[12345], 200000)),
    ("list.append", "O(1) 摊还", per_call(lambda: data.append(0) or data.pop(), 200000)),
    ("deque.popleft", "O(1)", per_call(lambda: dq.appendleft(dq.popleft()), 200000)),
    ("sorted(data)", "O(n log n)", per_call(lambda: sorted(data), 20)),
]
print(f"n = {N:,} 时单次操作的耗时")
for name, big_o, sec in rows:
    print(f"  {name:22s} {big_o:12s} {sec * 1e6:9.2f} 微秒")
print()
M = 30_000
print(f"把 {M:,} 个元素当队列全部取出来：")
q = data[:M]
t0 = time.perf_counter()
while q:
    q.pop(0)                                          # every time, all of the following elements shift forward
t_list = time.perf_counter() - t0
q = deque(data[:M])
t0 = time.perf_counter()
while q:
    q.popleft()
t_deque = time.perf_counter() - t0
print(f"  list.pop(0)：  {t_list * 1e3:8.1f} ms（O(n²)）")
print(f"  deque.popleft：{t_deque * 1e3:8.1f} ms（O(n)），快 {t_list / t_deque:.0f} 倍")
print()
print("用错数据结构差多少：求两个 20000 元素列表的交集")
a, b = list(range(20000)), list(range(10000, 30000))
t0 = time.perf_counter()
naive = [x for x in a if x in b]                      # O(n x m): every element scans the other list once
t_naive = time.perf_counter() - t0
t0 = time.perf_counter()
fast = sorted(set(a) & set(b))                        # O(n + m)
t_fast = time.perf_counter() - t0
print(f"  列表里查找：{t_naive * 1e3:8.1f} ms")
print(f"  转成集合：  {t_fast * 1e3:8.1f} ms，快 {t_naive / t_fast:.0f} 倍；结果一致：{sorted(naive) == fast}")
```

```text title="output (on this machine)"
n = 100,000 时单次操作的耗时
  x in list（未命中）         O(n)           1172.54 微秒
  x in set（未命中）          O(1)              0.05 微秒
  d[x]                   O(1)              0.06 微秒
  list.append            O(1) 摊还           0.07 微秒
  deque.popleft          O(1)              0.07 微秒
  sorted(data)           O(n log n)    19676.73 微秒

把 30,000 个元素当队列全部取出来：
  list.pop(0)：    1236.6 ms（O(n²)）
  deque.popleft：     1.2 ms（O(n)），快 1003 倍

用错数据结构差多少：求两个 20000 元素列表的交集
  列表里查找：  1806.4 ms
  转成集合：       2.9 ms，快 630 倍；结果一致：True
```

Three facts to remember:

- **The wrong data structure costs hundreds or thousands of times**. A list lookup is O(n) while a set or dict is O(1); `list.pop(0)` is O(n) while `deque.popleft` is O(1). Using a hash table to bring O(n squared) down to O(n) is the most common optimisation in an interview.
- **Python does about ten million simple operations per second**, which gives this table:

| Data range | Feasible complexity | Typical approach |
| --- | --- | --- |
| n <= 10 | O(n!) | all permutations, backtracking |
| n <= 20 | O(2^n) | enumerating subsets, bitmask DP |
| n <= 100 | O(n cubed) | interval DP, Floyd |
| n <= 1000 | O(n squared) | two-dimensional DP, a plain double loop |
| n <= 1e5 | O(n log n) | sorting, a heap, binary search, union-find |
| n <= 1e7 | O(n) | two pointers, a sliding window, prefix sums |
| n >= 1e9 | O(log n) or O(1) | binary search on the answer, a closed form |

- **State the space complexity too**. A recursion's stack depth counts; O(1) extra space is a bonus requirement in many problems (in-place swaps, two pointers, bit tricks).

## A map of the problem types {#题型地图}

There are only so many types that come up often in interviews, and the next five chapters follow this order:

| Chapter | Type | The core pattern | The counterpart in an inference system |
| --- | --- | --- | --- |
| [Arrays and strings](array-string.md) | two pointers, sliding window, prefix sums, binary search | use monotonicity to turn two loops into one | the window of chunked prefill, interval statistics over KV blocks |
| [Linked lists, stacks and hashing](linked-stack-hash.md) | reversal, fast and slow pointers, a monotonic stack, LRU | trade extra structure for time | the free list of a block allocator, LRU eviction |
| [Trees and graphs](tree-graph.md) | traversal, BFS/DFS, topological sort, union-find, shortest paths | state transitions plus a visited mark | the prefix tree (radix cache), the topological order of a compute graph |
| [Sorting, heaps and greedy](sort-heap-greedy.md) | custom sorting, top-k, merging intervals, scheduling | sort first, then be greedy | request scheduling, top-k sampling, expert-parallel load balancing as bin packing |
| [Dynamic programming and backtracking](dp-backtrack.md) | one- and two-dimensional DP, knapsack, string DP, pruning | define the state plus the transition | the verification tree of speculative decoding, chunking policies |

Every chapter ends with exercises you can have graded in the browser, ordered by difficulty. Practising each type's standard form until you can write it without thinking matters far more than the number of problems you do.

## Python's traps {#python-写题的坑}

```python title="pitfalls.py"
# a few Python-specific traps when writing code in an interview, each able to trip someone up on the spot
import sys

# 1. a default argument is evaluated at definition time, so a mutable one is shared by every call
def collect(x, acc=[]):
    acc.append(x)
    return acc


print("可变默认参数：", collect(1), collect(2), "  <- 第二次调用带着上一次的结果")


def collect_ok(x, acc=None):
    acc = [] if acc is None else acc
    acc.append(x)
    return acc


print("正确写法：    ", collect_ok(1), collect_ok(2))

# 2. a two-dimensional array cannot be [[0] * n] * m: the outer one is m references to the same list
grid = [[0] * 3] * 2
grid[0][0] = 1
print("\n[[0]*3]*2 改一个元素：", grid, " <- 两行都变了")
grid = [[0] * 3 for _ in range(2)]
grid[0][0] = 1
print("列表推导式：          ", grid)

# 3. a slice is a copy: slicing in a loop adds a factor of n to the complexity
def has_dup_slow(a):
    return any(x in a[i + 1:] for i, x in enumerate(a))    # every slice makes a copy


def has_dup_fast(a):
    return len(set(a)) != len(a)


print("\n切片会拷贝：两种写法结果一致 =", has_dup_slow([1, 2, 3, 2]) == has_dup_fast([1, 2, 3, 2]))

# 4. recursion depth: the default limit is 1000, which linked-list and tree problems exceed easily
def depth(n):
    return 0 if n == 0 else 1 + depth(n - 1)


print("默认递归上限：", sys.getrecursionlimit())
try:
    depth(2000)
except RecursionError:
    print("递归 2000 层：RecursionError —— 链表和退化成链的树要用迭代写法")

# 5. integer division and rounding with negatives
print("\n-7 // 2 =", -7 // 2, "（向下取整），int(-7 / 2) =", int(-7 / 2), "（向零取整）")
print("-7 % 3 =", -7 % 3, "（Python 的余数跟除数符号），C 语言里是 -1")

# 6. the stability of sorting, and a custom key
words = [("b", 2), ("a", 2), ("c", 1)]
print("\n按第二个元素排序（稳定，相等的保持原序）：", sorted(words, key=lambda t: t[1]))
print("按第二个降序、第一个升序：", sorted(words, key=lambda t: (-t[1], t[0])))
```

```text title="output"
可变默认参数： [1, 2] [1, 2]   <- 第二次调用带着上一次的结果
正确写法：     [1] [2]

[[0]*3]*2 改一个元素： [[1, 0, 0], [1, 0, 0]]  <- 两行都变了
列表推导式：           [[1, 0, 0], [0, 0, 0]]

切片会拷贝：两种写法结果一致 = True
默认递归上限： 1000
递归 2000 层：RecursionError —— 链表和退化成链的树要用迭代写法

-7 // 2 = -4 （向下取整），int(-7 / 2) = -3 （向零取整）
-7 % 3 = 2 （Python 的余数跟除数符号），C 语言里是 -1

按第二个元素排序（稳定，相等的保持原序）： [('c', 1), ('b', 2), ('a', 2)]
按第二个降序、第一个升序： [('a', 2), ('b', 2), ('c', 1)]
```

A few more beyond those:

- **Use the standard library**: `collections.deque` (a double-ended queue), `heapq` (a min-heap, so negate for a max-heap), `bisect` (the insertion point from a binary search), `collections.Counter` (counting), `defaultdict` (automatic initialisation), `itertools` (permutations and combinations). Interviews allow these, and they save a lot of boilerplate.
- **Join strings with `"".join()`**: `s += x` in a loop is O(n squared).
- **Do not modify a container while iterating over it**: build a new list, or iterate backwards, when you have to delete.
- **Mind integer division**: `//` rounds down, which differs from C's and Java's `/` (rounding toward zero) for negatives; the same goes for the modulo.
- **`sort` is stable**, and a multi-key sort is simply `key=lambda x: (-x[1], x[0])`.

## A procedure for solving a problem {#一套解题流程}

How to spend the 40 minutes in an interview:

1. **Restate the problem (1 to 2 minutes)**: say it in your own words and confirm the input and output types, the data range, whether there are duplicates, whether it has to be in place, and whether the answer is unique. This step often draws out the key information.
2. **Give an example (2 minutes)**: write out one small example and one edge case (empty input, a single element, all identical). Many mistakes surface right here.
3. **State the brute force (1 minute)**: give a solution that is certainly correct along with its complexity, and say that it is the baseline you will improve on. This keeps you from scoring zero.
4. **State the optimisation (3 to 5 minutes)**: point out what the brute force does repeatedly, and which structure or which monotonicity removes it. Write only once that is agreed.
5. **Write the code (15 minutes)**: the signature and the edge-case handling first, then the body. Spell the variable names out (`left` and `right`, not `i` and `j`) and keep the intervals uniformly half-open, as in `[left, right)`.
6. **Test it yourself (5 minutes)**: walk through the examples from step 2 by hand, concentrating on the edges: empty input, a single element, out of range, an infinite loop.
7. **Analyse the complexity and say what could be better (2 minutes)**: the time and the space, whether the space can come down to O(1), and what to do if the data were ten times larger.

The follow-up questions that come up: **"what if the data does not fit in memory?"** (external sorting, chunked processing, a Bloom filter), **"what if it is multithreaded?"** (shard then merge, the granularity of the locks), **"what if it has to support online updates?"** (a structure that supports insertion and deletion, such as a heap, a balanced tree or a skip list). Interviewers for inference roles like these follow-ups, because they are closer to the engineering.

## How much to practise, and how {#练多少怎么练}

- **Quantity**: two passes over the classic Hot 100 plus this book's six chapters of exercises (about 80 problems) is basically enough for an inference role's algorithm round. On the second pass, only ask whether you can state the approach at a glance, and rewrite only the ones where you cannot.
- **Timing**: 20 minutes for a medium problem, 35 for a hard one. If you run over, read the solution and then rewrite it from scratch the next day.
- **By hand**: write at least half of them without autocomplete, because an interview is usually a shared document or a whiteboard.
- **Review**: which category you got wrong (misread the problem, an edge case, the complexity, could not solve it) matters more than which problem. Three mistakes in one category means practising that category specifically.

!!! interview "How to explain it"
    An algorithm round has no trick to it, only a procedure: restate the problem and the edge cases, give one small example and one edge case, state the brute force and its complexity, point out the repeated work and give the optimisation, write only once it is agreed, test the edges yourself, then analyse the complexity and volunteer what could be better. While writing, spell the variable names out, keep the intervals half-open and handle the edges first. On the follow-ups about data that does not fit in memory, concurrency and online updates, go toward external sorting and chunking, sharding and lock granularity, and heaps and balanced trees. The complexity has to match the data range: n <= 1e5 means O(n log n), and only n <= 20 allows enumerating subsets.

## Exercises {#练习}

**1. Guess the complexity.** A problem gives `1 <= n <= 200000` and asks for an integer. Which approaches would you rule out first? And if the range were `1 <= n <= 18`?

??? success "Answer"
    n <= 2e5: rule out O(n squared) (4e10 operations, certain to time out), and aim for O(n) or O(n log n). The usual suspects are sorting, prefix sums, a hash table, two pointers, a heap, or a binary search on the answer.

    n <= 18: this range is all but a hint to use a bitmask. Enumerating subsets is O(2^n) = 260 thousand, and even O(2^n x n squared) gets through. Seeing n <= 20 should send you toward bit tricks and bitmask DP.

**2. Find what is slow.** The code below counts the elements that appear more than once in a list, and it is very slow at n = 1e5. Where is it slow? How would you make it O(n)?

    ```python
    def dups(a):
        return [x for i, x in enumerate(a) if x in a[i + 1:]]
    ```

??? success "Answer"
    Two problems: `a[i+1:]` **copies** the rest of the list every time (O(n) in both space and time), and `x in list` is another O(n) lookup, so the whole thing is O(n squared) or worse.

    The fix: count with `Counter` in one pass, then filter for a count above 1 (watching out for duplicates and keeping the order):

    ```python
    from collections import Counter

    def dups(a):
        cnt = Counter(a)
        seen = set()
        return [x for x in a if cnt[x] > 1 and not (x in seen or seen.add(x))]
    ```

**3. Recursion to iteration.** Rewrite the linked-list reversal below iteratively, and say why that is preferable in Python.

    ```python
    def reverse(head):
        if head is None or head.next is None:
            return head
        new_head = reverse(head.next)
        head.next.next = head
        head.next = None
        return new_head
    ```

??? success "Answer"
    ```python
    def reverse(head):
        prev = None
        while head:
            head.next, prev, head = prev, head, head.next
        return prev
    ```

    The recursive version's stack depth equals the list's length, Python's default limit is 1000, and a list of ten thousand gives a `RecursionError`; every level of call also has overhead. The iterative version is O(1) extra space and easier to get right first time in an interview.

**4. Design the procedure.** The interviewer gives you a problem you have not seen, and 5 minutes later you still have no idea. What do you do next?

??? success "Answer"
    (1) Write out or state the brute force, so there is a correct baseline. (2) Work the target complexity back from the data range, which narrows the space of solutions. (3) Take a smaller example and look for the pattern by hand. (4) Ask yourself what the brute force does repeatedly: repeated computation usually means DP or prefix sums, repeated lookups mean a hash table, repeated sorting means a heap. (5) Talk to the interviewer: "I am thinking of a hash table to remove this lookup, does that direction look right to you?" Silence is the worst option, and showing your thinking is itself part of what is being assessed.

## Summary {#小结}

- [x] Complexity is about growth, but know the constants too: a list lookup against a set is twenty thousand times apart, and `list.pop(0)` against `deque.popleft` is a thousand.
- [x] Work the complexity back from the data range: 1e5 means O(n log n), 1e3 means O(n squared), 20 means O(2^n), 10 means O(n!).
- [x] Python's traps: a mutable default argument, `[[0]*n]*m`, a slice being a copy, the recursion limit of 1000, and division and modulo with negatives.
- [x] Make good use of the standard library: `deque`, `heapq`, `bisect`, `Counter`, `defaultdict`, `itertools`.
- [x] The interview procedure: restate, example, brute force, optimisation, write, test the edges, complexity and improvements.
