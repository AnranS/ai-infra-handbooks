# Dynamic programming and backtracking: removing the repeated work

<p class="lead">Dynamic programming is the family that makes people most nervous in an interview, but its pattern is actually fixed: define the state, write the transition, settle the initial values and the traversal order. Backtracking is trying every possibility, and it is won or lost on the pruning. This chapter works through DP in the order brute-force recursion, memoisation, iteration, rolling array, then fixes the backtracking templates with permutations, subsets, combination sum and n queens, and finally connects them to speculative decoding's verification tree and to chunking policies.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does a problem have to satisfy for dynamic programming to apply?
    2. Which four things have to be settled before writing DP?
    3. When a 0-1 knapsack is compressed to a one-dimensional array, why does the inner loop have to run backwards?
    4. What is the backtracking template? What is the trap in `out.append(path)`?
    5. What are the advantages and disadvantages of memoised recursion and of bottom-up iteration?

??? success "Answers for the self-test (answer first, then open this)"
    1. **Optimal substructure** (the large problem's optimum is composed of the subproblems' optima) and **overlapping subproblems** (the same subproblem is solved over and over). Optimal substructure without overlap is divide and conquer; greedy additionally requires that a local optimum leads to a global one.
    2. The state's definition (what `f[i]` means, being clear about "ending at i" against "the first i"), the transition, the initial values including the boundaries, and the traversal order (so that the subproblems used are already computed). Say those four sentences before writing, and the code is just a translation.
    3. Running backwards, `f[c - w]` is still the previous round's value (without the current item), which keeps each item to one use. Running forwards uses values already updated this round, which allows taking the same item several times, and that is exactly the **unbounded knapsack**.
    4. "Make a choice, recurse, undo the choice." The trap is that `path` is one list object, modified throughout, so collecting an answer has to copy it (`path[:]`), or you end up with all empty lists or the same result repeated.
    5. Memoised recursion is close to brute force with a cache, intuitive to write, and only computes the states actually used; the downsides are a limited recursion depth (1000 in Python) and a large constant. Iteration has no recursion overhead and can compress space with a rolling array, but you have to work out the traversal order yourself. In an interview you can write the memoised version to explain the idea and then rewrite it as iteration.

## From brute-force recursion to iteration {#从暴力递归到递推}

Iteration is filling the table in dependency order. Take edit distance and see where each cell comes from:

<div class="aig-widget" data-widget="edit-distance"></div>

```python title="dp.py"
# from brute-force recursion to memoisation to iteration: three ways to write one problem
import time
from functools import lru_cache


def fib_naive(n):
    return n if n < 2 else fib_naive(n - 1) + fib_naive(n - 2)


@lru_cache(maxsize=None)
def fib_memo(n):
    return n if n < 2 else fib_memo(n - 1) + fib_memo(n - 2)


def fib_iter(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


t0 = time.perf_counter()
fib_naive(28)
t_naive = time.perf_counter() - t0
t0 = time.perf_counter()
fib_memo(28)
t_memo = time.perf_counter() - t0
print(f"fib(28)：暴力递归 {t_naive * 1e3:.1f} ms，记忆化 {t_memo * 1e3:.4f} ms，"
      f"快 {t_naive / max(t_memo, 1e-9):.0f} 倍")
print(f"三种写法结果一致：{fib_naive(20) == fib_memo(20) == fib_iter(20)}")
print()

# the 0-1 knapsack: a two-dimensional table -> a one-dimensional rolling array (the inner loop has to run backwards)
def knapsack_2d(weights, values, cap):
    n = len(weights)
    f = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(cap + 1):
            f[i][c] = f[i - 1][c]                                   # do not take item i
            if c >= weights[i - 1]:
                f[i][c] = max(f[i][c], f[i - 1][c - weights[i - 1]] + values[i - 1])
    return f[n][cap]


def knapsack_1d(weights, values, cap):
    f = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):                             # backwards: keeps each item to one use
            f[c] = max(f[c], f[c - w] + v)
    return f[cap]


def knapsack_wrong(weights, values, cap):
    f = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):                                 # forwards: the same item gets used several times (the unbounded knapsack)
            f[c] = max(f[c], f[c - w] + v)
    return f[cap]


weights, values, cap = [2, 3, 4, 5], [3, 4, 5, 6], 8
print(f"0-1 背包（容量 {cap}）：二维 {knapsack_2d(weights, values, cap)}，"
      f"一维倒序 {knapsack_1d(weights, values, cap)}，一维正序 {knapsack_wrong(weights, values, cap)}（错：同一件拿了多次）")
print()

# edit distance: the standard template for two-dimensional DP
def edit_distance(a, b):
    m, n = len(a), len(b)
    f = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        f[i][0] = i                                                 # delete everything
    for j in range(n + 1):
        f[0][j] = j                                                 # insert everything
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                f[i][j] = f[i - 1][j - 1]
            else:
                f[i][j] = 1 + min(f[i - 1][j], f[i][j - 1], f[i - 1][j - 1])   # delete, insert, replace
    return f[m][n]


for a, b in [("horse", "ros"), ("intention", "execution"), ("", "abc")]:
    print(f"编辑距离('{a}', '{b}') = {edit_distance(a, b)}")
```

```text title="output (on this machine)"
fib(28)：暴力递归 46.1 ms，记忆化 0.0239 ms，快 1930 倍
三种写法结果一致：True

0-1 背包（容量 8）：二维 10，一维倒序 10，一维正序 12（错：同一件拿了多次）

编辑距离('horse', 'ros') = 3
编辑距离('intention', 'execution') = 5
编辑距离('', 'abc') = 3
```

Three points:

- **Memoisation is the easiest entry**: write the brute-force recursion, add an `@lru_cache`, and exponential becomes polynomial at once. Fibonacci fib(28) drops from 58 milliseconds to 0.04.
- **The direction of the one-dimensional rolling array** sets the semantics: backwards is the 0-1 knapsack (each item once) and forwards is the unbounded knapsack (each item without limit). In the comparison above the forward version gives 12 where the correct answer is 10.
- **The two-dimensional DP template** (edit distance) is worth memorising: the state is the fewest steps to turn a's first i characters into b's first j, the initial values are deleting everything and inserting everything, and the transition inherits the diagonal on a match or takes the smallest of delete, insert and replace plus one. The longest common subsequence, regular-expression matching and wildcard matching share the skeleton.

The common problems and their state definitions:

| Problem | State definition | Transition |
| --- | --- | --- |
| Climbing stairs, house robber | `f[i]`: the optimum up to step i or house i | `f[i] = max(f[i-1], f[i-2] + a[i])` |
| The longest increasing subsequence | `f[i]`: the longest length ending at i | `f[i] = max(f[j] + 1)`, O(n squared); greedy plus binary search is O(n log n) |
| The 0-1 knapsack | `f[i][c]`: the maximum value from the first i items at capacity c | take it or leave it |
| Edit distance, longest common subsequence | `f[i][j]`: the optimum between two prefixes | the best of three operations |
| Interval DP (burst balloons) | `f[i][j]`: the optimum over the interval `[i, j]` | enumerate the split point |
| Bitmask DP (the travelling salesman) | `f[mask][i]`: mask visited, stopped at i | enumerate the next one |

**How to define the state** is the only part that needs practice. Two rules of thumb: (1) write the brute-force recursive function's parameters first, and the parameters are the state; (2) "ending at i" and "the first i" are different definitions, the former usually for subarrays and substrings (where the answer is the maximum over all `f[i]`) and the latter for prefix decisions (where the answer is `f[n]`).

## Backtracking: choose, recurse, undo {#回溯做选择递归撤销}

```python title="backtrack.py"
# backtracking: choose -> recurse -> undo the choice. The pruning decides whether it can run at all
def permutations(nums):
    out, path, used = [], [], [False] * len(nums)

    def dfs():
        if len(path) == len(nums):
            out.append(path[:])                # you have to copy: path keeps being modified afterwards
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i], _ = True, path.append(x)
            dfs()
            used[i] = False                    # undo the choice
            path.pop()

    dfs()
    return out


def subsets(nums):
    out, path = [], []

    def dfs(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            dfs(i + 1)                         # start at i+1: do not pick the same one twice
            path.pop()

    dfs(0)
    return out


def combination_sum(candidates, target):
    """每个数可以用多次，结果不能重复"""
    out, path = [], []
    nums = sorted(candidates)

    def dfs(start, remain):
        if remain == 0:
            out.append(path[:])
            return
        for i in range(start, len(nums)):
            if nums[i] > remain:               # pruning: after sorting, the rest can only be larger
                break
            path.append(nums[i])
            dfs(i, remain - nums[i])           # still i: repeated use is allowed
            path.pop()

    dfs(0, target)
    return out


def n_queens(n):
    """统计 n 皇后的解数：用三个集合记录被占用的列和两条对角线"""
    cols, diag1, diag2 = set(), set(), set()
    count = 0

    def dfs(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or row - col in diag1 or row + col in diag2:
                continue
            cols.add(col), diag1.add(row - col), diag2.add(row + col)
            dfs(row + 1)
            cols.remove(col), diag1.remove(row - col), diag2.remove(row + col)

    dfs(0)
    return count


print("全排列 [1,2,3]：", permutations([1, 2, 3]))
print("子集 [1,2,3]：", subsets([1, 2, 3]))
print("组合总和（[2,3,6,7] 凑 7）：", combination_sum([2, 3, 6, 7], 7))
print("n 皇后的解数：", {n: n_queens(n) for n in range(4, 10)})
print()
print("模板都是：做选择 -> 递归 -> 撤销。区别在于起点（start）怎么传、什么时候剪枝。")
```

```text title="output"
全排列 [1,2,3]： [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
子集 [1,2,3]： [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
组合总和（[2,3,6,7] 凑 7）： [[2, 2, 3], [7]]
n 皇后的解数： {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352}

模板都是：做选择 -> 递归 -> 撤销。区别在于起点（start）怎么传、什么时候剪枝。
```

The three templates differ only in where the next level starts:

- **Permutations**: mark with a `used` array and scan from the beginning at every level (a different order is a different answer).
- **Subsets and combinations**: pass `start`, and the next level begins at `i + 1` (no repeats, order ignored).
- **Combinations with repetition**: the next level still begins at `i`.

With duplicate elements you need **sorting plus deduplication within a level** (`if i > start and nums[i] == nums[i-1]: continue`), or duplicate answers appear.

Pruning decides whether backtracking can run at all: in combination sum, `break` once `nums[i] > remain` after sorting; n queens checks conflicts in O(1) with three sets; sudoku fills the cell with the fewest candidates first. The complexity is usually O(the number of solutions x each solution's length), plus the branches that were pruned.

**How backtracking relates to DP**: both search a decision tree. DP remembers the subproblems' answers and computes each once; backtracking walks every path. When every arrangement has to be enumerated, only backtracking will do; when only the optimum or the count is wanted, use DP.

## In an inference system {#在推理系统里}

- **Speculative decoding's verification tree**: the draft model produces a tree of candidates, the target model verifies every branch in one forward pass, and the longest correct prefix is accepted. Building the tree and computing the expected accepted length is tree DP (see [Speculative decoding in depth](serving://topics/speculative/)).
- **Chunking and batch planning**: choosing each step's batch composition under memory and latency constraints is essentially knapsack or bin packing. Engineering uses a greedy approximation, but analysing the upper bound uses DP.
- **Dynamic programming inside a kernel**: FlashAttention's online softmax is a way of maintaining state incrementally, from the same root as DP's rolling array; beam search is a search tree with pruning.
- **Edit distance**: computing word or character error rates in an evaluation, and comparing the draft and target sequences in speculative decoding, both use it.

!!! interview "How to explain it"
    For a DP problem, say the four things first: "the state `f[i][j]` means..., the transition is..., the initial values are..., the traversal order is...", and then write the code. If the state will not come, write the brute-force recursion, add memoisation and translate it into iteration; that path is itself a good demonstration. Volunteer the space optimisation: "this only depends on the previous row, so it can be one-dimensional, and the 0-1 knapsack has to run backwards". For a backtracking problem, give the template (choose, recurse, undo) and stress two things: collecting an answer has to copy with `path[:]`, and where the pruning happens. Finish with the complexity: DP is O(the number of states x the transitions per state) and backtracking is O(the number of solutions x their length). Connecting it to an engineering case (speculative decoding's verification tree, the knapsack model of batch planning) counts in your favour.

## Summary {#小结}

- [x] DP needs optimal substructure plus overlapping subproblems; settle the state, the transition, the initial values and the traversal order before writing.
- [x] The path: brute-force recursion, add memoisation, convert to iteration, then a rolling array; the one-dimensional 0-1 knapsack runs backwards and forwards is the unbounded knapsack.
- [x] The two-dimensional DP skeleton (edit distance, longest common subsequence) is worth memorising.
- [x] Backtracking is choose, recurse, undo; collecting an answer has to copy, duplicates need sorting plus deduplication within a level, and the pruning decides whether it runs.
- [x] Use backtracking when every arrangement is wanted and DP when only the optimum or the count is; speculative decoding's verification tree and batch planning both reduce to these two methods.
