# Containers and data structures

<p class="lead">Pick the right data structure and the code gets shorter, faster and clearer at once. This chapter covers the built-in containers' more advanced uses, their performance characteristics, and <code>collections</code>, <code>heapq</code> and <code>bisect</code>, three tools that get overlooked.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the time complexity of `x in some_list` and of `x in some_set`?
    2. How do you sort a list of dicts "by score descending, then by name ascending for equal scores"?
    3. After `first, *rest = [1]`, what is `rest`?
    4. What does each of `defaultdict(list)` and `dict.setdefault` suit?
    5. What is the best fit for a buffer that "keeps only the last 100 entries"?

??? success "Answers (try it yourself first, then expand)"
    1. A `list` is O(n) and compares element by element; a `set` is O(1) on average and goes straight there by hash.
    2. `sorted(rows, key=lambda r: (-r["score"], r["name"]))`; when the score is not a number, sort by name ascending first and then by score descending (the sort is stable).
    3. An empty list, `[]`.
    4. `defaultdict(list)` suits appending into groups repeatedly (`groups[k].append(x)`), creating the entry automatically when the key is missing; `setdefault` suits an ordinary dict that needs a default occasionally, or where the dict's type should not change.
    5. `collections.deque(maxlen=100)`: once full it drops the oldest from the other end automatically, and both ends are O(1).

## Complexity at a glance {#复杂度速查}

Move n along the curves from the computer fundamentals handbook to see how much the complexities really differ:

<div class="aig-widget" data-widget="complexity"></div>

This table belongs in your head while writing code. The commonest performance problem is an `in` on a `list` or a `pop(0)` inside a loop.

| Operation | `list` | `dict` / `set` | `collections.deque` |
| --- | --- | --- | --- |
| index `x[i]` | O(1) | — | O(n) (O(1) at the ends) |
| append / pop at the end | O(1) | — | O(1) |
| insert / pop at the front | **O(n)** | — | O(1) |
| membership `x in c` | **O(n)** | O(1) | O(n) |
| insert / remove anywhere | O(n) | O(1) | O(n) |
| sort | O(n log n) | — | — |

!!! tip "Rules of thumb"
    - testing "is it in there" repeatedly: convert to a `set` first.
    - pushing and popping at both ends (a queue, a sliding window): use a `deque`.
    - looking up by key: use a `dict` rather than searching a list linearly.

## Sequences, beyond the basics {#序列操作进阶}

### Slicing {#切片}

A slice `seq[start:stop:step]` always returns a **new object** (a shallow copy) and never raises for being out of range.

```pycon
>>> s = "abcdefgh"
>>> s[::2], s[::-1], s[-3:]
('aceg', 'hgfedcba', 'fgh')
>>> nums = list(range(10))
>>> nums[2:5] = ["x"]          # slice assignment: the lengths may differ
>>> nums
[0, 1, 'x', 5, 6, 7, 8, 9]
>>> del nums[::2]              # remove the even positions
>>> nums
[1, 5, 7, 9]
```

A slice can be named, which makes the code explain itself. That helps when parsing fixed-width data:

```pycon
>>> record = "20260924ERROR  disk full"
>>> DATE, LEVEL, MSG = slice(0, 8), slice(8, 15), slice(15, None)
>>> record[DATE], record[LEVEL].strip(), record[MSG]
('20260924', 'ERROR', 'disk full')
```

### Unpacking {#解包}

```pycon
>>> first, *middle, last = [1, 2, 3, 4, 5]
>>> first, middle, last
(1, [2, 3, 4], 5)
>>> first, *rest = [1]
>>> rest
[]
>>> a, b = 1, 2
>>> a, b = b, a                # a swap, with no temporary
>>> a, b
(2, 1)
>>> for name, (x, y) in [("p1", (0, 1)), ("p2", (3, 4))]:
...     print(name, x + y)
...
p1 1
p2 7
```

`*` and `**` also merge data inside a literal:

```pycon
>>> [*range(3), *"ab"]
[0, 1, 2, 'a', 'b']
>>> defaults = {"host": "localhost", "port": 80}
>>> {**defaults, "port": 8080}
{'host': 'localhost', 'port': 8080}
```

### Sorting {#排序}

`sorted()` returns a new list and `list.sort()` sorts in place and returns `None`. Both are **stable sorts**: elements with equal keys keep their original relative order.

```pycon
>>> users = [
...     {"name": "bob", "score": 90},
...     {"name": "amy", "score": 95},
...     {"name": "cat", "score": 90},
... ]
>>> [u["name"] for u in sorted(users, key=lambda u: (-u["score"], u["name"]))]
['amy', 'bob', 'cat']
```

A tuple key gives a multi-level sort, and negating a numeric field gives a descending one. When a field cannot be negated (a string to be sorted descending, say), use the stability and sort twice: by the secondary key first, then by the primary.

```pycon
>>> from operator import itemgetter
>>> rows = [("b", 2), ("a", 2), ("c", 1)]
>>> rows.sort(key=itemgetter(0), reverse=True)   # the secondary key: name descending
>>> rows.sort(key=itemgetter(1))                 # the primary key: the number ascending
>>> rows
[('c', 1), ('b', 2), ('a', 2)]
```

`min` and `max` take a `key` too, and a `default` for an empty sequence:

```pycon
>>> max(users, key=lambda u: u["score"])["name"]
'amy'
>>> max([], default=None) is None
True
```

### `enumerate` and `zip` {#enumerate-与-zip}

```pycon
>>> for i, ch in enumerate("abc", start=1):
...     print(i, ch)
...
1 a
2 b
3 c
>>> names, scores = ["amy", "bob"], [95, 90]
>>> dict(zip(names, scores))
{'amy': 95, 'bob': 90}
>>> list(zip(*[(1, "a"), (2, "b")]))       # "unzipping": a transpose
[(1, 2), ('a', 'b')]
```

`zip` stops when the shortest sequence ends by default, which **silently drops data** when the lengths disagree. Where both sides should be the same length, add `strict=True`:

```pycon
>>> list(zip([1, 2, 3], "ab", strict=True))
Traceback (most recent call last):
  ...
ValueError: zip() argument 2 is shorter than argument 1
```

## Comprehensions {#推导式}

The comprehension is one of Python's most distinctive pieces of syntax. There is one each for lists, dicts and sets, and a generator expression uses parentheses.

```pycon
>>> words = ["apple", "Bob", "cat", "Apple"]
>>> [w.lower() for w in words if len(w) > 3]
['apple', 'apple']
>>> {w.lower() for w in words}  == {"apple", "bob", "cat"}
True
>>> {w: len(w) for w in words}
{'apple': 5, 'Bob': 3, 'cat': 3, 'Apple': 5}
>>> sum(len(w) for w in words)            # a generator expression: no intermediate list
16
```

Nested loops come in the same order as the `for` statements would:

```pycon
>>> matrix = [[1, 2, 3], [4, 5, 6]]
>>> [x for row in matrix for x in row]              # flatten
[1, 2, 3, 4, 5, 6]
>>> [[row[i] for row in matrix] for i in range(3)]  # transpose
[[1, 4], [2, 5], [3, 6]]
```

!!! warning "A longer comprehension is not a better one"
    Beyond two levels of loop, or with a complicated condition, go back to an ordinary `for` loop or split a function out. A comprehension is only for **building a collection** and should never be written for a side effect (`[print(x) for x in xs]`).

## Dict techniques {#字典技巧}

From 3.7, a dict **keeps its insertion order**, which is part of the language specification.

```pycon
>>> stock = {"apple": 3}
>>> stock.get("pear", 0)                 # a default when the key is absent
0
>>> stock.setdefault("pear", 0)          # inserted and returned when absent
0
>>> stock | {"apple": 5, "kiwi": 1}      # merged, the right side winning
{'apple': 5, 'pear': 0, 'kiwi': 1}
>>> stock |= {"kiwi": 2}                 # merged in place
>>> stock
{'apple': 3, 'pear': 0, 'kiwi': 2}
```

The idiomatic form of a few common operations:

```pycon
>>> scores = {"amy": 95, "bob": 90, "cat": 90}
>>> {v: k for k, v in scores.items()}              # inverted (a repeated value is overwritten by the later one)
{95: 'amy', 90: 'cat'}
>>> sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:2]   # the top 2 by value
[('amy', 95), ('bob', 90)]
>>> scores.keys() & {"amy", "dan"}                 # the keys view supports set operations
{'amy'}
```

!!! warning "Do not add or remove keys while iterating"
    Adding or removing a key of `d` inside `for k in d:` raises `RuntimeError: dictionary changed size during iteration`. To delete, iterate over a copy, `for k in list(d):`, or build a new dict with a comprehension.

## `collections`: the specialized containers {#collections专用容器}

### `Counter`: counting {#counter计数}

```pycon
>>> from collections import Counter
>>> c = Counter("mississippi")
>>> c.most_common(2)
[('i', 4), ('s', 4)]
>>> c["z"]                     # a missing key gives 0 rather than an error
0
>>> c.update("ssz")
>>> c["s"], c["z"]
(6, 1)
>>> Counter(a=3, b=1) - Counter(a=1, b=2)    # subtraction keeps only the positive counts
Counter({'a': 2})
```

### `defaultdict`: automatic initialization {#defaultdict自动初始化}

Especially handy for grouping, building an index and constructing an adjacency list:

```pycon
>>> from collections import defaultdict
>>> words = ["apple", "avocado", "banana", "blueberry", "cherry"]
>>> by_letter = defaultdict(list)
>>> for w in words:
...     by_letter[w[0]].append(w)
...
>>> dict(by_letter)
{'a': ['apple', 'avocado'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}
```

A `defaultdict` inserts the default value when **reading** a missing key too, which is sometimes a surprise. To query only, use `d.get(k)`.

### `deque`: a double-ended queue {#deque双端队列}

O(1) at both ends. With a `maxlen` it becomes a ring buffer that discards the old data automatically:

```pycon
>>> from collections import deque
>>> recent = deque(maxlen=3)
>>> for line in ["a", "b", "c", "d", "e"]:
...     recent.append(line)
...
>>> recent
deque(['c', 'd', 'e'], maxlen=3)
>>> q = deque([1, 2, 3])
>>> q.appendleft(0); q.pop()
3
>>> q.rotate(1)
>>> q
deque([2, 0, 1])
```

### `ChainMap`: layered lookup {#chainmap分层查找}

Looks through several dicts in order, which suits layered configuration like "command-line arguments > environment variables > defaults":

```pycon
>>> from collections import ChainMap
>>> defaults = {"debug": False, "port": 80}
>>> env = {"port": 8080}
>>> cli = {"debug": True}
>>> cfg = ChainMap(cli, env, defaults)
>>> cfg["debug"], cfg["port"]
(True, 8080)
```

### `OrderedDict`: when the order has to move {#ordereddict需要移动顺序时}

An ordinary dict is already ordered, and `OrderedDict` is now mainly for the cases that need `move_to_end` and `popitem(last=False)`, an LRU cache say (see the exercises).

`namedtuple` is covered in [modelling data](data-classes.md).

## `heapq`: a priority queue and top-K {#heapq优先队列与-top-k}

`heapq` maintains a **min-heap** over an ordinary list: `heap[0]` is always the smallest and pushing and popping are O(log n).

```pycon
>>> import heapq
>>> tasks = []
>>> heapq.heappush(tasks, (2, "write docs"))
>>> heapq.heappush(tasks, (1, "fix bug"))
>>> heapq.heappush(tasks, (3, "refactor"))
>>> heapq.heappop(tasks)
(1, 'fix bug')
>>> heapq.nlargest(2, [5, 1, 8, 3, 9])
[9, 8]
>>> heapq.nsmallest(1, [{"n": "a", "p": 3}, {"n": "b", "p": 1}], key=lambda d: d["p"])
[{'n': 'b', 'p': 1}]
```

!!! tip "Comparison when the priorities are equal"
    Comparing tuples moves on to the second element when the priorities are equal. An incomparable second element (a dict, say) then raises. The usual way is to insert an increasing sequence number: `(priority, next(counter), item)`.

To take the top K of a large set: `nlargest`/`nsmallest` when K is small, `sorted(...)[:k]` when K approaches n, and `max`/`min` when only one is needed.

`heapq.merge` lazily merges several **already sorted** sequences:

```pycon
>>> list(heapq.merge([1, 4, 7], [2, 5], [3, 6]))
[1, 2, 3, 4, 5, 6, 7]
```

## `bisect`: a binary search in a sorted list {#bisect在有序列表里二分查找}

```pycon
>>> import bisect
>>> def grade(score, cutoffs=(60, 70, 80, 90), grades="FDCBA"):
...     return grades[bisect.bisect_right(cutoffs, score)]
...
>>> [grade(s) for s in (33, 60, 77, 90, 100)]
['F', 'D', 'C', 'A', 'A']
>>> xs = [1, 3, 5]
>>> bisect.insort(xs, 4)         # insert and stay sorted
>>> xs
[1, 3, 4, 5]
```

The `bisect` functions take a `key` too, which allows a binary search over a list of objects by one field.

## How to choose {#怎么选}

| Need | Choice |
| --- | --- |
| ordered, duplicates allowed, accessed by position | `list` |
| an immutable group of values, or a dict key | `tuple` |
| deduplication, fast membership, set operations | `set` / `frozenset` |
| lookup by key | `dict` |
| counting | `Counter` |
| grouping, building an index | `defaultdict(list)` |
| a queue, a sliding window, the last N | `deque` |
| taking the smallest or largest repeatedly | `heapq` |
| searching and inserting in a sorted list | `bisect` |
| many numbers of one type | `array.array`, or NumPy from outside the standard library |

!!! interview "How to explain it"
    Start with complexity: a `list`'s `in` and its insertions and removals at the front are O(n), while `set` / `dict` lookup is O(1) on average, so deduplication and membership go to a set; the sort is stable and a multi-level sort uses a tuple key (with numbers negated for descending). Then choose a specialized container by the case: `Counter` for counting, `defaultdict(list)` for grouping, `deque(maxlen=...)` for a queue and a sliding window, `heapq.nlargest` for top-K (or keeping a min-heap of size K), `bisect` for a sorted lookup. Examples in an inference service: the scheduler's waiting queue is a `deque`, dequeuing by priority is a heap, and LRU is an ordered dict.

## Exercises {#练习}

**1. Word frequency.** Given a passage of English text, ignoring case and punctuation, return the 3 most frequent words and their counts.

??? success "Answer"
    ```python
    import re
    from collections import Counter

    def top_words(text, n=3):
        words = re.findall(r"[a-z']+", text.lower())
        return Counter(words).most_common(n)

    text = "The cat and the hat. The cat sat! And then? The end."
    assert top_words(text) == [("the", 4), ("cat", 2), ("and", 2)]
    ```

    `most_common` orders ties by first appearance, which is why `cat` comes before `and`.

**2. Grouping anagrams.** Turn `["eat", "tea", "tan", "ate", "nat", "bat"]` into `[["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]`.

??? success "Answer"
    ```python
    from collections import defaultdict

    def group_anagrams(words):
        groups = defaultdict(list)
        for w in words:
            groups["".join(sorted(w))].append(w)
        return list(groups.values())

    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [
        ["eat", "tea", "ate"], ["tan", "nat"], ["bat"]
    ]
    ```

    The sorted string is the "fingerprint", and equal fingerprints are anagrams. A dict keeps its insertion order, so the groups' order is deterministic too.

**3. The sliding window maximum.** Given a list `nums` and a window size `k`, return each window's maximum in O(n). For `[1, 3, -1, -3, 5, 3, 6, 7]` with `k=3` that is `[3, 3, 5, 5, 6, 7]`.

??? success "Answer"
    Keep the **indices** in a `deque` with their values decreasing monotonically. The front is always the current window's maximum.

    ```python
    from collections import deque

    def max_sliding_window(nums, k):
        dq, out = deque(), []
        for i, x in enumerate(nums):
            while dq and nums[dq[-1]] <= x:   # an element smaller than x can never be the maximum again
                dq.pop()
            dq.append(i)
            if dq[0] <= i - k:                # the front has slid out of the window
                dq.popleft()
            if i >= k - 1:
                out.append(nums[dq[0]])
        return out

    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    ```

    Each index enters and leaves the queue at most once, so the whole thing is O(n).

**4. An LRU cache with `OrderedDict`.** Support `get(key)` (returning `None` when absent) and `put(key, value)`, evicting the least recently used key when it is full.

??? success "Answer"
    ```python
    from collections import OrderedDict

    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self._data = OrderedDict()

        def get(self, key):
            if key not in self._data:
                return None
            self._data.move_to_end(key)          # mark it as most recently used
            return self._data[key]

        def put(self, key, value):
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)   # pop the least recently used

    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")          # a becomes the most recently used
    cache.put("c", 3)       # b is evicted
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3
    ```

    To cache a function in a real project, use `functools.lru_cache` directly; see [functions in depth](functions.md).

## Summary {#小结}

- [x] Remember the complexities: a `list`'s `in` and front operations are O(n), and `set`/`dict` lookup is O(1).
- [x] A multi-level sort uses a tuple key; the sort is stable.
- [x] `zip(strict=True)` keeps data from being dropped silently.
- [x] Counting goes to `Counter`, grouping to `defaultdict`, queues to `deque` and top-K to `heapq`.
- [x] A comprehension builds a collection; once it gets complicated, go back to a loop.
