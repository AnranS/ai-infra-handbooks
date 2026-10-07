# Iterators and generators

<p class="lead">Behind the <code>for</code> loop is a simple protocol. Once you have it, you can write data pipelines whose memory use is constant, compress a dozen lines of loop into one with <code>itertools</code>, and read asyncio's ancestor.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is the difference between an iterable and an iterator?
    2. Why does iterating the same generator twice give nothing the second time?
    3. What does `yield from` do?
    4. How do `sum([x * x for x in data])` and `sum(x * x for x in data)` differ?
    5. When does `itertools.groupby` give "wrong" groups?

??? success "Answers (try it yourself first, then expand)"
    1. An iterable can produce an iterator through `iter()` (it implements `__iter__`) and can be traversed many times; an iterator implements `__next__`, remembers how far it got, and is single-use.
    2. A generator is an iterator, and the first traversal exhausted it, so the second `next` is an immediate `StopIteration`. Either call the generator function again, or store the results in a list first.
    3. It `yield`s the values another iterable produces one by one (the equivalent of `for x in sub: yield x`), and it also passes `send`, `throw` and the subgenerator's return value through correctly.
    4. The first builds a complete list and then sums it, taking O(n) memory; the second is a generator expression consumed as it is produced, in constant memory.
    5. It groups only **adjacent** equal keys; when the data is not sorted by that key, the same key is split across several groups. Sort by the same key first.

## The iteration protocol {#迭代协议}

- **an iterable**: implements `__iter__`, and `iter(obj)` gives an iterator. Lists, strings, dicts and files all are.
- **an iterator**: implements `__next__` (returning the next element each time and raising `StopIteration` when there is none), and its `__iter__` returns itself.

What `for x in obj` really does is:

```python
it = iter([10, 20])
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    print(x)
```

```pycon
>>> nums = [1, 2]
>>> it = iter(nums)
>>> next(it), next(it)
(1, 2)
>>> next(it)
Traceback (most recent call last):
  ...
StopIteration
>>> next(it, "done")          # a default stops it raising
'done'
```

**An iterator is single-use.** A list gives a new iterator from every `iter()`, so it can be traversed repeatedly; but the iterator itself is empty once traversed:

```pycon
>>> squares = (x * x for x in range(3))     # a generator is an iterator too
>>> list(squares)
[0, 1, 4]
>>> list(squares)
[]
```

This is a very common source of bugs: storing the result of `map`, `filter`, `zip` or a generator in a variable and using it twice. Where several traversals are needed, `list()` it first.

## Implementing an iterator yourself {#自己实现迭代器}

Doing it with a class means maintaining state and two methods:

```python
class Countdown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

assert list(Countdown(3)) == [3, 2, 1]
assert list(Countdown(3)) == [3, 2, 1]     # every iter() is a new generator, so it can be traversed repeatedly
```

There is a shortcut here: `__iter__` is itself written as a generator function. That is the recommended way to implement an **iterable container**, far simpler than writing `__next__` by hand and naturally supporting repeated traversal.

![Figure: a generator pipeline and a generator's state](../assets/figures/generator-pipeline.svg){.aig-svg}

## Generator functions {#生成器函数}

With a `yield` in its body, calling the function does not run the body but returns a **generator object**. Each `next()` runs to the next `yield`, pauses and hands the value over; the locals are all kept and the next call resumes where it paused.

```pycon
>>> def fib():
...     a, b = 0, 1
...     while True:           # an infinite sequence is fine too: computed on demand
...         yield a
...         a, b = b, a + b
...
>>> gen = fib()
>>> [next(gen) for _ in range(10)]
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

### Why use a generator: laziness {#为什么要用生成器惰性求值}

A list puts every element in memory at once; a generator produces one at a time.

```pycon
>>> import sys
>>> sys.getsizeof([x for x in range(1_000_000)]) > 8_000_000
True
>>> sys.getsizeof(x for x in range(1_000_000)) < 500
True
```

So when the result goes straight to a consumer like `sum`, `max`, `any` or `"".join`, a generator expression (dropping the square brackets) is all that is needed:

```py
total = sum(x * x for x in data)
has_error = any("ERROR" in line for line in lines)   # stops at the first one found
```

## Building a data pipeline with generators {#用生成器搭建数据管道}

Generators chain together like Unix pipes, each step handling one line, with memory use independent of the file's size. Below, an `io.StringIO` stands in for a log file:

```python
import io

LOG = io.StringIO("""\
2026-09-24 10:00:01 INFO  user=amy action=login
2026-09-24 10:00:05 ERROR user=bob action=pay msg=timeout
# comment line
2026-09-24 10:01:10 ERROR user=amy action=pay msg=declined
2026-09-24 10:02:00 INFO  user=cat action=logout
""")

def read_lines(f):
    for line in f:
        yield line.rstrip("\n")

def skip_comments(lines):
    return (ln for ln in lines if ln and not ln.startswith("#"))

def parse(lines):
    for ln in lines:
        date, time, level, *pairs = ln.split()
        yield {"level": level, **dict(p.split("=", 1) for p in pairs)}

def only(level, records):
    return (r for r in records if r["level"] == level)

errors = only("ERROR", parse(skip_comments(read_lines(LOG))))
assert [e["user"] for e in errors] == ["bob", "amy"]
```

Each function does one thing and can be tested on its own and combined freely. Not one line is read until the whole pipeline is consumed at the end.

## `yield from`: delegating to a subgenerator {#yield-from委托给子生成器}

`yield from iterable` is the equivalent of `for x in iterable: yield x`, and it also passes `send()`, exceptions and the subgenerator's return value through correctly. Its commonest use is a recursive generator:

```pycon
>>> def flatten(items):
...     for x in items:
...         if isinstance(x, (list, tuple)):
...             yield from flatten(x)
...         else:
...             yield x
...
>>> list(flatten([1, [2, [3, (4, 5)]], 6]))
[1, 2, 3, 4, 5, 6]
```

## A generator's other abilities {#生成器的其他能力}

A generator also supports `send(value)` (passing a value into the pause), `throw(exc)` (raising at the pause) and `close()`. That is how Python implemented coroutines early on. Asynchronous code is now written with `async`/`await` (see [asyncio](../concurrency/asyncio.md)) and `send` is rarely used directly, so knowing of it is enough.

One detail that is still used often: when a generator is closed or reclaimed, a `GeneratorExit` is raised at the pause, so the cleanup in a `try/finally` always runs. `contextlib.contextmanager` rests on exactly that, see [exceptions and context managers](errors-context.md).

## `itertools`: the iterator toolbox {#itertools迭代器工具箱}

Every function in `itertools` returns an iterator: lazy, memory-thrifty, implemented in C and fast.

### Infinite iterators {#无限迭代器}

```pycon
>>> from itertools import count, cycle, repeat, islice
>>> list(islice(count(10, 5), 4))          # 10, 15, 20, ...
[10, 15, 20, 25]
>>> list(islice(cycle("AB"), 5))
['A', 'B', 'A', 'B', 'A']
>>> list(zip("abc", repeat(0)))
[('a', 0), ('b', 0), ('c', 0)]
```

`islice` is how an iterator is sliced (an iterator does not support `[a:b]`).

### Chaining, splitting, filtering {#连接切分过滤}

```pycon
>>> from itertools import chain, batched, pairwise, takewhile, dropwhile, compress
>>> list(chain([1, 2], (3,), "ab"))
[1, 2, 3, 'a', 'b']
>>> list(chain.from_iterable([[1, 2], [3]]))       # flatten one level
[1, 2, 3]
>>> list(batched(range(7), 3))                     # batching, 3.12+
[(0, 1, 2), (3, 4, 5), (6,)]
>>> list(pairwise([1, 4, 9, 16]))                  # pair up the neighbours
[(1, 4), (4, 9), (9, 16)]
>>> [b - a for a, b in pairwise([1, 4, 9, 16])]    # the differences
[3, 5, 7]
>>> list(takewhile(lambda x: x < 3, [1, 2, 5, 1]))
[1, 2]
>>> list(dropwhile(lambda x: x < 3, [1, 2, 5, 1]))
[5, 1]
>>> list(compress("abcd", [1, 0, 1, 0]))
['a', 'c']
```

`batched` <span class="since">3.12+</span> is very handy for bulk database writes and batched API calls; from 3.13 it also takes `strict=True`, requiring the last batch to be full.

### `groupby`: grouping **adjacent** equal keys {#groupby对相邻的相同键分组}

```pycon
>>> from itertools import groupby
>>> data = ["apple", "avocado", "banana", "apricot"]
>>> [(k, list(g)) for k, g in groupby(data, key=lambda w: w[0])]
[('a', ['apple', 'avocado']), ('b', ['banana']), ('a', ['apricot'])]
>>> data.sort(key=lambda w: w[0])
>>> [(k, list(g)) for k, g in groupby(data, key=lambda w: w[0])]
[('a', ['apple', 'avocado', 'apricot']), ('b', ['banana'])]
```

`groupby` merges only **consecutive** equal keys, so the data usually has to be sorted by the same key first. To avoid sorting, group with a `defaultdict(list)`. Note also that each group `g` is an iterator too and has to be used up before moving to the next group.

### Accumulating and combining {#累积与组合}

```pycon
>>> from itertools import accumulate, product, permutations, combinations
>>> list(accumulate([1, 2, 3, 4]))                 # a prefix sum
[1, 3, 6, 10]
>>> list(accumulate([3, 1, 4, 1, 5], max))         # a running maximum
[3, 3, 4, 4, 5]
>>> list(product("ab", [0, 1]))                    # the Cartesian product, in place of nested loops
[('a', 0), ('a', 1), ('b', 0), ('b', 1)]
>>> list(combinations("abc", 2))
[('a', 'b'), ('a', 'c'), ('b', 'c')]
>>> len(list(permutations(range(4), 2)))
12
```

### `tee` and `zip_longest` {#tee-和-zip_longest}

```pycon
>>> from itertools import tee, zip_longest
>>> a, b = tee(iter([1, 2, 3]))       # duplicate one iterator into two
>>> list(a), list(b)
([1, 2, 3], [1, 2, 3])
>>> list(zip_longest("ab", [1], fillvalue="-"))
[('a', 1), ('b', '-')]
```

After a `tee`, stop using the original iterator. If one copy runs far ahead of the other, `tee` has to buffer everything in between, and a plain `list()` is better.

!!! interview "Answering in an interview"
    On iterators: an iterable produces an iterator through `iter()` and an iterator is single-use, which is why the same generator is empty on a second traversal; a generator function produces on demand, which suits writing large-file processing as a pipeline of constant memory; `yield from` delegates the iteration (and `send` and the return value) to a subgenerator; and a generator expression builds no intermediate list when feeding `sum`, `any` or `join`. `groupby` merges only adjacent equal keys, so sort before grouping. Streaming output in an inference service (returning token by token) is the typical use of a generator or an asynchronous generator.

## Exercises {#练习}

**1. A sliding window.** Write a generator `window(iterable, n)` yielding tuples of length `n`. `window("abcde", 3)` yields `('a','b','c')`, `('b','c','d')` and `('c','d','e')`. It has to work on an infinite iterator too.

??? success "Answer"
    ```python
    from collections import deque
    from itertools import count, islice

    def window(iterable, n):
        it = iter(iterable)
        buf = deque(islice(it, n), maxlen=n)
        if len(buf) == n:
            yield tuple(buf)
        for x in it:
            buf.append(x)          # maxlen pushes the oldest element out automatically
            yield tuple(buf)

    assert list(window("abcde", 3)) == [("a", "b", "c"), ("b", "c", "d"), ("c", "d", "e")]
    assert list(window("ab", 3)) == []
    assert next(window(count(), 2)) == (0, 1)
    ```

    The recipes at the end of `itertools`'s official documentation include a `sliding_window` with the same idea.

**2. Order-preserving deduplication.** Write `unique(iterable, key=None)` lazily yielding each element's first appearance, with `key` deciding what counts as a duplicate.

??? success "Answer"
    ```python
    def unique(iterable, key=None):
        seen = set()
        for x in iterable:
            k = x if key is None else key(x)
            if k not in seen:
                seen.add(k)
                yield x

    assert list(unique([3, 1, 3, 2, 1])) == [3, 1, 2]
    assert list(unique(["a", "B", "A", "b"], key=str.lower)) == ["a", "B"]
    ```

    When laziness is unnecessary and the elements are all hashable, `list(dict.fromkeys(items))` is a one-line order-preserving deduplication.

**3. Reading a large file in chunks.** Write `read_chunks(f, size)` reading `size` characters at a time from a file object until it ends. A hint: the built-in `iter` has a little-known two-argument form, `iter(callable, sentinel)`.

??? success "Answer"
    ```python
    import io
    from functools import partial

    def read_chunks(f, size=8192):
        return iter(partial(f.read, size), "")      # stops at an empty string

    f = io.StringIO("abcdefg")
    assert list(read_chunks(f, 3)) == ["abc", "def", "g"]
    ```

    `iter(callable, sentinel)` calls `callable()` repeatedly until the return value equals the sentinel. For a binary file the sentinel becomes `b""`.

**4. An infinite prime generator.** Write `primes()` yielding primes forever. Then use `itertools` to take the first 10 primes in one line, and to count the primes below 100.

??? success "Answer"
    ```python
    from itertools import count, islice, takewhile

    def primes():
        found = []
        for n in count(2):
            if all(n % p for p in takewhile(lambda p: p * p <= n, found)):
                found.append(n)
                yield n

    assert list(islice(primes(), 10)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert sum(1 for _ in takewhile(lambda p: p < 100, primes())) == 25
    ```

    Only the primes already found whose square does not exceed `n` have to be tried as divisors. `sum(1 for _ in ...)` is the idiom for counting an iterator's length.

## Summary {#小结}

- [x] An iterable produces an iterator; an iterator is single-use.
- [x] Writing a container's `__iter__` as a generator function is the simplest way.
- [x] Feeding `sum`/`any`/`join` through a generator expression saves memory.
- [x] Chaining several small generators into a pipeline processes a large file in constant memory.
- [x] Know `islice`, `chain`, `batched`, `pairwise`, `groupby` (sort first), `accumulate` and `product` by heart.
