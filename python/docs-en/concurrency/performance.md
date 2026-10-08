# Profiling and optimization

<p class="lead">The sentence after "premature optimization is the root of all evil" is "yet we should not pass up our opportunities in that critical 3%". This chapter covers finding that 3%: measure first, locate the hot spots, then optimize the right way. Most of the time the most effective optimization is a better data structure or algorithm.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why time a small piece of code with `timeit` rather than by subtracting `time.time()`?
    2. How do you find a program's most time-consuming function?
    3. What is wrong with building a string with `+=` in a loop?
    4. How do you find which line allocates the most memory?
    5. A function is called a million times and is fast each time, yet the whole thing is slow. What are the ways to optimize it?

??? success "Answers (try it yourself first, then expand)"
    1. `timeit` repeats many times and takes a statistic, uses the high-resolution `perf_counter`, and disables garbage collection by default, which removes the noise of a single measurement and the timer's resolution.
    2. Run it under `cProfile` (`python -m cProfile -s cumtime`) and read the most expensive functions by cumulative time; a production process takes py-spy's sampling, with no code change and no restart.
    3. A string is immutable, so every `+=` may create a new string and copy what is already there, and n concatenations copy O(n²) characters in total. Collect the pieces in a list and `"".join(parts)` at the end.
    4. Use `tracemalloc`: take a snapshot after `start()` and use `snapshot.statistics("lineno")` for the memory allocated per source line.
    5. Call it fewer times (a different algorithm or data structure, avoiding repeated work, adding a cache); hand the loop to a built-in function, the standard library or a vectorized library like numpy; localize frequently accessed attributes and globals; batch the work; and only then consider a compiled extension or several processes.

## The order of optimization {#优化的顺序}

![Figure: the order of optimization - measure before touching anything](../assets/figures/optimize-order.svg){.aig-svg}

1. **Make it correct first.** Optimization comes after there are tests to fall back on.
2. **Measure, do not guess.** A programmer's intuition about where the hot spot is is often wrong.
3. **Change the algorithm and the data structure first.** Turning O(n²) into O(n) beats any micro-optimization.
4. **Use the efficient implementations that exist.** The built-in functions, the standard library and vectorized libraries like NumPy and Polars are all written in C.
5. **Only then consider micro-optimization**, or a different tool (several processes, Cython, a Rust extension, PyPy).

## Measuring: `timeit` {#测量timeit}

`timeit` runs the code many times and takes a statistic, chooses a suitable loop count automatically, and removes garbage collection's interference, which is far more reliable than timing by hand.

From the command line it is easiest:

```bash
python -m timeit -s "data = list(range(10_000))" "9_999 in data"
# an example: 5000 loops, best of 5: 45.2 usec per loop
python -m timeit -s "data = set(range(10_000))" "9_999 in data"
# an example: 10000000 loops, best of 5: 21.3 nsec per loop
```

In code, `timeit.timeit` or `timeit.repeat`:

```python
import timeit

setup = "data = list(range(10_000)); s = set(data)"
t_list = min(timeit.repeat("9_999 in data", setup=setup, number=1_000, repeat=5))
t_set = min(timeit.repeat("9_999 in s", setup=setup, number=1_000, repeat=5))
print(f"list: {t_list * 1e3:.3f} ms, set: {t_set * 1e3:.3f} ms per 1000 lookups")
assert t_set * 50 < t_list          # a set's lookup is orders of magnitude faster
```

Take the **minimum** of the repetitions: it is closest to the code's own speed, and the larger values reflect interference from the other processes on the system.

For a longer stretch, time it with `time.perf_counter()` (high resolution and monotonic) rather than `time.time()` (which is affected by adjustments to the system clock).

## Finding the hot spots: `cProfile` {#定位热点cprofile}

`cProfile` records how often each function was called and how long it took. The simplest use is from the command line:

```bash
python -m cProfile -s cumtime my_script.py | head -20
```

It can also profile one stretch from inside the code:

```python
import cProfile
import io
import pstats

def parse(lines):
    return [line.split(",") for line in lines]

def slow_dedupe(rows):
    unique = []
    for r in rows:
        if r not in unique:                 # a list's in: O(n), O(n^2) overall
            unique.append(r)
    return unique

def pipeline():
    lines = [f"{i % 500},{i % 7}" for i in range(20_000)]
    return slow_dedupe(parse(lines))

profiler = cProfile.Profile()
with profiler:                               # a Profile object works as a context manager
    result = pipeline()

out = io.StringIO()
pstats.Stats(profiler, stream=out).sort_stats("cumulative").print_stats(5)
report = out.getvalue()
print(report)
assert "slow_dedupe" in report
```

The key columns of the report:

| Column | Meaning |
| --- | --- |
| `ncalls` | the number of calls |
| `tottime` | the time in the function **itself** (excluding what it called) |
| `cumtime` | the cumulative time (including everything it called) |

Sort by `cumtime` first to find "which call chain is slow", then by `tottime` to find "which function is slow".

!!! tip "Better third-party profilers"
    - **py-spy**: a sampling profiler that attaches to **a running** process (`py-spy top --pid 1234`) with no code change, and can produce flame graphs. The first choice in production.
    - **Scalene**: profiles CPU, memory and GPU at once and distinguishes time spent in Python code from time in C.
    - **line_profiler**: times line by line, finding the slow statements inside a function.
    - **memray**: Bloomberg's memory profiler, which traces every allocation.

    Python 3.15 will include a statistical sampling profiler, `profiling.sampling`, whose overhead is low enough for production.

## Memory: `tracemalloc` {#内存分析tracemalloc}

```python
import tracemalloc

tracemalloc.start()

big_list = [str(i) * 10 for i in range(100_000)]      # the line that allocates a lot of memory
small = {i: i for i in range(100)}

snapshot = tracemalloc.take_snapshot()
top = snapshot.statistics("lineno")[0]                # counted per source line, taking the largest
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(top)
print(f"current={current / 1e6:.1f} MB, peak={peak / 1e6:.1f} MB")
assert current > 5_000_000
```

`statistics("lineno")` tells you which line allocated the most memory. To find a leak, take two snapshots at different times and read `snapshot2.compare_to(snapshot1, "lineno")` for where the growth is.

## The common optimizations {#常见的优化手段}

### Choosing the right data structure {#选对数据结构}

This is the class of optimization that pays most, and the complexity table in [containers and data structures](../core/containers.md#复杂度速查) belongs in your head:

```python
import timeit

def common_slow(a, b):
    return [x for x in a if x in b]          # b is a list: O(len(a) * len(b))

def common_fast(a, b):
    b_set = set(b)                           # build the set once: O(len(b))
    return [x for x in a if x in b_set]      # each lookup is O(1)

a = list(range(0, 4_000, 2))
b = list(range(0, 4_000, 3))
assert common_slow(a, b) == common_fast(a, b)

t_slow = timeit.timeit(lambda: common_slow(a, b), number=3)
t_fast = timeit.timeit(lambda: common_fast(a, b), number=3)
print(f"speedup: {t_slow / t_fast:.0f}x")
assert t_fast * 20 < t_slow
```

### Build strings with `join` {#字符串拼接用-join}

A string is immutable, so `s += piece` in a loop creates a new string each time in principle, which is O(n²) overall. CPython optimizes this case to a degree, but it cannot be relied on. The standard form collects the pieces in a list and `"".join(parts)`s them, or writes a generator expression directly:

```python
rows = [("amy", 95), ("bob", 90)]
csv_text = "\n".join(f"{name},{score}" for name, score in rows)
assert csv_text == "amy,95\nbob,90"
```

To write a lot of text, use an `io.StringIO` or write to the file line by line.

### Use the built-in functions and the standard library {#用内置函数和标准库}

The built-in functions and most of the standard library are implemented in C and far faster than the equivalent Python loop:

| Rather than a loop by hand | Use |
| --- | --- |
| summing | `sum()`, `math.fsum()`, `math.prod()` |
| the maximum or minimum | `max()`, `min()`, `heapq.nlargest()` |
| testing existence | `any()`, `all()` |
| counting | `collections.Counter`, `str.count()` |
| grouping after sorting | `itertools.groupby` |
| finding a substring | `in`, `str.find()` rather than comparing character by character |

### Avoid repeated work {#避免重复计算}

- **Move the invariant computation out of the loop**: `re.compile` a regular expression beforehand; compute the constants a loop uses in advance.
- **Cache a pure function's result**: `functools.cache`, `cached_property`.
- **Be lazy**: when only the first few results are needed, use a generator with `itertools.islice`, or `next()` for the first element that matches, rather than building the whole list.

```python
import re

LOG_LINE = re.compile(r"(?P<level>ERROR|WARN) (?P<msg>.*)")     # compiled once

def first_error(lines):
    return next((m["msg"] for line in lines if (m := LOG_LINE.match(line)) and m["level"] == "ERROR"), None)

assert first_error(["INFO ok", "WARN disk 80%", "ERROR disk full", "ERROR again"]) == "disk full"
```

### Use less memory {#减少内存占用}

- Use a generator in place of a list and process the data as a stream, see [iterators and generators](../core/iterators.md).
- Many small objects take `__slots__` or `@dataclass(slots=True)`.
- Many numbers of one type take an `array.array` or a NumPy array rather than a `list`.

### Numerics: vectorize {#数值计算向量化}

A Python loop over the elements of a numeric array is one to two orders of magnitude slower than NumPy's vectorized operations:

```py
import numpy as np

prices = np.random.rand(1_000_000)
# slow: a Python loop
total = sum(p * 1.1 for p in prices)
# fast: one vectorized expression, with the loop done in C
total = (prices * 1.1).sum()
```

Tabular data goes to pandas or Polars, where a Python loop over the rows (`iterrows`, or complicated logic inside `apply`) is likewise to be avoided.

## When that is still not fast enough {#还不够快怎么办}

| Approach | When it fits |
| --- | --- |
| several processes / free-threaded Python | CPU-bound work that splits into independent tasks, see [threads, processes and the GIL](threads-processes.md) |
| PyPy | a long-running pure Python program, where the JIT gives several times the speed; its C extension support is weaker |
| Cython / mypyc | annotate a hot module and compile it into a C extension |
| Rust (PyO3 + maturin) or a C/C++ extension | a performance-critical core module; ruff, pydantic-core and Polars all do this |

CPython itself keeps getting faster: 3.11 averaged about 25% quicker and later versions have kept optimizing; from 3.13 there is an experimental JIT as well. **Upgrading the Python version is itself a cheap optimization.**

!!! interview "How to explain it"
    The order to explain performance in: measure before optimizing, with `timeit` for a small piece (many runs, with GC's interference removed), `cProfile` for the whole program's hot spots, py-spy's sampling for a production process, and `tracemalloc` to find the line that allocates most; the biggest gains come from the algorithm and the data structure (a `list`'s `in` becoming a `set`, O(n²) becoming O(n)); then the built-in functions and the standard library, `join` for building strings, and fewer attribute lookups and function calls; numerics get vectorized, CPU-bound work gets several processes, and Cython or a C extension comes last. In an inference engine, Python's own cost (a few hundred microseconds of scheduling and batching per step) shows up directly in TPOT, which is why vLLM and SGLang both overlap scheduling and move the hot path into C++ / Rust.

!!! info "Related chapters"
    - [Linux profiling tools](cs://os/perf-tools/) (computer fundamentals)
    - [Profiling an inference engine](serving://perf/profiling/) (inference systems: what Python's cost looks like inside an engine)

## Exercises {#练习}

**1. Find and fix the hot spot.** The function below counts how many lines each word appears in. Confirm where it is slow with `cProfile` or `timeit` first, then make it at least 10 times faster and verify the result is unchanged.

```py
def word_line_counts(lines):
    words = []
    for line in lines:
        for w in line.split():
            if w not in words:
                words.append(w)
    result = {}
    for w in words:
        result[w] = sum(1 for line in lines if w in line.split())
    return result
```

??? success "Answer"
    ```python
    import timeit
    from collections import Counter

    def word_line_counts(lines):
        words = []
        for line in lines:
            for w in line.split():
                if w not in words:
                    words.append(w)
        result = {}
        for w in words:
            result[w] = sum(1 for line in lines if w in line.split())
        return result

    def word_line_counts_fast(lines):
        counts = Counter()
        for line in lines:
            counts.update(set(line.split()))     # deduplicate each line and count: one pass
        return dict(counts)

    lines = [" ".join(f"w{(i * j) % 97}" for j in range(8)) for i in range(400)]
    assert word_line_counts(lines) == word_line_counts_fast(lines)

    slow = timeit.timeit(lambda: word_line_counts(lines), number=1)
    fast = timeit.timeit(lambda: word_line_counts_fast(lines), number=1)
    assert slow > 10 * fast
    ```

    The original has two problems: `words` is a list, so `in` is O(n); worse, it re-`split`s every line for every word, which is O(words × lines × line length). The optimized version walks every line once. The two dicts' key order matches too, because `Counter` likewise inserts in order of first appearance.

**2. Measuring memory.** Create 100,000 instances of an ordinary class and 100,000 of a `@dataclass(slots=True)` (3 fields each) and compare their memory with `tracemalloc`.

??? success "Answer"
    ```python
    import tracemalloc
    from dataclasses import dataclass

    class Plain:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    @dataclass(slots=True)
    class Slotted:
        x: int
        y: int
        z: int

    def measure(cls, n=100_000):
        tracemalloc.start()
        objs = [cls(i, i, i) for i in range(n)]
        size, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return size, objs

    plain_size, _ = measure(Plain)
    slotted_size, _ = measure(Slotted)
    print(f"plain {plain_size / 1e6:.1f} MB, slots {slotted_size / 1e6:.1f} MB")
    assert slotted_size < plain_size
    ```

    `__slots__` removes each instance's `__dict__`. Recent CPython versions have optimized ordinary instances' attribute storage a good deal, so the gap is smaller than it used to be, but it is still clear when the objects are numerous.

## Summary {#小结}

- [x] Measure before optimizing: `timeit` for a small piece, `cProfile` for the whole program, py-spy in production, `tracemalloc` for memory.
- [x] The biggest gains come from the algorithm and the data structure: a `list`'s `in` becoming a `set`, O(n²) becoming O(n).
- [x] Use the built-in functions and the standard library; build strings with `join`; compile regular expressions in advance.
- [x] Numerics go to a vectorized library; CPU-bound work goes to several processes; a compiled extension comes last.
- [x] Upgrading the Python version alone brings free performance.
