# CuTe's layout algebra

<p class="lead">Writing a GEMM or a FlashAttention kernel, what goes wrong is usually not the algorithm but the indices: which tile this block moves, which elements this thread holds, how to arrange shared memory without bank conflicts, which position of the matrix each Tensor Core register holds. CuTe, from CUTLASS 3.x on, unifies all of that into one mathematical object, the Layout, and an algebra for combining layouts. This chapter first implements every operation in under two hundred lines of Python (with the same algorithms as CUTLASS's own pycute), then compiles and runs the real CuTe from CUTLASS and compares line by line, and finally uses it to compute the memory efficiency of a thread partition and the register layout of mma.sync.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Where does the layout `(4,(2,4)):(2,(1,8))` map the coordinate (1, 5)? And the one-dimensional coordinate 5?
    2. What does `coalesce` preserve? Why can `(4,8):(1,4)` collapse to `32:1` while `(4,8):(8,1)` cannot?
    3. What decides the shape of a composition A∘B? How would you use composition to get a transposed view of a row-major matrix?
    4. How is `logical_divide(A, B)` built from composition and complement? What do the two modes of the result mean?
    5. What do `local_tile` and `local_partition` each carve out? For a warp reading a column-major fp32 tile, which way should the thread layout run?

??? success "Answers (try it yourself first, then expand)"
    1. Coordinate (1, 5): mode 0 contributes 1 × 2 = 2; the 5 in mode 1 splits into (1, 2) by its shape (2,4) and contributes 1 × 1 + 2 × 8 = 17, for 19 in all. The one-dimensional coordinate 5 splits first by mode 0's length 4 into (1, 1), and that 1 splits again into (1, 0), giving 2 + 1 = 3.
    2. It preserves the function "one-dimensional coordinate to index", rewriting it with as few modes as possible. `(4,8):(1,4)`'s mode 0 walks 4 steps to index 4, which equals mode 1's stride, so the two modes join end to end into 32 contiguous elements; `(4,8):(8,1)`'s mode 0 walks to 32, which is not mode 1's stride of 1, so they do not join. As a function of a one-dimensional coordinate it was never i ↦ i.
    3. B decides it: R(c) = A(B(c)), so R's domain is B's and its shape is B's shape. A row-major 4×8 matrix `(4,8):(8,1)` composed with `(8,4):(4,1)` gives `(8,4):(1,8)`, satisfying R(i, j) = A(j, i) without moving any data.
    4. A ⊘ B = A ∘ (B, B\*), where B\* = complement(B, size(A)) describes how the indices B does not cover are arranged. (B, B\*) rearranges all of A's one-dimensional coordinates into two modes: mode 0 is the elements B selects (within a tile) and mode 1 is which tile.
    5. Both start with a `zipped_divide` into ((within a tile), (which tile)). `local_tile` fixes the tile index by the CTA's coordinate, giving the whole tile this CTA owns; `local_partition` fixes the within-tile mode by the thread's coordinate in the thread layout, giving this thread's element in every tile. The thread layout's fast mode should run along the direction the data is contiguous in memory: column-major data with a column-major (16,2) thread layout has 32 threads read 128 contiguous bytes in one instruction (4 sectors, 100% utilization), while one 2×2 tile per thread without a vector instruction gets only 50%.

A six-panel strip before the text:

<!-- comic ../assets/comics/cute-layout.webp is in Chinese; put it back once the English version exists -->

## Why a layout algebra {#为什么需要布局代数}

Recall [the road to a fast GEMM](../kernels/gemm.md#v4二维寄存器分块): moving a tile of A from global memory into shared memory, each thread taking its 8×8 elements out of shared memory, and writing C back, each with a hand-written index formula. Those formulas are coupled: change the tile size, the data type, add vectorization, add a swizzle, and every one has to be derived again, with one mistake giving a silently wrong answer.

Look closely and they are all the same kind of thing: **a function from a logical coordinate to a memory index**:

- how data is laid out: row-major, column-major, stored in tiles, swizzled shared memory;
- who handles which tile: which tile of C a CTA owns, which tile along K the main loop takes;
- which elements a thread holds: which it moves during a copy, which positions of the matrix its registers hold during an MMA;
- an instruction's shape: one `cp.async` moves 16 bytes, one `mma.sync` computes 16×8×16.

CuTe expresses them all as a **Layout**, and a tensor is "a pointer plus a layout". Tiling, partitioning across threads and taking a different view are all operations between layouts, and that set of operations is the layout algebra. With the shapes and strides as compile-time constants, those operations happen at compile time: an illegal partition is a compile error, and the index arithmetic generated is as tight as a hand-written one.

[The ecosystem chapter](../tools/ecosystem.md#cutlass-与-cute) already printed a few layouts with CuTe. This chapter works through the algebra behind it: first implemented in Python so every operation can be stepped through, and finally compiled against the real CuTe from CUTLASS to confirm the two agree line by line.

## A layout: shape and stride {#布局形状与步长}

Start by moving things around: the same shape with different strides, and where a coordinate lands in memory:

<div class="aig-widget" data-widget="cute-layout"></div>

A layout is a **Shape** and a **Stride**, written `shape:stride`. The two are tuples of the same (possibly nested) structure, and each position is a **mode**:

- `(4,8):(1,4)`: 4 × 8, mode 0 with stride 1 and mode 1 with stride 4, which is column-major; `(4,8):(8,1)` is row-major;
- `(4,(2,4)):(2,(1,8))`: mode 1 is itself a (2,4) layout, which is a **hierarchical layout**.

A layout is a function: **the index is the inner product of the coordinate with the stride**. It accepts three kinds of coordinate:

- **one-dimensional**: an integer from 0 to size − 1, split into a multidimensional coordinate with "mode 0 varying fastest" (colexicographic, matching column-major order);
- **top-level**: one integer per top-level mode, with an integer falling on a nested mode split by the same rule, as in (1, 5);
- **fully expanded**: matching the shape's nesting exactly, as in (1, (1, 2)).

A few other quantities: **size** is the number of coordinates (the product of the shape's modes), **cosize** is the largest index plus 1 (the storage the layout touches), and **rank** is the number of top-level modes. Here is this chapter's implementation, with the layout itself and the simplest operation, coalescing:

```python title="layout_core.py"
"""layout_core.py —— CuTe 的布局：形状 + 步长，坐标到下标的映射，以及合并（coalesce）。

布局 = 形状（Shape）: 步长（Stride），两者都可以是嵌套的元组。布局是一个函数：把坐标映射成一维下标。
一维坐标按"第 0 维变化最快"（colexicographic，和列优先一致）拆成多维坐标，这是 CuTe 的约定。
打印格式和 CuTe 的 print 相同（只是不区分编译期常量）。
"""

from math import prod


def is_tuple(x):
    return isinstance(x, tuple)


def flatten(t):
    return sum((flatten(x) for x in t), ()) if is_tuple(t) else (t,)


def size(shape):
    return prod(flatten(shape))


def unflatten(flat, like):
    """按 like 的嵌套结构把扁平的元组重新分组，返回 (结果, 剩下的元素)"""
    if not is_tuple(like):
        return flat[0], flat[1:]
    out = []
    for x in like:
        y, flat = unflatten(flat, x)
        out.append(y)
    return tuple(out), flat


def crd2idx(crd, shape, stride):
    if is_tuple(crd):                                         # a tuple coordinate: map each mode and add
        return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
    if is_tuple(shape):                                       # an integer coordinate landing on a multidimensional mode: split it first (mode 0 fastest)
        idx = 0
        for s, d in zip(shape[:-1], stride[:-1]):
            idx += crd2idx(crd % size(s), s, d)
            crd //= size(s)
        return idx + crd2idx(crd, shape[-1], stride[-1])     # no modulo on the last mode: a coordinate past the end extends along it
    return crd * stride


def fmt(t):
    return "(" + ",".join(fmt(x) for x in t) + ")" if is_tuple(t) else str(t)


class Layout:
    def __init__(self, shape, stride=None):
        if stride is None:                                    # the default stride: a compact column-major layout
            flat, acc = [], 1
            for n in flatten(shape):
                flat.append(acc)
                acc *= n
            stride = unflatten(tuple(flat), shape)[0]
        self.shape, self.stride = shape, stride

    def __call__(self, coord):
        """坐标 → 下标。coord 可以是整数（一维坐标），也可以是和形状对应的（嵌套）元组"""
        return crd2idx(coord, self.shape, self.stride)

    def size(self):                                           # the domain's size: how many coordinates there are
        return size(self.shape)

    def cosize(self):                                         # the codomain's size: the largest index plus 1
        return self(self.size() - 1) + 1

    def __len__(self):                                        # the rank: how many top-level modes
        return len(self.shape) if is_tuple(self.shape) else 1

    def __getitem__(self, i):                                 # mode i on its own is a layout too
        return Layout(self.shape[i], self.stride[i]) if is_tuple(self.shape) else self

    def __eq__(self, other):
        return (self.shape, self.stride) == (other.shape, other.stride)

    def __repr__(self):
        return f"{fmt(self.shape)}:{fmt(self.stride)}"


def make_layout(*layouts):
    """把几个布局并排成一个多维布局"""
    return Layout(tuple(l.shape for l in layouts), tuple(l.stride for l in layouts))


def coalesce(layout):
    """合并：把"接得上"的相邻维（形状 × 步长 = 下一维的步长）并成一维，去掉长度为 1 的维；一维坐标上的函数不变"""
    shape, stride = [1], [0]
    for n, d in zip(flatten(layout.shape), flatten(layout.stride)):
        if n == 1:
            continue
        if shape[-1] == 1:
            shape[-1], stride[-1] = n, d
        elif shape[-1] * stride[-1] == d:
            shape[-1] *= n
        else:
            shape.append(n)
            stride.append(d)
    return Layout(shape[0], stride[0]) if len(shape) == 1 else Layout(tuple(shape), tuple(stride))
```

Drawing three layouts as tables (row i, column j is the index coordinate (i, j) maps to):

```python title="layout_basics.py"
from layout_core import Layout, coalesce


def show(layout, rows, cols):
    """把二维布局画成表格：第 i 行第 j 列是坐标 (i, j) 映射到的下标"""
    print(layout)
    for i in range(rows):
        print("  " + " ".join(f"{layout((i, j)):3d}" for j in range(cols)))


col = Layout((4, 8))                                 # the default stride: column-major, (4,8):(1,4)
show(col, 4, 8)
show(Layout((4, 8), (8, 1)), 4, 8)                   # row-major
h = Layout((4, (2, 4)), (2, (1, 8)))                 # the column mode split into (2,4): columns in pairs, adjacent within a pair, 8 apart between pairs
show(h, 4, 8)
print("h((1,5)) =", h((1, 5)), " h((1,(1,2))) =", h((1, (1, 2))), " h(5) =", h(5))
print("一维坐标 0..7 →", [h(i) for i in range(8)])
print("size =", h.size(), " cosize =", h.cosize(), " rank =", len(h), " h[1] =", h[1])
print("coalesce((2,(1,6)):(1,(6,2))) =", coalesce(Layout((2, (1, 6)), (1, (6, 2)))))
print("coalesce((4,8):(1,4)) =", coalesce(col), " coalesce((4,8):(8,1)) =", coalesce(Layout((4, 8), (8, 1))))
```

```text title="output"
(4,8):(1,4)
    0   4   8  12  16  20  24  28
    1   5   9  13  17  21  25  29
    2   6  10  14  18  22  26  30
    3   7  11  15  19  23  27  31
(4,8):(8,1)
    0   1   2   3   4   5   6   7
    8   9  10  11  12  13  14  15
   16  17  18  19  20  21  22  23
   24  25  26  27  28  29  30  31
(4,(2,4)):(2,(1,8))
    0   1   8   9  16  17  24  25
    2   3  10  11  18  19  26  27
    4   5  12  13  20  21  28  29
    6   7  14  15  22  23  30  31
h((1,5)) = 19  h((1,(1,2))) = 19  h(5) = 3
一维坐标 0..7 → [0, 2, 4, 6, 1, 3, 5, 7]
size = 32  cosize = 32  rank = 2  h[1] = (2,4):(1,8)
coalesce((2,(1,6)):(1,(6,2))) = 12:1
coalesce((4,8):(1,4)) = 32:1  coalesce((4,8):(8,1)) = (4,8):(8,1)
```

A few things to notice:

- `Layout((4, 8))` without a stride defaults to the column-major `(4,8):(1,4)`: a one-dimensional coordinate already expands with mode 0 fastest, and column-major makes the one-dimensional coordinate and the index identical.
- In the third layout the columns come in pairs: within a pair the two columns are adjacent (stride 1), within a column neighbouring rows differ by 2, and pairs are 8 apart. Those 4 rows × 2 columns, 8 elements, are contiguous in memory and laid out by row. That is "storing in tiles": the shape is still 4×8 and the logical coordinates are unchanged; only the stride is different.
- The one-dimensional coordinates 0..7 map to 0, 2, 4, 6, 1, 3, 5, 7: mode 0 (4 rows) first, then mode 1. A layout need be neither monotone nor injective (a stride of 0 maps several coordinates to one index, which is how broadcasting is expressed).

## Coalescing: the simplest form of the same function {#合并同一个函数的最简写法}

`coalesce` flattens a layout, drops the modes of length 1, and merges two neighbouring modes that **join end to end**: when mode k's "shape × stride" equals mode k+1's stride, walking mode k lands exactly on mode k+1's start and the two can be written as one. What it preserves is **the function on one-dimensional coordinates**: L(i) is the same before and after for every i.

- `(2,(1,6)):(1,(6,2))`: dropping the mode of length 1 gives `(2,6):(1,2)`, and 2 × 1 = 2 equals the next stride, so it merges to `12:1`;
- `(4,8):(1,4)` merges to `32:1`: a whole column-major tile is one contiguous stretch of memory;
- `(4,8):(8,1)` cannot merge: 4 × 8 = 32 ≠ 1.

One direct use of coalescing is **deciding whether vectorization is possible**: if a thread's elements coalesce with mode 0 equal to `n:1`, those n elements are contiguous and one 8-byte or 16-byte instruction moves them all. That is exactly what CuTe's `copy` does: `max_common_vector(src, dst)` gives the number of contiguous elements source and destination share and the vector width follows (up to 128 bits).

## Composition: a layout over a layout {#复合在布局上再套一层布局}

Composition is simply defined: **R = A ∘ B, R(c) = A(B(c))**. B's output is A's one-dimensional coordinate, so B decides "which elements of A, in what order", and R's shape is B's shape.

When B has one mode n:d, A ∘ B is "take every d-th of A's one-dimensional coordinates, n of them". The algorithm is **skip d first, then take n**: starting from mode 0 of A (coalesced), if the shape divides d, divide that mode's shape by d and multiply its stride by d; if not, that mode is already skipped and the remaining d carries to the next; after skipping, take n out of what is left. With several modes in B, each composes with A separately and they sit side by side. Here is the implementation, along with complement and division:

```python title="layout_algebra.py"
"""layout_algebra.py —— 布局代数：复合、补集、逻辑划分。算法与 CUTLASS 自带的 pycute 一致，另外像 CuTe 一样检查整除条件。"""

from layout_core import Layout, coalesce, flatten, is_tuple, make_layout


def composition(a, b):
    """复合：R(c) = A(B(c))，R 的形状就是 B 的形状。B 也可以是逐维作用的元组（每一维一个布局）"""
    if isinstance(b, tuple):                                  # compose mode by mode: A's mode i with B's mode i
        return make_layout(*[composition(a[i], b[i]) for i in range(len(b))], *[a[i] for i in range(len(b), len(a))])
    if is_tuple(b.shape):                                     # B has several modes: each composes with A separately
        return make_layout(*[composition(a, b[i]) for i in range(len(b))])
    if b.stride == 0:
        return Layout(b.shape, 0)
    rest_n, rest_d = b.shape, b.stride                        # B = n:d - take every d-th element of A, n of them
    shape, stride = [], []
    fa = coalesce(a)
    fn, fd = flatten(fa.shape), flatten(fa.stride)
    for n, d in zip(fn[:-1], fd[:-1]):                        # mode by mode: skip d first, then take n
        assert n % rest_d == 0 or rest_d % n == 0, "步长不整除，复合没有定义"
        take = min(max(1, n // rest_d), rest_n)
        assert rest_n % take == 0, "形状不整除，复合没有定义"
        if take != 1:
            shape.append(take)
            stride.append(rest_d * d)
        rest_n //= take
        rest_d = -(-rest_d // n)
    if rest_n != 1 or not shape:                              # A's last mode extends without limit
        shape.append(rest_n)
        stride.append(rest_d * fd[-1])
    return Layout(shape[0], stride[0]) if len(shape) == 1 else Layout(tuple(shape), tuple(stride))


def complement(layout, max_idx=1):
    """补集：A 没有覆盖的下标怎么排。(A, 补集) 合起来恰好把 [0, max_idx) 每个下标覆盖一次"""
    shape, stride, cur = [], [], 1
    for d, n in sorted(zip(flatten(layout.stride), flatten(layout.shape))):
        if d == 0 or n == 1:
            continue
        assert d % cur == 0, "A 的各维互相交叠，补集没有定义"
        shape.append(d // cur)
        stride.append(cur)
        cur = n * d
    shape.append(-(-max_idx // cur))
    stride.append(cur)
    return coalesce(Layout(tuple(shape), tuple(stride)))


def logical_divide(a, b):
    """逻辑划分：按 B 把 A 切成块。第 0 维是块内（B 选中的元素），第 1 维是第几块。B 可以是逐维作用的元组"""
    if isinstance(b, tuple):
        return make_layout(*[logical_divide(a[i], b[i]) for i in range(len(b))], *[a[i] for i in range(len(b), len(a))])
    return composition(a, make_layout(b, complement(b, a.size())))


def zipped_divide(a, tiler):
    """逐维划分之后重新分组：((块内的各维), (块号的各维))"""
    d = logical_divide(a, tiler)
    return make_layout(make_layout(*[d[i][0] for i in range(len(tiler))]),
                       make_layout(*[d[i][1] for i in range(len(tiler))]))
```

With A = `(6,2):(8,2)` and B = `(4,3):(3,1)`, by hand:

- B's mode 0, `4:3`: take every 3rd of A, 4 of them. A's mode 0 has 6 elements, and taking every 3rd leaves 6 / 3 = 2 with a stride of 3 × 8 = 24; 4 / 2 = 2 more come from A's mode 1 (stride 2), giving `(2,2):(24,2)`;
- B's mode 1, `3:1`: take A's first 3 elements, all within A's mode 0, giving `3:8`;
- together R = `((2,2),3):((24,2),8)`, and the program below checks R(i) = A(B(i)) one by one.

Composition's commonest use is **a different view without moving data**:

```python title="layout_compose.py"
from layout_algebra import composition, complement, logical_divide
from layout_core import Layout, make_layout

a = Layout((6, 2), (8, 2))
b = Layout((4, 3), (3, 1))
r = composition(a, b)
print("A =", a, " B =", b, " A∘B =", r)
print("逐个核对 R(i) == A(B(i))：", all(r(i) == a(b(i)) for i in range(b.size())))
for bad in (Layout(4, 2), Layout(4, 4)):                 # a composition that fails the divisibility condition
    try:
        composition(a, bad)
    except AssertionError as e:
        print(f"A∘{bad}：{e}")

m = Layout((4, 8), (8, 1))                           # a 4x8 row-major matrix
t = composition(m, Layout((8, 4), (4, 1)))
print("转置视图：", t, " t((5,2)) =", t((5, 2)), " m((2,5)) =", m((2, 5)))
e = composition(m, Layout((2, 8), (2, 4)))
print("只取偶数行：", e, " 第 1 行 =", [e((1, j)) for j in range(8)])

c = complement(Layout(4, 2), 24)
print("complement(4:2, 24) =", c, " 拼起来：", make_layout(Layout(4, 2), c),
      " 覆盖 0..23 各一次：", sorted(make_layout(Layout(4, 2), c)(i) for i in range(24)) == list(range(24)))
print("complement((2,2):(1,6), 24) =", complement(Layout((2, 2), (1, 6)), 24))

d = logical_divide(Layout(24), Layout(4, 2))
print("logical_divide(24:1, 4:2) =", d)
for k in range(d[1].size()):
    print(f"  第 {k} 块：", [d((i, k)) for i in range(4)])
```

```text title="output"
A = (6,2):(8,2)  B = (4,3):(3,1)  A∘B = ((2,2),3):((24,2),8)
逐个核对 R(i) == A(B(i))： True
A∘4:2：形状不整除，复合没有定义
A∘4:4：步长不整除，复合没有定义
转置视图： (8,4):(1,8)  t((5,2)) = 21  m((2,5)) = 21
只取偶数行： (2,8):(16,1)  第 1 行 = [16, 17, 18, 19, 20, 21, 22, 23]
complement(4:2, 24) = (2,3):(1,8)  拼起来： (4,(2,3)):(2,(1,8))  覆盖 0..23 各一次： True
complement((2,2):(1,6), 24) = (3,2):(2,12)
logical_divide(24:1, 4:2) = (4,(2,3)):(2,(1,8))
  第 0 块： [0, 2, 4, 6]
  第 1 块： [1, 3, 5, 7]
  第 2 块： [8, 10, 12, 14]
  第 3 块： [9, 11, 13, 15]
  第 4 块： [16, 18, 20, 22]
  第 5 块： [17, 19, 21, 23]
```

- **a transposed view**: the row-major 4×8 matrix m = `(4,8):(8,1)` composed with `(8,4):(4,1)` gives `(8,4):(1,8)`, where t(5, 2) = m(2, 5) = 21.
- **every other row**: composing with `(2,8):(2,4)` takes every second row and gives `(2,8):(16,1)`, whose row 1 is the original's row 2.
- The swizzle in [the ecosystem chapter](../tools/ecosystem.md#cutlass-与-cute) is a composition too: `composition(Swizzle<2,0,3>{}, tile)` wraps an xor function around the index the layout computes.
- Composition is not always defined. Still with A = `(6,2):(8,2)`: ∘ `4:2` takes every 2nd of A's mode 0 and can only produce 3, so the 4 wanted cannot be split into whole counts per mode; ∘ `4:4`'s third element is A's one-dimensional coordinate 8, which falls in the middle of column 1 and cannot be expressed with one stride. This chapter's implementation raises on both (lines 3 and 4 of the output). CuTe checks what it can at compile time: the first is a "Shape Divisibility Condition" `static_assert` error; the second compiles fine in CUTLASS 4.8 and gives `(_2,_2):(_32,_2)`, which does not satisfy R(i) = A(B(i)) (R(2) = 2 while A(8) = 18). So when you give CuTe a tile size or a tiler, make sure the divisibility holds yourself.

## Complement and division: cutting data into tiles {#补集与划分把数据切成块}

**Complement** answers: a layout A covers only some of the indices, so how are the gaps arranged? `complement(A, M)` returns a layout A\* such that (A, A\*) covers every index of [0, M) exactly once. In the output above, `4:2` covers {0, 2, 4, 6}, and its complement within 24 is `(2,3):(1,8)`: an offset of 0 or 1 fills the odd positions, repeated 3 times with a period of 8; together `(4,(2,3)):(2,(1,8))` covers 0..23 once each.

With complement, **logical division** is one line:

$$A \oslash B = A \circ (B,\ B^*),\qquad B^* = \text{complement}(B,\ \text{size}(A))$$

(B, B\*) rearranges A's one-dimensional coordinates into two modes: mode 0 is the elements B selects (**within a tile**) and mode 1 is **which tile**. `logical_divide(24:1, 4:2)` splits 24 elements into 6 tiles of 4 elements spaced 2 apart: tile 0 is 0, 2, 4, 6, tile 1 is 1, 3, 5, 7, tile 2 starts at 8, and so on. B need not be a contiguous stretch, so a "tile" can be any regular set of elements.

Beyond one dimension, division takes a **per-mode tuple** (which CuTe calls a tiler): mode 0 is divided by the first layout, mode 1 by the second. `zipped_divide` then regroups the result into **((the within-tile modes), (the tile-index modes))**, which is the most useful form. CuTe also has `tiled_divide` and `flat_divide`, the same content with the tile-index modes grouped differently. The dual of division is the **product**: `logical_product`, `blocked_product` and `raked_product` replicate a small layout according to another, which the thread partitions below use.

## From tiles to threads: local_tile and local_partition {#从块到线程local_tile-与-local_partition}

With `zipped_divide`, both levels of partitioning in a kernel are "divide, then fix one mode" (`cute/tensor_impl.hpp`):

- **`local_tile(tensor, tiler, coord)`**: divide by the tiler and fix the **tile index** by the CTA's coordinate, leaving the whole tile this CTA owns;
- **`local_partition(tensor, thr_layout, tid)`**: divide by the thread layout's shape and fix the **within-tile** mode by the thread's coordinate in that layout, leaving the tile-index mode as this thread's values. Thread tid takes the same position in every tile.

The body of CuTe's tutorial `examples/cute/tutorial/sgemm_1.cu` is these few lines (excerpted; the comments are this book's):

```cpp
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k): this CTA's A
Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k): the part this thread moves from global memory
Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K):   where it goes in shared memory
copy(tAgA(_,_,k_tile), tAsA);                                        // two tensors of the same shape, copied elementwise
```

How the thread layout is chosen decides the memory efficiency directly. Below, a 16×8 column-major fp32 tile is split among 32 threads two ways: one 2×2 tile per thread (the tile index being the thread index), and `local_partition` with a (16,2) thread layout. By the model in [the memory chapter](../basics/memory.md#全局内存合并访问), one memory instruction costs however many 32-byte sectors the 32 threads' addresses land in:

```python title="layout_partition.py"
from layout_algebra import zipped_divide
from layout_core import Layout

tile = Layout((16, 8))                               # a 16x8 fp32 tile, column-major: a column's 16 elements are adjacent
print("tile =", tile)


def blocked(val_shape):
    """每个线程拿一个 val_shape 的小块：按小块划分，"块号"那一维就是线程号"""
    d = zipped_divide(tile, tuple(Layout(n) for n in val_shape))
    print(f"每个线程一个 {val_shape[0]}×{val_shape[1]} 的小块：", d)
    return lambda t, v: d((v, t))


def local_partition(thr):
    """CuTe 的 local_partition：按线程布局的形状划分，线程在块里的坐标固定"块内"那一维，剩下的"块号"是它的各个值"""
    d = zipped_divide(tile, tuple(Layout(n) for n in thr.shape))
    print(f"按线程布局 {thr} 划分：", d)
    pos = [thr(j) for j in range(thr.size())]
    return lambda t, v: d((pos.index(t), v))


def request(part, values):
    """一条访存指令：32 个线程各读自己编号为 values 的值，数一数落在几个 32 字节扇区里"""
    addr = sorted(part(t, v) for t in range(32) for v in values)
    sectors = len({a * 4 // 32 for a in addr})
    print(f"  读编号 {values} 的值：地址 {addr[:6]}…，{sectors} 个扇区，利用率 {len(addr) * 4 / (sectors * 32):.0%}")


p = blocked((2, 2))
print("  线程 0、1、8 的值：", [[p(t, v) for v in range(4)] for t in (0, 1, 8)])
request(p, [0])                                      # 4 bytes per thread
request(p, [0, 1])                                   # its first two values are adjacent in memory: one 8-byte vector instruction reads both
q = local_partition(Layout((16, 2)))
print("  线程 0、1、16 的值：", [[q(t, v) for v in range(4)] for t in (0, 1, 16)])
request(q, [0])
```

```text title="output"
tile = (16,8):(1,16)
每个线程一个 2×2 的小块： ((2,2),(8,4)):((1,16),(2,32))
  线程 0、1、8 的值： [[0, 1, 16, 17], [2, 3, 18, 19], [32, 33, 48, 49]]
  读编号 [0] 的值：地址 [0, 2, 4, 6, 8, 10]…，8 个扇区，利用率 50%
  读编号 [0, 1] 的值：地址 [0, 1, 2, 3, 4, 5]…，8 个扇区，利用率 100%
按线程布局 (16,2):(1,16) 划分： ((16,2),(1,4)):((1,16),(0,32))
  线程 0、1、16 的值： [[0, 32, 64, 96], [1, 33, 65, 97], [16, 48, 80, 112]]
  读编号 [0] 的值：地址 [0, 1, 2, 3, 4, 5]…，4 个扇区，利用率 100%
```

- **one 2×2 tile per thread**: the thread-index mode is `(8,4):(2,32)`, so neighbouring threads' addresses differ by 2 and a 4-byte read uses only half of each sector, 50% utilization, exactly the "stride 2" row of the memory chapter's table.
- But this thread's value mode is `(2,2):(1,16)`, so its first two values are adjacent in memory. Read both with one 8-byte vector instruction and the same 8 sectors are full of useful data, back to 100%. **Vectorization and thread arrangement are two complementary tools**, and CuTe's `copy` picks the vector width from the layout automatically.
- **`local_partition` with a (16,2) thread layout**: the within-tile mode `(16,2):(1,16)` is indexed by the thread, so neighbouring threads' addresses differ by 1 and 32 threads read 128 contiguous bytes in one instruction, 4 sectors, 100% utilization. The rule: **the thread layout's fast mode must run along the direction the data is contiguous in memory**.

Real CuTe kernels rarely write these partitions by hand and instead use `make_tiled_copy(copy_atom, thr_layout, val_layout)` to say "each thread moves val_layout worth at a time, with the threads arranged as thr_layout", from which CuTe computes the **TV layout** with `raked_product(thr_layout, val_layout)` and `right_inverse`: (thread, value) to a within-tile coordinate. CuTe's tutorial `sgemm_sm80.cu` copies A with 128 threads arranged as `(16,8):(8,1)` (adjacent along K) and `(1,8)` halves per thread (one 16-byte `cp.async`), and its TV layout is printed by the C++ program at the end of this chapter; exercise 2 asks you to read it.

## A Tensor Core's register layout is a layout too {#tensor-core-的寄存器布局也是一个布局}

[The Tensor Core chapter](../advanced/tensor-core.md#mmasync寄存器布局是明确的) gave a table of which elements of A, B and C each thread holds for `mma.sync.m16n8k16`. In CuTe that table is a TV layout, written in `MMA_Traits` in `cute/atom/mma_traits_sm80.hpp`:

```cpp
template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>   // the FP32-accumulating SM80_16x8x16_F32F16F16F32_TN inherits it and only changes the value type
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using CLayout = SM80_16x8_Row;
};
```

where `SM80_16x8_Row` is `((_4,_8),(_2,_2)):((_32,_1),(_16,_8))`.

All three layouts map (thread, value) to a **column-major index within the tile**: A is M×K (16×16, index = row + 16 × column), B is N×K (8×16, index = n + 8k) and C is M×N (16×8). For A, the thread mode `(4,8):(32,1)` splits the lane into (lane % 4, lane / 4), the table's t and g: g walks the rows (stride 1) and each increment of t moves two columns right (stride 32 = 2 × 16); the value mode `(2,2,2):(16,8,128)` is "one column right", "eight rows down", "eight columns right" in turn. Expanding all three with this chapter's implementation and checking against the table item by item:

```python title="layout_mma.py"
from layout_core import Layout

# CUTLASS's cute/atom/mma_traits_sm80.hpp: the TV layouts of SM80_16x8x16_F32F16F16F32_TN (mma.sync.m16n8k16),
# mapping (thread, value) to a column-major index within the tile
A = Layout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))    # A: 16x16, M by K
B = Layout(((4, 8), (2, 2)), ((16, 1), (8, 64)))            # B: 8x16, N by K, index = n + 8k
C = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))            # C: 16x8, M by N


def rc(idx):
    return idx % 16, idx // 16                       # a column-major index to (row, column)


def ptx_a(lane, i):                                  # the table from the PTX documentation (copied in the Tensor Core chapter): g = lane / 4, t = lane % 4
    g, t = lane // 4, lane % 4
    return g + 8 * (i // 2 % 2), 2 * t + i % 2 + 8 * (i // 4)


def ptx_b(lane, i):                                  # B given as (k, n)
    g, t = lane // 4, lane % 4
    return 2 * t + i % 2 + 8 * (i // 2), g


def ptx_c(lane, i):
    g, t = lane // 4, lane % 4
    return g + 8 * (i // 2), 2 * t + i % 2


for lane in (0, 1, 4):
    print(f"lane {lane}：A 的 8 个值在", [rc(A((lane, i))) for i in range(8)])
print("A 与 PTX 的表一致：", all(rc(A((l, i))) == ptx_a(l, i) for l in range(32) for i in range(8)))
print("B 与 PTX 的表一致：", all((B((l, i)) // 8, B((l, i)) % 8) == ptx_b(l, i) for l in range(32) for i in range(4)))
print("C 与 PTX 的表一致：", all(rc(C((l, i))) == ptx_c(l, i) for l in range(32) for i in range(4)))
owner = {rc(C((l, i))): l for l in range(32) for i in range(4)}
print("C 的每个元素归哪个线程（行 0～3）：")
for r in range(4):
    print("  " + " ".join(f"{owner[(r, c)]:2d}" for c in range(8)))
```

```text title="output"
lane 0：A 的 8 个值在 [(0, 0), (0, 1), (8, 0), (8, 1), (0, 8), (0, 9), (8, 8), (8, 9)]
lane 1：A 的 8 个值在 [(0, 2), (0, 3), (8, 2), (8, 3), (0, 10), (0, 11), (8, 10), (8, 11)]
lane 4：A 的 8 个值在 [(1, 0), (1, 1), (9, 0), (9, 1), (1, 8), (1, 9), (9, 8), (9, 9)]
A 与 PTX 的表一致： True
B 与 PTX 的表一致： True
C 与 PTX 的表一致： True
C 的每个元素归哪个线程（行 0～3）：
   0  0  1  1  2  2  3  3
   4  4  5  5  6  6  7  7
   8  8  9  9 10 10 11 11
  12 12 13 13 14 14 15 15
```

One layout says everything the table says, and it takes part in the algebra: `make_tiled_mma(mma_atom, Layout<Shape<_2,_2,_1>>{})` replicates one warp's MMA across 2×2 warps with a product; `thr_mma.partition_A(sA)` and `partition_fragment_C(...)` carve out each thread's share of shared memory and registers through the TV layout. ldmatrix's, wgmma's and tcgen05's layouts live in their own traits the same way, so reading a CUTLASS kernel no longer means looking up tables in the PTX documentation.

## Checking against the real CuTe {#和真正的-cute-对照}

The program below redoes every example of this chapter with CUTLASS's own CuTe, running on the host only with no GPU needed. Shapes and strides are compile-time constants `_N` (that is, `Int<N>{}`) and print with a leading underscore:

```cuda title="cute_algebra.cu"
// cute_algebra.cu - checking this chapter's layout algebra against the real CuTe from CUTLASS: host only, no GPU needed
// build: nvcc -std=c++17 -arch=sm_80 -I${CUTLASS_DIR}/include cute_algebra.cu -o cute_algebra
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
using namespace cute;

template <class T>
void show(const char* name, T const& x) {
  printf("%-20s", name);
  print(x);
  printf("\n");
}

int main() {
  // 1. the layout algebra: shapes and strides as compile-time constants _N (that is, Int<N>{}), so the simplification happens at compile time
  show("coalesce", coalesce(make_layout(make_shape(_2{}, make_shape(_1{}, _6{})),
                                        make_stride(_1{}, make_stride(_6{}, _2{})))));
  show("coalesce (int)", coalesce(make_layout(make_shape(2, make_shape(1, 6)),   // ordinary ints: only flattening is possible
                                              make_stride(1, make_stride(6, 2)))));
  auto a = make_layout(make_shape(_6{}, _2{}), make_stride(_8{}, _2{}));
  auto b = make_layout(make_shape(_4{}, _3{}), make_stride(_3{}, _1{}));
  show("composition", composition(a, b));
  auto m = make_layout(make_shape(_4{}, _8{}), LayoutRight{});                 // 4x8 row-major
  show("transpose view", composition(m, make_layout(make_shape(_8{}, _4{}), LayoutRight{})));
  show("even rows", composition(m, make_layout(make_shape(_2{}, _8{}), make_stride(_2{}, _4{}))));
  show("complement", complement(make_layout(_4{}, _2{}), _24{}));
  show("logical_divide", logical_divide(make_layout(_24{}), make_layout(_4{}, _2{})));
  auto tile = make_layout(make_shape(_16{}, _8{}));                             // 16x8 column-major
  show("zipped_divide 2x2", zipped_divide(tile, make_shape(_2{}, _2{})));
  show("zipped_divide 16x2", zipped_divide(tile, make_shape(_16{}, _2{})));

  // 2. the thread partition: copying A in CuTe's tutorial sgemm_sm80.cu, 128 threads as 16x8 (adjacent along k) with 8 halves each
  using CopyA = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
                                         Layout<Shape<_16, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{}));
  show("TiledCopy tile", typename CopyA::Tiler_MN{});
  show("TiledCopy TV", typename CopyA::TiledLayout_TV{});

  // 3. Tensor Cores: mma.sync.m16n8k16's register layout, (thread, value) to a column-major index within the tile
  using Traits = MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>;
  show("MMA A (thr,val)", typename Traits::ALayout{});
  show("MMA B (thr,val)", typename Traits::BLayout{});
  show("MMA C (thr,val)", typename Traits::CLayout{});
  return 0;
}
```

Built and run with CUTLASS 4.8 (nvcc 12.9 and 13.4 give the same result):

```text
coalesce            _12:_1
coalesce (int)      (2,1,6):(1,6,2)
composition         ((_2,_2),_3):((_24,_2),_8)
transpose view      (_8,_4):(_1,_8)
even rows           (_2,_8):(_16,_1)
complement          (_2,_3):(_1,_8)
logical_divide      (_4,(_2,_3)):(_2,(_1,_8))
zipped_divide 2x2   ((_2,_2),(_8,_4)):((_1,_16),(_2,_32))
zipped_divide 16x2  ((_16,_2),(_1,_4)):((_1,_16),(_0,_32))
TiledCopy tile      (_16,_64)
TiledCopy TV        ((_8,_16),_8):((_128,_1),_16)
MMA A (thr,val)     ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
MMA B (thr,val)     ((_4,_8),(_2,_2)):((_16,_1),(_8,_64))
MMA C (thr,val)     ((_4,_8),(_2,_2)):((_32,_1),(_16,_8))
```

Apart from the underscores, every line matches the Python above. Only the second differs: built from ordinary `int`s, the same layout only gets flattened by `coalesce`, not merged. The reason is that a CuTe layout's type is fixed at compile time, while an ordinary `int`'s value is only known at run time, so which mode is 1 and which two join cannot be decided then. That is why tile sizes and thread layouts in a CuTe kernel are nearly always constants like `_128{}` or `Int<N>{}`: the simplification and the divisibility checks happen at compile time and only the final multiply-adds remain in the generated code. Only run-time quantities such as a matrix's M, N and K use ordinary integers.

CUTLASS 4.x's **CuTe DSL** writes kernels in Python with function names matching one to one: `cute.make_layout`, `cute.coalesce`, `cute.composition`, `cute.complement`, `cute.logical_divide`, `cute.zipped_divide`, `cute.local_tile`, `cute.local_partition`, `cute.make_tiled_copy_tv`. Once the layout algebra makes sense, the C++ and Python interfaces read the same.

A suggested order for continuing with CuTe: `01_layout.md` and `02_layout_algebra.md` (this chapter's original source), `03_tensor.md` and `0t_mma_atom.md` under `media/docs/cpp/cute/` in the CUTLASS repository; then `sgemm_1.cu` (only `local_tile` / `local_partition`) and `sgemm_sm80.cu` (`TiledCopy` plus `TiledMMA` plus `cp.async`) in `examples/cute/tutorial/`; and finally the Hopper and Blackwell examples (TMA, wgmma, tcgen05). When a layout stops making sense, print it on the host with `print` / `print_layout`, or work it out with pycute as this chapter does.

!!! interview "How to explain it"
    To explain "what is a CuTe Layout and why use it": a layout is shape : stride, a function from a (possibly nested) logical coordinate to a one-dimensional index, the index being the inner product of the coordinate with the stride; a tensor is a pointer plus a layout. How data is laid out, how CTAs tile it, how threads partition it and how an MMA arranges registers are all layouts, combined by one algebra: `coalesce` simplifies (preserving the function), a composition A∘B gives a different view (taking its shape from B), complement fills the gaps, `logical_divide` = A∘(B, complement(B)), and `zipped_divide` gives ((within a tile), (which tile)); `local_tile` fixes the tile index for a CTA and `local_partition` fixes the within-tile position for a thread; and an MMA atom's TV layout maps (thread, value) to a within-tile coordinate. With shapes as compile-time constants, the simplification and the divisibility checks happen at compile time. Giving an example of a thread layout affecting coalescing (neighbouring threads need stride 1 for one instruction to read a full 128 bytes) counts for extra.

## Exercises {#练习}

**1. Every other column.** From the 8×8 row-major matrix A = `(8,8):(8,1)`, take columns 0, 2, 4 and 6 as an 8×4 view. What should B be? What is A∘B?

??? success "Answer"
    The view's coordinate (i, j) is the original's (i, 2j), whose one-dimensional coordinate in A (with mode 0 fastest) is i + 8 × 2j = i + 16j, so B = `(8,4):(1,16)`. Composing, B's mode 0 `8:1` takes A's first 8 elements, exactly A's whole mode 0, giving `8:8`; B's mode 1 `4:16` takes every 16th: A's mode 0 has only 8 elements and is skipped entirely, carrying 16 / 8 = 2 into A's mode 1 (stride 1), where every 2nd of 4 is taken, giving `4:2`. The result is `(8,4):(8,2)`, that is, R(i, j) = 8i + 2j = A(i, 2j).

**2. Reading a TiledCopy.** The program at the end of this chapter prints `sgemm_sm80.cu`'s configuration for copying A: the tile is `(_16,_64)` and the TV layout is `((_8,_16),_8):((_128,_1),_16)` (the values being column-major indices within the 16×64 tile, of type half, with K contiguous in memory). Which elements does thread 9 copy? Which rows does one warp's 32 threads cover? Why exactly 8 threads per row?

??? success "Answer"
    The thread mode `(8,16):(128,1)` splits t into (t % 8, t / 8) with index = 128 × (t % 8) + t / 8; the value mode `8:16` adds 16 per value. Since index = row + 16 × column, thread t owns row t / 8, columns 8 × (t % 8) through 8 × (t % 8) + 7. Thread 9 owns row 1, columns 8-15. One warp is threads 0-31 and covers rows 0-3, 64 halves per row. 8 threads × 8 halves × 2 bytes = 128 bytes per row, exactly one whole row and one complete 128-byte segment: neighbouring threads run along K (the contiguous direction), so a warp's one 16-byte `cp.async` reads 4 complete 128-byte segments with nothing wasted. That is why the thread layout is `(16,8):(8,1)` (adjacent along K).

**3. One element of C.** With `CLayout = ((4,8),(2,2)):((32,1),(16,8))`, work out where lane 5's value 3 (c3) sits in C, and check it against the PTX table.

??? success "Answer"
    Lane 5 splits into (5 % 4, 5 / 4) = (1, 1), contributing 32 × 1 + 1 × 1 = 33; value 3 splits into (1, 1), contributing 16 + 8 = 24. The index 57 = 9 + 16 × 3, that is, row 9 column 3. From the PTX table: g = 5 / 4 = 1, t = 5 % 4 = 1, and c3 is at (g + 8, 2t + 1) = (9, 3). They agree.

## Summary {#小结}

- [x] A layout is shape : stride, a function from a (possibly nested) coordinate to an index; a one-dimensional coordinate expands with mode 0 fastest, the default stride is column-major, and a tensor is a pointer plus a layout.
- [x] `coalesce` preserves the function on one-dimensional coordinates and merges modes that join end to end; the contiguous mode after coalescing decides how wide a vector instruction can be.
- [x] A composition R = A∘B takes its shape from B and gives a different view (a transpose, every other row) or layers on a swizzle, without moving data.
- [x] Complement fills the indices a layout does not cover; `logical_divide` = A∘(B, complement(B)), and `zipped_divide` gives ((within a tile), (which tile)).
- [x] `local_tile` fixes the tile index for a CTA and `local_partition` the within-tile position for a thread; the thread layout's fast mode must run along the contiguous direction.
- [x] An MMA atom's TV layout is the register table from the PTX documentation; with shapes as compile-time constants, the simplification and the checks happen at compile time.
