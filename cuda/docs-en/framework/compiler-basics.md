# A compiler crash course: the concepts you need to read an AI compiler

<p class="lead">"Familiar with compiler construction" is a common line in inference job postings, but what is wanted is not that you write a C compiler yourself: it is that you can read what an AI compiler is doing. What exactly does `torch.compile` transform, what are Triton's ttir and ttgir, why does fusion save memory traffic, what is auto-tuning searching for, what does MLIR's "dialect" mean. This chapter uses three small runnable programs to cover the front end (lexing, parsing, the syntax tree), intermediate representations and optimization passes (SSA, constant folding, common subexpression elimination, dead code elimination), and loop transformations (interchange, tiling), then maps each one onto Inductor, Triton, TVM and MLIR.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What are the classic stages of a compiler? What does an AI compiler have in their place?
    2. What is an intermediate representation (IR)? Why is there an SSA form?
    3. What waste does each of constant folding, common subexpression elimination and dead code elimination remove?
    4. Why does loop tiling speed things up? Why is a larger tile not always better?
    5. What problem do MLIR's dialects solve?
    6. What is the difference between Triton's ttir and ttgir?

??? success "Answers (try it yourself first, then expand)"
    1. Lexing (a character stream to tokens) → parsing (tokens to a syntax tree) → semantic analysis and type checking → generating an intermediate representation → optimizing (a series of passes) → code generation and register allocation. The AI compiler's counterparts: the front end is graph capture at the Python level (Dynamo's bytecode analysis, `torch.fx`'s symbolic tracing), the middle is passes on a graph IR and a loop IR, and the back end generates Triton / CUDA / LLVM code.
    2. An IR sits between the source and the target code, and all the optimization happens on it, so N languages by M hardware targets need N + M parts rather than N × M. SSA (static single assignment) requires every variable to be assigned once, which turns "are these two expressions the same value" into a simple comparison of names and makes dataflow analysis and CSE much simpler.
    3. Constant folding: an expression computable at compile time should not be left to run time (`2 * 3 + x` → `6 + x`). Common subexpression elimination: compute the same expression once. Dead code elimination: delete instructions whose results nobody uses. In this chapter's example 11 instructions drop to 8 with identical results.
    4. Tiling keeps a small block of data in fast storage (cache, shared memory, registers) and reuses it many times, turning "fetch from memory every time" into "fetch once and use many times". Too large a tile and one block's working set no longer fits in that level, which is worse: in this chapter's simulation a 16×16 tile is three times worse than 8×8. So the tile size has to match the capacity of that level of storage, which is exactly the parameter auto-tuning searches.
    5. A traditional compiler has one level of IR (LLVM IR, say), while deep learning needs optimization at several levels of abstraction: the graph level (operator fusion), the loop level (tiling), the vector level and the hardware instruction level. MLIR lets several IRs (dialects) coexist and defines the lowering rules between them, and Triton, IREE and many domestic accelerators' compilers are built on it.
    6. ttir is block-level, hardware-independent tensor operations; ttgir adds a **layout** to every tensor (how this data is distributed across threads) and inserts shared-memory allocation, asynchronous copies and pipelining. When performance is off, look at ttgir's layouts and pipelining first, then check whether the PTX uses `mma` / `wgmma`.

## The front end: from characters to a syntax tree {#前端从字符到语法树}

A compiler's first stage turns a string of characters into a structured tree. Here is a tiny expression language handling the four arithmetic operations, exponentiation and variables:

```python title="minilang.py"
# a compiler front end's three steps: lexing (characters -> tokens), parsing (tokens -> a syntax tree), evaluating or generating code
import re
from dataclasses import dataclass

TOKEN = re.compile(r"\s*(?:(\d+\.?\d*)|([A-Za-z_]\w*)|(\*\*|[-+*/()=,;]))")


def tokenize(src):
    tokens, pos = [], 0
    while pos < len(src):
        m = TOKEN.match(src, pos)
        if not m:
            raise SyntaxError(f"无法识别的字符：{src[pos]!r}")
        pos = m.end()
        num, name, op = m.groups()
        tokens.append(("num", float(num)) if num else ("name", name) if name else ("op", op))
    tokens.append(("eof", None))
    return tokens


@dataclass
class Num:
    value: float


@dataclass
class Var:
    name: str


@dataclass
class BinOp:
    op: str
    left: object
    right: object


PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 3}     # the larger, the tighter it binds


class Parser:
    """Pratt 解析器：用优先级表处理二元运算符，比写一堆递归函数短得多"""

    def __init__(self, tokens):
        self.tokens, self.i = tokens, 0

    def peek(self):
        return self.tokens[self.i]

    def next(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def parse(self, min_prec=0):
        kind, value = self.next()
        if kind == "num":
            node = Num(value)
        elif kind == "name":
            node = Var(value)
        elif (kind, value) == ("op", "("):
            node = self.parse()
            assert self.next() == ("op", ")"), "括号没有闭合"
        elif (kind, value) == ("op", "-"):
            node = BinOp("-", Num(0.0), self.parse(3))     # unary minus
        else:
            raise SyntaxError(f"意外的记号：{value!r}")
        while True:
            kind, op = self.peek()
            if kind != "op" or op not in PRECEDENCE or PRECEDENCE[op] < min_prec:
                return node
            self.next()
            right = self.parse(PRECEDENCE[op] + (0 if op == "**" else 1))   # ** is right-associative
            node = BinOp(op, node, right)


def show(node):
    if isinstance(node, Num):
        return f"{node.value:g}"
    if isinstance(node, Var):
        return node.name
    return f"({show(node.left)} {node.op} {show(node.right)})"


def fold(node):
    """常量折叠：两个操作数都是常量就直接算出来；还顺手做几个代数化简"""
    if not isinstance(node, BinOp):
        return node
    left, right = fold(node.left), fold(node.right)
    if isinstance(left, Num) and isinstance(right, Num):
        value = {"+": left.value + right.value, "-": left.value - right.value,
                 "*": left.value * right.value, "/": left.value / right.value,
                 "**": left.value ** right.value}[node.op]
        return Num(value)
    if node.op == "*" and isinstance(right, Num) and right.value == 1:
        return left                                        # x * 1 -> x
    if node.op == "*" and isinstance(right, Num) and right.value == 0:
        return Num(0.0)                                    # x * 0 -> 0
    if node.op == "+" and isinstance(right, Num) and right.value == 0:
        return left                                        # x + 0 -> x
    return BinOp(node.op, left, right)


for src in ["2 * 3 + x * 1", "a + 2 * 3 * 4", "x * (2 - 2)", "2 ** 3 ** 2", "-x + 1 - 1"]:
    tree = Parser(tokenize(src)).parse()
    print(f"{src:18s} 语法树 {show(tree):28s} 折叠后 {show(fold(tree))}")
```

```text title="output"
2 * 3 + x * 1      语法树 ((2 * 3) + (x * 1))          折叠后 (6 + x)
a + 2 * 3 * 4      语法树 (a + ((2 * 3) * 4))          折叠后 (a + 24)
x * (2 - 2)        语法树 (x * (2 - 2))                折叠后 0
2 ** 3 ** 2        语法树 (2 ** (3 ** 2))              折叠后 512
-x + 1 - 1         语法树 (((0 - x) + 1) - 1)          折叠后 (((0 - x) + 1) - 1)
```

Three things:

- **Lexing** is a regex-driven loop: numbers, identifiers and operators each in their own class. A real compiler also has to handle comments, strings and indentation (Python).
- **Parsing** here uses a **Pratt parser** (precedence climbing): one precedence table decides "who binds first in `2 * 3 + x`", which is far shorter than a recursive function per precedence level. Note that `**` is **right-associative**, so the minimum precedence passed down the recursion is not incremented: the syntax tree of `2 ** 3 ** 2` is `2 ** (3 ** 2)` = 512, not `(2 ** 3) ** 2` = 64.
- **Constant folding** is the first optimization done on the syntax tree: compute directly when both operands are constants, plus a few algebraic simplifications along the way (`x * 1 → x`, `x * 0 → 0`). Note that the last example, `-x + 1 - 1`, was not simplified to `-x`: the syntax tree is left-associative, `((0-x) + 1) - 1`, and cancelling would need "reassociation", which floating-point arithmetic does not support. A compiler dares not do that transformation by default, unless you turn on `-ffast-math`. This is exactly what [deterministic inference](serving://topics/deterministic/) has to be careful about.

What Dynamo does is conceptually the same, except that its input is not characters but **Python bytecode**: it interprets the bytecode while recording the operations on tensors, and where it meets something it cannot handle it "breaks the graph", handing the part it did capture to the back end (see [torch.compile](compile.md)).

## Intermediate representations and optimization passes {#中间表示与优化-pass}

![Figure: a compiler's successive lowerings - graph-level IR, loop-level IR, target IR, machine code](../assets/figures/ir-lowering.svg){.aig-svg}

Optimization happens not on the syntax tree but on an **intermediate representation**. The most common form is three-address code: at most two operands and one result per instruction. Add **SSA** (every variable assigned exactly once) and dataflow analysis becomes very simple.

```python title="passes.py"
# the intermediate representation (SSA-style three-address code) and a few of the most basic optimization passes
from dataclasses import dataclass


@dataclass
class Instr:
    dst: str
    op: str                                    # "const" / "+" / "*" / "load" / "store"
    args: tuple

    def __str__(self):
        if self.op == "const":
            return f"{self.dst} = {self.args[0]:g}"
        if self.op in ("load", "store"):
            return f"{self.dst} = {self.op} {self.args[0]}" if self.op == "load" else f"store {self.args[0]} -> {self.dst}"
        return f"{self.dst} = {self.args[0]} {self.op} {self.args[1]}"


# a hand-written IR: storing (a+b)*(a+b) + (a+b)*2 into out, recomputing a+b a few times on purpose, plus one instruction nobody uses
program = [
    Instr("t1", "load", ("a",)),
    Instr("t2", "load", ("b",)),
    Instr("t3", "+", ("t1", "t2")),
    Instr("t4", "+", ("t1", "t2")),            # exactly the same as t3
    Instr("t5", "*", ("t3", "t4")),
    Instr("t6", "const", (2.0,)),
    Instr("t7", "+", ("t1", "t2")),            # once more
    Instr("t8", "*", ("t7", "t6")),
    Instr("t9", "+", ("t5", "t8")),
    Instr("t10", "*", ("t1", "t2")),           # computed but unused
    Instr("out", "store", ("t9",)),
]


def cse(instrs):
    """公共子表达式消除：同样的 (op, 参数) 只算一次，后面的引用改成第一次的结果"""
    seen, rename, out = {}, {}, []
    for ins in instrs:
        args = tuple(rename.get(a, a) for a in ins.args)
        key = (ins.op, args)
        if ins.op in ("+", "*") and key in seen:
            rename[ins.dst] = seen[key]        # a repeated expression: reuse it
            continue
        seen[key] = ins.dst
        out.append(Instr(ins.dst, ins.op, args))
    return out


def dce(instrs):
    """死代码消除：从最后的输出反向标记用到的值，没被用到的指令删掉"""
    live = {ins.dst for ins in instrs if ins.op == "store"} | {
        a for ins in instrs if ins.op == "store" for a in ins.args}
    for ins in reversed(instrs):
        if ins.dst in live or ins.op == "store":
            live |= set(ins.args)
    return [ins for ins in instrs if ins.dst in live or ins.op == "store"]


def run(instrs, a, b):
    """解释执行这段 IR，用来验证优化前后结果一致"""
    env = {"a": a, "b": b}
    for ins in instrs:
        if ins.op == "const":
            env[ins.dst] = ins.args[0]
        elif ins.op == "load":
            env[ins.dst] = env[ins.args[0]]
        elif ins.op == "store":
            env[ins.dst] = env[ins.args[0]]
        else:
            x, y = env[ins.args[0]], env[ins.args[1]]
            env[ins.dst] = x + y if ins.op == "+" else x * y
    return env["out"]


print("优化前：")
for ins in program:
    print("   ", ins)
after_cse = cse(program)
after_dce = dce(after_cse)
print("\n经过公共子表达式消除和死代码消除：")
for ins in after_dce:
    print("   ", ins)
print(f"\n指令数 {len(program)} -> {len(after_dce)}；结果一致："
      f"{all(run(program, a, b) == run(after_dce, a, b) for a, b in [(1, 2), (3, 5), (-2, 7)])}")
print("\n这三步（建 IR、在 IR 上跑一串 pass、验证语义不变）就是所有编译器的骨架。")
print("SSA（每个变量只赋值一次）让'这两个表达式是不是同一个值'变成简单的比较，上面的 CSE 正是靠它。")
```

```text title="output"
优化前：
    t1 = load a
    t2 = load b
    t3 = t1 + t2
    t4 = t1 + t2
    t5 = t3 * t4
    t6 = 2
    t7 = t1 + t2
    t8 = t7 * t6
    t9 = t5 + t8
    t10 = t1 * t2
    store t9 -> out

经过公共子表达式消除和死代码消除：
    t1 = load a
    t2 = load b
    t3 = t1 + t2
    t5 = t3 * t3
    t6 = 2
    t8 = t3 * t6
    t9 = t5 + t8
    store t9 -> out

指令数 11 -> 8；结果一致：True

这三步（建 IR、在 IR 上跑一串 pass、验证语义不变）就是所有编译器的骨架。
SSA（每个变量只赋值一次）让'这两个表达式是不是同一个值'变成简单的比较，上面的 CSE 正是靠它。
```

That is a compiler's skeleton: **build an IR → run a series of passes over it → each pass preserving the semantics**. A real compiler has tens to hundreds of passes (inlining, loop-invariant code motion, strength reduction, tail call elimination, alias analysis and so on), but the structure is always this.

A few points that bear directly on AI compilers:

- **Why SSA**: the CSE above can be written so briefly because `t3` never changes once defined, so the conclusion "`t1 + t2` has been computed" holds forever. When variables can be reassigned, a complicated reaching-definitions analysis has to come first.
- **The order of the passes matters**: CSE before DCE deletes more; some passes have to run repeatedly until they reach a fixed point.
- **A cost model** decides whether to do a transformation at all. Graph-level fusion is a typical example: fusion saves memory traffic but may increase register pressure (see [the AI compiler landscape](compilers.md#图级融合省下的是访存)).

## Loop transformations: tiling, interchange and fusion {#循环变换分块交换与融合}

Loops are the protagonists of operator-level optimization. For the same matrix multiply, a different loop order and tiling scheme change the memory traffic by tens of times:

```python title="loops.py"
# loop transformations: the same matrix multiply, a different loop order plus a level of tiling, tens of times the difference in memory traffic
def traffic(m, n, k, cache_lines, tile=None, order="ijk"):
    """用一个全相联 LRU 缓存模拟这段矩阵乘要从内存搬多少条缓存行（每行装 8 个 float64）"""
    from collections import OrderedDict
    cache, misses = OrderedDict(), 0

    def touch(addr):
        nonlocal misses
        line = addr // 8
        if line in cache:
            cache.move_to_end(line)
            return
        misses += 1
        if len(cache) == cache_lines:
            cache.popitem(last=False)
        cache[line] = True

    def body(i, j, p):
        touch(0 + i * k + p)                   # A[i][p]
        touch(m * k + p * n + j)               # B[p][j]
        touch(m * k + k * n + i * n + j)       # C[i][j]

    if tile is None:
        loops = {"ijk": lambda: ((i, j, p) for i in range(m) for j in range(n) for p in range(k)),
                 "ikj": lambda: ((i, j, p) for i in range(m) for p in range(k) for j in range(n))}[order]
        for i, j, p in loops():
            body(i, j, p)
    else:
        for i0 in range(0, m, tile):           # tiling: use up a small tile's data before moving on
            for j0 in range(0, n, tile):
                for p0 in range(0, k, tile):
                    for i in range(i0, min(i0 + tile, m)):
                        for p in range(p0, min(p0 + tile, k)):
                            for j in range(j0, min(j0 + tile, n)):
                                body(i, j, p)
    return misses


n = 48
lines = 16                                     # a very small cache: 16 lines = 128 elements
print(f"{n}x{n} 的矩阵乘，缓存只有 {lines} 条行（{lines * 8} 个元素）")
base = traffic(n, n, n, lines, order="ijk")
for name, misses in [("i,j,p 顺序（B 按列访问）", base),
                     ("i,p,j 顺序（B 按行访问）", traffic(n, n, n, lines, order="ikj")),
                     ("分块 8x8", traffic(n, n, n, lines, tile=8)),
                     ("分块 16x16", traffic(n, n, n, lines, tile=16))]:
    print(f"  {name:24s} 缓存缺失 {misses:7d} 次，相对第一种 {misses / base:5.2f}x")
print()
print("循环交换让 B 变成按行访问（连续），分块让一小块数据留在缓存里被反复使用。")
print("注意 16x16 反而比 8x8 差：块太大，一块的工作集装不进这个缓存了。")
print("块大小要和缓存（或共享内存、寄存器）的容量匹配——这正是自动调优要搜的参数。")
```

```text title="output"
48x48 的矩阵乘，缓存只有 16 条行（128 个元素）
  i,j,p 顺序（B 按列访问）         缓存缺失  124704 次，相对第一种  1.00x
  i,p,j 顺序（B 按行访问）         缓存缺失   14400 次，相对第一种  0.12x
  分块 8x8                   缓存缺失    5184 次，相对第一种  0.04x
  分块 16x16                 缓存缺失   15552 次，相对第一种  0.12x

循环交换让 B 变成按行访问（连续），分块让一小块数据留在缓存里被反复使用。
注意 16x16 反而比 8x8 差：块太大，一块的工作集装不进这个缓存了。
块大小要和缓存（或共享内存、寄存器）的容量匹配——这正是自动调优要搜的参数。
```

The four basic loop transformations:

| Transformation | What it does | Why |
| --- | --- | --- |
| **interchange** | swap the nesting order of the loops | make the innermost access contiguous, improving locality and the chance of vectorization |
| **tiling** (blocking) | split one loop level into "across tiles + within a tile" | keep a tile of data in cache / shared memory / registers and reuse it |
| **unrolling** | duplicate the loop body a few times | cut the loop overhead and create instruction-level parallelism (see [computer fundamentals: CPU architecture](root://cs/arch/cpu/)) |
| **fusion** | merge two loops over the same range into one | keep the intermediate result in registers rather than writing it back to memory |

The GPU counterparts: tiling corresponds to "one block handles one tile", with shared memory as the programmer-managed level of cache; unrolling corresponds to `#pragma unroll` and to each thread handling several elements; fusion corresponds to folding elementwise operators into a GEMM's epilogue. Triton letting you write block-level code is exactly "hand the tiling level to the programmer and the thread mapping within a tile to the compiler".

**A larger tile is not always better**: in the simulation above the 16×16 tile is three times worse than 8×8, because one tile's working set no longer fits that cache. On a real GPU the tile size is constrained three ways at once, by shared-memory capacity, register count and occupancy, which is why **auto-tuning** exists (measure several candidates and keep the fastest; `triton.autotune`, TVM's Ansor and CUTLASS's tactic search all do this).

## Code generation and register allocation {#代码生成与寄存器分配}

The last stage turns the IR into target instructions. Two core problems:

- **instruction selection**: which machine instructions one IR instruction becomes. Merging a multiply-add into one FMA, turning `x * 8` into a shift (strength reduction) and turning a chain of elementwise operations into one SIMD instruction all happen here.
- **register allocation**: the IR's virtual registers (`t1`, `t2`, …) are unlimited while the physical ones are not. The classic approach connects variables that are "live at the same time" into an interference graph and colors it, with the variables that cannot be colored **spilled** to memory.

On a GPU the consequences are more immediate: the more registers a thread uses, the fewer warps an SM can hold (lower occupancy); go over and it spills to local memory (device memory in practice) and performance falls off a cliff. The "Used N registers, M bytes spill stores" that `nvcc --ptxas-options=-v` or Nsight Compute shows is this stage's product (see [the execution model and performance fundamentals](../basics/execution.md#占用率occupancy)).

## Multi-level IR: MLIR and the shape of AI compilers {#多层-irmlir-与-ai-编译器的形状}

A traditional compiler has one level of IR (LLVM IR), but deep learning needs optimization at many levels. MLIR's answer is the **dialect**: let several IRs coexist, each describing one level of abstraction, and define the **lowering** rules between them.

One Triton kernel's lowering path is the typical example:

<!-- i18n:diagram 515b0d7989 -->
```text
Python (@triton.jit)
  → Triton IR (ttir)         block-level tensor operations, hardware independent
  → TritonGPU IR (ttgir)     plus layouts, shared memory, asynchronous copies and pipelining
  → LLVM IR                  scalar and vector instructions
  → PTX → cubin              target machine code
```

Side by side, the shape of each compiler becomes clear:

| System | Graph-level IR | Loop / block-level IR | Back end |
| --- | --- | --- | --- |
| PyTorch Inductor | FX graph (captured by Dynamo) | its own loop IR | generates Triton (GPU) / C++ + OpenMP (CPU) |
| Triton | — | ttir → ttgir | LLVM → PTX |
| TVM | Relay / Relax | TensorIR (schedulable) | LLVM, CUDA C, various accelerators |
| XLA | HLO | — | LLVM, TPU instructions |
| TensorRT | its own network graph | several tactics per operator | measured selection + fusion |

When reading source, recognizing "which level this IR is and which class of transformation this pass is doing" is far more useful than memorizing the class names.

!!! interview "Answering in an interview"
    Asked about compiler construction: give the classic stages first (lexing → parsing → semantics → IR → optimization → code generation), then map them straight onto an AI compiler. Dynamo's graph capture is the front end, fusion and constant folding happen on the graph-level IR, tiling, interchange and vectorization on the loop-level IR, and the back end generates Triton or LLVM. Explain why SSA makes CSE and dataflow analysis simpler. For fusion, give a quantitative reason (decode is bandwidth-bound and fusion saves memory traffic and kernel launches). For tiling, stress that the tile size has to match that level of storage's capacity and that too large is worse, hence auto-tuning. Finally, be able to name MLIR's dialects and lowering, Triton's ttir → ttgir → LLVM → PTX chain, and the fact that register allocation shows up on a GPU as occupancy and spilling.

## Exercises {#练习}

**1. Optimize a stretch of IR by hand.** The three-address code below computes `(x + y) * (x + y) + (x - y)`. Which instructions can go? How many remain?

    ```text
    t1 = x + y
    t2 = x + y
    t3 = t1 * t2
    t4 = x - y
    t5 = x + y
    t6 = t3 + t4
    t7 = t1 * t4
    out = t6
    ```

??? success "Answer"
    `t2` and `t5` are both recomputations of `x + y` (CSE points the references at `t1`); `t7` is computed and unused (DCE deletes it). What remains:

    ```text
    t1 = x + y
    t3 = t1 * t1
    t4 = x - y
    t6 = t3 + t4
    out = t6
    ```

    8 instructions down to 5. Note the order: CSE first makes `t7`'s operands `t1` and `t4`, and only then can DCE decide nobody uses it.

**2. Why a compiler dares not reassociate floating-point addition.** `(a + b) + c` and `a + (b + c)` can differ in floating point; give an example. What does that mean for an inference system?

??? success "Answer"
    Take `a = 1e16`, `b = -1e16`, `c = 1`: `(a + b) + c = 1`, while `a + (b + c) = 0` (`b + c` is still `-1e16` in double). Floating-point addition is not associative, so a compiler does not reassociate by default (`-ffast-math` allows it).

    The effect on an inference system: with the same model and the same weights, a different batch size gives a different reduction order, so results differ slightly and the sampled token may change. That is the problem "deterministic inference" sets out to solve (a fixed reduction order, batch-invariant kernels).

**3. Choosing a tile size.** A GPU kernel has each block produce a `BM × BN` output tile, which needs a `BM × BK` and a `BK × BN` input tile in shared memory (BF16). Shared memory is at most 228 KB per SM and you want at least 2 blocks resident per SM. With `BK = 64`, what is the largest `BM = BN`?

??? success "Answer"
    Shared memory per block = `(BM × 64 + 64 × BN) × 2 bytes`. With `BM = BN = x`: `(64x + 64x) × 2 = 256x` bytes. Two resident blocks require `2 × 256x ≤ 228 × 1024`, giving `x ≤ 456`, and in practice a power of two, `x = 256` (multi-stage pipelining multiplies by the number of stages, so with 3 stages `x = 128` is more realistic).

    The point of the exercise: the tile size's upper bound comes from storage capacity and the residency requirement, its lower bound from "a matrix multiply has to be large enough to keep the Tensor Cores fed", and the few candidates in between are chosen by measurement. That is auto-tuning's search space.

**4. Name the level.** At which stage of compilation does each of these happen? (a) `torch.compile` reports a "graph break"; (b) Nsight shows "Used 168 registers, 32 bytes spill stores"; (c) two elementwise operators became one kernel; (d) a tensor's layout in the ttgir is `blocked<{sizePerThread = [1, 8]}>`.

??? success "Answer"
    (a) The front end: Dynamo met a construct it cannot handle while capturing bytecode and cut the graph; (b) the back end's register allocation: 168 registers used and 32 bytes spilled anyway, which lowers occupancy; (c) graph-level optimization's operator fusion; (d) the data layout decided when Triton's block-level IR lowers to the GPU IR, with each thread taking 8 contiguous elements along this dimension, exactly matching a vectorized access.

## Summary {#小结}

- [x] The classic stages: lexing → parsing → semantics → IR → optimization passes → code generation; an AI compiler has one of each (Dynamo's capture, graph-level and loop-level passes, generating Triton/LLVM).
- [x] SSA turns "the same value" into a comparison of names, which simplifies CSE and dataflow analysis; the order of the passes affects the result.
- [x] Floating point is not associative, so compilers do not reassociate by default, which is the root of the deterministic-inference problem.
- [x] The four loop transformations: interchange, tiling, unrolling, fusion; the tile size has to match the storage capacity and too large is worse, hence auto-tuning.
- [x] Register allocation shows up on a GPU directly as occupancy and spilling.
- [x] MLIR supports multi-level IR through dialects; Triton's path is ttir → ttgir (adding layouts and pipelining) → LLVM → PTX.
