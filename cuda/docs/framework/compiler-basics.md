# 编译原理速成：读懂 AI 编译器需要的那些概念

<p class="lead">推理岗的招聘要求里常见"熟悉编译原理"，但要的不是自己写一个 C 编译器，而是能读懂 AI 编译器在做什么：`torch.compile` 到底变换了什么、Triton 的 ttir/ttgir 是什么、为什么融合能省访存、自动调优在搜什么、MLIR 的"方言"是什么意思。这一章用三个能跑的小程序把前端（词法、语法、语法树）、中间表示与优化 pass（SSA、常量折叠、公共子表达式消除、死代码消除）、循环变换（交换、分块）讲清楚，再把它们和 Inductor、Triton、TVM、MLIR 一一对上号。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个编译器的经典流程分哪几段？AI 编译器与之对应的是什么？
    2. 什么是中间表示（IR）？为什么要有 SSA 形式？
    3. 常量折叠、公共子表达式消除、死代码消除各消除了什么浪费？
    4. 循环分块（tiling）为什么能加速？块大小为什么不是越大越好？
    5. MLIR 的"方言"（dialect）解决了什么问题？
    6. Triton 的 ttir 和 ttgir 差在哪？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 词法分析（字符流 → 记号）→ 语法分析（记号 → 语法树）→ 语义分析与类型检查 → 生成中间表示 → 优化（一串 pass）→ 代码生成与寄存器分配。AI 编译器的对应：前端是 Python 层的图捕获（Dynamo 的字节码分析、`torch.fx` 的符号追踪），中间是图 IR 与循环 IR 上的 pass，后端是生成 Triton / CUDA / LLVM 代码。
    2. IR 是介于源码和目标代码之间的表示，优化都在它上面做——这样 N 种语言 × M 种硬件只需要 N + M 个部件，而不是 N × M 个。SSA（静态单赋值）要求每个变量只被赋值一次，于是"这两个表达式是不是同一个值"变成了简单的名字比较，数据流分析和 CSE 都因此变简单。
    3. 常量折叠：编译期就能算出来的表达式不要留到运行时（`2 * 3 + x` → `6 + x`）。公共子表达式消除：同一个表达式只算一次。死代码消除：算了但没人用的指令删掉。本章的例子里 11 条指令降到 8 条，而结果完全一致。
    4. 分块让一小块数据留在快的存储里（缓存、共享内存、寄存器）被反复使用，把"每次都从内存取"变成"取一次用很多次"。块太大时一块的工作集装不进那层存储，反而更差——本章的模拟里 16×16 的块比 8×8 差三倍。所以块大小要和存储容量匹配，这正是自动调优要搜索的参数。
    5. 传统编译器只有一层 IR（比如 LLVM IR），而深度学习需要在多个抽象层次上做优化：图级（算子融合）、循环级（分块）、向量级、硬件指令级。MLIR 允许同时存在多套 IR（方言），并定义它们之间的下降（lowering）规则，Triton、IREE、很多国产加速卡的编译器都建在它上面。
    6. ttir 是块级的、与硬件无关的张量运算；ttgir 在此基础上给每个张量标上**布局**（这块数据在线程之间怎么分布），并插入共享内存分配、异步拷贝和流水线。性能不对时先看 ttgir 的 layout 和流水线，再看 PTX 里有没有用上 `mma` / `wgmma`。

## 前端：从字符到语法树

编译器的第一段工作是把一串字符变成结构化的树。写一个能处理四则运算、幂和变量的微型表达式语言：

```python title="minilang.py"
# 编译器前端的三步：词法分析（字符 -> 记号）、语法分析（记号 -> 语法树）、求值或生成代码
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


PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "**": 3}     # 越大结合得越紧


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
            node = BinOp("-", Num(0.0), self.parse(3))     # 一元负号
        else:
            raise SyntaxError(f"意外的记号：{value!r}")
        while True:
            kind, op = self.peek()
            if kind != "op" or op not in PRECEDENCE or PRECEDENCE[op] < min_prec:
                return node
            self.next()
            right = self.parse(PRECEDENCE[op] + (0 if op == "**" else 1))   # ** 右结合
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

```text title="输出"
2 * 3 + x * 1      语法树 ((2 * 3) + (x * 1))          折叠后 (6 + x)
a + 2 * 3 * 4      语法树 (a + ((2 * 3) * 4))          折叠后 (a + 24)
x * (2 - 2)        语法树 (x * (2 - 2))                折叠后 0
2 ** 3 ** 2        语法树 (2 ** (3 ** 2))              折叠后 512
-x + 1 - 1         语法树 (((0 - x) + 1) - 1)          折叠后 (((0 - x) + 1) - 1)
```

三件事：

- **词法分析**就是一个正则驱动的循环：数字、标识符、运算符各归一类。真实编译器还要处理注释、字符串、缩进（Python）。
- **语法分析**这里用的是 **Pratt 解析器**（优先级爬升）：用一张优先级表决定"`2 * 3 + x` 里谁先结合"，比给每个优先级写一个递归函数短得多。注意 `**` 是**右结合**的，所以递归时传的最小优先级不加一——`2 ** 3 ** 2` 的语法树是 `2 ** (3 ** 2)` = 512，不是 `(2 ** 3) ** 2` = 64。
- **常量折叠**是在语法树上做的第一个优化：两个操作数都是常量就直接算，再顺手做几个代数化简（`x * 1 → x`、`x * 0 → 0`）。注意最后一个例子 `-x + 1 - 1` 没有被化简成 `-x`：因为语法树是左结合的 `((0-x) + 1) - 1`，要消掉需要"结合律重排"，而浮点运算不满足结合律——编译器默认不敢做这种变换，除非你开 `-ffast-math`。这正是[确定性推理](serving://topics/deterministic/)要小心的地方。

Dynamo 做的事情在概念上相同，只是输入不是字符而是 **Python 字节码**：它一边解释字节码一边记录对张量的操作，遇到无法处理的东西就"图中断"（graph break），把能捕获的部分交给后端（见 [torch.compile](compile.md)）。

## 中间表示与优化 pass

![图：编译器的层层下降——图级 IR、循环级 IR、目标 IR、机器码](../assets/figures/ir-lowering.svg){.aig-svg}

优化不在语法树上做，而在**中间表示**上做。最常见的形式是三地址码：每条指令最多两个操作数、一个结果。再加上 **SSA**（每个变量只被赋值一次），数据流分析就变得很简单。

```python title="passes.py"
# 中间表示（SSA 风格的三地址码）和几个最基本的优化 pass
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


# 一段手写的 IR：把 (a+b)*(a+b) + (a+b)*2 存到 out，中间故意重复算了几次 a+b，还有一条没人用的指令
program = [
    Instr("t1", "load", ("a",)),
    Instr("t2", "load", ("b",)),
    Instr("t3", "+", ("t1", "t2")),
    Instr("t4", "+", ("t1", "t2")),            # 和 t3 完全一样
    Instr("t5", "*", ("t3", "t4")),
    Instr("t6", "const", (2.0,)),
    Instr("t7", "+", ("t1", "t2")),            # 又一次
    Instr("t8", "*", ("t7", "t6")),
    Instr("t9", "+", ("t5", "t8")),
    Instr("t10", "*", ("t1", "t2")),           # 算了但没人用
    Instr("out", "store", ("t9",)),
]


def cse(instrs):
    """公共子表达式消除：同样的 (op, 参数) 只算一次，后面的引用改成第一次的结果"""
    seen, rename, out = {}, {}, []
    for ins in instrs:
        args = tuple(rename.get(a, a) for a in ins.args)
        key = (ins.op, args)
        if ins.op in ("+", "*") and key in seen:
            rename[ins.dst] = seen[key]        # 重复的表达式：直接复用
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

```text title="输出"
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

这就是编译器的骨架：**建 IR → 在 IR 上跑一串 pass → 每个 pass 保证语义不变**。真实编译器里的 pass 有几十上百个（内联、循环不变量外提、强度削弱、尾递归消除、别名分析……），但结构都一样。

几个和 AI 编译器直接相关的点：

- **为什么要 SSA**：上面的 CSE 之所以能写得这么短，是因为 `t3` 一旦定义就不会变，"`t1 + t2` 算过了"这个结论永远有效。变量可以重新赋值时，就必须先做复杂的到达定值分析。
- **pass 的顺序很重要**：先 CSE 再 DCE 才能删掉更多东西；有些 pass 要反复跑到不动点。
- **代价模型**决定要不要做一个变换。图级的融合就是一个典型例子：融合省下访存，但可能增加寄存器压力（见 [AI 编译器全景](compilers.md#图级融合省下的是访存)）。

## 循环变换：分块、交换与融合

算子级优化的主角是循环。同一个矩阵乘，循环顺序和分块方式不同，访存量能差几十倍：

```python title="loops.py"
# 循环变换：同一个矩阵乘，换一种循环顺序、加一层分块，访存量差几十倍
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
        for i0 in range(0, m, tile):           # 分块：先把一小块的数据全用完再换
            for j0 in range(0, n, tile):
                for p0 in range(0, k, tile):
                    for i in range(i0, min(i0 + tile, m)):
                        for p in range(p0, min(p0 + tile, k)):
                            for j in range(j0, min(j0 + tile, n)):
                                body(i, j, p)
    return misses


n = 48
lines = 16                                     # 很小的缓存：16 条行 = 128 个元素
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

```text title="输出"
48x48 的矩阵乘，缓存只有 16 条行（128 个元素）
  i,j,p 顺序（B 按列访问）         缓存缺失  124704 次，相对第一种  1.00x
  i,p,j 顺序（B 按行访问）         缓存缺失   14400 次，相对第一种  0.12x
  分块 8x8                   缓存缺失    5184 次，相对第一种  0.04x
  分块 16x16                 缓存缺失   15552 次，相对第一种  0.12x

循环交换让 B 变成按行访问（连续），分块让一小块数据留在缓存里被反复使用。
注意 16x16 反而比 8x8 差：块太大，一块的工作集装不进这个缓存了。
块大小要和缓存（或共享内存、寄存器）的容量匹配——这正是自动调优要搜的参数。
```

四个基本的循环变换：

| 变换 | 做什么 | 为什么 |
| --- | --- | --- |
| **交换**（interchange） | 调换循环的嵌套顺序 | 让最内层的访问连续，提高局部性和向量化的可能 |
| **分块**（tiling / blocking） | 把一层循环拆成"块外 + 块内"两层 | 让一块数据留在缓存 / 共享内存 / 寄存器里被反复使用 |
| **展开**（unrolling） | 把循环体复制几份 | 减少循环开销，制造指令级并行（见[计算机基础：CPU 体系结构](root://cs/arch/cpu/)） |
| **融合**（fusion） | 把两个遍历同样范围的循环合成一个 | 中间结果留在寄存器里，不写回内存 |

GPU 上的对应：分块对应"每个 block 处理一个 tile"、共享内存就是程序员管理的那一级缓存；展开对应 `#pragma unroll` 和每个线程处理多个元素；融合对应把逐元素算子并进 GEMM 的尾声（epilogue）。Triton 让你写块级的代码，就是把"分块"这一层交给程序员、把块内的线程映射交给编译器。

**块大小不是越大越好**：上面的模拟里 16×16 的块反而比 8×8 差三倍，因为一块的工作集装不进那个缓存了。在真实 GPU 上，块大小同时受共享内存容量、寄存器数量、占用率三重约束——这就是为什么要**自动调优**（在几组候选里实测挑最快的，`triton.autotune`、TVM 的 Ansor、CUTLASS 的 tactic 搜索都是在做这件事）。

## 代码生成与寄存器分配

最后一段是把 IR 变成目标指令。两个核心问题：

- **指令选择**：一条 IR 指令对应哪几条机器指令。乘加合成一条 FMA、把 `x * 8` 变成移位（强度削弱）、把一串逐元素运算变成一条 SIMD 指令，都在这一步。
- **寄存器分配**：IR 里的虚拟寄存器（`t1`、`t2`……）数量不限，物理寄存器是有限的。经典做法是把"同时活跃"的变量连成冲突图再做图着色，着不上色的变量**溢出**到内存（spill）。

GPU 上这件事的后果更直接：每个线程用的寄存器越多，一个 SM 能驻留的 warp 越少（占用率下降）；用超了就溢出到本地内存（实际在显存里），性能断崖式下跌。用 `nvcc --ptxas-options=-v` 或 Nsight Compute 看到的 "Used N registers, M bytes spill stores" 就是这一步的产物（见[执行模型与性能基础](../basics/execution.md#占用率occupancy)）。

## 多层 IR：MLIR 与 AI 编译器的形状

传统编译器只有一层 IR（LLVM IR），但深度学习要在很多个层次上优化。MLIR 的答案是**方言**（dialect）：允许同时存在多套 IR，每套描述一个抽象层次，并定义它们之间的**下降**（lowering）规则。

一个 Triton kernel 的下降路径就是典型例子：

```text
Python（@triton.jit）
  → Triton IR（ttir）        块级张量运算，与硬件无关
  → TritonGPU IR（ttgir）    加上布局（layout）、共享内存、异步拷贝与流水线
  → LLVM IR                  标量与向量指令
  → PTX → cubin              目标机器码
```

对照着看，各家编译器的形状就清楚了：

| 系统 | 图级 IR | 循环 / 块级 IR | 后端 |
| --- | --- | --- | --- |
| PyTorch Inductor | FX 图（Dynamo 捕获） | 自家的循环 IR | 生成 Triton（GPU）/ C++ + OpenMP（CPU） |
| Triton | — | ttir → ttgir | LLVM → PTX |
| TVM | Relay / Relax | TensorIR（可调度） | LLVM、CUDA C、各种加速卡 |
| XLA | HLO | — | LLVM、TPU 指令 |
| TensorRT | 自家的网络图 | 每个算子多个 tactic | 实测挑选 + 融合 |

读源码时，认出"这是哪一层的 IR、这个 pass 在做哪类变换"，比记住具体的类名有用得多。

!!! interview "面试怎么答"
    被问编译原理：先给经典流程（词法 → 语法 → 语义 → IR → 优化 → 代码生成），再立刻映射到 AI 编译器——Dynamo 捕获图相当于前端，图级 IR 上做融合与常量折叠，循环级 IR 上做分块、交换、向量化，后端生成 Triton 或 LLVM。解释 SSA 为什么让 CSE 和数据流分析变简单。讲融合时给出量化理由（decode 受带宽限制，融合省的是访存和 kernel 启动）。讲分块时强调"块大小要匹配那一级存储的容量，太大反而更差，所以需要自动调优"。最后能说出 MLIR 的方言与下降、Triton 的 ttir → ttgir → LLVM → PTX 这条链路，以及寄存器分配在 GPU 上表现为占用率与溢出。

## 练习

**1. 手工优化一段 IR。** 下面这段三地址码计算 `(x + y) * (x + y) + (x - y)`，哪些指令可以删？优化后剩几条？

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

??? success "参考答案"
    `t2` 和 `t5` 都是 `x + y` 的重复计算（CSE 后把引用改成 `t1`）；`t7` 算了但没人用（DCE 删掉）。剩下：

    ```text
    t1 = x + y
    t3 = t1 * t1
    t4 = x - y
    t6 = t3 + t4
    out = t6
    ```

    8 条降到 5 条。注意顺序：先 CSE 让 `t7` 的操作数变成 `t1`、`t4`，再 DCE 才能判定它没人用。

**2. 为什么编译器不敢重排浮点加法。** `(a + b) + c` 和 `a + (b + c)` 在浮点下可能不等，举一个例子。这对推理系统意味着什么？

??? success "参考答案"
    取 `a = 1e16`、`b = -1e16`、`c = 1`：`(a + b) + c = 1`，而 `a + (b + c) = 0`（`b + c` 在 double 下仍是 `-1e16`）。浮点加法不满足结合律，所以编译器默认不做这种重排（`-ffast-math` 会放开）。

    对推理系统的影响：同一个模型、同一份权重，batch 大小不同会让归约的顺序不同，结果就有微小差别，进而可能改变采样出的 token——这就是"确定性推理"要解决的问题（固定归约顺序、batch 无关的 kernel）。

**3. 选块大小。** 一个 GPU kernel 每个 block 处理一个 `BM × BN` 的输出块，需要把 `BM × BK` 和 `BK × BN` 两块输入放进共享内存（BF16）。共享内存每个 SM 最多 228 KB，希望每个 SM 至少驻留 2 个 block。`BK = 64` 时，`BM = BN` 最大能取多少？

??? success "参考答案"
    每个 block 的共享内存 = `(BM × 64 + 64 × BN) × 2 字节`。取 `BM = BN = x`：`(64x + 64x) × 2 = 256x` 字节。两个 block 驻留要求 `2 × 256x ≤ 228 × 1024`，得 `x ≤ 456`，实践中取 2 的幂即 `x = 256`（多级流水还要再乘上级数，比如 3 级流水时 `x = 128` 更现实）。

    这题的意义在于：块大小的上界由存储容量和驻留要求决定，下界由"矩阵乘要足够大才能喂饱 Tensor Core"决定，中间的几个候选靠实测选——这就是自动调优的搜索空间。

**4. 认一认这是哪一层。** 下面四个现象各发生在编译的哪一层？（a）`torch.compile` 报告 "graph break"；（b）Nsight 显示 "Used 168 registers, 32 bytes spill stores"；（c）两个逐元素算子被合成了一个 kernel；（d）ttgir 里某个张量的 layout 是 `blocked<{sizePerThread = [1, 8]}>`。

??? success "参考答案"
    （a）前端：Dynamo 在捕获字节码时遇到无法处理的构造，把图切开了；（b）后端的寄存器分配：用了 168 个寄存器还溢出了 32 字节，会压低占用率；（c）图级优化的算子融合；（d）Triton 的块级 IR 下降到 GPU IR 时决定的数据布局——每个线程在这个维度上拿 8 个连续元素，正好对应向量化访存。

## 小结

- [x] 经典流程：词法 → 语法 → 语义 → IR → 优化 pass → 代码生成；AI 编译器一一对应（Dynamo 捕获、图级与循环级 pass、生成 Triton/LLVM）。
- [x] SSA 让"同一个值"变成名字比较，CSE 和数据流分析因此变简单；pass 的顺序影响效果。
- [x] 浮点不满足结合律，编译器默认不重排——这正是确定性推理的根源问题。
- [x] 循环变换四件套：交换、分块、展开、融合；块大小要匹配存储容量，太大反而更差，所以要自动调优。
- [x] 寄存器分配在 GPU 上直接表现为占用率与溢出。
- [x] MLIR 用方言支持多层 IR；Triton 的路径是 ttir → ttgir（加 layout 与流水线）→ LLVM → PTX。
