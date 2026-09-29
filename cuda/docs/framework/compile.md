# torch.compile：Dynamo、AOTAutograd 与 Inductor

<p class="lead">eager 模式下，一个 RMSNorm 是 6 个 kernel，每个都把张量读一遍写一遍；decode 一步有几百个小 kernel，CPU 提交的开销占掉一大块。<code>torch.compile</code> 把 Python 代码捕获成一张图，融合算子、生成 kernel，还可以套上 CUDA Graph。vLLM 和 SGLang 都在用它。这一章讲它的三层结构，以及在推理框架里用它时最常遇到的问题：图断开、重新编译和动态形状。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `torch.compile` 的三个组件 Dynamo、AOTAutograd、Inductor 各做什么？
    2. 什么是 graph break？怎么找出它们？
    3. 同一个编译过的函数，换一个 batch 大小调用，会重新编译吗？
    4. Inductor 的"算子融合"在 RMSNorm 上能省下多少访存？
    5. 推理框架为什么常常把模型"分段"编译，并把注意力留在图外？

## 三层结构

1. **Dynamo**（图捕获）：在 Python 字节码层面"符号执行"你的函数，把遇到的张量运算记录成一张 FX 图，同时生成一组 **guard**（比如"输入的形状是 `(4, 16)`、dtype 是 float32"）。下次调用时 guard 都满足就直接复用编译结果，否则重新编译。遇到捕获不了的东西（依赖数据的控制流、未注册的 C 扩展、`print`），就在那里**断开图**，前后各编译一段，中间回到 Python 执行；
2. **AOTAutograd**：把前向图和对应的反向图一起提前生成出来（训练时需要），并做"函数化"——把原地修改改写成纯函数形式，方便后面优化；推理时它主要负责后者；
3. **Inductor**（代码生成）：把图降到循环级别的中间表示，融合能融合的运算，在 GPU 上生成 Triton kernel、在 CPU 上生成 C++ / OpenMP 代码，矩阵乘则调用 cuBLAS / CUTLASS 或自动调优的 Triton 模板。

每一层都可以单独换掉：`backend="eager"` 只做 Dynamo 捕获、不做代码生成，`backend="aot_eager"` 再加上 AOTAutograd——排查问题时常用它们二分"是哪一层出的错"。

## 看 Dynamo 捕获了什么

`backend` 可以是任意一个接收 FX 图的函数。写一个只打印图的后端，就能看到 Dynamo 捕获的结果：

```python title="show_graph.py"
import torch


def show_graph(gm, example_inputs):
    print(gm.code.strip())
    return gm.forward          # 返回一个可调用对象：这里直接用未优化的图


def mlp(x, w):
    return torch.nn.functional.silu(x @ w)


torch.compile(mlp, backend=show_graph)(torch.randn(4, 16), torch.randn(16, 16))
```

```text title="输出"
def forward(self, L_x_ : torch.Tensor, L_w_ : torch.Tensor):
    l_x_ = L_x_
    l_w_ = L_w_
    matmul = l_x_ @ l_w_;  l_x_ = l_w_ = None
    silu = torch.nn.functional.silu(matmul);  matmul = None
    return (silu,)
```

## graph break

依赖张量**值**的 Python 控制流，Dynamo 没法在编译期决定走哪个分支，只能断开图：

```python title="graph_break.py"
import torch


def step(x, w):
    h = x @ w
    if h.sum().item() > 0:      # 依赖数据的 Python 控制流
        h = h * 2
    return torch.relu(h)


exp = torch._dynamo.explain(step)(torch.randn(4, 16), torch.randn(16, 16))
print("捕获了", exp.graph_count, "张图，断开", exp.graph_break_count, "次")
for r in exp.break_reasons:
    print("原因：", str(r.reason).splitlines()[0])
```

```text title="输出"
捕获了 2 张图，断开 1 次
原因： Unsupported Tensor.item() call with capture_scalar_outputs=False
```

每断开一次，就多一次"回到 Python、同步、再进入编译代码"，也失去了跨越断点的融合；在 GPU 上，`.item()` 本身还是一次 CPU-GPU 同步（上一章）。找 graph break 的办法：

- `torch._dynamo.explain(fn)(*args)`，或者设置环境变量 `TORCH_LOGS="graph_breaks"`；
- `torch.compile(fn, fullgraph=True)`：遇到任何断开都直接报错，适合推理框架里"这一段必须是一张完整的图"的场景；
- 修法通常是把依赖数据的判断改写成张量运算（`torch.where`），或者把不支持的调用注册成自定义算子（上一章）。

## guard 与重新编译

guard 里包含输入的形状。形状变了，guard 失败，就要重新编译。PyTorch 的默认策略是：第一次按具体形状编译；某一维第二次出现不同的值时，把这一维标成**动态**，编译一个对任意大小都成立的版本：

```python title="recompile.py"
import torch

compiles = []


def counting_backend(gm, example_inputs):
    compiles.append(len(example_inputs))
    return gm.forward


def mlp(x, w):
    return torch.nn.functional.silu(x @ w)


f = torch.compile(mlp, backend=counting_backend)
w = torch.randn(16, 16)
for batch in (4, 4, 8, 16, 32):
    f(torch.randn(batch, 16), w)
    print(f"batch={batch:<3} 累计编译 {len(compiles)} 次")
```

```text title="输出"
batch=4   累计编译 1 次
batch=4   累计编译 1 次
batch=8   累计编译 2 次
batch=16  累计编译 2 次
batch=32  累计编译 2 次
```

第二次编译出的是动态形状的图（batch 维变成了一个符号），之后的任意 batch 都不再编译。推理服务里，batch 大小、序列长度每一步都在变，常见的做法是：

- 在第一次调用前用 `torch._dynamo.mark_dynamic(x, 0)` 显式标出会变的维度，避免"先按具体形状编一次"；
- 用 `TORCH_LOGS="recompiles"` 查看每次重新编译的原因，防止线上请求触发编译（一次编译几秒到几十秒，就是一次严重的延迟毛刺）；
- 服务启动时用典型形状把所有路径预热一遍。

## Inductor 做了什么：融合

还是 RMSNorm + SiLU。eager 下是 7 个独立的 aten 算子（上一章看到过前 6 个），Inductor 把它们融合成**一个** kernel：

```python title="inductor_fusion.py"
import re

import torch
from torch._inductor.utils import run_and_get_code


def rmsnorm_silu(x, w):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return torch.nn.functional.silu(h)


torch.manual_seed(0)
x, w = torch.randn(8, 4096), torch.randn(4096)
out, codes = run_and_get_code(torch.compile(rmsnorm_silu), x, w)   # 同时拿到结果和生成的代码
print("与 eager 结果一致：", torch.allclose(out, rmsnorm_silu(x, w), atol=1e-5))
print("生成的 kernel：", sorted(set(re.findall(r"cpp_fused_\w+", "\n".join(codes)))))
```

```text title="输出"
与 eager 结果一致： True
生成的 kernel： ['cpp_fused_add_mean_mul_pow_rsqrt_silu_0']
```

kernel 的名字就是被融合进去的运算。在 CPU 上生成的是 C++，在 GPU 上是一个 Triton kernel（`triton_per_fused_...`）。
访存量的变化：eager 下 `pow`、`mul`、`mul`、`silu` 各读写一遍整个张量，`mean` 读一遍，加上中间结果的写回，大约是 10 倍于输入的读写量；融合后每行只读一次输入和权重、写一次输出。对 decode 这种带宽瓶颈的场景，这就是接近 5～10 倍的差距。

设置 `TORCH_LOGS="output_code"` 可以直接打印生成的代码；`TORCH_COMPILE_DEBUG=1` 会把每一步的中间表示都写到磁盘上。

## 编译模式与推理框架

| 用法 | 做了什么 |
| --- | --- |
| `torch.compile(model)` | 默认：捕获 + 融合 + 生成 kernel |
| `mode="reduce-overhead"` | 再套上 CUDA Graph：消除 kernel 提交的 CPU 开销，适合小 batch 的 decode |
| `mode="max-autotune"` | 对矩阵乘等算子做自动调优（在多个 Triton 模板和 cuBLAS 之间测速选最快），编译更慢 |
| `dynamic=True / False` | 一开始就按动态 / 静态形状编译 |

推理框架不会直接 `torch.compile(model)` 了事：

- **分段编译**：注意力要读写分页 KV、依赖块表和每个请求不同的长度，通常作为一个不透明的自定义算子留在图外；vLLM 把模型在注意力处切成若干段，每段单独编译，并为每段、每个 batch 大小录制 CUDA Graph；
- **自定义 pass**：在 FX 图上做特定的模式替换，比如把 "all-reduce + RMSNorm" 融合成一个通信和计算融合的 kernel、把量化与前一个算子融合；
- **编译缓存**：把编译结果缓存到磁盘，避免每次启动都等几分钟。

这些都在推理系统手册的 [CUDA Graphs 与 torch.compile](serving://engine/graphs-compile/) 一章里有更具体的讨论。

## 练习

1. 把本章 `graph_break.py` 里的 `step` 改写成没有 graph break 的版本（结果与原来一致），并用 `fullgraph=True` 验证。

??? success "参考答案"
    把 Python 的 `if` 改写成张量运算：条件和两个分支都在图里计算，用 `torch.where` 选择：

    ```python title="no_break.py"
    import torch


    def step(x, w):
        h = x @ w
        if h.sum().item() > 0:
            h = h * 2
        return torch.relu(h)


    def step_nobreak(x, w):
        h = x @ w
        h = torch.where(h.sum() > 0, h * 2, h)   # 条件是一个 0 维张量，留在图里
        return torch.relu(h)


    torch.manual_seed(0)
    compiled = torch.compile(step_nobreak, fullgraph=True, backend="eager")
    ok = all(torch.allclose(compiled(x, w), step(x, w)) for x, w in [(torch.randn(4, 16), torch.randn(16, 16)) for _ in range(5)])
    print("fullgraph 编译成功，结果一致：", ok)
    ```

    ```text title="输出"
    fullgraph 编译成功，结果一致： True
    ```

    代价是两个分支都要算。对推理来说这通常划算：多算一个逐元素乘法，远比一次同步和一次图断开便宜。

2. 线上服务偶尔出现几秒的延迟毛刺，日志里能看到 Dynamo 的 recompile 信息。可能的原因有哪些？怎么排查和避免？

??? success "参考答案"
    原因：某个输入的形状（batch、序列长度、LoRA 的数量）出现了新值而那一维还没被标成动态；guard 里检查了某个 Python 对象的值（比如一个会变化的配置整数、全局变量）；输入的 dtype 或设备变了；超过了重新编译的上限后回退到 eager。
    排查：`TORCH_LOGS="recompiles"` 看每次失败的是哪个 guard。避免：`mark_dynamic` 标出会变的维度；把会变的标量改成张量输入；启动时用所有典型形状预热；推理框架里通常还会把 batch 补齐到固定的几档（和 CUDA Graph 的档位一致）。

## 小结

- [x] Dynamo 在字节码层面捕获 FX 图并生成 guard；AOTAutograd 生成反向图并函数化；Inductor 融合并生成 Triton / C++ 代码。
- [x] 依赖数据的控制流、`.item()`、未注册的扩展会断开图；用 `explain`、`TORCH_LOGS` 和 `fullgraph=True` 找出来，用张量运算或自定义算子消除。
- [x] guard 失败会重新编译；会变的维度要标成动态，启动时预热，避免线上编译。
- [x] Inductor 把一串逐元素运算和归约融合成一个 kernel，对带宽瓶颈的推理收益最大。
- [x] 推理框架分段编译、把注意力留在图外、为每段录制 CUDA Graph，并在 FX 图上做自定义的融合 pass。
