# dispatcher 与自定义算子

<p class="lead">调用 <code>torch.add(a, b)</code> 时，PyTorch 要决定：要不要记录 autograd、要不要做自动混合精度的类型转换、最后调用 CPU 还是 CUDA 的 kernel。负责这件事的是 dispatcher。理解它，才能正确地注册自定义算子，让它同时支持 autograd、<code>torch.compile</code>、meta 设备上的形状推导——推理框架里的每一个自定义 kernel 都要过这一关。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一次 `torch.mm` 调用要经过 dispatcher 的哪几层？
    2. aten 算子是什么？怎样看到一段模型代码实际调用了哪些 aten 算子？
    3. 注册自定义算子时，为什么要声明哪些参数会被原地修改？
    4. 什么是 meta 设备、fake tensor？它们在推理框架里用来做什么？
    5. 用 pybind11 直接暴露一个函数，和用 `torch.library` 注册一个算子，在 `torch.compile` 下有什么区别？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 从外到里依次经过：自动混合精度（Autocast，需要时转换输入精度）→ Autograd（输入需要梯度时记录反向节点）→ ADInplaceOrView（维护原地修改和视图的版本计数）→ 设备 kernel（CPU / CUDA）。
    2. aten 是 PyTorch 的底层算子库（`aten::mm`、`aten::silu`……），模型代码最终都落到它们上。用 `TorchDispatchMode` 拦截每一次落到最底层的调用，打印 `func` 即可看到。
    3. 编译器和函数化（functionalization）靠 schema 判断算子有没有副作用：声明了原地修改，编译器就不会删掉、重排或复用它的结果；声明错了，eager 下没问题，`torch.compile` 下会静默地算错。
    4. meta 设备上的张量只有形状和类型、没有数据；fake tensor 更进一步，假装在某个真实设备上。推理框架用它们不占内存地构建模型、数参数、规划显存和切分，`torch.compile` 也用 fake tensor 推导所有中间结果的形状。
    5. pybind11 暴露的函数对 torch.compile 是黑盒，Dynamo 只能在这里断开图；`torch.library` 注册的算子有 schema、fake 实现（和可选的反向），可以被完整地捕获进图里，也能在 meta 设备上推导形状。

## 从 Python 调用到 kernel

![图：一次 torch.add 怎么走到 kernel——按 dispatch key 一层层分发](../assets/figures/dispatcher-keys.svg){.aig-svg}

PyTorch 的每个算子（`aten::mm`、`aten::silu`……）在 dispatcher 里有一张表：按**分发键**（dispatch key）登记了不同的实现。一次调用会按优先级依次经过张量身上的每个键：

```python title="dispatch_keys.py"
import torch

x = torch.randn(3)
print(torch._C._dispatch_keys(x))
```

```text title="输出"
DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU, AutocastCPU)
```

从外往里：`AutocastCPU`（自动混合精度：需要时先把输入转成低精度）→ `AutogradCPU`（输入需要梯度时记录反向节点）→ `ADInplaceOrView`（给原地修改和视图维护版本计数）→ `CPU`（真正的 kernel）。每一层处理完自己的事，把调用"再分发"给下一层。张量在 GPU 上时，最后一层就是 `CUDA`；在 meta 设备上就是 `Meta`（只算形状、不算数据）。

模型代码最终都会落到这一组 **aten 算子**上。用 `TorchDispatchMode` 可以拦截每一次落到最底层的调用，看清一段代码到底执行了什么：

```python title="log_aten_ops.py"
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class LogOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


torch.manual_seed(0)
x = torch.randn(2, 16)
w = torch.ones(16)
lin = torch.nn.Linear(16, 32, bias=False)
with LogOps() as log:
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w   # RMSNorm
    y = torch.nn.functional.silu(lin(h))                               # Linear + SiLU
for op in log.ops:
    print(op)
```

```text title="输出"
aten.pow.Tensor_Scalar
aten.mean.dim
aten.add.Tensor
aten.rsqrt.default
aten.mul.Tensor
aten.mul.Tensor
aten.t.default
aten.mm.default
aten.silu.default
```

一个 RMSNorm 在 eager 模式下是 6 个独立的 kernel，每个都要把整个张量读一遍、写一遍——这就是推理框架要写融合的 RMSNorm kernel、要用 `torch.compile` 的原因（下一章）。
`TorchDispatchMode` 也是很多工具的基础：统计 FLOPs、检查数值、记录算子序列做回放，都是这样拦截的。

## 注册自定义算子

推理框架的自定义 kernel（融合的 RMSNorm、写 KV Cache、MoE 的路由和排序）都要注册成 dispatcher 里的算子。PyTorch 2.4 起推荐用 `torch.library.custom_op`：

```python title="custom_ops.py"
import torch


@torch.library.custom_op("demo::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # 真实系统里这里调用自己的 CUDA / Triton kernel；这里用 PyTorch 代替
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@rms_norm.register_fake
def _(x, w, eps):
    return torch.empty_like(x)   # 只描述输出的形状和类型，不做计算


@torch.library.custom_op("demo::store_kv", mutates_args=("cache",))
def store_kv(cache: torch.Tensor, slots: torch.Tensor, value: torch.Tensor) -> None:
    cache.index_copy_(0, slots, value)   # 把新 token 的 KV 写进缓存的指定槽位


print(torch.ops.demo.rms_norm.default._schema)
print(torch.ops.demo.store_kv.default._schema)

torch.manual_seed(0)
x, w = torch.randn(2, 16), torch.ones(16)
print("CPU 上计算：", tuple(torch.ops.demo.rms_norm(x, w, 1e-6).shape))

m = torch.ops.demo.rms_norm(torch.empty(4096, 8192, device="meta"), torch.empty(8192, device="meta"), 1e-6)
print("meta 设备上只推导形状：", tuple(m.shape), m.device)

cache = torch.zeros(8, 4)
torch.ops.demo.store_kv(cache, torch.tensor([1, 5]), torch.ones(2, 4))
print("写入的槽位：", cache.sum(dim=1).nonzero().flatten().tolist())


def block(x, w):
    return torch.ops.demo.rms_norm(x, w, 1e-6).relu()


compiled = torch.compile(block, fullgraph=True, backend="eager")   # fullgraph：整段代码必须被捕获成一张图
print("torch.compile 能完整捕获：", torch.allclose(compiled(x, w), block(x, w)))
```

```text title="输出"
demo::rms_norm(Tensor x, Tensor w, float eps) -> Tensor
demo::store_kv(Tensor(a0!) cache, Tensor slots, Tensor value) -> ()
CPU 上计算： (2, 16)
meta 设备上只推导形状： (4096, 8192) meta
写入的槽位： [1, 5]
torch.compile 能完整捕获： True
```

几个要点：

- **schema** 是从类型注解生成的：`Tensor(a0!)` 表示 `cache` 会被原地修改。编译器和函数化（functionalization）靠它知道这个算子有副作用，不能被删除、重排或者当成纯函数缓存结果——写错了会得到静默的错误结果；
- **fake 实现**（`register_fake`）只根据输入的形状、类型计算输出的形状、类型。`torch.compile` 在编译期用它推导形状，meta 设备也用它；没有它，编译器只能在这里断开图；
- 需要反向时用 `torch.library.register_autograd` 注册，和上一章的 `autograd.Function` 同理；
- C++ / CUDA 实现用 `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` 注册，效果完全一样（见 C++ 手册的 [pybind11 与 PyTorch C++ 扩展](cpp://engineering/python-binding/)）。vLLM 的 `torch.ops._C.*`、SGLang 的 `sgl_kernel` 都是这样注册的。

## meta 设备与 fake tensor

**meta 设备**上的张量只有元数据、没有数据，任何运算都只推导输出的形状和类型。推理框架用它来：

- **不分配内存地构建模型**：先在 meta 设备上实例化，数清参数量、规划显存和张量并行的切分，再把真正的权重一块块加载到目标设备上（避免"先在 CPU 上建一个完整模型再搬到 GPU"的双倍内存）；
- **估算峰值显存**：用假输入走一遍前向，记录每个中间张量的大小。

**fake tensor** 更进一步：它假装自己在某个真实设备上（比如 `cuda:0`），但不分配内存。`torch.compile` 在编译期就是用 fake tensor 跑一遍模型来推导所有形状的：

```python title="meta_model.py"
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

with torch.device("meta"):   # 不分配任何内存地构建一个 70B 模型量级的 MLP
    mlp = torch.nn.Sequential(torch.nn.Linear(8192, 28672, bias=False), torch.nn.Linear(28672, 8192, bias=False))
n = sum(p.numel() for p in mlp.parameters())
print(f"参数量 {n}，bf16 下 {n * 2 / 2**30:.2f} GiB，权重在 {mlp[0].weight.device} 上")

with FakeTensorMode():
    x = torch.empty(4, 128, 8192)        # 一个 batch 的隐藏状态，但不占内存
    h = torch.nn.functional.silu(x @ torch.empty(8192, 28672))
    print(type(h).__name__, tuple(h.shape), f"这个中间激活在 bf16 下需要 {h.numel() * 2 / 2**20:.0f} MiB")
```

```text title="输出"
参数量 469762048，bf16 下 0.88 GiB，权重在 meta 上
FakeTensor (4, 128, 28672) 这个中间激活在 bf16 下需要 28 MiB
```

!!! interview "怎么讲清楚"
    讲 dispatcher：一次调用按分发键逐层处理——自动混合精度 → autograd → 版本计数 → 设备 kernel；模型最终都落到 aten 算子上，`TorchDispatchMode` 能拦截并记录它们（统计 FLOPs、回放算子序列都靠它）。推理框架的自定义 kernel 用 `torch.library.custom_op` 或 C++ 的 `TORCH_LIBRARY` 注册：schema 要如实声明原地修改（否则 torch.compile 下会静默算错），要提供 fake 实现，需要时注册反向。meta 设备和 fake tensor 只有形状没有数据，用来不占内存地建模型、规划显存，也是编译器推导形状的方式。

## 练习

1. 用 `TorchDispatchMode` 写一个"算子计数器"，统计 `torch.nn.functional.scaled_dot_product_attention` 在 CPU 上（输入 `[1, 8, 128, 64]`，因果掩码）落到了哪些 aten 算子上。

??? success "参考答案"
    ```python title="sdpa_ops.py"
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode


    class CountOps(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = {}

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            self.counts[name] = self.counts.get(name, 0) + 1
            return func(*args, **(kwargs or {}))


    q = k = v = torch.randn(1, 8, 128, 64)
    with CountOps() as c:
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    for name, n in sorted(c.counts.items()):
        print(name, n)
    ```

    ```text title="输出"
    aten._scaled_dot_product_flash_attention_for_cpu.default 1
    ```

    CPU 上 SDPA 直接落到一个融合的 FlashAttention 实现上，只有一个 aten 算子；如果手写 `softmax(q @ k^T / sqrt(d)) @ v`，就会看到 `mm`、`div`、`masked_fill`、`softmax`、`mm` 等一串算子。这正是融合注意力的意义。

2. 一个自定义算子 `demo::append_kv(Tensor cache, Tensor new) -> Tensor` 在实现里原地写了 `cache`，但注册时写成了 `mutates_args=()`。在 eager 模式下一切正常，在 `torch.compile` 下可能出什么问题？

??? success "参考答案"
    编译器认为它是纯函数：输入相同输出就相同、没有副作用。于是它可能把两次调用合并成一次、把调用挪到别的位置、或者在结果"没人用"时把整个调用删掉——KV 就没写进缓存，后面的 decode 读到的是旧数据，结果静默地错了。
    PyTorch 在调试模式下会检查"声明为不修改的输入是否被修改了"并报错；声明要和实现一致，是注册自定义算子最重要的一条规则。

## 小结

- [x] dispatcher 按分发键逐层处理一次调用：自动混合精度 → autograd → 版本计数 → 设备 kernel。
- [x] 模型最终落到 aten 算子上；`TorchDispatchMode` 可以拦截并记录它们，是很多分析工具的基础。
- [x] 自定义算子用 `torch.library.custom_op`（或 C++ 的 `TORCH_LIBRARY`）注册：schema 要如实声明原地修改，提供 fake 实现，需要时注册反向。
- [x] meta 设备和 fake tensor 只有形状、没有数据：用来不占内存地构建模型、规划显存，也是 `torch.compile` 推导形状的方式。
