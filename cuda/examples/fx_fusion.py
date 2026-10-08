import operator

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

# 逐元素运算和归约：本例里所有算子都属于这两类
POINTWISE_FN = {operator.add, operator.mul, operator.truediv, torch.rsqrt, torch.nn.functional.silu}
POINTWISE_METHOD = {"pow"}
REDUCE_METHOD = {"mean", "sum"}


def rmsnorm_silu_mul(x, w, up):
    h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    return torch.nn.functional.silu(h) * up


def nbytes(node):
    meta = node.meta.get("tensor_meta")
    return meta.shape.numel() * meta.dtype.itemsize if meta is not None else 0


def fusible(node):
    if node.op == "call_function":
        return node.target in POINTWISE_FN
    if node.op == "call_method":
        return node.target in POINTWISE_METHOD | REDUCE_METHOD
    return False


gm = fx.symbolic_trace(rmsnorm_silu_mul)
x, w, up = torch.randn(8, 4096), torch.randn(4096), torch.randn(8, 4096)
ShapeProp(gm).propagate(x, w, up)   # 在图上推导每个节点的形状和类型

eager = 0
ops = [n for n in gm.graph.nodes if fusible(n)]
for n in ops:   # eager：每个算子读自己的全部输入、写自己的输出
    read = sum(nbytes(a) for a in n.all_input_nodes)
    eager += read + nbytes(n)
    name = n.target if isinstance(n.target, str) else n.target.__name__
    print(f"{name:<8} 读 {read:>7} 字节，写 {nbytes(n):>7} 字节")

# 融合：整条链只读图的输入、只写最终输出，中间结果留在寄存器里
group = set(ops)
inputs = {a for n in ops for a in n.all_input_nodes if a not in group}
outputs = [n for n in ops if any(u not in group for u in n.users)]
fused = sum(nbytes(a) for a in inputs) + sum(nbytes(o) for o in outputs)
print(f"{len(ops)} 个算子逐个执行：共 {eager} 字节；融合成 1 个 kernel：{fused} 字节，是原来的 {fused / eager:.1%}")
