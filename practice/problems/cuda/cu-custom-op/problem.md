---
title: 自定义算子：原地修改的 schema、fake 实现与反向
chapter: framework/dispatcher.md
difficulty: 中等
tags: [torch.library, custom_op, fake tensor, torch.compile]
requires: [torch]
---
推理框架里两个最常见的融合算子，用 `torch.library.custom_op` 注册到命名空间 `practice` 下（模块里的 `silu_and_mul`、`fused_add_rms_norm` 就是装饰后的对象；真实系统里函数体会调用 CUDA / Triton kernel，这里用 PyTorch 写）：

1. `practice::silu_and_mul(Tensor x) -> Tensor`：SwiGLU 的激活部分。`x` 的最后一维是 `2d`，前一半是 gate、后一半是 up，输出 `silu(gate) * up`，最后一维是 `d`。要注册 **fake 实现**（只推导形状）和**反向**（`register_autograd`），能被 `gradcheck` 检查；
2. `practice::fused_add_rms_norm(Tensor x, Tensor residual, Tensor weight, float eps) -> None`：vLLM 的写法，**原地**修改两个输入：先 `residual += x`，再把 `rmsnorm(residual) * weight` 写回 `x`（归一化在 fp32 里算）。没有返回值。

测试会用 `torch.library.opcheck` 检查 schema 与实现是否一致（声明了不修改、实现却改了输入，会被查出来）、fake 实现的形状是否正确、反向是否注册；还会用 `torch.compile(fullgraph=True)` 编译一段同时调用两个算子的代码，结果要和 eager 一致。

```python
x = torch.randn(4, 16)
torch.ops.practice.silu_and_mul(x).shape          # torch.Size([4, 8])
```

<!-- 题解 -->
`mutates_args=("x", "residual")` 让 schema 变成 `Tensor(a0!) x, Tensor(a1!) residual`：编译器据此知道这个调用有副作用，不能删掉、不能重排、不能当成纯函数去重。声明错了在 eager 下看不出来，`torch.compile` 下却会静默地算错：把 `fused_add_rms_norm` 声明成 `mutates_args=()` 再跑这里的测试，eager 的两个用例照样通过，编译后的输出却和 eager 明显不同（最大误差超过 1）——编译器以为 `h` 没被改过，直接用了旧值。`opcheck` 的 `test_schema` 在运行时比较输入前后有没有被改，能提前查出这类错误。

fake 实现只描述输出：`silu_and_mul` 返回 `x.new_empty(*x.shape[:-1], d)`；`fused_add_rms_norm` 没有输出，fake 实现什么都不用做。反向用 `register_autograd(backward, setup_context=...)`：$\frac{d}{da}\,\mathrm{silu}(a) = \sigma(a)\,(1 + a(1 - \sigma(a)))$，gate 的梯度是 `grad * up * silu'(gate)`，up 的梯度是 `grad * silu(gate)`，两者拼回 `2d`。
vLLM 的 `torch.ops._C.silu_and_mul`、`fused_add_rms_norm` 就是这两个算子的 CUDA 版本，注册方式相同（C++ 里用 `TORCH_LIBRARY`）。
