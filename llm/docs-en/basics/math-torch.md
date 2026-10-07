# Math and PyTorch basics

<p class="lead">Understanding large models takes less math than you might think: matrix multiplication, softmax, dot products, plus a feel for tensor shapes and a basic grasp of numerical precision. This chapter covers these tools in one go and fixes the shape notation used throughout the book. Every later chapter uses them again and again.</p>

!!! tip "To learn the math systematically"
    This chapter covers only the minimal toolkit needed to read model code. The more systematic material is its own book, [Math Fundamentals](math://): linear algebra and low rank, probability and sampling, information theory, backpropagation, floating-point error, performance and queueing theory, with every concept measured on a real model. Look things up there when you need them; no need to read it first.

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Applying `nn.Linear(4096, 11008)` to a tensor of shape `[2, 100, 4096]`, what is the output shape? How many floating-point operations does it take?
    2. What is the difference between `view` and `reshape`? Why is `contiguous()` sometimes needed after `transpose`?
    3. Why does softmax subtract the maximum first?
    4. How many exponent and mantissa bits do FP16 and BF16 have? Why do large models prefer BF16?
    5. Why use `torch.no_grad()` or `torch.inference_mode()` at inference time?

??? success "Answers (try first, then expand to compare)"
    1. The output is `[2, 100, 11008]`; the cost is $2 \times 2 \times 100 \times 4096 \times 11008 \approx 1.8 \times 10^{10}$ floating-point operations (each output element takes 4096 multiply-adds, and a multiply-add counts as 2 operations).
    2. `view` only changes metadata and requires that the memory layout can be described with the new strides, otherwise it raises an error; `reshape` returns a view when it can and copies when it cannot. `transpose` only swaps strides, so the result is no longer contiguous, and many kernels and `view` require contiguity, hence `contiguous()`.
    3. $e^x$ overflows for large $x$; softmax is unchanged when the same constant is added to every input, and after subtracting the maximum the largest exponential is $e^0 = 1$, so nothing overflows.
    4. FP16: 5 exponent bits and 10 mantissa bits, better precision but a maximum of only 65504; BF16: 8 exponent bits and 7 mantissa bits, the same range as FP32. In large-model training and inference, overflow is deadlier than precision loss, and BF16 needs no loss scaling, so it is more popular.
    5. They turn off gradient recording: no intermediates are saved for the backward pass (saving memory) and no graph is built (saving time); `inference_mode` also skips version counting.

## Shape notation {#形状记号}

The whole book uses the following notation:

| Symbol | Meaning | Value for Qwen3-0.6B |
| --- | --- | --- |
| $B$ | batch size (number of sequences processed together) | — |
| $T$ | sequence length (number of tokens) | — |
| $V$ | vocabulary size | 151936 |
| $d$ | hidden dimension (hidden_size) | 1024 |
| $n_h$ | number of attention heads | 16 |
| $n_{kv}$ | number of KV heads (GQA) | 8 |
| $d_h$ | dimension per head (head_dim), usually $d / n_h$, configured separately in Qwen3 | 128 |
| $d_{ff}$ | intermediate dimension of the feed-forward network (intermediate_size) | 3072 |
| $L$ | number of layers | 28 |

The most common tensor shapes in a Transformer are `[B, T, d]` (one d-dimensional vector per token) and `[B, n_h, T, d_h]` (after splitting into heads). **When reading code, write the tensor shape next to every line**: it is the most effective habit for understanding a model.

## Matrix multiplication: the linear layer {#矩阵乘法线性层}

`nn.Linear(in_features, out_features)` computes $y = xW^\top + b$, where the weight $W$ has shape `[out, in]`. It acts only on the **last dimension** and treats all earlier dimensions as batch:

```pycon
>>> import torch
>>> import torch.nn as nn
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at 0x...>
>>> lin = nn.Linear(4096, 11008, bias=False)
>>> lin.weight.shape
torch.Size([11008, 4096])
>>> x = torch.randn(2, 100, 4096)
>>> lin(x).shape
torch.Size([2, 100, 11008])
```

**Cost**: a `[M, K] × [K, N]` matrix multiplication takes $M \times N \times K$ multiply-adds, that is, $2MNK$ floating-point operations (FLOPs). In the example above M = 2 × 100 = 200, so it is 2 × 200 × 11008 × 4096 ≈ 18 billion.

This gives a very useful conclusion: **a linear layer's cost per token = 2 × its parameter count**. Nearly all of a model's parameters are in linear layers, so

$$
\text{forward cost per token} \approx 2 \times \text{parameter count}
$$

A 7-billion-parameter model needs about 14 billion floating-point operations per generated token (plus the attention part; see [estimation](../inference/estimation.md)).

### einsum: describing tensor operations with indices {#einsum用下标描述张量运算}

`torch.einsum` describes an operation with an index string, which makes complex tensor operations especially clear:

```pycon
>>> q = torch.randn(2, 14, 5, 64)             # [B, n_h, T, d_h]
>>> k = torch.randn(2, 14, 5, 64)
>>> scores = torch.einsum("bhqd,bhkd->bhqk", q, k)   # within each head, dot every query with every key
>>> scores.shape
torch.Size([2, 14, 5, 5])
>>> torch.allclose(scores, q @ k.transpose(-2, -1), atol=1e-5)
True
```

Indices that appear in the inputs but not in the output (`d` here) are summed over.

## Reshaping: view, reshape, transpose {#变形viewreshapetranspose}

The most common operation in multi-head attention is splitting `[B, T, n_h × d_h]` into `[B, n_h, T, d_h]`:

```pycon
>>> B, T, nh, hd = 2, 5, 14, 64
>>> x = torch.randn(B, T, nh * hd)
>>> heads = x.view(B, T, nh, hd).transpose(1, 2)   # split the last dim first, then move the head dim forward
>>> heads.shape
torch.Size([2, 14, 5, 64])
>>> heads.is_contiguous()
False
>>> back = heads.transpose(1, 2).reshape(B, T, nh * hd)   # merge back
>>> torch.equal(back, x)
True
```

- `view` only changes how the memory is interpreted, copies no data, and requires a compatible memory layout;
- `transpose` copies no data either; it just swaps the strides, so the result is no longer contiguous in memory;
- On a non-contiguous tensor, `view` raises an error, while `reshape` makes a copy when it must. `contiguous()` explicitly copies into a contiguous layout.

!!! inference "Inference view"
    These "no-copy" reshapes cost almost nothing on a GPU, but once a copy is triggered (`contiguous()`, `reshape`), it is a full read and write. High-performance kernels usually read data directly in the layout they need, or fuse the transpose into a neighboring operator to avoid a separate transpose kernel; see [matrix transpose](cuda://kernels/transpose/) in the CUDA book.

## Broadcasting {#广播}

When tensors of different shapes are combined elementwise, they are aligned from the last dimension, and dimensions of size 1 are "copied" to the matching size:

```pycon
>>> x = torch.randn(2, 5, 896)       # [B, T, d]
>>> w = torch.randn(896)             # [d]: the RMSNorm weight
>>> (x * w).shape                    # w is broadcast to every batch and position
torch.Size([2, 5, 896])
>>> mean_sq = x.pow(2).mean(dim=-1, keepdim=True)   # keepdim keeps the size-1 dim for broadcasting
>>> mean_sq.shape
torch.Size([2, 5, 1])
```

Broadcasting errors are among the most common errors when writing model code. There are only two rules; try a few common shapes and they will stick:

<div class="aig-widget" data-widget="broadcast"></div>

## Softmax and numerical stability {#softmax-与数值稳定}

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

It turns any real vector into a probability distribution. Computing it directly overflows:

```pycon
>>> z = torch.tensor([1000.0, 1001.0, 1002.0])
>>> torch.exp(z) / torch.exp(z).sum()      # e^1000 overflows to inf
tensor([nan, nan, nan])
>>> zs = z - z.max()                        # subtract the max; mathematically the same
>>> torch.exp(zs) / torch.exp(zs).sum()
tensor([0.0900, 0.2447, 0.6652])
>>> torch.softmax(z, dim=0)                 # the library does exactly this internally
tensor([0.0900, 0.2447, 0.6652])
```

Two useful properties: softmax depends only on the **differences** between the logits (adding a constant to all of them changes nothing); and dividing all logits by a number greater than 1 makes the distribution "flatter", which is the **temperature** in sampling (see [decoding and sampling](../inference/decoding.md)). To compute a loss, use `log_softmax` or `F.cross_entropy`, which combine the log and the softmax internally and are numerically more stable.

## Dot products and similarity {#点积与相似度}

The dot product of two vectors, $a \cdot b = \|a\| \|b\| \cos\theta$, measures how "similar" they are. The heart of attention is to use the dot product of the query vector with each key vector to decide which positions to read from (see [attention](../transformer/attention.md)). Dot products of high-dimensional random vectors have an important property, and it is the reason attention divides by $\sqrt{d_h}$:

```pycon
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at 0x...>
>>> for d in (16, 64, 256, 1024):
...     a, b = torch.randn(10000, d), torch.randn(10000, d)
...     print(d, round((a * b).sum(-1).std().item(), 1))
...
16 4.0
64 8.0
256 16.2
1024 32.1
```

For two d-dimensional random vectors with standard normal components, the dot product has a standard deviation of about $\sqrt{d}$. The higher the dimension, the larger the dot products; fed straight into softmax, they make the distribution extremely sharp and the gradients nearly zero. Dividing by $\sqrt{d}$ brings them back to unit scale.

## Number formats {#数值格式}

| Format | Sign/exponent/mantissa | Bytes per number | Maximum | Relative precision (machine epsilon) |
| --- | --- | --- | --- | --- |
| FP32 | 1/8/23 | 4 | about 3.4e38 | about 1.2e-7 |
| FP16 | 1/5/10 | 2 | 65504 | about 9.8e-4 |
| BF16 | 1/8/7 | 2 | about 3.4e38 | about 7.8e-3 |

```pycon
>>> for dt in (torch.float32, torch.float16, torch.bfloat16):
...     fi = torch.finfo(dt)
...     print(str(dt), fi.bits, f"{fi.max:.3g}", f"{fi.eps:.2g}")
...
torch.float32 32 3.4e+38 1.2e-07
torch.float16 16 6.55e+04 0.00098
torch.bfloat16 16 3.39e+38 0.0078
>>> torch.tensor(70000.0).half()          # out of FP16's range
tensor(inf, dtype=torch.float16)
>>> torch.tensor(70000.0).bfloat16()      # BF16 has the range, but coarse precision
tensor(70144., dtype=torch.bfloat16)
```

A large model's activations are occasionally very large, so FP16 overflows easily; BF16 has the same range as FP32, only with lower precision, and neural networks tolerate imprecision quite well. So today almost all large-model training and inference stores weights and activations in BF16, and computes the critical accumulations (the accumulators of matrix multiplications, softmax, normalization statistics) in FP32.

!!! inference "Inference view"
    **The number of bytes per parameter directly determines memory use and decode speed**. In BF16 each parameter takes 2 bytes, so the weights of a 7-billion-parameter model are about 14 GB; quantized to INT4 they need only about 3.5 GB (plus a few scales), and each decode step also reads three quarters less data; see [how quantization works](../inference/quantization.md).

## nn.Module and parameters {#nnmodule-与参数}

A PyTorch model is a nested structure of `nn.Module`s, with its parameters kept in a `state_dict` and named by hierarchy. Counting parameters and inspecting the structure are both easy:

```pycon
>>> class TinyBlock(nn.Module):
...     def __init__(self, d, dff):
...         super().__init__()
...         self.up = nn.Linear(d, dff, bias=False)
...         self.down = nn.Linear(dff, d, bias=False)
...     def forward(self, x):
...         return self.down(torch.relu(self.up(x)))
...
>>> block = TinyBlock(896, 4864)
>>> sum(p.numel() for p in block.parameters())
8716288
>>> [(k, tuple(v.shape)) for k, v in block.state_dict().items()]
[('up.weight', (4864, 896)), ('down.weight', (896, 4864))]
```

Loading a pretrained model essentially means filling the tensors in a file into the `state_dict` by name. We do this by hand in [assembling a large model from scratch](../transformer/build-llm.md).

## Turn off gradients at inference time {#推理时关掉梯度}

In training, PyTorch saves the intermediates each operation needs for backpropagation. Inference needs no backpropagation, so saving them only wastes memory and time:

```pycon
>>> w = torch.randn(4, 4, requires_grad=True)
>>> y = (w @ torch.randn(4)).sum()
>>> y.requires_grad
True
>>> with torch.inference_mode():
...     y2 = (w @ torch.randn(4)).sum()
...
>>> y2.requires_grad
False
```

Both `torch.no_grad()` and `torch.inference_mode()` turn off gradient recording; the latter is more thorough and cheaper.

!!! interview "In an interview"
    This chapter's material usually shows up as mental arithmetic: a linear layer applied to `[B, T, d_in]` outputs `[B, T, d_out]` at a cost of $2 \cdot B \cdot T \cdot d_{in} \cdot d_{out}$, so the whole model takes about 2 × the parameter count in operations per token; BF16 has 8 exponent bits and 7 mantissa bits, the same range as FP32 and no need for loss scaling, while FP16 has better precision but a maximum of only 65504; softmax subtracts the maximum to avoid overflow; and inference uses `torch.inference_mode()` to skip gradient recording and version counting. Writing out the tensor shapes as you answer comes across as very solid.

## Exercises {#练习}

**1. Cost.** A linear layer `nn.Linear(896, 4864)` takes input `[4, 512, 896]`. What is the output shape? How many FLOPs does it need? How many bytes do its weights take in BF16?

??? success "Answer"
    The output is `[4, 512, 4864]`. M = 4 × 512 = 2048, FLOPs = 2 × 2048 × 4864 × 896 ≈ 17.8 billion. The weight has 4864 × 896 ≈ 4.36 million parameters, about 8.7 MB in BF16.

    ```python
    M, K, N = 4 * 512, 896, 4864
    assert 2 * M * N * K == 17_850_957_824
    assert N * K * 2 == 8_716_288   # bytes
    ```

**2. Reshaping.** Expand K of shape `[B, T, n_kv, d_h]` into `[B, n_h, T, d_h]` so that each KV head is shared by $n_h / n_{kv}$ consecutive query heads (GQA). Write the code.

??? success "Answer"
    ```python
    import torch
    B, T, n_h, n_kv, d_h = 2, 5, 14, 2, 64
    k = torch.randn(B, T, n_kv, d_h)
    k_heads = k.transpose(1, 2).repeat_interleave(n_h // n_kv, dim=1)   # [B, n_h, T, d_h]
    assert k_heads.shape == (B, n_h, T, d_h)
    assert torch.equal(k_heads[:, 6], k_heads[:, 0]) and torch.equal(k_heads[:, 7], k.transpose(1, 2)[:, 1])
    ```

    Query heads 0–6 share KV head 0, and query heads 7–13 share KV head 1. Real inference kernels do not actually copy K; instead several query heads read the same K. See [attention variants](../transformer/attention-variants.md).

## Summary {#小结}

- [x] Write tensor shapes next to the code; the most common shapes in a Transformer are `[B, T, d]` and `[B, n_h, T, d_h]`.
- [x] A linear layer's cost per token is 2 × its parameter count, so the whole model takes about 2 × the parameter count in operations per token.
- [x] Softmax subtracts the maximum first; dot products grow with dimension, so attention divides by $\sqrt{d_h}$.
- [x] Large models store in BF16 and accumulate in FP32; the bytes per parameter determine memory use and decode speed.
- [x] Use `torch.inference_mode()` at inference time to turn off gradient recording.
