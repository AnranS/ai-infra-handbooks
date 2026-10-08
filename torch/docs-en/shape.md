# 2. Working with shapes: view, broadcasting and einsum

<p class="lead">More than half the time spent reading model code goes to "what shape is this now". This chapter settles the shape questions at once: how view differs from reshape, why you cannot view after a permute, the broadcasting rules and the silent bugs they create, and how einsum replaces a string of transposes with one line.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do `view` and `reshape` differ? When does `view` fail?
    2. Is a tensor still contiguous after `permute(0, 2, 1)`? What do the strides become?
    3. What are the broadcasting rules? What shape do `(3,1,5)` and `(4,5)` add to?
    4. How do `expand` and `repeat` differ?
    5. What does `torch.einsum("bqd,bkd->bqk", q, k)` compute?

??? success "Answers (try first, then expand to compare)"
    1. `view` only changes metadata and never copies, so it requires the original to be interpretable contiguously in the new shape; `reshape` views when it can and copies when it cannot. Viewing a non-contiguous tensor (after a permute, say) raises.
    2. No longer contiguous. The strides go from `(12,4,1)` to `(12,1,4)` — the data did not move, only the order in which it is read.
    3. Align from the right; each dimension must either match or be 1 (which is stretched). The result is `(3,4,5)`.
    4. `expand` does not copy — it uses a stride of 0 — and can only stretch dimensions of length 1; `repeat` really copies. Prefer `expand`.
    5. A batched $QK^\top$: for each batch, the dot product of every row of q with every row of k, giving `(b, q, k)` attention scores.

## view, reshape and permute {#viewreshape-与-permute}

```python title="shapes.py"
"""形状的功夫：view 和 reshape 的区别、permute 之后为什么不连续、广播规则、einsum"""
import torch

x = torch.arange(24).reshape(2, 3, 4)
print("x:", tuple(x.shape), "stride", x.stride(), "连续", x.is_contiguous())

print("\n—— view / reshape / permute ——")
v = x.view(6, 4)                                              # no data is copied, only the metadata changes
print("view(6, 4) 和 x 共享存储：", v.data_ptr() == x.data_ptr())
p = x.permute(0, 2, 1)                                        # swap the last two dimensions; only the strides change
print("permute(0,2,1):", tuple(p.shape), "stride", p.stride(), "连续", p.is_contiguous())
try:
    p.view(2, 12)
except RuntimeError as e:
    print("非连续的张量不能 view：", str(e).split(".")[0])
print("reshape 会在需要时自己复制一份：", tuple(p.reshape(2, 12).shape),
      "共享存储", p.reshape(2, 12).data_ptr() == x.data_ptr())
print("-1 让 PyTorch 自己算那一维：", tuple(x.view(-1, 4).shape))
print("squeeze / unsqueeze 增删长度为 1 的维：",
      tuple(x.unsqueeze(0).shape), "->", tuple(x.unsqueeze(0).squeeze(0).shape))

print("\n—— 广播：从右往左对齐，维度相等或其中一个是 1 ——")
a = torch.ones(3, 1, 5)
b = torch.ones(4, 5)
print(f"{tuple(a.shape)} + {tuple(b.shape)} = {tuple((a + b).shape)}")
print("广播不复制数据，靠 stride=0 实现：", a.expand(3, 4, 5).stride())
try:
    torch.ones(3, 4) + torch.ones(3, 5)
except RuntimeError as e:
    print("对不上就报错：", str(e).split(".")[0])
scores = torch.zeros(2, 3)
mask = torch.tensor([True, False, True])
print("最常见的用法是给某一维加 1：", (scores + mask.unsqueeze(0)).tolist())

print("\n—— 静默的 bug：形状对得上，但不是你想要的 ——")
pred = torch.tensor([1.0, 2.0, 3.0])                          # shape (3,)
target = torch.tensor([[1.0], [2.0], [3.0]])                  # shape (3, 1)
print(f"(3,) - (3,1) 会广播成 {tuple((pred - target).shape)}，均方误差算出来是 "
      f"{((pred - target) ** 2).mean():.4f}，而不是 0")
print("写损失函数时养成习惯：两个张量的形状打印出来对一遍，或者用 assert")

print("\n—— einsum：把下标写出来，省得数 permute ——")
q = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(1))   # (batch, sequence, dimension)
k = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(2))
by_hand = q @ k.transpose(1, 2)
by_einsum = torch.einsum("bqd,bkd->bqk", q, k)
print("注意力分数两种写法一致：", torch.allclose(by_hand, by_einsum), tuple(by_einsum.shape))
print("多头的版本一行就写完：", tuple(torch.einsum("bhqd,bhkd->bhqk",
                                              q.view(2, 2, 2, 8), k.view(2, 2, 2, 8)).shape))
```

```text title="output"
x: (2, 3, 4) stride (12, 4, 1) 连续 True

—— view / reshape / permute ——
view(6, 4) 和 x 共享存储： True
permute(0,2,1): (2, 4, 3) stride (12, 1, 4) 连续 False
非连续的张量不能 view： view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)
reshape 会在需要时自己复制一份： (2, 12) 共享存储 False
-1 让 PyTorch 自己算那一维： (6, 4)
squeeze / unsqueeze 增删长度为 1 的维： (1, 2, 3, 4) -> (2, 3, 4)

—— 广播：从右往左对齐，维度相等或其中一个是 1 ——
(3, 1, 5) + (4, 5) = (3, 4, 5)
广播不复制数据，靠 stride=0 实现： (5, 0, 1)
对不上就报错： The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 1
最常见的用法是给某一维加 1： [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]

—— 静默的 bug：形状对得上，但不是你想要的 ——
(3,) - (3,1) 会广播成 (3, 3)，均方误差算出来是 1.3333，而不是 0
写损失函数时养成习惯：两个张量的形状打印出来对一遍，或者用 assert

—— einsum：把下标写出来，省得数 permute ——
注意力分数两种写法一致： True (2, 4, 4)
多头的版本一行就写完： (2, 2, 2, 2)
```

The rules:

- **view never copies, reshape might**. Prefer `reshape` in library code (it is always correct) and `view` in performance-sensitive code (it raises if a copy would be needed, which surfaces the problem for you);
- **permute and transpose only change strides** — not one byte of data moves. To `view` afterwards you need `.contiguous()` first, and that is the step that really copies;
- **Broadcasting does not copy data**: the broadcast dimensions have a stride of 0, so every position reads the same element. That is also why you cannot write into the result of `expand` in place;
- **`(3,) - (3,1)` broadcasts to `(3,3)`**, and the loss comes out as 1.3333 instead of 0 — a legal shape with an entirely wrong result. Bugs like this never raise; they just leave you staring at a loss curve. Make it a habit to print both shapes before computing a loss, or simply `assert pred.shape == target.shape`.

!!! tip "einsum is the shortcut for reading and writing attention"
    `"bqd,bkd->bqk"` says: b and d appear on both sides, b is kept, d is summed away, q and k are kept. Multiple heads only need an h in front: `"bhqd,bhkd->bhqk"`. Compared with counting which two dimensions `transpose(1,2)` should swap, writing the subscripts is harder to get wrong and reads more like the formula. The price is that einsum can be slightly slower than a hand-written matmul on some backends, so measure on hot paths.

!!! interview "How to explain it"
    To explain "view versus reshape": `view` only changes metadata and demands a compatible memory layout, `reshape` copies when it is not compatible. Then move to **contiguity**: `permute` changes strides without moving data, so a `view` afterwards fails and you need `.contiguous()` or `reshape`. For broadcasting, stress two things: the rule is **align from the right, dimensions equal or 1**; and it does not copy data (`expand` uses a stride of 0). Finish with a real trap: a legal shape is not a correct one — `(N,)` minus `(N,1)` broadcasts to `(N,N)`, so loss functions deserve an explicit assert.

## Exercises {#练习}

**1. Split the heads.** A `(B, T, D)` tensor has to become `(B, H, T, D/H)` for multi-head attention. Write the two steps, and explain why the order cannot be reversed.

??? success "Answer"
    `x.view(B, T, H, D // H).transpose(1, 2)`. Split the last dimension into `(H, D/H)` first — because the channels of one head are contiguous in memory — and only then move H forward. Transposing first and viewing afterwards produces "heads" that straddle different channels, which is wrong, and very likely will not raise.

**2. Broadcast by hand.** Without broadcasting, use `expand` to align `(3,1,5)` and `(4,5)` into `(3,4,5)` explicitly and add them; check the result matches plain `+`.

??? success "An approach"
    `a.expand(3, 4, 5) + b.unsqueeze(0).expand(3, 4, 5)`. Doing it once makes clear that broadcasting only saves you the `unsqueeze` and `expand`; there is no magic — and its behaviour stops being surprising.

**3. Write a few things with einsum.** Express a batched matrix multiply, taking the diagonal, and a weighted sum over the last dimension.

??? success "Answer"
    `"bij,bjk->bik"`, `"ii->i"` and `"...d,d->..."`. When you are stuck, ask: which subscripts appear on both sides but not in the output (those are summed away), and which are kept.

## Summary {#小结}

- [x] `view` only changes metadata, `reshape` copies when it must; `permute` only changes strides and leaves the tensor non-contiguous.
- [x] Broadcasting aligns from the right with dimensions equal or 1, and copies nothing — `expand` uses a stride of 0.
- [x] A legal shape is not a correct one: `(N,)` minus `(N,1)` quietly broadcasts to `(N,N)`.
- [x] einsum writes the subscripts out, which beats counting transposes; multi-head attention fits on one line.
