# 3. Indexing, reductions and masks

<p class="lead">Cross-entropy takes one column per label, attention masks out the future, sampling takes the top k — all of these are indexing and reduction. This chapter covers them together, and settles one thing along the way: which forms of indexing return a view and which return a copy, because that decides whether writing to the result also writes to someone else's tensor.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which of `x[1:, 2:]` and `x[[0, 2]]` is a view and which is a copy?
    2. How do you take, from `(N, C)` logits, the column named by `(N,)` labels?
    3. Why does a causal mask fill with `-inf` rather than a large negative number?
    4. What does `max(dim=1)` return? What is `keepdim=True` for?
    5. What does the trailing underscore in `scatter_` mean?

??? success "Answers (try first, then expand to compare)"
    1. Slicing gives a view (shared storage); indexing with a list or tensor of integers, and boolean indexing, both give copies.
    2. `logits.gather(1, labels.unsqueeze(1)).squeeze(1)`, or `logits[torch.arange(N), labels]`.
    3. `-inf` becomes exactly 0 after softmax; something like `-1e9` overflows to `-inf` or `nan` in fp16 and does not mask cleanly.
    4. A named tuple with `values` and `indices`. `keepdim=True` keeps the reduced dimension with length 1, which makes the result broadcast against the original.
    5. In-place. The trailing-underscore methods modify their input, which needs care under autograd.

## Views, copies, and taking by index {#视图拷贝以及按下标取数}

```python title="indexing.py"
"""索引、归约与掩码：切片是视图、高级索引是拷贝，以及 gather / scatter / masked_fill 怎么用"""
import torch

x = torch.arange(12).reshape(3, 4)
print("x =\n", x)

print("\n—— 切片是视图，高级索引是拷贝 ——")
s = x[1:, 2:]
s[0, 0] = -1
print("改切片会改到原张量：\n", x)
a = x[[0, 2]]                                                 # a list or tensor of integers = advanced indexing
a[0, 0] = -999
print("改高级索引的结果不影响原张量：x[0,0] =", x[0, 0].item())
print("布尔索引同样是拷贝，而且会把结果拉平：", x[x > 8].tolist())

print("\n—— 按下标取：gather ——")
logits = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
labels = torch.tensor([1, 0])
picked = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
print("每一行取 label 那一列（交叉熵里天天用）：", picked.round(decimals=3).tolist())
print("等价的高级索引写法：", logits[torch.arange(2), labels].round(decimals=3).tolist())

print("\n—— 按下标写：scatter_ ——")
onehot = torch.zeros(2, 3).scatter_(1, labels.unsqueeze(1), 1.0)
print("做 one-hot：\n", onehot)
print("带下划线的方法是**原地**修改，会改掉输入，autograd 里要小心")

print("\n—— 掩码 ——")
scores = torch.tensor([[1.0, 2.0, 3.0], [2.0, 5.0, 1.0], [0.0, 3.0, 4.0]])
causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))       # True on the lower triangle
masked = scores.masked_fill(~causal, float("-inf"))
print("因果掩码之后的分数：\n", masked)
print("softmax 之后被屏蔽的位置正好是 0：\n", masked.softmax(-1).round(decimals=3))
print("用 -inf 而不是很大的负数：softmax 之后严格为 0，而且不会在 fp16 下溢出成 nan")
print("torch.where 按条件二选一：", torch.where(scores > 3, scores, torch.zeros_like(scores)).tolist())

y = torch.arange(12).reshape(3, 4)                            # a fresh one; the previous tensor was modified above

print("\n—— 归约：dim 和 keepdim ——")
print("y.sum() 把所有元素加起来：", y.sum().item())
print("y.sum(dim=0) 沿着第 0 维加，形状", tuple(y.sum(dim=0).shape), "：", y.sum(dim=0).tolist())
print("keepdim=True 保留那一维，方便广播：", tuple(y.sum(dim=1, keepdim=True).shape))
print("减去每行最大值（softmax 的标准做法）：", (y - y.max(dim=1, keepdim=True).values)[0].tolist())
print("max 返回 (值, 下标) 两个张量，argmax 只返回下标：", y.max(dim=1).indices.tolist(), y.argmax(dim=1).tolist())
print("topk 取前 k 大：", logits.topk(2, dim=1).indices.tolist())
print("cumsum 做前缀和（变长序列的偏移量常这么算）：", torch.tensor([2, 3, 1]).cumsum(0).tolist())
```

```text title="output"
x =
 tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

—— 切片是视图，高级索引是拷贝 ——
改切片会改到原张量：
 tensor([[ 0,  1,  2,  3],
        [ 4,  5, -1,  7],
        [ 8,  9, 10, 11]])
改高级索引的结果不影响原张量：x[0,0] = 0
布尔索引同样是拷贝，而且会把结果拉平： [9, 10, 11]

—— 按下标取：gather ——
每一行取 label 那一列（交叉熵里天天用）： [0.699999988079071, 0.800000011920929]
等价的高级索引写法： [0.699999988079071, 0.800000011920929]

—— 按下标写：scatter_ ——
做 one-hot：
 tensor([[0., 1., 0.],
        [1., 0., 0.]])
带下划线的方法是**原地**修改，会改掉输入，autograd 里要小心

—— 掩码 ——
因果掩码之后的分数：
 tensor([[1., -inf, -inf],
        [2., 5., -inf],
        [0., 3., 4.]])
softmax 之后被屏蔽的位置正好是 0：
 tensor([[1.0000, 0.0000, 0.0000],
        [0.0470, 0.9530, 0.0000],
        [0.0130, 0.2650, 0.7210]])
用 -inf 而不是很大的负数：softmax 之后严格为 0，而且不会在 fp16 下溢出成 nan
torch.where 按条件二选一： [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 4.0]]

—— 归约：dim 和 keepdim ——
y.sum() 把所有元素加起来： 66
y.sum(dim=0) 沿着第 0 维加，形状 (4,) ： [12, 15, 18, 21]
keepdim=True 保留那一维，方便广播： (3, 1)
减去每行最大值（softmax 的标准做法）： [-3, -2, -1, 0]
max 返回 (值, 下标) 两个张量，argmax 只返回下标： [3, 3, 3] [3, 3, 3]
topk 取前 k 大： [[1, 2], [0, 1]]
cumsum 做前缀和（变长序列的偏移量常这么算）： [2, 5, 6]
```

Reading that output:

- **Slicing gives a view, advanced indexing gives a copy**. A simple test if you cannot remember: **a contiguous range (`a:b`) gives a view, a bag of indices or a boolean mask gives a copy** — the elements the latter picks are not contiguous in memory and cannot be expressed with strides, so they have to be copied;
- **`gather` is "one element per row, chosen by index"**, which is how cross-entropy pulls out the log-probability of the correct class. `logits[torch.arange(N), labels]` is equivalent and reads better, but `gather` is more general (any dimension);
- **`scatter_` is the inverse of `gather`**: write by index. One-hot encoding and writing variable-length results back into a fixed buffer both use it;
- **Mask with `-inf`**: softmax then gives exactly 0. `-1e9` is tolerable in fp32, but fp16 tops out at 65504, so `-1e9` becomes `-inf` outright and can produce `nan` downstream;
- **`dim` and `keepdim`**: `dim=k` means "collapse dimension k". `keepdim=True` leaves it with length 1 so the result broadcasts against the original — subtracting the max in softmax and the mean in LayerNorm are both this pattern.

!!! tip "Do you want the index or the value"
    `max` returns `(values, indices)`, `argmax` returns only indices, `topk` gives both. Sampling wants the index (to look up the vocabulary); a loss wants the value. Being clear about which you want removes half the shape errors.

!!! interview "How to explain it"
    To explain "indexing in PyTorch": split it in two — **basic indexing (slicing) returns a view** with shared storage, so writes propagate; **advanced indexing (integer tensors, boolean masks) returns a copy**, because the selected elements are not contiguous. Then give the two operators you must know: `gather` takes by index (cross-entropy pulling out the correct class's probability is exactly this) and `scatter_` writes by index (one-hot, writing results back into a buffer). Finish with masking: causal attention uses `masked_fill(mask, -inf)`, and `-inf` rather than a large negative keeps softmax **exactly** zero and avoids fp16 overflow.

## Exercises {#练习}

**1. Write cross-entropy yourself.** Implement `nn.CrossEntropyLoss` with `log_softmax` and `gather`, and compare against the official one (note that it averages over the batch by default).

??? success "An approach"
    `-logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1).mean()`. It should agree with `nn.CrossEntropyLoss()(logits, labels)` to within 1e-6. Notice along the way why it is `log_softmax` and not `softmax().log()`: the former is far more numerically stable.

**2. A mask for variable lengths.** Given `(B,)` true lengths, build the `(B, T)` boolean mask.

??? success "Answer"
    `torch.arange(T)[None, :] < lengths[:, None]`. A textbook use of broadcasting: `(1,T)` compared with `(B,1)` gives `(B,T)`. The DataLoader chapter uses it.

**3. The view trap.** Explain why `x.expand(3, 4)[0, 0] = 1` raises while `x.repeat(3, 1)[0, 0] = 1` does not.

??? success "Answer"
    The expanded dimension has a stride of 0, so several positions point at the same memory; writing would change "several" elements at once, and PyTorch forbids it outright. `repeat` really copied the data, so every position is independent and writable.

## Summary {#小结}

- [x] Slicing returns a view; integer-tensor and boolean indexing return copies.
- [x] `gather` takes by index, `scatter_` writes by index; trailing-underscore methods modify in place.
- [x] Mask with `masked_fill(..., -inf)`: exactly zero after softmax, and no fp16 overflow.
- [x] `dim` says which dimension to collapse; `keepdim=True` leaves a length-1 dimension for broadcasting.
