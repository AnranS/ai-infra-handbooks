# （三）索引、归约与掩码

<p class="lead">交叉熵要按标签取一列，注意力要屏蔽未来的位置，采样要取前 k 大——这些都是索引和归约。这一章把它们一次讲完，顺带说清一件事：哪些索引返回视图、哪些返回拷贝，这决定了你改它的时候会不会改到别人。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `x[1:, 2:]` 和 `x[[0, 2]]` 哪个是视图、哪个是拷贝？
    2. 怎么从 `(N, C)` 的 logits 里按 `(N,)` 的标签取出对应的那一列？
    3. 因果掩码为什么填 `-inf` 而不是一个很大的负数？
    4. `max(dim=1)` 返回什么？`keepdim=True` 有什么用？
    5. `scatter_` 结尾的下划线意味着什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 切片是视图（共享存储），整数列表/张量索引和布尔索引都是拷贝。
    2. `logits.gather(1, labels.unsqueeze(1)).squeeze(1)`，或者 `logits[torch.arange(N), labels]`。
    3. `-inf` 经过 softmax 严格得到 0；用 `-1e9` 这类大负数在 fp16 下会溢出成 `-inf` 或者 `nan`，而且屏蔽得不彻底。
    4. 返回一个有 `values` 和 `indices` 两个字段的具名元组。`keepdim=True` 保留被归约掉的那一维（长度为 1），方便接下来广播。
    5. 原地修改。带下划线的方法会改掉输入本身，在 autograd 里要小心。

## 视图、拷贝，以及按下标取数

```python title="indexing.py"
"""索引、归约与掩码：切片是视图、高级索引是拷贝，以及 gather / scatter / masked_fill 怎么用"""
import torch

x = torch.arange(12).reshape(3, 4)
print("x =\n", x)

print("\n—— 切片是视图，高级索引是拷贝 ——")
s = x[1:, 2:]
s[0, 0] = -1
print("改切片会改到原张量：\n", x)
a = x[[0, 2]]                                                 # 整数列表 / 张量索引 = 高级索引
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
causal = torch.tril(torch.ones(3, 3, dtype=torch.bool))       # 下三角为 True
masked = scores.masked_fill(~causal, float("-inf"))
print("因果掩码之后的分数：\n", masked)
print("softmax 之后被屏蔽的位置正好是 0：\n", masked.softmax(-1).round(decimals=3))
print("用 -inf 而不是很大的负数：softmax 之后严格为 0，而且不会在 fp16 下溢出成 nan")
print("torch.where 按条件二选一：", torch.where(scores > 3, scores, torch.zeros_like(scores)).tolist())

y = torch.arange(12).reshape(3, 4)                            # 重新来一份干净的，前面那个被改过

print("\n—— 归约：dim 和 keepdim ——")
print("y.sum() 把所有元素加起来：", y.sum().item())
print("y.sum(dim=0) 沿着第 0 维加，形状", tuple(y.sum(dim=0).shape), "：", y.sum(dim=0).tolist())
print("keepdim=True 保留那一维，方便广播：", tuple(y.sum(dim=1, keepdim=True).shape))
print("减去每行最大值（softmax 的标准做法）：", (y - y.max(dim=1, keepdim=True).values)[0].tolist())
print("max 返回 (值, 下标) 两个张量，argmax 只返回下标：", y.max(dim=1).indices.tolist(), y.argmax(dim=1).tolist())
print("topk 取前 k 大：", logits.topk(2, dim=1).indices.tolist())
print("cumsum 做前缀和（变长序列的偏移量常这么算）：", torch.tensor([2, 3, 1]).cumsum(0).tolist())
```

```text title="输出"
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

读这段输出：

- **切片是视图，高级索引是拷贝**。记不住的话有个简单判据：**用连续区间（`a:b`）取的是视图，用一堆下标或布尔掩码取的是拷贝**——后者选出来的元素在内存里不连续，没法用 stride 表达，只能复制；
- **`gather` 是"每行按下标取一个"**，交叉熵内部就是这么取出正确类别的对数概率的。`logits[torch.arange(N), labels]` 是等价写法，可读性更好，但 `gather` 更通用（可以在任意维上取）；
- **`scatter_` 是 `gather` 的反向**：按下标写进去。做 one-hot、把变长序列的结果写回定长缓冲区都靠它；
- **掩码用 `-inf`**：softmax 之后严格是 0。用 `-1e9` 在 fp32 下也能凑合，但 fp16 的最大值才 65504，`-1e9` 会直接变成 `-inf` 甚至在后续运算里产生 `nan`；
- **`dim` 和 `keepdim`**：`dim=k` 的意思是"把第 k 维压掉"。`keepdim=True` 把它留成长度 1，于是可以直接和原张量广播——softmax 里减最大值、LayerNorm 里减均值，都是这个套路。

!!! tip "需要的是下标还是值"
    `max` 返回 `(values, indices)` 两个张量，`argmax` 只返回下标，`topk` 两个都给。采样时要的是下标（再去查词表），算损失时要的是值。写的时候明确自己要哪一个，能省掉一半的形状错误。

!!! interview "怎么讲清楚"
    讲"PyTorch 的索引"：先分两类——**基础索引（切片）返回视图**，共享存储，改了会互相影响；**高级索引（整数张量、布尔掩码）返回拷贝**，因为选出来的元素在内存里不连续。然后给两个必会的算子：`gather` 按下标取（交叉熵取正确类别的概率就是它），`scatter_` 按下标写（one-hot、把结果写回缓冲区）。最后讲掩码：因果注意力用 `masked_fill(mask, -inf)`，用 `-inf` 而不是大负数是为了 softmax 之后**严格**为 0，并且避免 fp16 溢出。

## 练习

**1. 自己写一遍交叉熵。** 用 `log_softmax` 和 `gather` 实现 `nn.CrossEntropyLoss`，和官方实现对比（注意默认是对 batch 求平均）。

??? success "参考思路"
    `-logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1).mean()`。和 `nn.CrossEntropyLoss()(logits, labels)` 对比应当在 1e-6 以内。顺带体会一下为什么是 `log_softmax` 而不是 `softmax().log()`：前者在数值上稳定得多。

**2. 变长序列的掩码。** 给定 `(B,)` 的真实长度，造出 `(B, T)` 的布尔掩码。

??? success "参考答案"
    `torch.arange(T)[None, :] < lengths[:, None]`。这是广播的典型用法：`(1,T)` 和 `(B,1)` 比较得到 `(B,T)`。下一章的 DataLoader 里会用到它。

**3. 视图的坑。** 解释为什么 `x.expand(3, 4)[0, 0] = 1` 会报错，而 `x.repeat(3, 1)[0, 0] = 1` 不会。

??? success "参考答案"
    `expand` 出来的维度 stride 是 0，多个位置指向同一块内存，往里写会同时改掉"好几个"元素，所以 PyTorch 直接禁止原地写。`repeat` 真的复制了数据，每个位置独立，可以写。

## 小结

- [x] 切片返回视图，整数张量索引和布尔索引返回拷贝。
- [x] `gather` 按下标取、`scatter_` 按下标写；带下划线的方法原地修改。
- [x] 掩码用 `masked_fill(..., -inf)`，softmax 之后严格为 0，也不会在 fp16 下溢出。
- [x] `dim` 指定压掉哪一维，`keepdim=True` 留下长度 1 的维方便广播。
