# （二）形状的功夫：view、广播与 einsum

<p class="lead">读模型代码时一大半时间花在"这一步形状变成什么了"。这一章把形状相关的几件事一次说清：view 和 reshape 差在哪、permute 之后为什么不能 view、广播的规则和它制造的静默 bug，以及用 einsum 把一串 transpose 写成一行。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `view` 和 `reshape` 有什么区别？什么时候 `view` 会报错？
    2. `permute(0, 2, 1)` 之后张量还连续吗？stride 变成什么样？
    3. 广播的规则是什么？`(3,1,5)` 和 `(4,5)` 相加得到什么形状？
    4. `expand` 和 `repeat` 有什么区别？
    5. `torch.einsum("bqd,bkd->bqk", q, k)` 算的是什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `view` 只改元数据、绝不复制，所以要求原张量在新形状下是连续可解释的；`reshape` 在能 view 时就 view，不能时复制一份。非连续的张量（比如 permute 之后）直接 `view` 会报错。
    2. 不连续了。stride 从 `(12,4,1)` 变成 `(12,1,4)`——数据没动，只是读的顺序变了。
    3. 从右往左对齐，每一维要么相等，要么其中一个是 1（会被拉伸）。结果是 `(3,4,5)`。
    4. `expand` 不复制数据，靠 stride=0 实现，只能拉伸长度为 1 的维；`repeat` 真的复制。能用 `expand` 就别用 `repeat`。
    5. 批量的 $QK^\top$：对每个 batch，把 q 的每一行和 k 的每一行点积，得到 `(b, q, k)` 的注意力分数。

## view、reshape 与 permute

```python title="shapes.py"
"""形状的功夫：view 和 reshape 的区别、permute 之后为什么不连续、广播规则、einsum"""
import torch

x = torch.arange(24).reshape(2, 3, 4)
print("x:", tuple(x.shape), "stride", x.stride(), "连续", x.is_contiguous())

print("\n—— view / reshape / permute ——")
v = x.view(6, 4)                                              # 不复制数据，只改元数据
print("view(6, 4) 和 x 共享存储：", v.data_ptr() == x.data_ptr())
p = x.permute(0, 2, 1)                                        # 交换后两维，只改 stride
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
pred = torch.tensor([1.0, 2.0, 3.0])                          # 形状 (3,)
target = torch.tensor([[1.0], [2.0], [3.0]])                  # 形状 (3, 1)
print(f"(3,) - (3,1) 会广播成 {tuple((pred - target).shape)}，均方误差算出来是 "
      f"{((pred - target) ** 2).mean():.4f}，而不是 0")
print("写损失函数时养成习惯：两个张量的形状打印出来对一遍，或者用 assert")

print("\n—— einsum：把下标写出来，省得数 permute ——")
q = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(1))   # (batch, 序列, 维度)
k = torch.randn(2, 4, 8, generator=torch.Generator().manual_seed(2))
by_hand = q @ k.transpose(1, 2)
by_einsum = torch.einsum("bqd,bkd->bqk", q, k)
print("注意力分数两种写法一致：", torch.allclose(by_hand, by_einsum), tuple(by_einsum.shape))
print("多头的版本一行就写完：", tuple(torch.einsum("bhqd,bhkd->bhqk",
                                              q.view(2, 2, 2, 8), k.view(2, 2, 2, 8)).shape))
```

```text title="输出"
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

几条规则：

- **view 不复制，reshape 可能复制**。写库函数时优先 `reshape`（总是对），写性能敏感的代码时用 `view`（复制了会直接报错，等于帮你把问题暴露出来）；
- **permute / transpose 只改 stride**，数据一个字节都没动。之后要 `view` 就得先 `.contiguous()`，那一步才是真的复制；
- **广播不复制数据**：`expand` 出来的维度 stride 是 0，走到哪都读同一个元素。这也是为什么 `expand` 的结果不能直接原地修改；
- **`(3,) - (3,1)` 广播成 `(3,3)`**，损失算出来是 1.3333 而不是 0——形状合法，结果全错。这类 bug 不会报错，只会让你盯着收敛曲线发呆。养成习惯：算损失前把两个张量的形状打出来，或者直接 `assert pred.shape == target.shape`。

!!! tip "einsum 是读写注意力的捷径"
    `"bqd,bkd->bqk"` 的意思是：b 和 d 在两边都有，b 保留、d 求和消掉，q 和 k 保留。多头只要在前面加一个 h：`"bhqd,bhkd->bhqk"`。比起数 `transpose(1,2)` 到底该转哪两维，下标写出来不容易错，读起来也更像公式。代价是 einsum 在某些后端上比手写 matmul 慢一点，热路径上要实测。

!!! interview "怎么讲清楚"
    讲"view 和 reshape"：`view` 只改元数据、要求内存布局兼容，`reshape` 在不兼容时会复制。再引到**连续性**：`permute` 只改 stride 不动数据，所以之后不能 `view`，要么 `.contiguous()` 要么 `reshape`。讲广播时强调两点：规则是**从右往左对齐、维度相等或为 1**；以及它不复制数据（`expand` 用 stride=0）。最后补一个真实的坑：形状合法不等于语义正确，`(N,)` 和 `(N,1)` 相减会广播成 `(N,N)`，损失函数里要显式 assert。

## 练习

**1. 拆多头。** 一个 `(B, T, D)` 的张量，要变成 `(B, H, T, D/H)` 喂给多头注意力。写出这两步，并说明为什么顺序不能反。

??? success "参考答案"
    `x.view(B, T, H, D // H).transpose(1, 2)`。先把最后一维拆成 `(H, D/H)`——因为同一个头的那些通道在内存里是连着的——再把 H 换到前面。如果先 transpose 再 view，拆出来的"头"会横跨不同的通道，结果是错的（而且很可能不报错）。

**2. 手写一次广播。** 不用广播，用 `expand` 显式地把 `(3,1,5)` 和 `(4,5)` 对齐成 `(3,4,5)` 再相加，验证结果和直接 `+` 一样。

??? success "参考思路"
    `a.expand(3, 4, 5) + b.unsqueeze(0).expand(3, 4, 5)`。做一遍就会发现广播只是帮你省了 `unsqueeze` 和 `expand` 这两步，没有任何魔法——也就不会再对它的行为感到意外。

**3. 用 einsum 写几个常见操作。** 用 einsum 表达：批量矩阵乘、取对角线、按最后一维加权求和。

??? success "参考答案"
    依次是 `"bij,bjk->bik"`、`"ii->i"`、`"...d,d->..."`。写不出来时就问自己：哪些下标两边都有且不在输出里（求和消掉），哪些保留。

## 小结

- [x] `view` 只改元数据，`reshape` 必要时复制；`permute` 只改 stride，之后不连续。
- [x] 广播从右往左对齐，维度相等或为 1；它不复制数据，`expand` 用 stride=0。
- [x] 形状合法不等于语义正确：`(N,)` 和 `(N,1)` 相减会悄悄广播成 `(N,N)`。
- [x] einsum 把下标写出来，比数 transpose 更不容易错，多头注意力一行就能表达。
