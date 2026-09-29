# 张量的内存模型：storage、stride 与视图

<p class="lead">写自定义算子、读推理框架的代码，第一件要弄清楚的事是：一个 <code>torch.Tensor</code> 在内存里到底长什么样。答案是"一块存储 + 一组描述怎么看它的元数据"。转置、切片、<code>expand</code> 都不拷贝数据，只是换了一种看法；而 kernel 往往假设数据是连续的。这一章把这层关系讲清楚。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个 `(3, 4)` 的连续张量转置之后，`stride` 是多少？它还连续吗？
    2. `view` 和 `reshape` 有什么区别？什么时候 `view` 会失败？
    3. `torch.repeat` 和 `torch.expand` 在内存上有什么差别？
    4. 往一个 `expand` 出来的张量里写一个元素，会发生什么？
    5. 自定义 kernel 收到一个非连续的输入，有哪两种处理方法？

## storage + 元数据

一个张量由两部分组成：

- **storage**：一块一维的、连续的原始内存（CPU 内存或显存），只知道自己有多少字节；
- **元数据**：`dtype`、`shape`、`stride`（每个维度走一步要跨过多少个**元素**）和 `storage_offset`（从 storage 的第几个元素开始）。

元素 `x[i, j]` 的地址就是 `storage_offset + i * stride[0] + j * stride[1]`。**视图**（view）就是共享同一块 storage、只有元数据不同的另一个张量：

```python title="storage_views.py"
import torch

x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print("原张量：shape", tuple(x.shape), "stride", x.stride(), "连续", x.is_contiguous())

t = x.t()                                  # 转置：只是交换了 shape 和 stride
print("转置：shape", tuple(t.shape), "stride", t.stride(), "连续", t.is_contiguous())
print("共享存储：", t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())

s = x[:, 1:3]                              # 切片：换了 offset 和 shape
print("切片：shape", tuple(s.shape), "stride", s.stride(), "offset", s.storage_offset(), "连续", s.is_contiguous())

t[0, 1] = 100.0                            # 通过视图写，原张量也变了
print("通过视图修改后 x[1, 0] =", x[1, 0].item())

try:
    t.view(12)
except RuntimeError as e:
    print("view 失败：", str(e).split(" (")[0])
r = t.reshape(12)                          # reshape：能做视图就做视图，做不了就拷贝
print("reshape 拷贝了：", r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr())
print("contiguous() 之后 stride", t.contiguous().stride())
```

```text title="输出"
原张量：shape (3, 4) stride (4, 1) 连续 True
转置：shape (4, 3) stride (1, 4) 连续 False
共享存储： True
切片：shape (3, 2) stride (4, 1) offset 1 连续 False
通过视图修改后 x[1, 0] = 100.0
view 失败： view size is not compatible with input tensor's size and stride
reshape 拷贝了： True
contiguous() 之后 stride (3, 1)
```

- **连续**（contiguous）指按行主序排列、没有空隙：最后一维 stride 为 1，每一维的 stride 等于后面各维大小的乘积；
- `view` 只改元数据，要求新形状能用一组 stride 描述原来的内存布局，做不到就报错；`reshape` 在能做视图时返回视图，否则拷贝一份——所以 `reshape` 之后你**不知道**它是不是和原张量共享内存；
- `contiguous()` 在张量已经连续时直接返回自己，否则拷贝成连续的新张量。

## expand 与 repeat：零 stride 的广播

`expand` 把大小为 1 的维度"广播"到更大的尺寸，做法是把那一维的 stride 设为 **0**——沿着这一维走多少步都停在同一个位置，所以不分配任何内存。`repeat` 则真的把数据复制若干份：

```python title="expand_repeat.py"
import torch

bias = torch.tensor([1.0, 2.0, 3.0])
e = bias.expand(4, 3)              # 4 行都指向同一段内存
r = bias.repeat(4, 1)              # 真的复制成 4 行
print("expand：stride", e.stride(), "存储字节", e.untyped_storage().nbytes())
print("repeat：stride", r.stride(), "存储字节", r.untyped_storage().nbytes())

e[0, 0] = 5.0                      # 4 行共用这个元素：改一个等于改一列
print("写 e[0, 0] 之后 bias =", bias.tolist(), "e[3, 0] =", e[3, 0].item())
try:
    e.add_(1)                      # 原地运算会写同一个位置多次：直接报错
except RuntimeError as err:
    print("原地 add_ 失败：", str(err).split(":")[0])

win = torch.arange(6).unfold(0, 3, 1)   # 滑动窗口也是一个视图：相邻窗口的 stride 是 1
print("滑动窗口：stride", win.stride(), "内容", win.tolist())
```

```text title="输出"
expand：stride (0, 1) 存储字节 12
repeat：stride (3, 1) 存储字节 48
写 e[0, 0] 之后 bias = [5.0, 2.0, 3.0] e[3, 0] = 5.0
原地 add_ 失败： unsupported operation
滑动窗口：stride (1, 1) 内容 [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
```

"`torch.repeat` 和 `torch.expand` 有什么区别"是面试里的常见问题：`expand` 零拷贝、结果不能安全地原地写；`repeat` 分配新内存。`unfold` 这样的滑动窗口视图常用于 n-gram 匹配（比如用 prompt 里已有的片段做投机解码的草稿），不拷贝就能得到所有窗口。

### 例子：GQA 的 KV 头要不要复制

GQA 里 8 个 query 头共享 2 个 KV 头。最直接的写法是把 KV 头复制成 8 份再做普通的多头注意力；另一种写法是把 query 按组 reshape，让 KV 在"组"这一维上广播：

```python title="gqa_broadcast.py"
import torch

torch.manual_seed(0)
b, hq, hkv, s, d = 1, 8, 2, 16, 64
g = hq // hkv
q = torch.randn(b, hq, 1, d)        # decode：每个头一个 query
k = torch.randn(b, hkv, s, d)

k_rep = k.repeat_interleave(g, dim=1)                    # 写法一：KV 复制成 8 个头
s1 = q @ k_rep.transpose(-1, -2)
s2 = (q.view(b, hkv, g, 1, d) @ k.view(b, hkv, 1, s, d).transpose(-1, -2)).view(b, hq, 1, s)   # 写法二：按组广播
print("两种写法结果一致：", torch.allclose(s1, s2, atol=1e-5))
print("复制出来的 K 是原来的", k_rep.untyped_storage().nbytes() // k.untyped_storage().nbytes(), "倍")
```

```text title="输出"
两种写法结果一致： True
复制出来的 K 是原来的 4 倍
```

在 PyTorch 层面，写法二的 `matmul` 内部仍可能为了调用批量矩阵乘而把广播的操作数展开；真正省内存靠的是注意力 kernel **原生支持 GQA**：kernel 里用 `kv_head = q_head / group` 算出该读哪个 KV 头，从来不把 KV 复制成 query 的头数。FlashAttention、FlashInfer 都是这样做的——对 decode 这种纯带宽瓶颈的场景，这意味着 KV 读取量直接少了 `g` 倍。

## 非连续的张量与 kernel

自定义 kernel 几乎都按"指针 + 连续下标"访问数据，所以收到非连续的输入时有两种做法：

1. **先 `.contiguous()`**：简单，代价是一次额外的拷贝（读一遍写一遍）；
2. **把 stride 传进 kernel**：kernel 按 `base + i * stride0 + j * stride1` 访问，不拷贝——vLLM 的 `reshape_and_cache` 把 `key.stride(0)` 传进 kernel，就是因为 `key` 常常是从合并的 QKV 张量里切出来的视图（见 C++ 手册的[读懂推理基础库的 C++](cpp://engineering/reading-code/)）。

另外两个容易被忽略的元数据：

- **`dtype` 与 `element_size()`**：bf16 / fp16 每元素 2 字节，fp8 1 字节；kernel 按字节算地址时要乘上它；
- **存储的对齐**：`untyped_storage().data_ptr()` 通常按 64 字节以上对齐，但一个 `storage_offset` 不为 0 的视图，起始地址可能只按元素大小对齐——kernel 想做 16 字节的向量化读取之前，要先检查地址是否对齐。

## 练习

1. 一个形状 `(2, 3, 4)` 的连续张量，`permute(2, 0, 1)` 之后的 `stride` 是多少？它连续吗？再对它 `view(4, 6)` 会怎样？

??? success "参考答案"
    原张量 stride 是 `(12, 4, 1)`；`permute(2, 0, 1)` 把维度重排，stride 跟着重排成 `(1, 12, 4)`，形状 `(4, 2, 3)`，不连续。`view(4, 6)` 要把后两维合并成一维，要求 `stride[1] == shape[2] * stride[2]`，即 `12 == 3 * 4`，恰好成立，所以可以 view。

    ```python title="permute_stride.py"
    import torch

    x = torch.arange(24).reshape(2, 3, 4)
    p = x.permute(2, 0, 1)
    print(tuple(p.shape), p.stride(), p.is_contiguous())
    v = p.view(4, 6)
    print(tuple(v.shape), v.stride(), v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())
    ```

    ```text title="输出"
    (4, 2, 3) (1, 12, 4) False
    (4, 6) (1, 4) True
    ```

2. 一个 decode 步里有 `batch` 个请求，每个请求要从 KV Cache 里取出自己的块。为什么推理引擎不把每个请求的 KV "拼成"一个连续的大张量再调用注意力，而是把块表（block table）传给 kernel？

??? success "参考答案"
    拼接就是一次完整的拷贝：每一步都要把所有请求的全部 KV 读一遍、写一遍，而 decode 本身就是带宽瓶颈，相当于把注意力的访存量翻了几倍。
    把块表传进 kernel，kernel 按"逻辑位置 → 块号 → 物理地址"间接寻址，数据原地不动——这是分页注意力的核心，也是本章"传 stride / 传索引而不是拷贝"思路的延伸（见推理系统手册的[分页 KV Cache](serving://engine/paged-kv/)）。

## 小结

- [x] 张量 = storage（一块连续内存）+ 元数据（dtype、shape、stride、offset）；视图共享 storage，只换元数据。
- [x] `view` 只改元数据、可能失败；`reshape` 能视图就视图、否则拷贝；`contiguous()` 必要时拷贝。
- [x] `expand` 用零 stride 广播、零拷贝、不能安全地原地写；`repeat` 真的复制。
- [x] 注意力 kernel 原生支持 GQA、分页 KV 传块表，都是为了不拷贝数据。
- [x] kernel 处理非连续输入：先 `contiguous()`，或者把 stride 传进去；向量化读取前检查地址对齐。
