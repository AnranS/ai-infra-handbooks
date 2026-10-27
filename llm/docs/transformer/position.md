# 位置编码与 RoPE

<p class="lead">注意力本身不知道 token 的先后顺序。"猫追狗"和"狗追猫"在没有位置信息的注意力眼里是一样的。位置编码把顺序信息注入模型。今天的大模型几乎都用旋转位置编码（RoPE），它的设计直接影响长上下文扩展、KV Cache 的存储方式和推理 kernel 的写法。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么注意力需要位置编码？
    2. RoPE 对 Q、K 做了什么变换？为什么说它编码的是相对位置？
    3. `rope_theta`（基数）从 10000 调到 1000000 有什么作用？
    4. 为什么同一个模型的 RoPE 有两种写法（相邻两维一组 vs 前后两半一组）？
    5. KV Cache 里存的 K 是旋转前的还是旋转后的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 注意力对输入顺序是置换等变的：打乱 token 的顺序，输出只是跟着打乱，模型本身分不出先后，所以必须显式地注入位置信息。
    2. 把 q、k 的每两维看成一个复数（一对坐标），按 token 所在的位置旋转：第 m 个 token 的 q 转 $m\theta_i$，第 n 个 token 的 k 转 $n\theta_i$（m 是发出 query 的位置，n 是被看的 key 的位置，i 是第几对维度）。两者的点积只取决于转角之差 $(n - m)\theta_i$，所以编码的是相对位置。
    3. 频率 $\theta_i = \text{base}^{-2i/d}$，base 越大，低频维度转得越慢，在更长的距离内不会"转回原点"，模型能区分更远的位置，用于长上下文。
    4. 一种把相邻两维配成一对，另一种把前一半和后一半配对。两者数学上等价，差别只是维度的排列，但必须和权重的维度顺序配套（从某个格式转换权重时，q、k 的投影矩阵会相应重排）。
    5. 旋转后的：K 在写入缓存前就已经按它自己的位置旋转好了，之后直接和新的 q 做点积。

先看一个六格小剧场，再读正文：

![漫画：RoPE 把位置变成旋转](../assets/comics/rope.webp){.aig-comic}

## 注意力对顺序"视而不见"

不加掩码、不加位置信息时，注意力对输入顺序是**置换等变**的：把输入的 token 打乱，输出也只是按同样的方式打乱，每个 token 得到的结果完全不变。用随机的"词向量"和随机的投影矩阵验证一下：

```python
import math
import torch

torch.manual_seed(0)
T, d = 6, 16
x = torch.randn(T, d)                                 # 6 个 token，每个 16 维，代替词向量
Wq, Wk, Wv = (torch.randn(d, d) for _ in range(3))    # 代替训练好的投影矩阵

def attn(x):                                          # 最朴素的单头注意力：没有掩码，也没有位置编码
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    return (q @ k.T / math.sqrt(d)).softmax(-1) @ v

perm = torch.randperm(T)                              # 0～5 的一个随机排列
a = attn(x)[perm]                                     # 先算注意力，再按 perm 打乱输出
b = attn(x[perm])                                     # 先按 perm 打乱输入，再算注意力
print("perm:", perm.tolist())
print("原来第 3 个 token 的输出:", [round(v, 3) for v in attn(x)[3, :4].tolist()])
print("打乱后排第 0 的输出:   ", [round(v, 3) for v in b[0, :4].tolist()])
print("两种算法一致:", torch.allclose(a, b, atol=1e-5))
assert torch.allclose(a, b, atol=1e-5)                # 打乱输入 = 打乱输出
```

```text title="输出"
perm: [3, 5, 1, 2, 4, 0]
原来第 3 个 token 的输出: [2.923, 2.794, 2.621, -0.517]
打乱后排第 0 的输出:    [2.923, 2.794, 2.621, -0.517]
两种算法一致: True
```

`x[perm]` 把输入重新排队：新顺序的第 0 个是原来的第 3 个 token，第 1 个是原来的第 5 个，依此类推。原来第 3 个 token 挪到了最前面，算出来的输出却一个数都没变（这里只打印了前 4 维）。整体比较两种算法，结果也一致；差别只在 $10^{-6}$ 量级，是求和顺序不同带来的浮点舍入，所以 `allclose` 要给一点容差。

为什么一定成立，看注意力的三步：

- q、k、v 都是每一行单独乘矩阵得到的，输入的行换了顺序，q、k、v 的行就跟着换；
- 分数矩阵 $QK^\top$ 的行和列按同一个顺序换，softmax 是按行做的，不受影响；
- 最后用权重对 v 加权求和，加法不在乎先后：权重的列和 v 的行一起换了顺序，正好抵消。

写成矩阵就是：对任意置换矩阵 $P$，$\text{attn}(Px) = P\,\text{attn}(x)$。每个 token 的输出只取决于"有哪些 token"，不取决于"它们怎么排"。

换成句子更直观。给"猫""追""狗"三个词随便配上词向量，看"猫"在"猫追狗"和"狗追猫"里各得到什么输出：

```python
emb = {w: torch.randn(d) for w in "猫追狗"}             # 三个词，随便配上词向量

def output_of(sentence, word, pos=None):              # word 在这句话里经过注意力之后的输出
    xs = torch.stack([emb[w] for w in sentence])
    if pos is not None:
        xs = xs + pos[: len(sentence)]                # 每个位置加上自己的位置向量
    return attn(xs)[sentence.index(word)]

show = lambda t: [round(v, 2) for v in t[:3].tolist()]
print("不加位置  猫追狗:", show(output_of("猫追狗", "猫")), " 狗追猫:", show(output_of("狗追猫", "猫")))
pos = torch.randn(T, d)                               # 每个位置一个随机向量，代替位置编码
print("加了位置  猫追狗:", show(output_of("猫追狗", "猫", pos)), " 狗追猫:", show(output_of("狗追猫", "猫", pos)))
```

```text title="输出"
不加位置  猫追狗: [6.0, -4.0, -1.43]  狗追猫: [6.0, -4.0, -1.43]
加了位置  猫追狗: [10.4, 1.78, -1.63]  狗追猫: [9.95, 0.01, -0.46]
```

不加位置时，两句话里"猫"的输出一模一样，模型分不出是猫追狗还是狗追猫，而语言显然依赖顺序。给每个位置加上一个不同的向量之后，同一个词在不同位置的输入不同了，输出也就不同。位置编码做的就是这件事，下面几节讲具体怎么加。

这里特意没加因果掩码。掩码让第 i 个 token 只能看到前面的 token，打乱之后"谁能看到谁"就变了，等式不再成立，所以因果掩码本身会带来一点隐式的位置信息。但这远不如显式的位置编码直接，主流的大模型都会加位置编码。

## 早期方案：绝对位置编码

- **正弦位置编码**（原始 Transformer）：给第 m 个位置一个固定的向量（不同频率的 sin、cos），加到词嵌入上；
- **可学习的绝对位置嵌入**（GPT-2、BERT）：像词嵌入一样，为每个位置学一个向量。

它们的问题是：模型学到的是"第 5 个位置"这种绝对信息，而语言里更重要的往往是**相对**距离（"前一个词"、"三个词之前"）；可学习的位置嵌入还无法处理比训练时更长的序列。

## RoPE：用旋转编码位置

旋转位置编码（Rotary Position Embedding，苏剑林等，2021）的想法非常优雅：**不把位置加到向量上，而是按位置把 q 和 k 旋转一个角度**。

![图：RoPE 把每一对维度按位置旋转，点积只剩下相对位置](../assets/figures/rope.svg){.aig-svg}

先看二维的情况。取序列里的两个 token：位置 m 的那个发出 query（"谁在看"），位置 n 的那个提供 key（"被看的"）；位置就是它在序列里是第几个 token，从 0 数。把 m 的 query 向量旋转 $m\theta$ 角度，n 的 key 向量旋转 $n\theta$ 角度：

$$
q'_m = R(m\theta)\, q,\qquad k'_n = R(n\theta)\, k,\qquad
R(\alpha) = \begin{pmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{pmatrix}
$$

旋转矩阵满足 $R(a)^\top R(b) = R(b - a)$，所以它们的点积：

$$
q'^\top_m k'_n = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R\big((n - m)\theta\big)\, k
$$

**只依赖相对位置 $n - m$**，与绝对位置无关。而且它是乘性的，不会像加法那样把位置信息和内容信息混在一起。

旋转后的点积 $q'^\top_m k'_n$ 就是分数矩阵第 m 行、第 n 列的那个数。举个例子：「我 爱 吃 苹果」四个 token 的位置是 0、1、2、3，「苹果」看「爱」时 m = 3、n = 1，q 转 $3\theta$，k 转 $\theta$，点积里只剩 $(1 - 3)\theta = -2\theta$。把这两个词挪到位置 103 和 101，还是 $-2\theta$，分数不变。有因果掩码时 n ≤ m，所以 n − m 总是 ≤ 0，表示往前看隔了几个 token。

对 $d_h$ 维的向量，把它分成 $d_h/2$ 组，每组两维独立旋转，第 i 组使用不同的频率：

$$
\theta_i = \text{base}^{-2i/d_h},\qquad i = 0, 1, \ldots, d_h/2 - 1
$$

前面的组转得快（高频，对近距离敏感），后面的组转得慢（低频，能区分很远的距离）。`base` 就是配置文件里的 `rope_theta`。

### 实现

实践中有两种等价的分组方式：

- **相邻两维一组**：(0, 1)、(2, 3)……，可以把每组看成一个复数，旋转就是乘以 $e^{i m\theta}$。Meta 发布的原始 LLaMA 代码是这样写的；
- **前后两半一组**：第 i 维和第 i + d_h/2 维一组，用 `rotate_half` 实现。Hugging Face transformers、Qwen 等采用这种写法。

```python title="rope.py"
"""rope.py —— 旋转位置编码的两种等价实现。"""

import torch


def inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))   # [d/2]


def rope_half(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """"前后两半一组"（transformers 的写法）。x: [..., T, d]，pos: [T]"""
    freqs = pos.float()[:, None] * inv_freq(x.shape[-1], base)[None, :]   # [T, d/2]
    cos, sin = torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)                                 # rotate_half
    return x * cos + rotated * sin


def rope_complex(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """"相邻两维一组"（原始 LLaMA 的写法）：把 (x0, x1) 看成复数 x0 + i·x1，乘以 e^{i·m·θ}。"""
    freqs = pos.float()[:, None] * inv_freq(x.shape[-1], base)[None, :]
    rot = torch.polar(torch.ones_like(freqs), freqs)                       # e^{i m θ}
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * rot).flatten(-2)
```

验证 RoPE 的关键性质：

```python
from rope import rope_complex, rope_half

torch.manual_seed(0)
d = 64
q, k = torch.randn(d), torch.randn(d)

def score(m, n, rope):
    """第 m 个 token 的 q（旋转 mθ）和第 n 个 token 的 k（旋转 nθ）的点积"""
    qm = rope(q[None, :], torch.tensor([m]))[0]
    kn = rope(k[None, :], torch.tensor([n]))[0]
    return (qm @ kn).item()

# 1) 点积只取决于相对位置：(5, 3)、(105, 103)、(1005, 1003) 的结果相同
for rope in (rope_half, rope_complex):
    s = [score(m, m - 2, rope) for m in (5, 105, 1005)]
    print(f"{rope.__name__:12} 相隔 2：", [round(v, 3) for v in s])
    assert max(s) - min(s) < 1e-3, s
print("相隔 0、1、2、3、10：", [round(score(10, 10 - r, rope_half), 3) for r in (0, 1, 2, 3, 10)])

# 2) 旋转不改变向量长度
x = torch.randn(10, d)
xr = rope_half(x, torch.arange(10))
print("旋转前后的长度：", round(x[9].norm().item(), 3), round(xr[9].norm().item(), 3))
assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), atol=1e-4)

# 3) 两种写法等价：只是维度的排列顺序不同
perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])   # 相邻排列 -> 前后两半排列
x = torch.randn(7, d)
pos = torch.arange(7)
same = torch.allclose(rope_complex(x, pos)[:, perm], rope_half(x[:, perm], pos), atol=1e-5)
print("两种写法一致：", same)
assert same
```

```text title="输出"
rope_half    相隔 2： [-11.249, -11.249, -11.249]
rope_complex 相隔 2： [-10.143, -10.143, -10.143]
相隔 0、1、2、3、10： [-11.434, -12.406, -11.249, -8.867, -4.946]
旋转前后的长度： 7.745 7.745
两种写法一致： True
```

逐行看：

- 前两行：q 在位置 5、k 在位置 3，q 在 105、k 在 103，q 在 1005、k 在 1003，三对都相隔 2，点积完全一样。绝对位置变了，分数不变，这就是"只依赖 n − m"。两种写法的数不一样，是因为它们把不同的维度配成一对，同一个 q、k 在两种写法里转出来的向量不同（第 3 条会看到，把维度重新排列之后两者完全一致）。
- 第三行：q 固定在位置 10，k 依次放在 10、9、8、7、0，也就是相隔 0、1、2、3、10，分数各不相同。相对距离一变分数就变，训练好的模型就靠这个分辨远近。
- 第四行：旋转只改方向、不改长度，RoPE 不会放大或缩小 q、k。
- 第五行：把维度从"相邻两维一组"的排法换成"前后两半一组"的排法，两种实现的结果完全一致。

第 3 条说明：两种写法的区别只是 q、k 维度的**排列**。这有一个实际后果：把 Meta 格式的 LLaMA 权重转换成 Hugging Face 格式时，转换脚本会对 `q_proj`、`k_proj` 的输出维度做一次重排。**权重和 RoPE 的写法必须配套**，否则模型能跑但输出全是乱码。

## 频率与"波长"

第 i 组维度转一整圈需要的位置数（波长）是 $2\pi / \theta_i$：

```pycon
>>> import math
>>> from rope import inv_freq
>>> for base in (10_000, 1_000_000):
...     f = inv_freq(64, base)
...     print(base, [round(2 * math.pi / f[i].item()) for i in (0, 8, 16, 24, 31)])
...
10000 [6, 63, 628, 6283, 47117]
1000000 [6, 199, 6283, 198692, 4080185]
```

最快的一组每 6 个 token 转一圈，最慢的一组要几万甚至几百万个 token。把 base 从 10000 增大到 1000000（Qwen2.5 的取值，LLaMA-3 用 500000），低频部分的波长被拉得更长，模型在很长的距离上仍能区分位置，这是支持长上下文的基础之一。

把一对维度随位置的变化画成三维的螺旋，就能同时看到"转得多快"（波长）和"q、k 的夹角只取决于 m − n"这两件事。拖动旋转，把 i 从 0 拉到 63，看螺旋怎么从弹簧变成直线：

<div class="aig-widget" data-widget="rope-helix"></div>

## 长上下文扩展

模型在 4K 长度上训练，推理时要用 32K 甚至 128K，超出训练范围的位置（旋转角度）模型没见过，效果会急剧下降。几种常见的扩展方法：

| 方法 | 做法 | 说明 |
| --- | --- | --- |
| 位置插值（PI） | 把位置 m 缩放成 m / s，所有位置都"压"回训练范围 | 简单，但高频部分被压缩后近距离分辨率下降，通常需要少量微调 |
| NTK-aware 缩放 | 增大 base，主要拉长低频部分的波长，高频部分基本不变 | 不微调也有一定效果 |
| YaRN | 按频率分段处理：高频不插值、低频插值、中间过渡，并对注意力分数做温度修正 | Qwen2.5 用它把上下文扩展到 128K，配置里写作 `rope_scaling: {"type": "yarn", "factor": 4.0, ...}` |
| LLaMA-3.1 的方式 | 类似 YaRN 的按频率分段缩放 | 配置里的 `rope_type: "llama3"` |

这些方法都只修改 cos/sin 表的计算方式（以及可能的注意力缩放），模型结构不变。**推理引擎必须按配置实现同样的 RoPE 变体**，否则长文本效果会明显变差。

另一种思路是**ALiBi**：不旋转向量，而是直接给注意力分数加上一个与距离成正比的负偏置，远的 token 被"惩罚"。BLOOM、MPT 等模型用过它，现在主流模型多用 RoPE。

!!! inference "推理视角"
    - **KV Cache 里存的是旋转之后的 K**：每个 token 的 K 在写入缓存时就按它的位置旋转好了，之后不再改变。decode 时只需要对新 token 的 q、k 做旋转；
    - 因为 K 已经按绝对位置旋转过，**同一段文本只有出现在相同的位置时，缓存才能复用**。前缀缓存天然满足这一点（共享的都是从位置 0 开始的前缀）；而丢弃中间 token 或者把缓存"平移"到别的位置时，就要重新处理位置；
    - RoPE 的计算量很小但访存不少，推理引擎通常把它和 QKV 投影之后的操作融合成一个 kernel（有时连同写 KV Cache 一起），而不是单独启动；
    - DeepSeek 的 MLA 为了能压缩 KV Cache，把 RoPE 单独拆出一小部分维度，见[注意力变体](attention-variants.md#mla多头潜在注意力)。

!!! interview "怎么讲清楚"
    讲 RoPE：按位置旋转 q、k 的每两维，点积只依赖相对位置；不同维度组的频率不同，base（`rope_theta`）越大低频转得越慢，用于长上下文；两种写法（相邻两维一组、前后两半一组）数学上等价，但必须和权重的维度排列配套——接入新模型时常见的 bug；KV Cache 里存的是旋转后的 K，所以缓存复用要求位置一致；PI、NTK、YaRN 等长上下文扩展只改 cos / sin 的计算，推理时 RoPE 常和相邻的操作融合。

## 练习

**1. 手算旋转。** 二维向量 q = (1, 0)，θ = π/2，位置 m = 1。RoPE 之后 q 变成什么？k = (1, 0) 在位置 n = 3 呢？它们的点积是多少，和 $q^\top R((n-m)\theta)k$ 一致吗？

??? success "参考答案"
    q' = R(π/2)(1, 0) = (0, 1)；k' = R(3π/2)(1, 0) = (0, −1)；点积 = −1。
    直接计算：$R(2 \cdot \pi/2) = R(\pi)$，$q^\top R(\pi) k = (1, 0) \cdot (−1, 0) = −1$，一致。

    ```python
    import math, torch
    def R(a): return torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = k = torch.tensor([1.0, 0.0])
    lhs = (R(math.pi / 2) @ q) @ (R(3 * math.pi / 2) @ k)
    rhs = q @ R(math.pi) @ k
    assert abs(lhs.item() + 1) < 1e-6 and abs(rhs.item() + 1) < 1e-6
    ```

**2. 思考题。** 如果推理引擎在 prefill 时 RoPE 用了一种写法，decode 时用了另一种写法，会发生什么？

??? success "参考答案"
    缓存中的 K 按一种维度分组方式旋转，新 token 的 q 按另一种方式旋转，两者的相对位置关系被破坏，注意力分数变成错误的值。表现通常是：前几个 token 正常（它们来自 prefill 的最后一个位置），之后很快开始输出重复或无意义的内容。这类 bug 很隐蔽，推理引擎的测试一般会对比"带缓存逐步生成"和"不带缓存整段计算"的 logits 是否一致，[KV Cache](../inference/kv-cache.md) 一章会用这种方法验证我们的实现。

## 小结

- [x] 注意力对顺序置换等变，必须显式注入位置信息。
- [x] RoPE 按位置旋转 q、k，点积只依赖相对位置；不同维度组使用不同频率，base 越大低频越慢。
- [x] 两种实现（相邻一组 / 前后两半一组）等价，但必须和权重的维度排列配套。
- [x] 长上下文扩展（PI、NTK、YaRN）只改 cos/sin 的计算，推理引擎要按配置实现。
- [x] KV Cache 存旋转后的 K，缓存复用要求位置一致；RoPE 常与其他操作融合成一个 kernel。
