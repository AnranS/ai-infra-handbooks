# 概率与采样

<p class="lead">语言模型输出的是一个概率分布，生成文本就是从这个分布中反复采样。温度、top-p、投机解码的正确性证明、RL 中的重要性采样比、压测结果的误差条，背后都是同一套概率工具。这一章从离散分布讲起，逐一推导推理中用到的采样算法，并用蒙特卡洛实验和真实模型的分布验证每一个结论。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 从一个离散分布中采样，有哪几种方法？Gumbel-max 技巧为什么成立？
    2. 用 n 次采样估计一个概率，误差大约是多少？
    3. 序列的概率怎么由每一步的条件概率算出？为什么贪心解码不一定得到概率最大的序列？
    4. 投机采样的接受率等于什么？
    5. 重要性采样是做什么的？为什么权重方差大时估计就不可靠？

## 离散分布、期望与方差

语言模型在每个位置给出一个定义在词表上的**离散分布** $p(x)$，$\sum_x p(x) = 1$。对任意函数 $f$，**期望** $\mathbb{E}_p[f] = \sum_x p(x) f(x)$，**方差** $\text{Var}[f] = \mathbb{E}[f^2] - \mathbb{E}[f]^2$。

logits 经过 softmax 变成分布，温度 $T$ 在 softmax 之前把 logits 除以 $T$：$p_T(x) \propto e^{z_x / T}$。$T < 1$ 让分布更尖锐，$T > 1$ 让分布更平坦。在真实模型的一个位置上看看温度的影响：

```python
import copy
import math
import re
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer

torch.set_num_threads(16)
path = "models/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
raw = re.sub(r"```.*?```", "", open("docs/basics/language-model.md").read(), flags=re.S)
ids = tok(re.sub(r"[#*`>|\-\[\]()!]", "", raw)).input_ids[:400]          # 大模型手册中的一段正文
with torch.no_grad():
    logits = model(torch.tensor([ids]))[0]                                 # 每个位置的 logits

for T in (0.5, 1.0, 1.5):
    p = (logits[49] / T).softmax(-1).sort(descending=True).values           # 第 50 个位置
    print(f"T = {T}：最可能的 token 概率 {p[0]:.3f}，凑够 90% 概率需要 {int((p.cumsum(0) < 0.9).sum()) + 1} 个 token")
```

```text
T = 0.5：最可能的 token 概率 0.975，凑够 90% 概率需要 1 个 token
T = 1.0：最可能的 token 概率 0.622，凑够 90% 概率需要 25 个 token
T = 1.5：最可能的 token 概率 0.188，凑够 90% 概率需要 2260 个 token
```

同一个位置，温度从 0.5 到 1.5，分布从"几乎确定"变成"两千多个 token 分享 90% 的概率"。这就是高温采样容易"胡言乱语"的原因：大量低概率 token 的总概率变得可观，这也是 top-p、min-p 这类截断要解决的问题（见[解码与采样](../inference/decoding.md)）。

## 采样算法

给定分布 $p$，怎样生成一个服从它的随机样本？三种方法：

1. **逆 CDF**：算出累积分布 $F(x) = \sum_{y \le x} p(y)$，取均匀随机数 $u \sim U(0,1)$，返回第一个满足 $F(x) \ge u$ 的 $x$。直观，但需要前缀和与查找；
2. **Gumbel-max**：给每个 logit 加上独立的 Gumbel 噪声 $g = -\log(-\log u)$，取 argmax。因为 $\arg\max_x (z_x + g_x)$ 恰好以概率 $\text{softmax}(z)_x$ 选中 $x$；
3. **指数竞赛**：取 $\arg\max_x p(x) / E_x$，$E_x$ 服从参数为 1 的指数分布。$E_x / p(x)$ 服从参数为 $p(x)$ 的指数分布，而若干独立指数分布中最小的那个是 $x$ 的概率为 $p(x) / \sum_y p(y)$。它与 Gumbel-max 其实是同一件事：$-\log E$ 就是 Gumbel 分布。vLLM 的采样器用的就是这个方法（见推理系统手册的[批量采样](serving://engine/sampler-api/#批量采样)）。

后两种方法只需要逐元素运算和一次 argmax，没有前缀和、没有 CPU-GPU 同步，适合 GPU 上批量执行。用蒙特卡洛实验验证三者得到相同的分布：

```python
torch.manual_seed(0)
p = torch.tensor([0.5, 0.25, 0.15, 0.07, 0.03])
n = 200_000
u = torch.rand(n, 1)
inverse_cdf = torch.searchsorted(p.cumsum(0), u).squeeze(1).clamp(max=len(p) - 1)
gumbel = (p.log() - torch.log(-torch.log(torch.rand(n, len(p))))).argmax(-1)
race = (p / torch.empty(n, len(p)).exponential_()).argmax(-1)
for name, samples in [("逆 CDF", inverse_cdf), ("Gumbel-max", gumbel), ("指数竞赛", race)]:
    freq = torch.bincount(samples, minlength=len(p)).float() / n
    print(f"{name:10s} {[round(f, 3) for f in freq.tolist()]}  与 p 的最大偏差 {(freq - p).abs().max():.4f}")
```

```text
逆 CDF      [0.499, 0.25, 0.151, 0.07, 0.03]  与 p 的最大偏差 0.0014
Gumbel-max [0.501, 0.249, 0.151, 0.07, 0.03]  与 p 的最大偏差 0.0015
指数竞赛       [0.501, 0.249, 0.15, 0.069, 0.03]  与 p 的最大偏差 0.0006
```

## 蒙特卡洛的误差

用 $n$ 次独立采样中事件发生的频率 $\hat{p}$ 估计它的概率 $p$，$\hat p$ 的标准差是 $\sqrt{p(1-p)/n}$：**误差随样本数的平方根下降**，样本数翻 100 倍，误差才缩小 10 倍。上面 20 万次采样，误差约 0.001 的量级，与看到的偏差一致。

```python
p_true = 0.3
for n in (100, 10_000):
    estimates = (torch.rand(2000, n) < p_true).float().mean(1)     # 重复 2000 次实验，每次采样 n 个
    print(f"n = {n:6d}：估计值的标准差 {estimates.std():.4f}，理论值 {math.sqrt(p_true * (1 - p_true) / n):.4f}")
```

```text
n =    100：估计值的标准差 0.0458，理论值 0.0458
n =  10000：估计值的标准差 0.0046，理论值 0.0046
```

这个结论在推理工作中随处可见：投机解码的接受率、压测中的 P99 延迟、评测集上的准确率，都是"用有限样本估计的量"，都有误差条。评测集只有 200 道题时，准确率 70% 的标准误约 3 个百分点，两个方案相差 2 个百分点并不能说明谁更好（性能测量的统计问题见[性能与服务中的数学](performance-math.md)）。

## 序列的概率

语言模型用**链式法则**给整个序列定义概率：

$$
p(x_1, \dots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}), \qquad \log p(x_{1:T}) = \sum_{t} \log p(x_t \mid x_{<t})
$$

实际计算都在对数空间中进行，否则几百个小于 1 的概率相乘会下溢为 0。一个重要的推论：**每一步都选概率最大的 token（贪心），不一定得到概率最大的序列**：

```pycon
>>> # 第一步：A 概率 0.6，B 概率 0.4；选 A 之后第二步最多只有 0.5，选 B 之后第二步有 0.9
>>> paths = {("A", "a1"): 0.6 * 0.5, ("A", "a2"): 0.6 * 0.5, ("B", "b1"): 0.4 * 0.9, ("B", "b2"): 0.4 * 0.1}
>>> max(paths, key=paths.get), round(max(paths.values()), 2)          # 概率最大的序列
(('B', 'b1'), 0.36)
>>> round(paths[("A", "a1")], 2)                                     # 贪心得到的序列
0.3
```

束搜索（beam search）同时保留若干条候选路径来缓解这个问题，但对话模型通常用采样而不是束搜索：概率最大的序列往往是重复、乏味的文本。

## 拒绝采样与投机解码

**拒绝采样**：想从分布 $p$ 采样，但只能方便地从 $q$ 采样。从 $q$ 取一个样本 $x$，以一定概率接受，拒绝时再用别的方式补救，使最终结果恰好服从 $p$。投机采样就是一个特例：草稿模型给出 $x \sim q$，以 $\min(1, p(x)/q(x))$ 接受，拒绝时从 $\text{norm}(\max(0, p - q))$ 重新采样。大模型手册的[推理服务](../inference/serving.md#投机解码)一章用蒙特卡洛验证了输出分布等于 $p$。

它的**接受率**是

$$
\sum_x q(x) \min\!\left(1, \frac{p(x)}{q(x)}\right) = \sum_x \min(p(x), q(x)) = 1 - \text{TV}(p, q)
$$

其中 $\text{TV}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ 是两个分布的**总变差距离**。所以草稿模型越"像"目标模型，接受率越高。在真实模型上测一测：目标是 FP32 的 Qwen2.5-0.5B，草稿是它的 INT4 量化版本，在一段文本的 400 个位置上计算接受率：

```python
from quant import fake_quant_int

draft = copy.deepcopy(model)
for layer in draft.layers:
    a, f = layer.self_attn, layer.mlp
    for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, f.gate_proj, f.up_proj, f.down_proj):
        lin.weight.data = fake_quant_int(lin.weight.data, 4, "group")
with torch.no_grad():
    p_target, q_draft = logits.softmax(-1), draft(torch.tensor([ids]))[0].softmax(-1)
accept = torch.minimum(p_target, q_draft).sum(-1)
tv = 0.5 * (p_target - q_draft).abs().sum(-1)
print(f"接受率：平均 {accept.mean():.3f}，中位数 {accept.median():.3f}，最低 {accept.min():.3f}")
print(f"1 - 总变差距离 的平均值：{(1 - tv).mean():.3f}")
assert torch.allclose(accept, 1 - tv, atol=1e-4)
```

```text
接受率：平均 0.706，中位数 0.693，最低 0.087
1 - 总变差距离 的平均值：0.706
```

平均每个草稿 token 有约 70% 的概率被接受，与"1 − 总变差距离"完全一致；但在少数位置，两个模型的分布差别很大，接受率低到 9%。

## 重要性采样

想计算 $\mathbb{E}_p[f]$，但样本来自另一个分布 $q$。利用

$$
\mathbb{E}_p[f] = \sum_x p(x) f(x) = \sum_x q(x) \frac{p(x)}{q(x)} f(x) = \mathbb{E}_q\!\left[w(x) f(x)\right], \quad w = \frac{p}{q}
$$

给每个样本乘上**重要性权重** $w$ 即可。RL 中的 PPO、GRPO 用的比值 $\pi_\theta / \pi_\text{old}$，以及修正"推理引擎与训练框架概率不一致"的比值（见推理系统手册的 [RL 训练中的推理](serving://topics/rl-rollout/#问题二训练与推理的概率不一致)），都是重要性权重。

它的问题在于：$q$ 与 $p$ 差别大时，少数样本的权重极大，估计的方差爆炸。一个衡量指标是**有效样本数** $\text{ESS} = (\sum w)^2 / \sum w^2$：

```python
torch.manual_seed(0)
values = torch.arange(10).float()                                   # f(x) = x
p = torch.softmax(torch.linspace(0, 2, 10), 0)                      # 目标分布
true_mean = (p * values).sum().item()
for shift in (0.0, 1.0, 3.0):
    q = torch.softmax(torch.linspace(0, 2, 10) - shift * torch.linspace(0, 1, 10), 0)   # 越来越不像 p 的采样分布
    x = torch.multinomial(q, 1000, replacement=True)
    w = p[x] / q[x]
    estimate = (w * values[x]).mean().item()
    ess = (w.sum() ** 2 / (w ** 2).sum()).item()
    clipped = (w.clamp(max=2.0) * values[x]).mean().item()           # 截断权重：方差更小，但有偏
    print(f"偏移 {shift}：估计 {estimate:.2f}（真值 {true_mean:.2f}），有效样本数 {ess:6.0f} / 1000，截断后估计 {clipped:.2f}")
```

```text
偏移 0.0：估计 6.18（真值 6.20），有效样本数   1000 / 1000，截断后估计 6.18
偏移 1.0：估计 6.11（真值 6.20），有效样本数    922 / 1000，截断后估计 6.11
偏移 3.0：估计 6.60（真值 6.20），有效样本数    489 / 1000，截断后估计 4.98
```

$q$ 越偏离 $p$，有效样本数越少，估计越不稳定；把权重截断在某个上限以内（PPO 的 clip、TIS），能控制方差，但会引入偏差。这正是 RL 算法中各种"比值截断"的数学背景。

## 练习

**1. top-k 之后的接受率。** 如果目标模型与草稿模型都先做 top-k 截断再采样，接受率会怎样变化？

??? success "参考思路"
    接受率仍然是截断后的两个分布之间的 $\sum_x \min(p'(x), q'(x))$。如果两者的 top-k 集合大部分重合，截断把尾部的概率重新分配到头部，两个分布往往更接近，接受率可能上升；如果 top-k 集合差别大，重合部分变少，接受率会下降。注意投机采样的正确性要求：验证时用的 $p$、草稿时用的 $q$ 必须都是**实际用来采样的分布**（包括温度、截断），否则输出分布就不再等于目标分布。

**2. 需要多少样本？** 你想比较两个投机解码配置的平均接受长度，它们大约是 2.5 和 2.6，每次验证的接受长度标准差约为 1.2。每个配置至少需要多少次验证，才能让两者的差距超过 2 倍标准误？

??? success "参考答案"
    两个均值之差的标准误约为 $\sqrt{2} \times 1.2 / \sqrt{n}$。要求 $0.1 > 2 \times \sqrt{2} \times 1.2 / \sqrt{n}$，得 $n > (2 \times 1.7 / 0.1)^2 \approx 1150$。每个配置至少要一千多次验证（也就是生成几千个 token），差距小于这个时的比较结论不可靠。

## 小结

- [x] 温度改变分布的尖锐程度；高温时大量低概率 token 的总概率可观，需要截断。
- [x] 逆 CDF、Gumbel-max、指数竞赛都能从离散分布中精确采样，后两者适合 GPU 批量执行。
- [x] 蒙特卡洛误差按 $1/\sqrt{n}$ 下降；任何用样本估计的量都有误差条。
- [x] 序列概率是条件概率的乘积（对数空间中求和）；贪心不一定得到概率最大的序列。
- [x] 投机采样的接受率 = $1 - \text{TV}(p, q)$；重要性采样用 $p/q$ 加权，权重方差大时需要截断，以偏差换方差。
