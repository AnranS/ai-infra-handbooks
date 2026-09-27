# 手撕代码题

<p class="lead">推理岗的面试几乎都有现场写代码的环节，题目集中在三类：用 PyTorch 写模型的核心组件（注意力、RoPE、RMSNorm、采样），写推理系统的数据结构与算法（LRU、前缀树、调度、all-reduce），写 GPU kernel（softmax、归约、GEMM）。这一章给出前两类中最常考的题目：每题先说明面试官在看什么，再给出可以直接运行、并与参考实现逐项核对的答案。GPU kernel 的题目见 CUDA 手册的[面试题库](cuda://career/interview/)和各算子章节。</p>

!!! tip "怎么练"
    先自己在白纸或空白编辑器里限时写（15～20 分钟一题），写完再对照答案，并把答案里的测试代码跑一遍。面试中写完之后，主动说出复杂度、边界情况和怎样测试，往往比代码本身更加分。

## 1. 带 KV Cache 的 GQA 注意力

**题目**：实现一个多头注意力模块，支持 GQA（KV 头数少于 query 头数）、因果掩码，以及增量解码时的 KV Cache。

**考察点**：形状变换（`view`/`transpose`）、缩放因子、GQA 的头复制、因果掩码在有缓存时的偏移、缓存的拼接。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GQAttention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.nh, self.nkv, self.hd = n_heads, n_kv_heads, dim // n_heads
        self.wq = nn.Linear(dim, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, dim, bias=False)

    def forward(self, x, cache=None):
        """x: [B, T, dim]；cache: None 或 (k, v)，形状 [B, n_kv, S_past, hd]。返回 (输出, 新的 cache)。"""
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd).transpose(1, 2)        # [B, nh, T, hd]
        k = self.wk(x).view(B, T, self.nkv, self.hd).transpose(1, 2)       # [B, nkv, T, hd]
        v = self.wv(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        S = k.shape[2]
        rep = self.nh // self.nkv
        kr, vr = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        scores = q @ kr.transpose(-2, -1) / math.sqrt(self.hd)             # [B, nh, T, S]
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)      # 新 token 位于序列末尾
        scores = scores.masked_fill(~mask, float("-inf"))
        out = scores.softmax(-1) @ vr                                       # [B, nh, T, hd]
        return self.wo(out.transpose(1, 2).reshape(B, T, -1)), (k, v)


torch.manual_seed(0)
attn = GQAttention(dim=64, n_heads=8, n_kv_heads=2)
x = torch.randn(2, 10, 64)
full, _ = attn(x)                                                           # 一次算完整个序列
out, cache = attn(x[:, :6])                                                 # 先 prefill 6 个 token
steps = [out]
for t in range(6, 10):                                                      # 再逐个 decode
    o, cache = attn(x[:, t:t + 1], cache)
    steps.append(o)
print("增量计算与一次计算的最大误差：", (torch.cat(steps, 1) - full).abs().max().item())
q = attn.wq(x).view(2, 10, 8, 8).transpose(1, 2)                            # 与 PyTorch 官方实现比较
k = attn.wk(x).view(2, 10, 2, 8).transpose(1, 2).repeat_interleave(4, 1)
v = attn.wv(x).view(2, 10, 2, 8).transpose(1, 2).repeat_interleave(4, 1)
ref = attn.wo(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 64))
assert torch.allclose(full, ref, atol=1e-5) and torch.allclose(torch.cat(steps, 1), full, atol=1e-5)
```

```text
增量计算与一次计算的最大误差： 1.1920928955078125e-07
```

**常见错误**：掩码写成 `tril()` 不带偏移（有缓存时 decode 的 query 什么都看不到或看错位置）；GQA 用 `repeat` 而不是 `repeat_interleave`（头的对应关系错乱）；忘记除以 $\sqrt{d_h}$。

## 2. RoPE

**题目**：实现旋转位置编码，并说明它为什么编码的是相对位置。

**考察点**：频率 $\theta^{-2i/d}$、两种等价写法（相邻两维一组，或前后两半一组）、位置从缓存长度开始。

```python
def rope(x, positions, theta=10000.0):
    """x: [..., T, d]，把第 i 维与第 i + d/2 维看成一个二维向量，按 位置 × 频率 旋转。"""
    d = x.shape[-1]
    inv_freq = theta ** (-torch.arange(0, d, 2).float() / d)               # [d/2]
    angles = positions.float()[:, None] * inv_freq[None, :]                 # [T, d/2]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

from mini_llm import apply_rope, rope_cos_sin
x = torch.randn(2, 4, 16, 64)
pos = torch.arange(100, 116)                                                # 有缓存时位置从缓存长度开始
cos, sin = rope_cos_sin(pos, 64, 10000.0)
print("与 mini_llm 的实现一致：", torch.allclose(rope(x, pos), apply_rope(x, cos, sin), atol=1e-5))

q, k = torch.randn(64), torch.randn(64)                                     # 相对位置：只取决于位置之差
s1 = rope(q[None], torch.tensor([10])) @ rope(k[None], torch.tensor([7])).T
s2 = rope(q[None], torch.tensor([110])) @ rope(k[None], torch.tensor([107])).T
print("位置 (10, 7) 与 (110, 107) 的注意力分数之差：", (s1 - s2).abs().item())
assert torch.allclose(rope(x, pos), apply_rope(x, cos, sin), atol=1e-5) and (s1 - s2).abs() < 1e-3
```

```text
与 mini_llm 的实现一致： True
位置 (10, 7) 与 (110, 107) 的注意力分数之差： 7.152557373046875e-06
```

## 3. RMSNorm

**题目**：实现 RMSNorm，并说明推理引擎中为什么要把它和残差相加融合。

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()                                                     # 统计量用 FP32 计算，防止 BF16 溢出和精度损失
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

norm = RMSNorm(64)
nn.init.normal_(norm.weight)
x = torch.randn(4, 64)
ref = F.rms_norm(x, (64,), norm.weight, eps=1e-6)
print("与 torch.nn.functional.rms_norm 的最大误差：", (norm(x) - ref).abs().max().item())
assert torch.allclose(norm(x), ref, atol=1e-6)
```

```text
与 torch.nn.functional.rms_norm 的最大误差： 0.0
```

**要点**：RMSNorm 是访存受限的算子（读一遍、写一遍，计算很少）。推理中 `residual = x + residual; y = rmsnorm(residual)` 如果分成两个 kernel，要多读写一遍整个张量；融合之后只读写一次，这就是 vLLM、SGLang 中的 fused add RMSNorm（见 CUDA 手册的 [Softmax 与归一化](cuda://kernels/softmax-norm/)）。

## 4. 在线 softmax 与分块注意力（FlashAttention 的核心）

**题目**：不物化完整的注意力矩阵，分块计算注意力。

**考察点**：在线 softmax 的递推（维护行最大值 $m$ 与分母 $l$，遇到更大的值时对之前的结果重新缩放）、最后再除以分母。

```python
def flash_attention(q, k, v, block=16):
    """q: [Tq, d]，k, v: [Tk, d]（单头、无掩码）。每次只处理一块 K/V，显存占用与序列长度无关。"""
    scale = q.shape[-1] ** -0.5
    m = torch.full((q.shape[0], 1), float("-inf"))                         # 每行目前见过的最大分数
    l = torch.zeros(q.shape[0], 1)                                          # 每行目前的 softmax 分母（相对于 m）
    acc = torch.zeros_like(q)                                               # 未归一化的输出
    for start in range(0, k.shape[0], block):
        s = q @ k[start:start + block].T * scale                            # [Tq, block]
        m_new = torch.maximum(m, s.max(-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        correction = torch.exp(m - m_new)                                   # 旧的累加结果按新的最大值重新缩放
        l = l * correction + p.sum(-1, keepdim=True)
        acc = acc * correction + p @ v[start:start + block]
        m = m_new
    return acc / l

q, k, v = torch.randn(32, 64), torch.randn(100, 64), torch.randn(100, 64)
ref = torch.softmax(q @ k.T / 8, -1) @ v
print("分块计算与标准注意力的最大误差：", (flash_attention(q, k, v) - ref).abs().max().item())
assert torch.allclose(flash_attention(q, k, v), ref, atol=1e-5)
```

```text
分块计算与标准注意力的最大误差： 2.384185791015625e-07
```

**追问**：加上因果掩码怎么办（完全在未来的块直接跳过，对角线上的块加掩码）？反向传播时为什么要保存 $m$ 和 $l$（或 log-sum-exp）？decode 时 query 只有一行，怎样并行（Flash-Decoding：沿 K/V 切分给多个线程块，最后用 LSE 合并，见[上下文并行](../distributed/pp-cp.md#ring-attention)）。

## 5. top-p 采样

**题目**：实现 temperature + top-p 采样，要求对一个 batch 一次完成。

```python
def sample_top_p(logits, temperature, top_p, generator=None):
    """logits: [B, V]；temperature, top_p: [B]。"""
    probs = (logits / temperature[:, None]).softmax(-1)
    sorted_probs, idx = probs.sort(-1, descending=True)
    remove = (sorted_probs.cumsum(-1) - sorted_probs) >= top_p[:, None]    # 前面的累计概率已经够 top_p
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(-1, keepdim=True), 1, generator=generator)
    return idx.gather(-1, choice).squeeze(-1)

logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, -3.0]])
g = torch.Generator().manual_seed(0)
draws = torch.stack([sample_top_p(logits, torch.tensor([1.0]), torch.tensor([0.8]), g) for _ in range(20000)])
freq = torch.bincount(draws.flatten(), minlength=5) / 20000
p = logits.softmax(-1)[0]
kept = p[:2] / p[:2].sum()                                                  # 前两个 token 的累计概率刚好超过 0.8
print("采样频率：", [round(f, 3) for f in freq.tolist()], " 理论值：", [round(x, 3) for x in kept.tolist()] + [0, 0, 0])
assert (freq[:2] - kept).abs().max() < 0.02 and freq[2:].sum() == 0
```

```text
采样频率： [0.733, 0.267, 0.0, 0.0, 0.0]  理论值： [0.731, 0.269, 0, 0, 0]
```

**要点**：判断条件用"排在它前面的累计概率 ≥ p"，保证至少保留一个 token；批量实现不要写 Python 循环；追问时可以提到 vLLM 用指数竞赛代替 `multinomial` 以避免 CPU-GPU 同步（[采样器一章](../engine/sampler-api.md#批量采样)）。

## 6. O(1) 的 LRU 缓存

**题目**：实现 `get` 和 `put` 都是 O(1) 的 LRU 缓存（前缀缓存、专家缓存、KV 卸载都会用到）。

**考察点**：哈希表 + 双向链表；更新时把节点移到链表头部；淘汰时删除尾部。面试时通常要求手写链表，而不是直接用 `OrderedDict`。

```python
class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key, self.value, self.prev, self.next = key, value, None, None


class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.map = capacity, {}
        self.head, self.tail = Node(), Node()                               # 哨兵：head 之后最新，tail 之前最旧
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node):
        node.prev, node.next = self.head, self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            return None
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key, value):
        if key in self.map:
            node = self.map[key]
            node.value = value
            self._unlink(node)
        else:
            if len(self.map) == self.capacity:
                oldest = self.tail.prev
                self._unlink(oldest)
                del self.map[oldest.key]
            node = self.map[key] = Node(key, value)
        self._push_front(node)

cache = LRUCache(2)
cache.put("a", 1); cache.put("b", 2); cache.get("a"); cache.put("c", 3)   # b 最久未用，被淘汰
print("a:", cache.get("a"), " b:", cache.get("b"), " c:", cache.get("c"))
assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)
```

```text
a: 1  b: None  c: 3
```

## 7. ring all-reduce

**题目**：n 个进程各有一个长度为 N 的向量，用环形算法求和，使每个进程最终都得到总和。说明通信量。

**考察点**：先 reduce-scatter（n−1 步，每步把一块发给下一个进程并累加），再 all-gather（n−1 步，把完整的块传一圈）；每个进程总共发送 $2\frac{n-1}{n}N$ 个元素，与进程数几乎无关，这就是 ring 算法带宽最优的原因。

```python
def ring_all_reduce(vectors):
    """模拟 n 个进程：vectors[r] 是进程 r 的数据。返回每个进程最终的数据，以及每个进程发送的元素数。"""
    n = len(vectors)
    chunks = [[c.clone() for c in v.chunk(n)] for v in vectors]             # chunks[r][c]：进程 r 的第 c 块（拷贝，不改输入）
    sent = 0
    for step in range(n - 1):                                               # reduce-scatter
        outgoing = [(r, (r - step) % n, chunks[r][(r - step) % n].clone()) for r in range(n)]
        for r, c, data in outgoing:
            chunks[(r + 1) % n][c] += data                                  # 发给下一个进程，由它累加
            sent += data.numel()
    # 此时进程 r 的第 (r + 1) % n 块是完整的和
    for step in range(n - 1):                                               # all-gather
        outgoing = [(r, (r + 1 - step) % n, chunks[r][(r + 1 - step) % n].clone()) for r in range(n)]
        for r, c, data in outgoing:
            chunks[(r + 1) % n][c] = data
            sent += data.numel()
    return [torch.cat(c) for c in chunks], sent // n

vectors = [torch.randn(12) for _ in range(4)]
total = torch.stack(vectors).sum(0)
results, per_rank = ring_all_reduce(vectors)
print("每个进程都得到了总和：", all(torch.allclose(r, total, atol=1e-6) for r in results),
      f"；每个进程发送 {per_rank} 个元素 = 2 × (n-1)/n × N = {2 * 3 / 4 * 12:.0f}")
assert all(torch.allclose(r, total, atol=1e-6) for r in results)
```

```text
每个进程都得到了总和： True ；每个进程发送 18 个元素 = 2 × (n-1)/n × N = 18
```

一个很容易踩的坑：`v.chunk(n)` 返回的是原张量的**视图**，如果不先拷贝，`+=` 会悄悄改掉调用者的输入。本书的第一版答案就犯了这个错误，而且因为参考答案 `total` 是在调用之后才计算的，比较结果莫名其妙地不一致。PyTorch 代码里的原地操作，面试时值得主动说明。

## 8. 估算题的代码化

**题目**：写一个函数，输入模型配置、batch、上下文长度和 GPU 参数，输出 decode 一步的时间下限，以及显存能容纳的最大并发。

这是把大模型手册的[估算方法](llm://inference/estimation/)变成代码，面试中也经常以"口算"的形式出现：

```python
def decode_estimate(params, n_layers, n_kv_heads, head_dim, batch, context, bw=3.35e12, mem=80e9,
                    weight_bytes=2, kv_bytes=2, reserve=0.1):
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * kv_bytes
    step_ms = (params * weight_bytes + batch * context * kv_per_token) / bw * 1e3
    free = mem * (1 - reserve) - params * weight_bytes
    return step_ms, int(free // (context * kv_per_token))

step, max_batch = decode_estimate(8.03e9, 32, 8, 128, batch=32, context=4096)
print(f"LLaMA-3-8B，batch 32、上下文 4K：decode 一步下限 {step:.1f} ms，单卡最多约 {max_batch} 个这样的请求")
```

```text
LLaMA-3-8B，batch 32、上下文 4K：decode 一步下限 9.9 ms，单卡最多约 104 个这样的请求
```

## 更多题目

| 题目 | 参考 |
| --- | --- |
| 分页注意力（给定块表） | [分页 KV Cache](../engine/paged-kv.md) |
| 连续批处理调度器 | [调度器](../engine/scheduler.md) |
| 基数树的插入、匹配与淘汰 | [前缀缓存](../engine/prefix-cache.md) |
| 张量并行的线性层 | [张量并行](../distributed/tensor-parallel.md) |
| MoE 路由与按专家分组 | 大模型手册的 [MoE](llm://transformer/moe/) |
| 投机解码的拒绝采样 | 大模型手册的[推理服务](llm://inference/serving/#投机解码) |
| BPE 分词的训练与编码 | 大模型手册的[分词](llm://basics/tokenization/) |
| CUDA：归约、softmax、GEMM、转置 | CUDA 手册的[经典算子](cuda://kernels/reduction/)与[面试题库](cuda://career/interview/) |

## 小结

- [x] 模型组件类题目的关键是形状、掩码偏移、GQA 的头对应和数值稳定性，写完用官方实现核对。
- [x] 在线 softmax 是 FlashAttention、Flash-Decoding、ring attention 的共同基础，务必能默写。
- [x] 系统类题目常考 LRU、前缀树、调度、all-reduce，要能说出复杂度与通信量。
- [x] 估算题要把公式变成代码或口算：权重字节 + KV 字节除以带宽，显存减去权重除以每请求 KV。
