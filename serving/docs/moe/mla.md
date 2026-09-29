# MLA 推理：两条计算路径与 FlashMLA

<p class="lead">大模型手册的<a href="llm://transformer/attention-variants/#mla多头潜在注意力">注意力变体</a>一章讲了 MLA 的原理：KV Cache 只存一个低维潜向量，推理时可以把"解压"矩阵吸收进 query 和输出投影。这一章从推理引擎的角度看它：带解耦 RoPE 的完整 MLA 有"展开"和"吸收"两条等价的计算路径，它们的计算量差了几个数量级、适用的场景正好相反；decode 的注意力因此从访存瓶颈变成了接近计算瓶颈，这就是 FlashMLA 这类专用 kernel 存在的原因。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. MLA 的 KV Cache 每个 token 每层存几个数？为什么 RoPE 的部分要单独缓存？
    2. "展开 K、V"和"权重吸收"两条路径，哪条用于 prefill、哪条用于 decode？为什么？
    3. 带前缀缓存的 extend（在长前缀后追加一段新 token）该用哪条路径？
    4. 为什么说 MLA 的 decode 注意力接近计算瓶颈？和 GQA 差多少？
    5. FlashMLA 解决了什么问题？MLA 为什么不适合张量并行？

## 两条等价的路径

DeepSeek-V3 的 MLA 里，每个 token 缓存两样东西：512 维的潜向量 $c$，以及 64 维、所有头共享的 RoPE 键 $k^R$。每个头的 query 分成非位置部分 $q^N$（128 维）和 RoPE 部分 $q^R$（64 维），注意力分数是

![图：MLA 只缓存潜向量，decode 时把上投影吸收进 query](../assets/figures/mla.svg){.aig-svg}

$$
s = q^{N\top} k^N + q^{R\top} k^R = q^{N\top} W_{UK}^\top c + q^{R\top} k^R
$$

于是有两种算法：

- **路径一（展开）**：用 $W_{UK}$、$W_{UV}$ 把每个缓存 token 的 $c$ 展开成每个头的 $k^N$ 和 $v$，拼上 $k^R$，做一次普通的多头注意力（每头 q/k 为 192 维、v 为 128 维）；
- **路径二（吸收）**：先把 query 变到潜空间 $\tilde q = W_{UK} q^N$（512 维），分数是 $\tilde q^\top c + q^{R\top} k^R$；加权求和也在潜空间里做，最后才乘 $W_{UV}$（它还可以并进输出投影）。这相当于 128 个 query 头共享同一份 576 维的"键"和 512 维的"值"——一个头维很大的 MQA。

用一个小尺寸的完整 MLA（含解耦 RoPE）验证两条路径的输出相同：

```python
import math

import torch

torch.manual_seed(0)
T, D, H = 6, 64, 4                                # token 数、隐藏维度、头数
NOPE, ROPE, DV, DC = 8, 4, 8, 16                  # 每头 q/k 的非位置部分、RoPE 部分、v 的维度、潜向量维度
W_q = torch.randn(D, H * (NOPE + ROPE)) / 8
W_dkv = torch.randn(D, DC) / 8                    # hidden → 潜向量 c（KV Cache 存它）
W_kr = torch.randn(D, ROPE) / 8                   # hidden → 所有头共享的 k 的 RoPE 部分（也要缓存）
W_uk = torch.randn(H, DC, NOPE) / 4               # 潜向量 → 每个头 k 的非位置部分
W_uv = torch.randn(H, DC, DV) / 4                 # 潜向量 → 每个头的 v
W_o = torch.randn(H * DV, D) / 8
h = torch.randn(T, D)
causal = torch.ones(T, T, dtype=torch.bool).tril()


def rope(x):                                      # x: [..., T, ROPE]，按位置旋转相邻两维
    pos = torch.arange(x.shape[-2], dtype=torch.float32)[:, None]
    ang = pos / 10000 ** (torch.arange(0, ROPE, 2) / ROPE)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)


c = h @ W_dkv                                     # [T, DC]
k_r = rope(h @ W_kr)                              # [T, ROPE]
q = (h @ W_q).view(T, H, NOPE + ROPE).transpose(0, 1)                   # [H, T, NOPE+ROPE]
q_n, q_r = q[..., :NOPE], rope(q[..., NOPE:])
scale = 1 / math.sqrt(NOPE + ROPE)


def softmax(s):
    return s.masked_fill(~causal, float("-inf")).softmax(-1)


# 路径一（prefill）：从潜向量展开出每个头的 K、V，做普通的多头注意力
k = torch.cat([torch.einsum("tc,hcd->htd", c, W_uk), k_r.expand(H, T, ROPE)], -1)
v = torch.einsum("tc,hcd->htd", c, W_uv)
out1 = softmax(torch.cat([q_n, q_r], -1) @ k.transpose(1, 2) * scale) @ v

# 路径二（decode）：把 W_uk 吸收进 query，直接和潜向量 c 算分数；在潜空间里加权求和，最后再乘 W_uv
q_lat = torch.einsum("htd,hcd->htc", q_n, W_uk)                          # [H, T, DC]
p = softmax((q_lat @ c.T + q_r @ k_r.T) * scale)                          # 非位置部分 + RoPE 部分
out2 = torch.einsum("hts,sc,hcd->htd", p, c, W_uv)

y1 = out1.transpose(0, 1).reshape(T, H * DV) @ W_o
y2 = out2.transpose(0, 1).reshape(T, H * DV) @ W_o
print("两条路径的输出一致：", torch.allclose(y1, y2, atol=1e-5))
print(f"每个 token 每层缓存：MHA 需要 {H * (NOPE + ROPE) + H * DV} 个数，MLA 只要 {DC + ROPE} 个（潜向量 + 共享的 RoPE 键）")
```

```text title="输出"
两条路径的输出一致： True
每个 token 每层缓存：MHA 需要 80 个数，MLA 只要 20 个（潜向量 + 共享的 RoPE 键）
```

RoPE 的部分之所以要单独拿出来：旋转矩阵依赖位置，$q^\top R_m^\top R_n W_{UK} c$ 中的 $R_n$ 挡在 $W_{UK}$ 和 $c$ 之间，$W_{UK}$ 就无法提前吸收进 query。把位置信息放在单独的 64 维上、所有头共享一份 $k^R$，才让吸收成立。DeepSeek-V3 每层每个 token 缓存 $512 + 64 = 576$ 个数，61 层、bf16 下是 70 KB——同尺寸的 MHA 要几 MB。

## 谁快：看新 token 和缓存 token 的数量

两条路径数学上等价，计算量却很不一样。按 DeepSeek-V3 的形状算：

```python
H, NOPE, ROPE, DV, DC = 128, 128, 64, 128, 512    # DeepSeek-V3 的 MLA 形状

expand = 2 * DC * H * (NOPE + DV)                 # 路径一：每个"键 token"展开出 K、V 的 FLOPs
pair_naive = 2 * H * (NOPE + ROPE) + 2 * H * DV   # 路径一：每个 (query, key) 对的 QK 与 PV
absorb = 2 * H * NOPE * DC + 2 * H * DC * DV      # 路径二：每个"query token"吸收 W_uk、最后乘 W_uv
pair_absorb = 2 * H * (DC + ROPE) + 2 * H * DC    # 路径二：每个 (query, key) 对在潜空间里的 QK 与 PV
print(f"每对 (query, key)：路径一 {pair_naive / 1e3:.0f}K FLOPs，路径二 {pair_absorb / 1e3:.0f}K FLOPs（{pair_absorb / pair_naive:.1f} 倍）")
print(f"路径一每个键 token 要展开 {expand / 1e6:.1f}M FLOPs；路径二每个 query token 多 {absorb / 1e6:.1f}M FLOPs")


def flops(q, k):                                   # q 个新 token，接在 k 个已缓存的 token 后面（k 含新 token）
    pairs = q * k - q * (q - 1) / 2                # 因果：新 token 只看自己之前的
    return k * expand + pairs * pair_naive, q * absorb + pairs * pair_absorb


for name, q, k in [("decode，上下文 8K", 1, 8192), ("prefill 8K", 8192, 8192),
                   ("在 32K 缓存前缀后追加 64 个 token", 64, 32768 + 64), ("在 32K 缓存前缀后追加 512 个 token", 512, 32768 + 512)]:
    a, b = flops(q, k)
    print(f"{name}：路径一 {a / 1e9:,.1f} GFLOPs，路径二 {b / 1e9:,.1f} GFLOPs → 用{'路径一（展开）' if a < b else '路径二（吸收）'}")
print(f"追加 q 个新 token 时，q 超过约 {expand / (pair_absorb - pair_naive):.0f} 就该改用展开的路径（k 远大于 q 时）")

kv_bytes = (DC + ROPE) * 2                         # decode 时每个缓存 token 要读的字节（bf16）
print(f"decode 算术强度：MLA {pair_absorb / kv_bytes:.0f} FLOP/字节；对比 GQA（64 个 q 头、8 个 KV 头、头维 128）"
      f"{2 * 64 * 128 * 2 / (8 * 128 * 2 * 2):.0f} FLOP/字节；H800 的屋脊点约 {989e12 / 3.35e12:.0f}")
```

```text title="输出"
每对 (query, key)：路径一 82K FLOPs，路径二 279K FLOPs（3.4 倍）
路径一每个键 token 要展开 33.6M FLOPs；路径二每个 query token 多 33.6M FLOPs
decode，上下文 8K：路径一 275.5 GFLOPs，路径二 2.3 GFLOPs → 用路径二（吸收）
prefill 8K：路径一 3,024.0 GFLOPs，路径二 9,621.9 GFLOPs → 用路径一（展开）
在 32K 缓存前缀后追加 64 个 token：路径一 1,273.6 GFLOPs，路径二 586.8 GFLOPs → 用路径二（吸收）
在 32K 缓存前缀后追加 512 个 token：路径一 2,501.8 GFLOPs，路径二 4,726.7 GFLOPs → 用路径一（展开）
追加 q 个新 token 时，q 超过约 171 就该改用展开的路径（k 远大于 q 时）
decode 算术强度：MLA 242 FLOP/字节；对比 GQA（64 个 q 头、8 个 KV 头、头维 128）8 FLOP/字节；H800 的屋脊点约 295
```

两条路径的代价结构正好相反：

- **展开**的固定开销在"每个缓存 token"上（把 $c$ 展开成 128 个头的 K、V，3360 万 FLOPs），每对 (query, key) 的计算便宜；
- **吸收**的固定开销在"每个新 token"上，每对 (query, key) 贵 3.4 倍（在 576 / 512 维的潜空间里做点积）。

所以 decode（1 个新 token、上万个缓存 token）必须吸收——展开的话每一步都要把整个上下文重新展开一遍，贵上百倍；prefill（新 token 和缓存 token 一样多）用展开，按普通多头注意力算，计算量只有吸收的三分之一。带前缀缓存的 extend 介于两者之间，分界大约在"新 token 数 ≈ 170"（缓存远长于新 token 时）。推理引擎因此要实现两套注意力路径，并按批次的形状选择：

- **SGLang**：DeepSeek 模型的注意力有 `MHA`、`MLA`、`MHA_CHUNKED_KV` 等几种前向方法（`srt/models/deepseek_common/attention_forward_methods/`）。decode 用吸收的 `MLA`；没有前缀的 prefill 用 `MHA`；有前缀时，前缀总长度低于阈值（默认 8192，环境变量 `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`）用 `MLA`，更长时用 `MHA_CHUNKED_KV`——把前缀分块展开、逐块做注意力，再用 LSE 合并各块的结果（和[上下文并行](train://model/context/)里的合并方法相同），避免一次展开整个前缀占用过多显存；
- **vLLM**：`model_executor/layers/attention/mla_attention.py` 把一个批次拆成 decode 和 prefill 两部分，decode 走各后端的 `forward_mqa`（吸收，FlashMLA、FlashInfer、CUTLASS MLA 等），prefill 走 `v1/attention/backends/mla/prefill/` 下的多头注意力实现，有上下文时同样分块展开。

## decode 注意力变成了计算问题

GQA 的 decode 注意力，每读一个缓存 token 的 K、V（4 KB）只做 8 FLOP/字节的计算，是彻底的访存瓶颈，kernel 的目标是跑满显存带宽。MLA 吸收之后，128 个头共享同一份 1152 字节的潜向量，每字节 242 FLOPs，已经接近 H800 的屋脊点（约 295）；KV 用 FP8 存放时每字节的计算再翻一倍，就成了计算瓶颈。这改变了 kernel 的写法：

- 要用 Tensor Core：把 128 个头当成矩阵乘的 M 维（像把"多个 query"拼起来），潜向量当 K 维，才能把算力用起来；
- 分块要兼顾计算和访存：一个线程块同时处理全部 128 个头和一段 KV，共享内存里放下 576 维的 KV 块；
- 长上下文、小 batch 时要沿 KV 维拆分到多个 SM（split-KV，即 flash-decoding），最后合并各段的部分结果。

**FlashMLA** 是 DeepSeek 开源的、针对 Hopper 的 MLA decode kernel：分页 KV（块大小 64）、按上下文长度预先计算调度元数据（`get_mla_metadata`，决定每个 SM 处理哪几段 KV）、支持 FP8 KV；官方给出的数据是在 H800 上访存受限的配置接近 3000 GB/s、计算受限的配置达到数百 TFLOPS。后续版本还加入了面向稀疏注意力（DeepSeek-V3.2 的 DSA，见[MTP 与稀疏注意力](mtp-sparse.md)）的 kernel。FlashInfer、CUTLASS 也提供了 MLA 的 decode kernel，SGLang、vLLM 都可以选择。

## MLA 与并行方式

MLA 的潜向量是所有头共享的一份。张量并行按头切分注意力时，每张卡都要保存完整的潜向量 KV Cache——TP=8 就是 8 份一样的缓存，MLA 省下的显存又被吐了回去。所以 DeepSeek 类模型的主流部署是**注意力部分做数据并行**（DP Attention：每张卡处理不同的请求、只存自己请求的 KV），MoE 部分做专家并行（见[专家并行与 DP Attention](../distributed/expert-parallel.md)）。另一个选择是把注意力的 TP 限制在很小的范围（比如 TP=2～4）并接受 KV 的复制，DeepSeek-V3 论文中 prefill 阶段就用了 TP4 的注意力（配合序列并行）。

!!! interview "面试怎么答"
    问 MLA 的推理实现，按三层回答：**缓存**（576 个数 / token / 层，其中 64 维 RoPE 键单独存，因为 RoPE 挡住了吸收）；**两条路径**（decode 吸收、prefill 展开，给出"每对 3.4 倍、展开每 token 3360 万 FLOPs"这类数字，说明 extend 要按形状选择，SGLang 用前缀长度阈值和分块展开）；**kernel 与并行**（decode 算术强度约 240，接近计算瓶颈，所以 FlashMLA 要用 Tensor Core、split-KV；潜向量不能按头切，TP 会复制 KV，所以用 DP Attention）。

## 练习

**1. 为什么吸收后每对 (query, key) 更贵？** 用维度解释路径二每对 279K FLOPs、路径一 82K FLOPs 的来源。

??? success "参考答案"
    路径一的每个头：QK 点积在 $128 + 64 = 192$ 维上，PV 在 128 维上，$128 \times (192 + 128) \times 2 \approx 82$K。路径二的每个头：QK 在潜空间的 $512 + 64 = 576$ 维上，PV 在 512 维上，$128 \times (576 + 512) \times 2 \approx 279$K。吸收把每个头的"小维度点积"换成了"大维度点积"，换来的是不用展开缓存——每对更贵，但省掉了每个缓存 token 3360 万 FLOPs 的展开。

**2. FP8 的 KV Cache。** MLA 的 KV 改用 FP8 存放后，decode 注意力的算术强度是多少？对 kernel 设计意味着什么？

??? success "参考答案"
    每个缓存 token 读 576 字节（不计缩放因子），算术强度约 $278528 / 576 \approx 484$ FLOP/字节，超过 H800 BF16 的屋脊点（约 295），变成计算瓶颈。这时 kernel 的瓶颈从"读 KV"转到"Tensor Core 的吞吐"，要么用 FP8 的 Tensor Core 做 QK（精度要仔细处理），要么接受计算瓶颈、把优化重点放在提高 Tensor Core 利用率上。对系统而言，FP8 KV 让同样的显存放下两倍的上下文，而 decode 延迟不会同比例下降——瓶颈已经不在带宽上了。

**3. 为什么吸收路径对长上下文的 prefill 也可能有用？**

??? success "参考思路"
    展开路径要把整个上下文的 K、V 实体化：128 个头、每头 192 + 128 维，每个 token 约 80 KB（bf16），32K 的前缀就是 2.6 GB，还只是一层。显存吃紧时，要么分块展开（SGLang 的 `MHA_CHUNKED_KV`、vLLM 的分块上下文）并用 LSE 合并，要么在新 token 不多时直接用吸收路径。所以真实系统的选择不只看 FLOPs，还看显存和 kernel 的可用性。

## 小结

- [x] MLA 每 token 每层缓存 512 维潜向量 + 64 维共享 RoPE 键；RoPE 必须解耦出来，吸收才成立。
- [x] 两条等价路径：展开（每个缓存 token 固定开销大、每对便宜）用于 prefill；吸收（每个新 token 固定开销、每对贵 3.4 倍）用于 decode；extend 按新 token 数和前缀长度选择，长前缀分块展开并用 LSE 合并。
- [x] 吸收后 decode 注意力的算术强度约 240 FLOP/字节（GQA 约 8），接近计算瓶颈；FlashMLA 等 kernel 用 Tensor Core、split-KV 和分页 KV 来适配。
- [x] 潜向量不能按头切分，TP 会复制 KV Cache；主流部署是 DP Attention + EP。
