# The new generation of open models: compressed attention, hyper-connections and new MoE

<p class="lead">vLLM 0.30 and SGLang 0.5.20 already support the new generation of open models: DeepSeek-V4 (Flash and Pro), Kimi-K3, GLM-5, MiniMax-M3, Tencent's HY4, Gemma 4, and Qwen3.5, which this book uses as its example. Compared with the previous generation, they change three things at once: attention (compression + sparsity + sliding windows, or hybrids with linear attention), residual connections (multi-stream "hyper-connections"), and MoE (routing by token id, computing experts in a lower-dimensional space). Each brings new work for inference engines: several kinds of cache within one model, extra per-request state to save and roll back, and a batch of new kernels. Starting from each model's config.json and both frameworks' implementations, this chapter explains these changes and their effect on inference.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What parts make up DeepSeek-V4's attention? For a request with 1M of context, how many KV entries does one decode step look at per layer?
    2. Why is V4's KV per token only about a tenth of V3.2's? What fixed costs does each request take on in exchange?
    3. What does mHC turn the residual into? Why is the mixing matrix between streams constrained to be doubly stochastic? What does it cost at inference time?
    4. How does a "hash-routed" MoE layer differ from an ordinary MoE layer? What must an inference engine do for it?
    5. What does Kimi-K3's latent MoE save?

??? success "Answers (try first, then expand to compare)"
    1. A 128-token sliding window + compressed KV: C4 layers compress every 4 tokens into one entry, and an indexer picks the top-k (512 entries) to attend to; C128 layers compress every 128 tokens into one entry and attend to all of them. At 1M of context, a C4 layer looks at 128 + 512 = 640 entries per layer, and a C128 layer at 128 + 8192 = 8320.
    2. Most of the history is compressed into entries (1/4 in C4 layers, 1/128 in C128 layers), so KV per token is only about 3.8 KB, versus about 47 KB for V3.2 (MLA plus indexer keys). The cost is a fixed overhead per request: every layer keeps the raw KV of a sliding window, plus the state of the compressor's not-yet-full entry.
    3. The residual goes from one stream to 4 parallel streams, with a mixing matrix exchanging information between streams from layer to layer. Mixing matrices multiplied over dozens of layers amplify or shrink the signal; constraining them to be doubly stochastic (spectral norm 1) preserves the identity path. At inference time, activations and PP communication become 4×, and the related small operators must be fused.
    4. The first few layers decide routing by looking up the token id in a table (hash routing), ignoring the hidden state: routing can be computed in advance and load is predictable, but it is skewed by high-frequency tokens. The engine must support such token-id routing tables, and exploit "knowing in advance" to prepare dispatch early.
    5. Experts are computed in a latent space of half the dimension (project down first, run the experts' FFN in the low dimension), so the vectors dispatch / combine transfer are low-dimensional too, halving communication; that makes room for more, finer experts.

## The new generation in one table {#一张表看新一代}

| Model | Attention | Residual | MoE and number formats |
| --- | --- | --- | --- |
| DeepSeek-V4-Flash / Pro (43 / 61 layers) | every layer has a 128-token sliding window; C4 layers compress every 4 tokens into one KV entry, then an indexer picks 512 / 1024 entries; C128 layers compress every 128 tokens into one entry and attend to all; Flash also has 2 sliding-window-only layers | mHC: 4 residual streams | 6 of 256 / 384 experts; the first 3 layers route by looking up the token id; expert weights in FP4, the rest in FP8 (scales are powers of 2, UE8M0); up to 1M |
| Kimi-K3 (93 layers) | 1 MLA layer in every 4, the other 69 layers linear attention (KDA) | ordinary residual | 16 of 896 experts + 2 shared experts; routed experts computed in a 3584-dimensional "latent space" (latent MoE), weights in MXFP4; up to 1M |
| GLM-5 | DSA on an MoE model (sparse attention with an indexer picking top-k) | ordinary residual | uses the DeepSeek-V2 implementation in vLLM; long-context decode defaults to all-to-all context parallelism |
| MiniMax-M3 | MSA: block-sparse attention with an indexer, with kernels targeting Blackwell | ordinary residual | an MXFP8 version is provided |
| HY4 (Tencent) | MLA + indexer-based sparse attention + attention sinks | iHC: several independent residual channels | MoE |
| Gemma 4 | sliding window and full attention 5:1; full-attention layers have a single 512-dimensional KV head, with K and V shared | ordinary residual | the 12B is dense (the config has an MoE switch) |
| Qwen3.5 (smallest 0.8B, this book's example) | Gated DeltaNet and gated full attention 3:1; full-attention layers have 2 KV heads with head dim 256, and apply RoPE to only the first 1/4 of the dims | ordinary residual | the 0.8B is dense; natively multimodal (one model with a vision encoder); ships with 1 MTP layer for speculative decoding out of the box |

A few common directions: **sparse attention has become standard** (after DeepSeek-V3.2's DSA, GLM-5, MiniMax-M3 and HY4 all pick tokens with an indexer); **KV keeps shrinking** (compression, sliding windows, linear attention, shared K=V); **residual connections are widening** (V4, HY4 and GLM-5's successor all use multiple residual streams); **MoE is getting finer and leaner on communication**; and **4 bits has become a release format**. Below we look at attention, residuals and MoE in turn, then sum up what inference engines need.

## Attention: sliding window + compression + sparsity {#注意力滑窗--压缩--稀疏}

DeepSeek-V4's attention is still of the MLA family (Q goes through a low-rank projection; all heads share one 512-dimensional KV vector, 64 dims of which carry RoPE), but the history each layer can see is split three ways:

- **Sliding window**: the raw KV of the last 128 tokens, in every layer;
- **Compressed KV**: a **compressor** compresses every $r$ tokens into one KV entry: each token computes a KV vector and a set of gate scores, a softmax-weighted sum by score is taken within the block, then RMSNorm and RoPE. In $r = 4$ layers (C4), each entry actually summarizes 8 tokens (this block's 4 plus the previous block's 4, so blocks overlap); in $r = 128$ layers (C128) there is no overlap;
- **Sparse selection**: C4 layers still have many compressed entries (260 thousand at 1M of context), so, like V3.2's DSA, a lightweight **indexer** (64 heads, head dim 128, with its own compressor) scores all compressed entries and attention covers only the top 512 (1024 for Pro); C128 layers have few entries (8192 at 1M of context) and attend to all of them.

`compress_ratios` in config.json gives the compression ratio per layer: Flash has 2 sliding-window-only layers, 21 C4 and 20 C128; Pro has 30 C4 and 31 C128, with C4 and C128 alternating. The example below uses a small compressor to show two things: compressing token by token gives the same result as compressing all at once, and tokens that have not filled a block yet stay in a "partial state" (the example omits C4's overlap):

```python
import torch

torch.manual_seed(0)
D, R = 8, 4                          # KV dim per token, compression ratio (V4: 512, 4 or 128)


class Compressor:
    """每 R 个 token 压成一条 KV：门控分数在块内做 softmax，对 KV 加权求和（权重随机，只演示结构）。
    还没凑满 R 个的 token 留在"部分状态"里——它属于每个请求，和 KV 一样要保存、传输、回滚。"""

    def __init__(self):
        self.w_kv, self.w_gate = torch.randn(D, D) / D**0.5, torch.randn(D, D) / D**0.5
        self.ape = torch.randn(R, D) / 4                      # bias for positions within a block (called ape in V4)
        self.partial_kv, self.partial_score = [], []          # partial state: tokens of this block that have arrived
        self.entries = []                                     # compressed KV entries

    def push(self, h):
        """decode 一步来一个 token：凑满 R 个就产出一条压缩 KV"""
        i = len(self.partial_kv)
        self.partial_kv.append(h @ self.w_kv)
        self.partial_score.append(h @ self.w_gate + self.ape[i])
        if len(self.partial_kv) == R:
            kv, score = torch.stack(self.partial_kv), torch.stack(self.partial_score)
            self.entries.append((score.softmax(0) * kv).sum(0))     # each dimension does its own softmax weighting over R tokens
            self.partial_kv, self.partial_score = [], []


def compress_all(h, c):
    """prefill：一次压完整段"""
    n = len(h) // R * R
    kv = (h[:n] @ c.w_kv).view(-1, R, D)
    score = (h[:n] @ c.w_gate).view(-1, R, D) + c.ape
    return (score.softmax(1) * kv).sum(1)


hidden = torch.randn(22, D)
c = Compressor()
for t in range(len(hidden)):
    c.push(hidden[t])
print(f"逐 token 压缩 {len(c.entries)} 条，与一次压完一致：{torch.allclose(torch.stack(c.entries), compress_all(hidden, c), atol=1e-6)}；"
      f"部分状态里还有 {len(c.partial_kv)} 个 token")


def attended(pos, ratio, topk=None, window=128):
    """位置 pos 的 query 要看多少条 KV：滑窗里的原始 token + 已经压好的条目（C4 层再用索引器挑 top-k）"""
    swa = min(window, pos + 1)
    if ratio <= 1:
        return swa                                            # sliding-window-only layers
    done = (pos + 1) // ratio
    return swa + (min(done, topk) if topk else done)


for L in (4_096, 131_072, 1_048_576):
    print(f"上下文 {L:>9,}：全注意力看 {L:>9,} 条；V4-Flash 的 C4 层看 {attended(L - 1, 4, 512):>3} 条（索引器给 {L // 4:>7,} 条打分），"
          f"C128 层看 {attended(L - 1, 128):>5,} 条，只有滑窗的层看 {attended(L - 1, 1)} 条")
```

```text title="output"
逐 token 压缩 5 条，与一次压完一致：True；部分状态里还有 2 个 token
上下文     4,096：全注意力看     4,096 条；V4-Flash 的 C4 层看 640 条（索引器给   1,024 条打分），C128 层看   160 条，只有滑窗的层看 128 条
上下文   131,072：全注意力看   131,072 条；V4-Flash 的 C4 层看 640 条（索引器给  32,768 条打分），C128 层看 1,152 条，只有滑窗的层看 128 条
上下文 1,048,576：全注意力看 1,048,576 条；V4-Flash 的 C4 层看 640 条（索引器给 262,144 条打分），C128 层看 8,320 条，只有滑窗的层看 128 条
```

- **Decode attention barely grows with context**: C4 layers always look at 512 compressed entries + 128 neighbors, and C128 layers' entries are 1/128 of the context; the only part that grows linearly with context is the indexer's scoring (head dim 128, FP8, MXFP4 on Blackwell), which is as cheap as in [DSA](../moe/mtp-sparse.md#稀疏注意力nsa-与-dsa);
- **Each branch has its job**: the sliding window handles exact neighbors, C4 + the indexer "precisely find the few relevant stretches in a very long history", and C128 gives a coarse global overview; Flash's first two layers have only the sliding window;
- **The compressor has state**: the partial state is the same kind of thing as [linear attention's state](linear-attn.md): it belongs to each request and updates during decode. vLLM implements it as a kind of sliding-window cache (C4 keeps the intermediate values of the last 8 tokens, C128 of 128, in float32), in the same memory pool as the KV.

## A new KV ledger {#新的-kv-账本}

Change the number of layers and the share of full attention to see how a hybrid model's memory ledger shifts (the same widget as in the linear attention chapter):

<div class="aig-widget" data-widget="linear-memory"></div>

Plug the structural parameters from each model's config.json into the KV of one request (V3.2 and V4 in FP8 format, Kimi-K3's MLA in bf16, KDA states in float32):

```python
KB, MB, GB = 1024, 1024**2, 1024**3
# per model: (bytes added per token, fixed bytes per request, max context)


def v4(c4, c128, swa_only, idx_bytes=132):
    """DeepSeek-V4：每层都存最近 128 个 token 的原始 KV（FP8 格式 584 B/token）；C4 层每 4 个 token 一条压缩 KV + 一条
    索引器 key（FP8 时 132 B）；C128 层每 128 个 token 一条压缩 KV；压缩器的部分状态是 float32，C4 层留 8 个 token、C128 层留 128 个"""
    per_token = c4 * (584 + idx_bytes) / 4 + c128 * 584 / 128
    swa = (c4 + c128 + swa_only) * 128 * 584
    state = c4 * 8 * (2 * 2 * 512 + 2 * 2 * 128) * 4 + c128 * 128 * (2 * 512) * 4
    return per_token, swa + state


MODELS = {
    "DeepSeek-V3.2": (61 * (656 + 132), 0, 163_840),              # 61 MLA layers (FP8, 656 B/token) + indexer keys
    "DeepSeek-V4-Flash": (*v4(c4=21, c128=20, swa_only=2), 1_048_576),
    "DeepSeek-V4-Pro": (*v4(c4=30, c128=31, swa_only=0), 1_048_576),
    "Kimi-K3": (24 * 576 * 2, 69 * 96 * 128 * 128 * 4, 1_048_576),   # 24 MLA layers (bf16) + 69 KDA layers (96 heads × 128×128 float32 state)
    "Gemma 4 12B": (8 * 512 * 2, 40 * 1024 * 8 * 256 * 2 * 2, 262_144),  # 8 full-attention layers (K=V sharing one 512-dim head) + 40 layers with a 1024 sliding window
    # 6 full-attention layers (2 KV heads × 256, bf16) + 18 Gated DeltaNet layers (16 heads × 128×128 float32 state + 3 columns of conv cache)
    "Qwen3.5-0.8B": (6 * 2 * 2 * 256 * 2, 18 * 16 * 128 * 128 * 4 + 18 * 6144 * 3 * 2, 262_144),
}
for name, (grow, fixed, max_len) in MODELS.items():
    total = [f"{(grow * L + fixed) / GB:.2f} GB" if L <= max_len else "不支持" for L in (131_072, 1_048_576)]
    print(f"{name}：每 token {grow / KB:.1f} KB，每请求固定 {fixed / MB:.0f} MB；一个 128K 的请求 {total[0]}，1M 的请求 {total[1]}")
```

```text title="output"
DeepSeek-V3.2：每 token 46.9 KB，每请求固定 0 MB；一个 128K 的请求 5.87 GB，1M 的请求 不支持
DeepSeek-V4-Flash：每 token 3.8 KB，每请求固定 15 MB；一个 128K 的请求 0.48 GB，1M 的请求 3.77 GB
DeepSeek-V4-Pro：每 token 5.4 KB，每请求固定 22 MB；一个 128K 的请求 0.69 GB，1M 的请求 5.40 GB
Kimi-K3：每 token 27.0 KB，每请求固定 414 MB；一个 128K 的请求 3.78 GB，1M 的请求 27.40 GB
Gemma 4 12B：每 token 8.0 KB，每请求固定 320 MB；一个 128K 的请求 1.31 GB，1M 的请求 不支持
Qwen3.5-0.8B：每 token 12.0 KB，每请求固定 19 MB；一个 128K 的请求 1.52 GB，1M 的请求 不支持
```

- **Two roads to 1M**: V4 relies on compression (one entry per 4 or 128 tokens), with KV per token about a tenth of V3.2's, about 4–5 GB for a 1M request; Kimi-K3 relies on linear attention to turn 3/4 of its layers into fixed-size states, while the remaining 24 MLA layers still grow per token, 5–7× V4 at 1M;
- **Fixed costs have grown**: V4 has a dozen to twenty-odd MB of sliding-window and compressor state per request, Kimi-K3 over 400 MB of KDA state, and Gemma 4's sliding-window layers 320 MB. **With short requests and high concurrency**, this part, not KV, decides how many requests can be served at once (the [linear attention](linear-attn.md#新的显存账本) chapter computed the crossover);
- **With smaller KV, the bottleneck moves**: a 128K request on V4 has only 0.5–0.7 GB of KV, so a GPU fits an order of magnitude more concurrent requests than with V3.2, and decode's bottleneck shifts more onto weight reads and expert communication, one reason V4 made its expert weights FP4.

## Residuals: from one stream to four {#残差从一条流到四条流}

A standard Transformer's residual has a single stream: $x \leftarrow x + f(\text{norm}(x))$. **Hyper-connections** widen it into $n$ parallel streams, mixing once before and once after each sublayer (attention or FFN):

- Before the sublayer: compute three sets of coefficients from the current $n$ streams (a small linear layer + sigmoid / softmax), and combine the $n$ streams into the sublayer's input with the `pre` weights;
- After the sublayer: the new stream $j$ $= \sum_i \text{comb}_{ij} \cdot$ stream $i$ $+ \text{post}_j \cdot$ the sublayer output.

DeepSeek-V4 uses $n = 4$, and constrains the mixing matrix between streams, `comb`, to be **doubly stochastic** (every row and column sums to 1) with Sinkhorn iterations (20 of them); this is where the name mHC comes from (manifold-constrained: constrained to the set of doubly stochastic matrices). Below is a version following vLLM's reference implementation (`model_executor/kernels/mhc/torch.py`), and a look at what the constraint does:

```python
import torch

torch.manual_seed(0)
N, H = 4, 64                                   # 4 residual streams (hc_mult), hidden size


def sinkhorn(logits, iters=20, eps=1e-6):
    """先按行 softmax，再交替按列、按行归一化：得到每行、每列之和都是 1 的双随机矩阵"""
    m = logits.softmax(-1) + eps
    m = m / (m.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(-1, keepdim=True) + eps)
        m = m / (m.sum(-2, keepdim=True) + eps)
    return m


def mhc_pre(streams, fn, base, scale):
    """子层之前：由 4 条流算出三组系数，并把 4 条流加权合成子层的输入"""
    x = streams.flatten()                                  # [N*H]
    mix = (fn @ x) * torch.rsqrt(x.square().mean() + 1e-6)
    pre = torch.sigmoid(mix[:N] * scale[0] + base[:N])                 # weights for combining the input
    post = torch.sigmoid(mix[N:2 * N] * scale[1] + base[N:2 * N]) * 2  # weights for writing the sublayer output back to each stream
    comb = sinkhorn((mix[2 * N:] * scale[2] + base[2 * N:]).view(N, N))  # the mixing matrix between streams
    return (pre[:, None] * streams).sum(0), post, comb


def mhc_post(out, streams, post, comb):
    """子层之后：新的第 j 条流 = Σ_i comb[i, j] · 第 i 条流 + post[j] · 子层输出"""
    return comb.T @ streams + post[:, None] * out[None, :]


streams = torch.randn(H).repeat(N, 1)                      # copy the embedding into 4 streams
fn, base, scale = torch.randn(2 * N + N * N, N * H) * 0.02, torch.randn(2 * N + N * N), torch.ones(3)
inp, post, comb = mhc_pre(streams, fn, base, scale)
print("comb 每行之和：", [round(v, 3) for v in comb.sum(1).tolist()], "每列之和：", [round(v, 3) for v in comb.sum(0).tolist()])
streams = mhc_post(torch.tanh(inp), streams, post, comb)
print("子层输入", tuple(inp.shape), "；残差流", tuple(streams.shape))

# why constrain to doubly stochastic: look only at the residual path and multiply 60 layers of mixing matrices together
depth, prod_free, prod_ds = 60, torch.eye(N), torch.eye(N)
for _ in range(depth):
    free = torch.eye(N) + torch.randn(N, N) * 0.15           # the original hyper-connections: unconstrained mixing matrices, close to the identity
    prod_free = free @ prod_free
    prod_ds = sinkhorn(torch.eye(N) * 3 + torch.randn(N, N) * 0.5).T @ prod_ds
print(f"{depth} 层之后残差路径的放大倍数（矩阵的最大奇异值）：不约束 {torch.linalg.matrix_norm(prod_free, 2):.3g}，"
      f"双随机约束 {torch.linalg.matrix_norm(prod_ds, 2):.3g}")
```

```text title="output"
comb 每行之和： [1.0, 1.0, 1.0, 1.0] 每列之和： [1.0, 1.0, 1.0, 1.0]
子层输入 (64,) ；残差流 (4, 64)
60 层之后残差路径的放大倍数（矩阵的最大奇异值）：不约束 11.5，双随机约束 1
```

- **Why constrain**: an ordinary residual's identity path does not change the signal's magnitude, which is key to training deep networks; unconstrained mixing matrices are close to the identity at each layer, but deviations accumulate when multiplied over 60 layers (amplified more than tenfold above, and it could shrink too). The product of doubly stochastic matrices is still doubly stochastic, with a largest singular value of exactly 1, so the identity path's property is preserved;
- **The cost at inference time**: the residual goes from $[T, H]$ to $[T, 4, H]$, so 4× the residual is read and written around every sublayer; under pipeline parallelism, the activations passed between adjacent stages are 4× too (in vLLM's implementation, PP's intermediate tensors keep the shape `(num_tokens, hc_mult, hidden_size)`);
- **A pile of small operators to fuse**: around each sublayer there is a small $24 \times 4H$ GEMV, a sigmoid, 20 Sinkhorn iterations and a weighted sum, each a small memory-bound kernel if launched separately. vLLM wrote `mhc_pre`, `mhc_post`, and a kernel fusing "the previous sublayer's post + the next sublayer's pre" in TileLang; SGLang's implementation is in `srt/layers/communicator_mhc.py`;
- **What draft models take as input**: MTP, EAGLE3 and DSpark need hidden states from certain layers of the target model, and V4's implementation averages the 4 streams before handing them to the draft model.

Tencent's HY4 uses another variant, iHC (independent hyper-connections, `vllm/models/hy_v4/nvidia/hc.py`), with the same structure: each sublayer first combines several channels into one input, scatters the result back into each channel afterwards, and a head layer merges them at the end.

## MoE: routing by token id, computing experts in a latent space {#moe按-token-编号路由在潜空间里算专家}

**Hash routing.** V4's first 3 MoE layers do not use the router to pick experts: the weight file contains a table `tid2eid` (vocabulary size × 6), and token $t$ always goes to the 6 experts written in the table; the router's output is used only to compute those 6 experts' weights (V4's scoring function is $\sqrt{\text{softplus}(\text{logit})}$, normalized and multiplied by `routed_scaling_factor`). The other layers pick experts with the router as usual (the biased noaux scheme, as in V3).

This has two direct effects on inference: first, **routing is known before the forward pass**: given the token ids, which experts each token goes to in the first 3 layers, and how many tokens each expert gets, can be computed in advance, so expert-parallel dispatch can be planned early; second, **the load depends only on the distribution of tokens in the batch**:

```python
import torch

torch.manual_seed(0)
VOCAB, E, K = 129_280, 256, 6                       # DeepSeek-V4-Flash: vocabulary, routed experts, experts picked per token
tid2eid = torch.stack([torch.randperm(E)[:K] for _ in range(VOCAB)])   # hash-routing layer: token id → 6 experts (a random table here)

# tokens in a batch of text roughly follow a Zipf distribution: a few high-frequency tokens (punctuation, function words) make up a large share
rank = torch.arange(1, VOCAB + 1, dtype=torch.float64)
freq = rank ** -1.1 / (rank ** -1.1).sum()
for batch in (256, 4096, 65536):
    ids = torch.multinomial(freq, batch, replacement=True)
    hashed = torch.bincount(tid2eid[ids].flatten(), minlength=E).float()
    learned = torch.bincount(torch.stack([torch.randperm(E)[:K] for _ in range(batch)]).flatten(), minlength=E).float()
    print(f"batch {batch:>6} 个 token：最忙的专家是平均的 哈希路由 {hashed.max() / hashed.mean():.1f} 倍，"
          f"理想的均衡路由 {learned.max() / learned.mean():.1f} 倍")

# under expert parallelism, bytes each token sends per MoE layer (dispatch in FP8)
for name, hidden, topk in [("DeepSeek-V3（7168 维，选 8 个）", 7168, 8), ("DeepSeek-V4-Pro（7168 维，选 6 个）", 7168, 6),
                           ("Kimi-K3（先降到 3584 维，选 16 个）", 3584, 16), ("假如 Kimi-K3 不降维", 7168, 16)]:
    print(f"{name}：分发 {hidden * topk / 1024:.0f} KB/token")
```

```text title="output"
batch    256 个 token：最忙的专家是平均的 哈希路由 8.0 倍，理想的均衡路由 2.2 倍
batch   4096 个 token：最忙的专家是平均的 哈希路由 7.6 倍，理想的均衡路由 1.3 倍
batch  65536 个 token：最忙的专家是平均的 哈希路由 7.5 倍，理想的均衡路由 1.1 倍
DeepSeek-V3（7168 维，选 8 个）：分发 56 KB/token
DeepSeek-V4-Pro（7168 维，选 6 个）：分发 42 KB/token
Kimi-K3（先降到 3584 维，选 16 个）：分发 56 KB/token
假如 Kimi-K3 不降维：分发 112 KB/token
```

- **High-frequency tokens decide the load**: with a random table under a Zipf distribution, the busiest expert is 7–8× the average, and larger batches don't even it out (high-frequency tokens always go to the same experts). The real table comes from training, and whether it also achieves balance must be measured on real traffic; the good news for engines is that these layers' load can be **predicted in advance**, and EPLB can replicate hot experts directly from token frequencies;
- **The engine must carry token ids**: V4's implementation raises an error outright when there are no `input_ids`. Calling with embeddings only (as some multimodal pipelines do) cannot work for these layers; V4's vision version reserves a few ids for image tokens, and those positions are routed with a separate bias instead;
- **Latent MoE**: Kimi-K3 first projects the 7168-dimensional hidden state down to 3584 dims, with routed experts' inputs and outputs all in this "latent space", projecting back up afterwards. Expert-parallel dispatch and combine traffic halves, and so does each expert's weight, so it can use 896 experts with 16 picked per token at the same communication volume as V3 picking 8. The cost is two extra projections; vLLM's implementation is `models/kimi_k3/nvidia/latent_moe_runner.py`, which concatenates the latent-space result with the shared experts' result for **one** all-reduce;
- **MegaMoE**: both V4 and Kimi-K3 can use FlashInfer's MegaMoE in vLLM and SGLang: it builds expert-parallel all-to-all dispatch, expert compute and combine into one large kernel (communicating through symmetric memory), saving the multiple kernel launches and syncs in an MoE layer (SGLang's wrapper is in `srt/layers/moe/flashinfer_megamoe.py`).

**Number formats.** V4's expert weights are FP4 (`expert_dtype: fp4`), and the rest is FP8 with 128×128 blocks, with UE8M0 scales (powers of 2 only, turning multiplication into exponent addition; see [FP8 GEMM](../moe/fp8-gemm.md) and [low-bit inference](low-bit.md)); Kimi-K3's routed expert weights are MXFP4 (one E8M0 scale per 32 numbers), while attention, shared experts and dense MLPs keep high precision; V4's SwiGLU clamps activations (`swiglu_limit: 10`) so low-precision activations don't overflow.

## What inference engines need {#对推理引擎的要求}

| Change | What the engine must do | Where to see the implementation |
| --- | --- | --- |
| Several caches per layer: sliding-window KV, compressed KV, indexer keys, compressor state; hybrid models add linear-attention state | the KV manager must support several cache specs sharing one memory pool (vLLM has compressor state and KV blocks share the same physical tensor, so page sizes must align) | vLLM `v1/kv_cache_interface.py` (`SlidingWindowMLASpec` and others), `models/deepseek_v4/compressor.py`; SGLang `srt/layers/attention/dsv4/` |
| Every request has state (the compressor's partial state, KDA state) | prefix caching can hit only where state is reusable (aligned to compression blocks, or with saved checkpoints); rejected speculative drafts must roll back state; PD disaggregation must transfer state too | the same class of problems as in the [linear attention](linear-attn.md#前缀缓存要重做) chapter |
| Sparse attention's indices and metadata change every step | leave the sparse indexer and attention as an eager segment during CUDA Graph capture (V4's `_prepare_and_attn` uses `eager_break_during_capture`), capturing the rest as usual | `models/deepseek_v4/attention.py` |
| Several independent small computations before attention | run the Q projection, compressor and indexer in parallel on different CUDA streams | same file, `execute_in_parallel` |
| The residual becomes 4 streams | activations and PP communication × 4; mHC's small operators must be fused | vLLM `model_executor/kernels/mhc/` |
| New kernels | V4's sparse MLA (FlashMLA and FlashInfer each have a DSV4 backend), a compressor written in CuTe DSL, mHC in TileLang, MegaMoE, the MXFP4 indexer cache on Blackwell | `v1/attention/backends/mla/` |
| Every vendor's hardware | V4 has four implementations in vLLM: nvidia, amd, xpu and cpu; SGLang also has one for Ascend NPUs | vLLM `models/deepseek_v4/{nvidia,amd,xpu,cpu}/`; SGLang `srt/hardware_backend/npu/dsv4/` |

Speculative decoding changed along with them: V4 and Kimi-K3 both have DSpark draft implementations in vLLM (one forward pass gives a whole block of drafts), and vLLM also added adaptive verification that decides how many tokens to verify by load; see [new approaches to speculative decoding](spec-next.md).

!!! interview "In an interview"
    When asked "how does the new generation of models affect inference engines", cover four areas, each with numbers: **attention**: V4 uses a sliding window + compression (one entry per 4 / 128 tokens) + an indexer picking top-k, so decode looks at only hundreds to thousands of KV entries per layer, KV per token is a tenth of V3.2's, and a 1M-context request is 4–5 GB; Kimi-K3 takes the linear-attention hybrid road, with over 400 MB of fixed state per request. **Residuals**: mHC turns the residual into 4 streams, with mixing matrices constrained by Sinkhorn to be doubly stochastic to preserve the identity path; at inference, activations and PP communication × 4, and the small operators must be fused. **MoE**: the first 3 layers route by token-id lookup, so routing can be computed early and load is predictable but skewed by high-frequency tokens; latent MoE computes experts in a half-dimension space, halving communication. **Engines**: several caches sharing a memory pool, state entering prefix caching / speculative decoding / PD disaggregation, segmented CUDA Graph capture, multi-stream parallelism, and one implementation per kind of hardware.

## Exercises {#练习}

**1. Estimate one request's KV for DeepSeek-V4-Pro at 256K of context.** Use the ledger above.

??? success "Answer"
    About 5.4 KB per token (5511 B exactly), so 256K tokens are about 1.34 GB, plus about 22 MB fixed per request (4.6 MB of sliding window + about 18 MB of compressor state), about 1.37 GB in total. For comparison, V3.2 tops out at 160K and is already 5.9 GB at 128K.

**2. Why do C4 layers pick top-k with an indexer, while C128 layers attend to everything?**

??? success "Answer"
    Count the entries: at 1M of context a C4 layer has 260 thousand compressed KV entries, and attending to all of them costs about as much as dense attention over 260 thousand tokens, so it must be sparse; a C128 layer has only 8192, costing about as much as attention over an 8K context, which is acceptable and gives each query a coarse global view that misses no position. C4 + the indexer "precisely find the few relevant stretches", C128 "misses nothing", and the sliding window handles neighbors: the three complement each other.

**3. The prefix cache hits a 1000-token prefix and the new request starts prefill at token 1001. What must a C128 layer prepare?**

??? success "Answer"
    The 7 compressed KV entries of the first 896 tokens (7 full blocks) can be reused directly; tokens 897–1000, those 104 tokens, have not filled a block yet, and their intermediate values live in the compressor's partial state, so either that state was cached along with the prefix, or it is recomputed starting from token 897. The sliding window also needs the raw KV of the last 128 tokens. So engines usually align prefix-cache hits to compression block boundaries (multiples of 128 for C128) and recompute the leftover tail, the same kind of trade-off as linear attention's "hit only where state was saved".

**4. What do hash-routed layers mean for load balancing under expert parallelism?**

??? success "Answer"
    In these layers, each expert's load = the total occurrences in the batch of "the token ids routed to it", which depends only on the text itself and can be computed before the forward pass, even right after tokenization. The downside is that high-frequency tokens always land on the same few experts, and larger batches do not even out naturally; the upside is predictability: EPLB can replicate hot experts from token frequency statistics, and the scheduler can know these layers' dispatch volume in advance, overlapping the communication with earlier compute.

## Summary {#小结}

- [x] The new generation of models changes attention, residuals and MoE at once: sparse attention becomes standard, KV is compressed or replaced by state, residuals widen, MoE gets finer, and 4 bits becomes a release format.
- [x] DeepSeek-V4's attention = a 128-token sliding window + compressed KV (C4 picks top-k with an indexer, C128 attends to all); KV per token is about a tenth of V3.2's, but each request carries fixed costs such as sliding-window and compressor state.
- [x] mHC turns the residual into 4 streams, with mixing matrices constrained to be doubly stochastic to preserve the identity path; at inference, activations and PP communication × 4, and small operators must be fused.
- [x] Hash routing makes the first few layers' routing known in advance and their load predictable; latent MoE computes experts in a half-dimension space, halving communication and making room for more, finer experts.
- [x] Engines must support several caches sharing a memory pool, carry state through prefix caching, speculative decoding and PD disaggregation, capture CUDA Graphs in segments, and provide implementations for each kind of hardware.
