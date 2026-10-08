# Inference for linear attention and hybrid architectures

<p class="lead">Sparse attention lets each query look at fewer tokens, but the KV Cache still grows linearly with context. Another path is <b>linear attention</b>: a fixed-size state matrix summarizes the entire history, so each decode step's compute and memory are independent of context length. Pure linear attention isn't good enough on its own, so the new generation of models uses <b>hybrid architectures</b>: most layers use linear attention while a few keep full attention (for example, Qwen3-Next and Qwen3.5 mix Gated DeltaNet with gated attention 3:1, Kimi mixes KDA with MLA 3:1, MiniMax uses lightning attention, and Nemotron-H and Jamba mix Mamba with attention). This chapter looks at them from the inference engine's side: two ways to write the same computation (recurrent for decode, chunked for prefill), the new memory accounting that states bring, and why mechanisms "designed for KV", such as prefix caching and speculative decoding, all have to be redone. Each section is checked on the real hybrid model Qwen3.5-0.8B.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What shape is linear attention's state? How does one decode step's compute relate to context length?
    2. Why doesn't prefill recur token by token, using a chunked formulation instead?
    3. Does a hybrid model always save memory compared with a full-attention model?
    4. Why can't the prefix cache of linear-attention layers be shared by block? What do vLLM and SGLang do?
    5. What extra trouble does speculative decoding have on linear-attention models?

??? success "Answers (try first, then expand to compare)"
    1. One $d_k \times d_v$ matrix per head (128 × 128 in Qwen3.5's linear layers, stored in fp32), independent of context length; one decode step reads the state once and does one rank-one update, so compute and memory access are constant, independent of context length.
    2. Token-by-token recurrence is serial, with tiny vector operations at each step that cannot use Tensor Cores; the chunked formulation does matrix multiplies within a chunk (like a short stretch of attention with a decay mask) and passes only the state between chunks, turning most of the compute into matrix multiplies that can also run in parallel.
    3. Not necessarily: it saves a lot with long contexts, but each request has a fixed-size state (18.8 MiB per request in Qwen3.5-0.8B, while KV is only 12 KiB per token), so short requests (under about 540 tokens) actually take more memory than with full attention. The paged KV pool and the state pool must be planned together against the distribution of request lengths.
    4. The state summarizes the entire prefix and can neither be split by block nor concatenated; it can only be reused at positions where a state happened to be saved, and checkpoints are expensive (over ten MiB each). vLLM's align mode saves states only where "a stretch of prefill just finished and lands on a block boundary"; SGLang's hybrid radix tree hangs states on specific nodes (such as the end of a prompt).
    5. A failed verification must roll back: rejected draft tokens have already updated the state in place, so it cannot simply drop the tail as with KV; one state must be kept per draft position (or replayed from a checkpoint); PD disaggregation also needs to transfer the states and convolution caches in addition.

## Two ways to write the same computation {#同一个计算的两种写法}

Softmax attention must keep every past token's K and V. Without softmax, $o_t = \sum_{s \le t} (q_t \cdot k_s)\, v_s = q_t \sum_{s \le t} k_s^\top v_s$, and the sum can accumulate into a $d_k \times d_v$ **state** $S_t$. Modern linear attention always adds "forgetting":

- Gated linear attention (GLA, RetNet, the SSD form of Mamba2): $S_t = \alpha_t S_{t-1} + k_t^\top v_t$, with $\alpha_t \in (0, 1)$ determined by the input;
- DeltaNet uses an "error-correcting" update $S_t = (I - \beta_t k_t^\top k_t) S_{t-1} + \beta_t k_t^\top v_t$ (first erase the old memory in the direction of $k_t$, then write the new one), and Gated DeltaNet multiplies in a decay gate $\alpha_t$ as well.

At inference time there are two equivalent algorithms, verified below with gated linear attention:

```python
import torch

torch.manual_seed(0)
T, DK, DV, C = 64, 16, 16, 16                      # sequence length, key / value dims, chunk size
q, k, v = torch.randn(T, DK) / 4, torch.randn(T, DK) / 4, torch.randn(T, DV)
alpha = torch.sigmoid(torch.randn(T) + 3)          # per-step decay gate (close to 1): S_t = α_t·S_{t-1} + k_tᵀ v_t


def recurrent(q, k, v, alpha):
    """decode 的形式：逐 token 更新固定大小的状态 S（DK×DV），输出 o_t = q_t S_t"""
    S, out = torch.zeros(DK, DV), []
    for t in range(len(q)):
        S = alpha[t] * S + k[t, :, None] * v[t, None, :]
        out.append(q[t] @ S)
    return torch.stack(out), S


def chunked(q, k, v, alpha):
    """prefill 的形式：块内像注意力一样用矩阵乘（带衰减的因果掩码），块间只传递状态"""
    S, out = torch.zeros(DK, DV), []
    for s in range(0, len(q), C):
        qc, kc, vc = q[s:s + C], k[s:s + C], v[s:s + C]
        g = torch.log(alpha[s:s + C]).cumsum(0)                     # cumulative log decay within the chunk
        decay = torch.exp(g[:, None] - g[None, :]).tril()           # decay when position i looks at position j (j ≤ i)
        intra = ((qc @ kc.T) * decay) @ vc                          # intra-chunk: a C×C "attention"
        inter = torch.exp(g)[:, None] * (qc @ S)                    # inter-chunk: all earlier tokens are summarized in S
        out.append(intra + inter)
        S = torch.exp(g[-1]) * S + (torch.exp(g[-1] - g)[:, None] * kc).T @ vc
    return torch.cat(out), S


o1, s1 = recurrent(q, k, v, alpha)
o2, s2 = chunked(q, k, v, alpha)
print("递推与分块的输出一致：", torch.allclose(o1, o2, atol=1e-5), "；最终状态一致：", torch.allclose(s1, s2, atol=1e-5))
print(f"每个头的状态：{DK}×{DV} 个数，与序列长度无关；同样长度的 KV Cache 要 {T}×({DK}+{DV}) 个数")
```

```text title="output"
递推与分块的输出一致： True ；最终状态一致： True
每个头的状态：16×16 个数，与序列长度无关；同样长度的 KV Cache 要 64×(16+16) 个数
```

- **Decode uses recurrence**: each step reads the state once and does one rank-one update, so compute and memory access are constant, regardless of how long the context is;
- **Prefill uses chunks**: token-by-token recurrence is serial, with tiny vector operations at each step that cannot use Tensor Cores. The chunked formulation does $C \times C$ matrix multiplies within a chunk (like a short stretch of attention with a decay mask) and passes only the state between chunks, for $O(T \cdot C)$ compute overall, mostly matrix multiplies. The Triton kernels of the flash-linear-attention (fla) library and Mamba2's SSD kernel have this structure; DeltaNet's intra-chunk part is more complex (the error-correcting updates within a chunk must first be written in matrix form), but the idea is the same;
- **Chunked prefill comes for free**: when an inference engine cuts a long prompt into several segments, it just hands the state at the end of one segment to the next, which is the same thing as passing the state between chunks.

A linear-attention layer is usually preceded by a short **causal convolution** (a window of about 4), which also has to keep the inputs of the last few tokens for each request; together with the state, it forms that layer's "cache".

### Checking on a real model {#在真实模型上核对}

The Qwen3.5 series is hybrid from the smallest 0.8B up: of 24 layers, 3 in every 4 are Gated DeltaNet and 1 is gated full attention. Load its text part with transformers (without the fla library installed, transformers' prefill takes a chunked pure-PyTorch implementation and decode takes a token-by-token recurrent implementation, exactly the two formulations above), and compare the logits the three algorithms give at the last position:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(16)
path = "models/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()   # load only the text part
cfg = model.config
print("".join("F" if t == "full_attention" else "L" for t in cfg.layer_types), "（L：Gated DeltaNet，F：全注意力）")

ids = tok("线性注意力用一个固定大小的状态汇总全部历史，decode 每一步的计算量与上下文长度无关。" * 4,
          return_tensors="pt").input_ids[:, :64]
with torch.no_grad():
    whole = model(ids).logits[0, -1]                                          # one prefill: the chunked algorithm
    out = model(ids[:, :32], use_cache=True)
    for t in range(32, 64):                                                   # decode the last 32 tokens one by one: the recurrent algorithm
        out = model(ids[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
    stepwise = out.logits[0, -1]
    out = model(ids[:, :20], use_cache=True)                                  # prefill in three segments: only the state passes between them
    out = model(ids[:, 20:45], past_key_values=out.past_key_values, use_cache=True)
    chunks = model(ids[:, 45:], past_key_values=out.past_key_values, use_cache=True).logits[0, -1]
for name, logits in [("前 32 个 prefill + 后 32 个逐个 decode", stepwise), ("分三段 prefill", chunks)]:
    print(f"{name}：与一次 prefill 的 logits 最大差 < 1e-4：{(logits - whole).abs().max().item() < 1e-4}")
```

```text title="output"
LLLFLLLFLLLFLLLFLLLFLLLF （L：Gated DeltaNet，F：全注意力）
前 32 个 prefill + 后 32 个逐个 decode：与一次 prefill 的 logits 最大差 < 1e-4：True
分三段 prefill：与一次 prefill 的 logits 最大差 < 1e-4：True
```

The three algorithms agree within floating-point error: inference engines can safely use chunked kernels for prefill and recurrent kernels for decode, and can also prefill a long prompt in several segments.

## A new memory ledger {#新的显存账本}

In a hybrid model, full-attention layers still have KV that grows with context, while linear-attention layers have one fixed-size state per request. States are usually kept in fp32 (numerical error in the recurrence accumulates), so one 128×128 head is 64 KB. Take an illustrative hybrid model (48 layers, 1 in every 4 being full attention):

```python
MiB = 2**20
LAYERS, FULL = 48, 12                              # an illustrative hybrid model: 48 layers, 1 full-attention and 3 linear-attention layers in every 4
KV = 2 * 2 * 256 * 2                               # per token in a full-attention layer: K, V × 2 KV heads × head dim 256 × bf16 = 2 KiB
STATE = 32 * 128 * 128 * 4                         # per request in a linear-attention layer: 32 heads × a 128×128 state × fp32 = 2 MiB
LINEAR = LAYERS - FULL

print("上下文    全部用全注意力    混合（KV + 固定状态）   混合 / 全注意力")
for L in (512, 1024, 4096, 32768, 262144):
    dense, hybrid = LAYERS * KV * L, FULL * KV * L + LINEAR * STATE
    print(f"{L:>7}   {dense / MiB:>10.0f} MiB   {hybrid / MiB:>12.0f} MiB   {hybrid / dense:>12.0%}")
print(f"分界点：上下文短于 {STATE // KV} 个 token 时，混合模型每个请求反而占得更多")

print("前缀缓存：每 B 个 token 存一个状态检查点，相对这 B 个 token 的 KV 要多占多少")
for B in (256, 1024, 4096):
    print(f"  B = {B:>4}：一个检查点 {LINEAR * STATE / MiB:.0f} MiB，这 {B} 个 token 的 KV {FULL * KV * B / MiB:.0f} MiB，"
          f"比例 {LINEAR * STATE / (FULL * KV * B):.1f} 倍")
```

```text title="output"
上下文    全部用全注意力    混合（KV + 固定状态）   混合 / 全注意力
    512           48 MiB             84 MiB           175%
   1024           96 MiB             96 MiB           100%
   4096          384 MiB            168 MiB            44%
  32768         3072 MiB            840 MiB            27%
 262144        24576 MiB           6216 MiB            25%
分界点：上下文短于 1024 个 token 时，混合模型每个请求反而占得更多
前缀缓存：每 B 个 token 存一个状态检查点，相对这 B 个 token 的 KV 要多占多少
  B =  256：一个检查点 72 MiB，这 256 个 token 的 KV 6 MiB，比例 12.0 倍
  B = 1024：一个检查点 72 MiB，这 1024 个 token 的 KV 24 MiB，比例 3.0 倍
  B = 4096：一个检查点 72 MiB，这 4096 个 token 的 KV 96 MiB，比例 0.8 倍
```

- **It saves a lot with long contexts**: the longer the context, the closer it gets to "only 1/4 of the layers have KV", and memory drops to a quarter;
- **Short requests actually cost more**: every request needs 72 MiB of state right away, and contexts shorter than about 1000 tokens take more than with full attention. In high-concurrency scenarios with lots of short requests, the number of requests served at once is limited by the size of the state pool, not by KV;
- So inference engines prepare **two memory pools** for hybrid models: a paged KV pool (full-attention layers) and a per-request state pool (linear-attention layers), whose capacities must be planned together against the load's length distribution. vLLM manages them as different KV cache groups, with linear-attention layers handled by `MambaManager` (`v1/core/single_type_kv_cache_manager.py`); SGLang uses `HybridReqToTokenPool`, `MambaPool` and `HybridLinearKVPool` (`srt/mem_cache/memory_pool.py`).

### The real model's ledger {#真实模型的账本}

Prefill 100, 1000 and 3000 tokens on Qwen3.5-0.8B and tally what the cache actually holds (converted at deployment precision: KV and convolution caches in bf16, recurrent states in fp32, i.e. `mamba_ssm_dtype` in the config):

```python
def cache_bytes(cache):
    """按部署时的精度折算：KV 和卷积缓存 bf16，递推状态 fp32"""
    kv = state = conv = 0
    for layer in cache.layers:
        if hasattr(layer, "recurrent_states"):                                # Gated DeltaNet layer
            state += sum(t.numel() for t in layer.recurrent_states.values()) * 4
            conv += sum(t.numel() for t in layer.conv_states.values()) * 2
        else:                                                                 # full-attention layer
            kv += (layer.keys.numel() + layer.values.numel()) * 2
    return kv, state, conv

long_ids = tok("推理引擎要为混合模型准备两种内存池。" * 400, return_tensors="pt").input_ids
for n in (100, 1000, 3000):
    with torch.no_grad():
        cache = model(long_ids[:, :n], use_cache=True).past_key_values
    kv, state, conv = cache_bytes(cache)
    print(f"{n:>5} 个 token：全注意力层的 KV {kv / 1024:>6.0f} KiB；线性层的状态 {state / MiB:.0f} MiB，卷积缓存 {conv / 1024:.0f} KiB")
full_layer = cache.layers[cfg.layer_types.index("full_attention")]
print("全注意力层缓存的 K：", tuple(full_layer.keys.shape), "  线性层的状态：", tuple(cache.layers[0].recurrent_states[0].shape))
per_token = kv // 3000
dense = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * 2    # if all 24 layers were full attention
print(f"假如 24 层都是全注意力：每 token {dense // 1024} KiB；混合：每 token {per_token // 1024} KiB + 每请求 {(state + conv) / MiB:.1f} MiB，"
      f"上下文短于约 {(state + conv) // (dense - per_token)} 个 token 时混合反而更占显存")
```

```text title="output"
  100 个 token：全注意力层的 KV   1200 KiB；线性层的状态 18 MiB，卷积缓存 864 KiB
 1000 个 token：全注意力层的 KV  12000 KiB；线性层的状态 18 MiB，卷积缓存 864 KiB
 3000 个 token：全注意力层的 KV  36000 KiB；线性层的状态 18 MiB，卷积缓存 864 KiB
全注意力层缓存的 K： (1, 2, 3000, 256)   线性层的状态： (1, 16, 128, 128)
假如 24 层都是全注意力：每 token 48 KiB；混合：每 token 12 KiB + 每请求 18.8 MiB，上下文短于约 536 个 token 时混合反而更占显存
```

- There are only 6 full-attention layers, each with 2 KV heads (head dim 256), for 12 KiB per token, growing linearly with context;
- Each of the 18 linear layers has a state of 16 heads × 128×128 × fp32 = 1 MiB, 18 MiB in total, **independent of context length**. Its shape has no sequence-length dimension at all, which is why prefix caching runs into trouble in the next section;
- The convolution cache is tiny (transformers keeps 4 columns, the window size; an inference engine needs to keep only 3);
- The conclusion matches the illustrative model, only with an earlier crossover: this small model's KV per token is small to begin with, so the state's "fixed cost" is amortized only at about 540 tokens.

Change the number of layers, the share of full attention and the state size to see how the crossover moves:

<div class="aig-widget" data-widget="linear-memory"></div>

## Prefix caching has to be redone {#前缀缓存要重做}

KV prefix caching can share by block because block $i$'s KV depends only on the tokens of the first $i$ blocks, and each block's KV is stored separately. A linear-attention state, however, is **one summary of the entire prefix**: the state at position 1000 cannot be "cut" out of the state at position 1024, and two blocks' states cannot be concatenated. To reuse a prefix, a state must have been **saved at exactly that position** (a checkpoint).

Demonstrate on Qwen3.5-0.8B: two requests share the same system prompt. Save a full cache (KV + states) at the end of the common prefix as a checkpoint, and let each request continue from its own **copy** of it; copying is mandatory, since the state is updated in place by later tokens:

```python
import copy

system = {"role": "system", "content": "你是推理引擎方面的专家，回答要简短。" * 3}
chats = [tok.apply_chat_template([system, {"role": "user", "content": q}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
         for q in ["什么是 KV Cache？", "什么是分块 prefill？"]]
ids = [tok(c, return_tensors="pt").input_ids for c in chats]
n = next(i for i in range(ids[0].shape[1]) if ids[0][0, i] != ids[1][0, i])    # length of the two requests' common prefix
with torch.no_grad():
    checkpoint = model(ids[0][:, :n], use_cache=True).past_key_values           # save a state at the end of the common prefix
    for x in ids:
        reference = model.generate(x, max_new_tokens=16, do_sample=False)
        reused = model.generate(x, past_key_values=copy.deepcopy(checkpoint), max_new_tokens=16, do_sample=False)
        print(f"公共前缀 {n} 个 token + 自己的 {x.shape[1] - n} 个：从检查点的拷贝继续生成，与从头计算一致：{torch.equal(reference, reused)}")
kv, state, conv = cache_bytes(checkpoint)
print(f"这份检查点：KV {kv / 1024:.0f} KiB（随前缀变长而增长），状态 + 卷积缓存 {(state + conv) / MiB:.1f} MiB（与前缀长度无关）")
```

```text title="output"
公共前缀 39 个 token + 自己的 12 个：从检查点的拷贝继续生成，与从头计算一致：True
公共前缀 39 个 token + 自己的 14 个：从检查点的拷贝继续生成，与从头计算一致：True
这份检查点：KV 468 KiB（随前缀变长而增长），状态 + 卷积缓存 18.8 MiB（与前缀长度无关）
```

The prefix is 39 tokens, yet the checkpoint takes 19 MiB, of which KV is only 468 KiB. The KV part can still be shared by block as usual, while the state part can only be saved whole and copied whole.

The ledger above shows checkpoints are expensive: saving one every 256 tokens takes 12 times that stretch's KV in state memory. So in practice they are saved selectively:

- **vLLM**'s `--mamba-cache-mode`: `all` saves a state at every block boundary; `align`, the default when prefix caching is on, saves only where a scheduling step ends exactly on a block boundary, effectively leaving checkpoints only where "a stretch of prefill just finished";
- **SGLang**'s `mamba_radix_cache.py` puts full attention's KV and the linear layers' states into the same radix tree, hanging states only on specific nodes (such as the end of a request's prompt), and `mamba_checkpoint_pool.py` manages these checkpoints.

The cost is coarser hit granularity: a prefix can only be reused "where a state was saved", and anything after that must be prefilled again. Multi-turn conversations fit nicely: where the previous turn ended is the next turn's prefix, so saving a state there lets the next turn hit completely.

## Speculative decoding and PD disaggregation {#投机解码与-pd-分离}

- **Speculative decoding must be able to roll back**: rolling back KV only requires dropping the KV of rejected draft tokens; but the state has already been updated by the draft tokens and cannot be "subtracted back". Either save one state per draft position (memory × number of drafts), or record each step's update and replay from some position when needed (vLLM's configuration has a replay buffer for this). transformers' cache works the same way: calling `crop` (truncating to an earlier position) on linear layers raises an error outright, and you must first `activate_past_recording` so it keeps each step's state before it can roll back;
- **PD disaggregation must transfer states**: when prefill ends, besides the full-attention layers' KV, each linear layer's state and convolution cache must be sent to the decode instance. States have a fixed size independent of prompt length, so for long prompts the transfer is much smaller than for pure-attention models;
- **CUDA Graphs and batching**: during decode, each request's state has a fixed slot in the state pool, and batched kernels read and write by slot index, the same idea as the block table of paged KV.

!!! interview "How to explain it"
    To explain "how do inference engines support hybrid architectures like Qwen3-Next", cover three things: **compute** (recurrent for decode, chunked for prefill, matrix multiplies within a chunk and states passed between chunks, with chunked prefill coming for free); **memory** (a paged KV pool + a per-request state pool, states often in fp32, short requests actually taking more memory, capacity planned against the length distribution); **mechanisms redone** (prefix caching can hit only where states were saved, vLLM's align mode and SGLang's hybrid radix tree; rolling back states in speculative decoding; transferring states in PD disaggregation). Numbers like "Qwen3.5-0.8B has 18.8 MiB of state per request and 12 KiB of KV per token, so under 540 tokens it takes more memory than full attention" show you have really done the math.

## Exercises {#练习}

**1. How to choose the chunk size?** In the chunked formulation, how do compute and efficiency change as the chunk size $C$ grows or shrinks?

??? success "Answer"
    The intra-chunk part is a $C \times C$ matrix multiply, with total compute about $T \cdot C \cdot d$, growing linearly with $C$; the inter-chunk part is one $d_k \times d_v$ state update per chunk, totaling about $(T/C) \cdot d_k d_v \cdot C = T d_k d_v$, independent of $C$, but with $T/C$ serial steps. With $C$ too small, the matrix multiplies are too small to fill the Tensor Cores and there are many serial steps; with $C$ too large, the intra-chunk "attention" compute is wasted (what was linear becomes quadratic within the chunk). Real kernels usually take about 64, matching the Tensor Core tile size.

**2. What precision for the state?** Why are linear-attention states usually kept in fp32 while the KV Cache can use bf16 or even FP8?

??? success "Answer"
    Each KV value is written only once, so quantization error does not accumulate; the state, though, is "multiplied by the decay and added to a new outer product" at every step, so bf16 rounding error accumulates token by token and can drift noticeably over long sequences. Also, the state is a summary whose value range changes with context, so low precision more easily overflows or loses small updates. The cost is double the state pool's memory, one reason for "short requests cost more" in this chapter's ledger.

## Summary {#小结}

- [x] Linear attention summarizes the entire history in a fixed-size state: recurrent for decode (constant compute and memory), chunked for prefill (matrix multiplies within a chunk, states passed between chunks), and the two are equivalent.
- [x] A hybrid model's memory = paged KV for the full-attention layers + a fixed state per request for the linear layers (often fp32); long contexts save a lot, short requests actually cost more, and the two memory pools must be planned together. Qwen3.5-0.8B: 12 KiB per token, 18.8 MiB per request.
- [x] Prefix caching can hit only at positions with saved state checkpoints; checkpoints are expensive, and vLLM's align mode and SGLang's hybrid radix tree both save them only at specific positions.
- [x] Speculative decoding needs state rollback (multiple states or replay), and PD disaggregation must also transfer states and convolution caches.
