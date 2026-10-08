# The complete journey of a token

<p class="lead">This chapter ties the whole handbook into one thread: starting from a sentence the user types, follow it through tokenization, embedding, 28 Transformer layers, the output layer, sampling and detokenization, recording the tensor shapes at every stop on a real model; then pause at every stop and ask "what does the inference engine optimize here". Finally, a roofline table answers "where does decode time actually go".</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. For Qwen3-0.6B, when prefilling 17 tokens, what is the output shape of `k_proj`? And the shape of layer 0's KV cache?
    2. During prefill, does the output layer need to compute logits for all 17 positions?
    3. As the batch grows from 1 to 64, which kinds of operators in decode gain arithmetic intensity and which do not? Why does this decide the bottleneck at large batch sizes?
    4. Which inference optimizations leave the output mathematically unchanged, and which change it? Does "mathematically unchanged" mean bitwise identical?

??? success "Answers (try first, then expand to compare)"
    1. `k_proj` outputs `[1, 17, 1024]` (8 KV heads × 128 dimensions); split into heads, layer 0's K and V are each `[1, 8, 17, 128]`.
    2. No. Generation uses only the last position's logits, and computing only that one saves most of the LM head's work (unless the prompt's logprobs must be returned).
    3. The arithmetic intensity of weight operators (projections, FFN) grows with the batch, because the same weights are reused by more tokens; attention reads each request's own KV, so its intensity is only the GQA group size, independent of the batch. So at large batch sizes, reading KV becomes the main bottleneck.
    4. Mathematically unchanged: the KV cache, paging, FlashAttention, prefix caching, chunked prefill, continuous batching, speculative decoding with greedy verification and so on; changing it: quantization, KV quantization, sparse attention, approximate draft acceptance rules and so on. "Mathematically unchanged" is not bitwise identical: a different order of computation and batch composition give tiny floating-point differences.

## Tracing tensor shapes {#追踪张量形状}

Use PyTorch forward hooks to record the input and output shapes of each module in layer 0, first prefilling the whole prompt and then decoding one step:

```python
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

text = tok.apply_chat_template([{"role": "user", "content": "中国的首都是哪里？"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="pt").input_ids

layer0 = model.layers[0]
watch = {
    "embed_tokens": model.embed_tokens, "input_layernorm": layer0.input_layernorm,
    "q_proj": layer0.self_attn.q_proj, "k_proj": layer0.self_attn.k_proj, "v_proj": layer0.self_attn.v_proj,
    "self_attn": layer0.self_attn, "post_attention_layernorm": layer0.post_attention_layernorm,
    "gate_proj": layer0.mlp.gate_proj, "up_proj": layer0.mlp.up_proj, "down_proj": layer0.mlp.down_proj,
    "norm": model.norm, "lm_head": model.lm_head,
}
shapes = {}

def recorder(name):
    def hook(module, args, output):
        shapes.setdefault(name, []).append((tuple(args[0].shape), tuple(output.shape)))
    return hook

handles = [m.register_forward_hook(recorder(n)) for n, m in watch.items()]
cache = KVCache(model.cfg.num_hidden_layers)
with torch.no_grad():
    logits = model(ids, cache)                                        # prefill
    next_id = logits[:, -1].argmax(-1)
    kv_after_prefill = tuple(cache.k[0].shape)
    logits = model(next_id[:, None], cache)                           # one decode step
for h in handles:
    h.remove()

print(f"{'模块':26s}{'prefill 输入 → 输出':34s}decode 输入 → 输出")
for name in watch:
    (pi, po), (di, do) = shapes[name]
    print(f"{name:26s}{str(pi) + ' → ' + str(po):34s}{di} → {do}")
print("第 0 层的 K Cache", kv_after_prefill, "→", tuple(cache.k[0].shape))
print("生成：", repr(tok.decode(next_id)), repr(tok.decode(logits[:, -1].argmax(-1))))
```

Running it in this handbook's environment gives:

```text
模块                        prefill 输入 → 输出                   decode 输入 → 输出
embed_tokens              (1, 17) → (1, 17, 1024)           (1, 1) → (1, 1, 1024)
input_layernorm           (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
q_proj                    (1, 17, 1024) → (1, 17, 2048)     (1, 1, 1024) → (1, 1, 2048)
k_proj                    (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
v_proj                    (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
self_attn                 (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
post_attention_layernorm  (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
gate_proj                 (1, 17, 1024) → (1, 17, 3072)     (1, 1, 1024) → (1, 1, 3072)
up_proj                   (1, 17, 1024) → (1, 17, 3072)     (1, 1, 1024) → (1, 1, 3072)
down_proj                 (1, 17, 3072) → (1, 17, 1024)     (1, 1, 3072) → (1, 1, 1024)
norm                      (1, 17, 1024) → (1, 17, 1024)     (1, 1, 1024) → (1, 1, 1024)
lm_head                   (1, 17, 1024) → (1, 17, 151936)   (1, 1, 1024) → (1, 1, 151936)
第 0 层的 K Cache (1, 8, 17, 128) → (1, 8, 18, 128)
生成： '中国的' '首'
```

A few things worth noting:

- Apart from attention, **every module computes each token independently**: prefill and decode shapes differ only in T (17 vs. 1). This is why prefill is matrix-matrix multiplication (GEMM) and decode matrix-vector multiplication (GEMV).
- `k_proj` and `v_proj` output 1024 dimensions (8 KV heads × 128), `q_proj` 2048 (16 heads × 128): this is [GQA](../transformer/attention-variants.md#mqa-与-gqa), with two query heads sharing one set of K and V. The KV cache stores 2 × 1024 numbers per token per layer. Note that `q_proj`'s output is wider than the hidden dimension of 1024: Qwen3 configures its head dimension separately as 128.
- Attention is the only "cross-token" operation: it connects the current token with every past token in the KV cache. In decode there is 1 query and 18 keys.
- `lm_head` outputs 151936 dimensions, the widest layer in the whole model.

## Stop by stop {#逐站解读}

The table below puts each stop's computation next to the corresponding optimizations in inference engines. The left columns are "what the model does", the right column "how engineering makes it faster".

![Figure: the journey of a token, with the tensor shapes at every stop for prefill and decode](../assets/figures/token-journey.svg){.aig-svg}

| Stop | Shape (prefill) | What the computation is | Optimizations in inference engines |
| --- | --- | --- | --- |
| Chat template, tokenization | text → `[1, 17]` | string processing on the CPU | the template must match training ([chat templates](../basics/tokenization.md#特殊-token-与对话模板)); tokenization runs in a separate process so it does not block GPU scheduling |
| Embedding | `[1, 17]` → `[1, 17, 1024]` | a lookup by id | split by vocabulary under tensor parallelism (vocab parallel) |
| RMSNorm + residual | `[1, 17, 1024]` | elementwise + reduction, memory intensive | fused with the residual addition into one kernel (fused add + RMSNorm); see [softmax and normalization](cuda://kernels/softmax-norm/) in the CUDA book |
| Q, K, V projections | → `[1, 17, 2048]`, `[1, 17, 1024]` × 2 | GEMM / GEMV | the three matrices concatenated into one `qkv_proj`, one GEMM; [weight quantization](../inference/quantization.md); split by head under tensor parallelism |
| QK-Norm, RoPE | shapes unchanged | elementwise (per-head normalization, rotation) | positions come from the KV cache length; fused with the QKV or attention kernel |
| Write to the KV cache | `[1, 8, 17, 128]` × 2 | a copy | written into [paged](../inference/kv-cache.md#kv-cache-的显存管理) blocks (`reshape_and_cache`); can be stored as FP8 |
| Attention | scores `[1, 16, 17, 17]` | prefill: compute intensive; decode: reading KV, memory intensive | FlashAttention for prefill, without materializing the scores; paged decode kernels and split-KV (Flash-Decoding) for decode; GQA shares KV inside the kernel; see [FlashAttention and inference operators](cuda://advanced/attention/) in the CUDA book |
| O projection | → `[1, 17, 1024]` | GEMM | split by rows under tensor parallelism, followed by one all-reduce |
| SwiGLU MLP | → `[1, 17, 3072]` → `[1, 17, 1024]` | three GEMMs, the most parameters | `gate_proj` and `up_proj` merged; `silu(gate) * up` fused into one kernel; MoE uses grouped GEMMs ([grouping by expert](../transformer/moe.md#实现逐-token-与按专家分组)) |
| Final RMSNorm + output layer | → `[1, 17, 151936]` | the widest GEMM | **prefill computes only the last position** (see below); the vocabulary split by columns |
| Sampling | `[1, 151936]` → 1 id | softmax, sorting, random numbers | [temperature, top-p, penalties](../inference/decoding.md) done in batches on the GPU; constrained decoding masks illegal tokens here; speculative decoding verifies here |
| Detokenization | id → text | CPU | [incremental detokenization](../basics/tokenization.md#流式输出与增量反分词), handling incomplete UTF-8 bytes, streaming back |

On top of this, every decode step launches hundreds of kernels (28 layers × a dozen or so per layer), and for small models the CPU's launch overhead can even exceed the GPU's compute time, so decode usually uses **CUDA Graphs** to record the whole step's forward pass and replay it at once; see [streams, concurrency and CUDA Graphs](cuda://tools/streams/) in the CUDA book.

### Prefill only needs the last position's logits {#prefill-时只需要最后一个位置的-logits}

The `mini_llm` above computes logits for all 17 positions, for teaching purposes. But generation only uses the last position (only training needs all positions, for the loss). For this model the saving is considerable:

```python
T, d, V = ids.shape[1], model.cfg.hidden_size, model.cfg.vocab_size
n_params = sum(p.numel() for p in model.parameters())                  # the shared embedding counts once
body_flops = 2 * T * (n_params - V * d)                                # the 28 Transformer layers
print(f"prefill {T} 个 token：主体 {body_flops / 1e9:.1f} GFLOP，"
      f"全部位置的输出层 {2 * T * d * V / 1e9:.1f} GFLOP，只算最后一个位置 {2 * d * V / 1e9:.2f} GFLOP")
```

```text title="output"
prefill 17 个 token：主体 15.0 GFLOP，全部位置的输出层 5.3 GFLOP，只算最后一个位置 0.31 GFLOP
```

In a small model the vocabulary takes a large share, so this optimization saves about a quarter of the prefill computation, as well as the memory of the `[17, 151936]` logits (with a very long prompt this tensor can reach several GB). The `LogitsProcessor` in inference engines first picks out the hidden state at each request's last position and then applies the output layer.

## Where decode time goes {#decode-的时间花在哪里}

Break one LLaMA-3-8B decode step into a few kinds of operators, compute each one's FLOPs, bytes and [arithmetic intensity](../inference/estimation.md#延迟的下限), and compare with the H100's ridge point (about 295 FLOP/byte). Each operator's lower bound on time is the larger of "the time to compute it" and "the time to read it":

```python
from estimate import H100

llama3_8b = dict(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=8)

def decode_ops(c, batch, context, wbytes=2, kvbytes=2):
    """decode 一步中各类算子的计算量（FLOP）与访存量（字节），对整个模型求和。"""
    d, dff, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
    hd = d // nh
    ops = {}

    def gemm(name, k, n, times=L):                                     # [batch, k] @ [k, n]
        flop = 2 * batch * k * n * times
        byte = (k * n * wbytes + batch * (k + n) * 2) * times         # weights + input and output activations
        ops[name] = (flop, byte)

    gemm("qkv_proj", d, (nh + 2 * nkv) * hd)
    gemm("o_proj", nh * hd, d)
    gemm("gate_up_proj", d, 2 * dff)
    gemm("down_proj", dff, d)
    gemm("lm_head", d, V, times=1)
    ops["attention"] = (4 * batch * nh * hd * context * L,            # QK^T and PV
                        2 * batch * nkv * hd * context * kvbytes * L)   # read K and V
    return ops

totals = {}
for batch in (1, 64):
    print(f"batch = {batch}，上下文 4096")
    print(f"{'算子':14s}{'GFLOP':>9s}{'GB':>8s}{'强度':>8s}{'时间下限 ms':>12s}")
    totals[batch] = {}
    for name, (flop, byte) in decode_ops(llama3_8b, batch, 4096).items():
        t = max(flop / (H100.tflops * 1e12), byte / (H100.bw_tbs * 1e12)) * 1e3
        totals[batch][name] = t
        print(f"{name:14s}{flop / 1e9:9.1f}{byte / 1e9:8.2f}{flop / byte:8.1f}{t:12.2f}")
    print(f"{'合计':14s}{'':25s}{sum(totals[batch].values()):12.2f}\n")
```

```text title="output"
batch = 1，上下文 4096
算子                GFLOP      GB      强度     时间下限 ms
qkv_proj            1.6    1.61     1.0        0.48
o_proj              1.1    1.07     1.0        0.32
gate_up_proj        7.5    7.52     1.0        2.24
down_proj           3.8    3.76     1.0        1.12
lm_head             1.1    1.05     1.0        0.31
attention           2.1    0.54     4.0        0.16
合计                                             4.64

batch = 64，上下文 4096
算子                GFLOP      GB      强度     时间下限 ms
qkv_proj          103.1    1.65    62.4        0.49
o_proj             68.7    1.11    62.1        0.33
gate_up_proj      481.0    7.65    62.9        2.28
down_proj         240.5    3.83    62.7        1.14
lm_head            67.2    1.07    63.0        0.32
attention         137.4   34.36     4.0       10.26
合计                                            14.83
```

This is the most important table in the book and worth understanding line by line:

1. **At batch = 1, every operator's intensity is far below 295**, so everything is memory bound and nearly all the time goes to reading weights (the MLP being the largest part). This is what [decode being memory bound](../inference/kv-cache.md#prefill-与-decode) concretely means.
2. **The intensity of weight operators (GEMMs) is about the batch size**: the weights are read once and shared by every request in the batch. Going from batch 1 to 64, their computation grows 64-fold while their time barely changes. This is why [batching](../inference/kv-cache.md#批处理让多个请求分摊权重读取) works.
3. **Attention's intensity is always the GQA group size (32 / 8 = 4), independent of the batch**: each request reads its own KV cache, with nothing shared. As the batch grows, the KV read grows in proportion. At batch = 64, attention takes 70% of the time and becomes the new bottleneck.
4. So with large batches and long contexts, the main battlefield of optimization is the **KV cache**:
    - Less KV: [GQA/MLA](../transformer/attention-variants.md) (MLA has 128 heads share one latent vector, raising attention's intensity dozens of times), KV cache quantization, sliding windows;
    - Faster KV reads: paged decode kernels, Flash-Decoding's split-KV parallelism;
    - KV shared by many requests: [prefix caching](../inference/serving.md#kv-cache-管理与前缀缓存) (the KV of a shared prefix can be read once, as in cascade attention).

!!! inference "Inference view"
    This table directly answers many interview questions. For example, "why does MoE inference depend more on large batches?": each MoE expert is used by only part of the batch's tokens, so the intensity of the weight GEMMs is "the tokens each expert receives" rather than the batch, and a larger batch (or expert parallelism gathering more requests together) is needed to escape the memory bottleneck. Or "why does speculative decoding gain less at large batch sizes?": at large batches the weight operators' intensity is already near the ridge point, so the computation of verifying several tokens is no longer "free".

## Tying the whole handbook together {#把整本手册串起来}

All inference optimizations can be derived from two fundamental facts:

<!-- i18n:diagram e0f6a4b020 -->
```text
Fact 1: a language model is autoregressive and generates one token at a time
  │
  ├─→ generation must be serial → every token needs a full forward pass
  │     └─→ past K and V do not change in the forward pass → KV cache
  │           ├─→ inference splits into prefill (compute bound) and decode (memory bound)
  │           │     ├─→ the two interfere → chunked prefill, PD disaggregation
  │           │     └─→ each decode step computes 1 token → speculative decoding: "guess several, verify once"
  │           └─→ the KV cache grows linearly with context and concurrency, the main consumer of memory and bandwidth
  │                 ├─→ less KV: GQA, MLA, KV quantization, sliding windows
  │                 ├─→ managing KV: PagedAttention, prefix caching, preemption and swapping
  │                 └─→ reading KV: FlashAttention, Flash-Decoding
  │
Fact 2: every decode step reads all the weights, with an arithmetic intensity only as large as the batch
  │
  ├─→ grow the batch to share weight reads → continuous batching (requests differ in length, so schedule step by step)
  ├─→ fewer bytes of weights → weight quantization (INT4/FP8); decode speed roughly inversely proportional to bit width
  ├─→ weights too large for one GPU → tensor / expert parallelism; communication becomes a new cost
  └─→ kernel launch overhead matters at small batch sizes → operator fusion, CUDA Graphs
```

Every chapter is a node in this picture. When you read an inference engine's source or a paper, first ask "where is it in this picture, which bottleneck does it address, and what price does it pay", and most designs become clear.

## Which optimizations change the output {#哪些优化会改变输出}

| Category | Optimizations | Output |
| --- | --- | --- |
| Mathematically equivalent | KV cache, prefix caching, chunked prefill, continuous batching, PagedAttention, FlashAttention, operator fusion, CUDA Graphs, tensor parallelism | the same as the original computation (within floating-point error) |
| Distribution-preserving | speculative decoding (token-identical under greedy decoding, the same distribution under sampling) | as above |
| Lossy | weight / activation / KV cache quantization, KV eviction (dropping unimportant tokens), sparse attention, pruning, distillation | different from the original model; accuracy must be evaluated |

"Mathematically equivalent" is not "bitwise identical". Floating-point addition is not associative, and different kernels, different parallel splits, and even **different batch sizes** change the order of accumulation:

```python
with torch.no_grad():
    alone = model(ids)[0, -1]
    batched = model(ids.repeat(4, 1))[0, -1]                           # the same sequence, placed in a batch of 4
print(f"单独计算与在 batch 中计算的 logits 最大差异：{(alone - batched).abs().max().item():.1e}，"
      f"逐位相同：{torch.equal(alone, batched)}")
assert (alone - batched).abs().max() < 1e-3
```

```text title="output"
单独计算与在 batch 中计算的 logits 最大差异：3.1e-05，逐位相同：False
```

With BF16 inference on a GPU the differences are larger. When two candidate tokens have nearly the same probability, a tiny difference sends greedy decoding down a different path, and once it diverges, the rest of the text is completely different. So in a live service, "the same request at temperature 0 giving two different results" is normal: the size of the batch the request lands in keeps changing. When strict reproducibility is needed (for example, aligning inference with training in reinforcement learning), use dedicated batch-invariant kernels, at some cost in performance.

!!! inference "Inference view"
    This also decides how inference optimizations are **tested**: mathematically equivalent optimizations are verified by "logits error against a reference implementation within a threshold" (as this handbook does for `mini_llm`), not by demanding word-for-word identical text; lossy optimizations must be evaluated on downstream tasks (perplexity, MMLU, GSM8K and so on).

!!! interview "How to explain it"
    This chapter is the standard account of "the full path from a request to a token": tokenization and the chat template → embedding (gather) → per layer RMSNorm, QKV projections (GEMM in prefill, GEMV in decode), RoPE, attention (FlashAttention in prefill, reading KV in decode), SwiGLU → logits for the last position only → sampling on the GPU → incremental detokenization. The key numbers: in decode the weight operators' arithmetic intensity is about the batch size, while attention's equals the GQA group size, independent of the batch, so at large batches reading KV becomes the bottleneck. Finish by saying which optimizations are mathematically equivalent and which are lossy, and that "equivalent is not bitwise identical".

## Exercises {#练习}

**1. Deriving shapes.** For LLaMA-3-8B (hidden 4096, 32 heads, 8 KV heads, head_dim 128, ffn 14336, 32 layers, vocabulary 128256), a batch of 8 requests each with 1000 tokens of context now decodes one step. Write the shapes of the `q_proj` output, layer 0's K cache (after the update), the attention scores, the `gate_proj` output and the `lm_head` output.

??? success "Answer"
    - `q_proj` output: `[8, 1, 4096]`, reshaped to `[8, 32, 1, 128]`;
    - Layer 0's K cache: `[8, 8, 1001, 128]`;
    - Scores: `[8, 32, 1, 1001]`;
    - `gate_proj` output: `[8, 1, 14336]`;
    - `lm_head` output: `[8, 1, 128256]`.

    Real inference engines do not use a padded `[B, T, ...]` layout; instead they concatenate all requests' tokens into one dimension, `[num_tokens, hidden]` (num_tokens = 8 here), and describe where the KV lives with each request's sequence length and block table. This lets prefill and decode tokens mix in the same batch.

**2. Deciding with the roofline table.** In the table above, if the KV cache is quantized to FP8, roughly what does the lower bound of one decode step at batch = 64 become? What if the weights are quantized to INT4 instead? Which gains more?

??? success "Answer"
    ```python
    fp8_kv = sum(totals[64].values()) - totals[64]["attention"] / 2
    int4_w = sum(totals[64].values()) - sum(t for n, t in totals[64].items() if n != "attention") * 0.75
    print(f"原始 {sum(totals[64].values()):.1f} ms，KV FP8 约 {fp8_kv:.1f} ms，权重 INT4 约 {int4_w:.1f} ms")
    ```

    ```text title="output"
    原始 14.8 ms，KV FP8 约 9.7 ms，权重 INT4 约 11.4 ms
    ```

    FP8 KV halves attention's reads (10.3 → 5.1 ms); INT4 weights cut the weight operators' reads to 1/4 (4.6 → 1.1 ms), but they took only 4.6 ms to begin with. So **with large batches and long contexts, KV quantization gains more; with small batches, weight quantization gains more**. (This is a rough estimate that ignores dequantization overhead; also, at batch = 64 the weight GEMMs' intensity is already 63, and after INT4 it quadruples but is still below the ridge point, so the estimate holds.)

## Summary {#小结}

- [x] Apart from attention, every module computes each token independently; prefill is GEMM and decode is GEMV.
- [x] Every stop has corresponding inference optimizations: fusion, merged projections, paged KV, FlashAttention, logits for the last position only, sampling on the GPU, incremental detokenization, CUDA Graphs.
- [x] In decode, the weight operators' intensity is about the batch size, while attention's equals the GQA group size, independent of the batch; at large batches the KV cache is the bottleneck.
- [x] All inference optimizations trace back to two facts: "autoregressive and serial" and "decode is memory bound".
- [x] Even mathematically equivalent optimizations do not guarantee bitwise identical results, since the batch size changes floating-point results; test with error thresholds, and evaluate accuracy for lossy optimizations.
