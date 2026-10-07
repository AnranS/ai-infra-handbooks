# Deterministic inference: results independent of the batch

<p class="lead">Set the temperature to 0, send the same request twice, and the results may still differ. Many people blame "GPU randomness", but the main cause is that <b>the batch changes</b>: when the same request is batched with different requests, kernels choose different ways to split the work, the order of floating-point additions changes with it, and the results differ in the last few bits; when two candidate tokens have close probabilities, that small difference sends generation down different paths. This chapter explains where this comes from, how to achieve "independence from the batch" (batch invariance), what it costs, and the switches in vLLM and SGLang. It matters especially for reinforcement learning: the probabilities computed on the inference side must match the training side for training to be truly on-policy.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. At temperature 0, what is the main reason the same request gives different results twice?
    2. Why do split-K in matrix multiplication and split-KV in attention make results depend on the batch?
    3. How do you make a kernel batch-invariant? Where is the cost?
    4. Besides matrix multiplication and attention, what else affects determinism?
    5. What do vLLM's `VLLM_BATCH_INVARIANT` and SGLang's `--enable-deterministic-inference` each do?

??? success "Answers (try first, then expand to compare)"
    1. The batch changes: kernels choose different splits (split-K, split-KV) depending on how many requests are in the batch, the order of floating-point additions changes accordingly, the same request's logits differ slightly, and when tokens are nearly tied, the argmax flips.
    2. With few requests, to use every SM, kernels split the K dimension of a matrix multiply or the KV of attention into several pieces computed in parallel and merged at the end; how many pieces depends on the batch size, so the merge order differs, and the result varies with the batch.
    3. Make each request's reduction order depend only on itself: no split-K, or a fixed way of splitting; split attention's KV by a fixed length (rather than a fixed number of pieces); have one thread block reduce each row of normalization and softmax in a fixed order. The cost is less parallelism, and longer latency, for small batches and long-context decode.
    4. The split points of chunked prefill and prefix caching (KV from different sources must be bitwise identical), the algorithm and channel count of tensor-parallel all-reduce, MoE grouping and sorting, configurations of fused kernels and autotuning, sampling randomness (which must be determined by the request's own seed and position), and CUDA Graphs choosing kernels by padded size.
    5. vLLM's `VLLM_BATCH_INVARIANT=1`: batch-invariant matrix multiplication (a Triton persistent matmul on SM80, cuBLAS split-K turned off on Hopper / Blackwell), batch-invariant attention backends and operators such as softmax, and NCCL fixed to the tree algorithm with a single channel. SGLang's `--enable-deterministic-inference`: attention backends split by fixed lengths (FlashInfer at 4096 for prefill and 2048 for decode, FA3 without splitting), all-reduce fusion turned off, and seeded sampling; `--rl-on-policy-target` turns it on automatically and aligns with the training side using `log_softmax`.

<!-- comic ../assets/comics/deterministic.webp is in Chinese; put it back once the English version exists -->

## The root: the order of floating-point additions {#根源浮点加法的顺序}

Floating-point addition is not associative: add in a different order and the result may differ. To use every SM, GPU kernels choose how to split based on input size: with few requests they split a matrix multiply's reduction dimension K into several pieces computed in parallel (split-K), split a long context's KV into several pieces for parallel attention (split-KV, also called flash-decoding), and merge at the end. **How many pieces depends on how many requests are in the batch**, so the same request's values depend on who else is in its batch:

```python
import torch

a, b, c = torch.tensor(1e8), torch.tensor(-1e8), torch.tensor(1.0)      # float32
print("浮点加法不满足结合律：(a + b) + c =", ((a + b) + c).item(), "；a + (b + c) =", (a + (b + c)).item())

torch.manual_seed(0)
K, N = 4096, 256
W = torch.randn(K, N)


def gemm(x, invariant):
    """模拟 GEMM kernel 的 split-K：请求少（M 小）时把 K 维切成几段并行算、最后再加起来，以便占满所有 SM；
    请求多时不切。切几段决定了加法的顺序——同一行的结果因此取决于 batch 里还有多少别的请求"""
    M = x.shape[0]
    splits = 1 if invariant or M >= 64 else 4
    bounds = torch.linspace(0, K, splits + 1).long().tolist()
    out = torch.empty(M, N)
    for i in range(M):                              # compute each row separately, ruling out differences from BLAS's own tiling
        parts = [x[i, s:e] @ W[s:e] for s, e in zip(bounds, bounds[1:])]
        out[i] = torch.stack(parts).sum(0)
    return out


mine = torch.randn(1, K)
for invariant in (False, True):
    alone = gemm(mine, invariant)[0]
    same = []
    for M in (8, 63, 64, 200):
        batch = torch.cat([mine, torch.randn(M - 1, K)])
        same.append(f"batch {M}：{torch.equal(gemm(batch, invariant)[0], alone)}")
    print("固定切分（batch 无关）" if invariant else "按 batch 大小切分  ", "——和单独跑逐位相同？", "，".join(same))
```

```text title="输出"
浮点加法不满足结合律：(a + b) + c = 1.0 ；a + (b + c) = 0.0
按 batch 大小切分   ——和单独跑逐位相同？ batch 8：True，batch 63：True，batch 64：False，batch 200：False
固定切分（batch 无关） ——和单独跑逐位相同？ batch 8：True，batch 63：True，batch 64：True，batch 200：True
```

When the batch goes from 63 to 64, the heuristic switches to another split and the result changes: online, the same request's output varies "randomly" with load. Attention is the same:

```python
import torch

torch.manual_seed(0)
D, S = 64, 8192                                     # head dim, context length
q, K, V = torch.randn(D), torch.randn(S, D), torch.randn(S, D)


def decode_attention(q, K, V, splits):
    """flash-decoding：把 KV 切成若干段并行算，每段得到 (最大值, 分母, 加权和)，最后按 LSE 合并"""
    parts = []
    for Kc, Vc in zip(K.tensor_split(splits), V.tensor_split(splits)):
        s = (Kc @ q) / D**0.5
        m = s.max()
        p = torch.exp(s - m)
        parts.append((m, p.sum(), p @ Vc))
    M = torch.stack([m for m, _, _ in parts]).max()
    num = sum(torch.exp(m - M) * o for m, _, o in parts)
    den = sum(torch.exp(m - M) * l for m, l, _ in parts)
    return num / den


def splits_by_load(batch, sms=132):
    """常见的启发式：请求越少，每个请求切得越多，好让所有 SM 都有活干"""
    return max(1, min(64, sms // batch))


def splits_fixed(seq_len, tile=2048):
    """与 batch 无关：按固定的长度切（SGLang 确定性模式下 FlashInfer decode 用 2048）"""
    return -(-seq_len // tile)


ref = decode_attention(q, K, V, splits_by_load(1))
print("按负载切分：", "，".join(f"batch {b} 切 {splits_by_load(b)} 段、与 batch 1 逐位相同 {torch.equal(decode_attention(q, K, V, splits_by_load(b)), ref)}"
                          for b in (1, 16, 64, 256)))
fixed = [decode_attention(q, K, V, splits_fixed(S)) for _ in (1, 16, 64, 256)]
print(f"固定按 2048 切：每个 batch 都切 {splits_fixed(S)} 段，结果全部相同 {all(torch.equal(f, fixed[0]) for f in fixed)}；"
      f"和不切分的结果只差浮点误差 {torch.allclose(fixed[0], torch.softmax(K @ q / D**0.5, 0) @ V, atol=1e-5)}")
```

```text title="输出"
按负载切分： batch 1 切 64 段、与 batch 1 逐位相同 True，batch 16 切 8 段、与 batch 1 逐位相同 False，batch 64 切 2 段、与 batch 1 逐位相同 False，batch 256 切 1 段、与 batch 1 逐位相同 False
固定按 2048 切：每个 batch 都切 4 段，结果全部相同 True；和不切分的结果只差浮点误差 True
```

Real libraries behave the same way: the LLM handbook's [the journey of a token](llm://synthesis/token-journey/#哪些优化会改变输出) chapter measured on CPU that computing the same request alone versus inside a batch makes the logits differ by $3 \times 10^{-5}$, bitwise different (CPU matrix-multiply libraries also choose their tiling by matrix size).

![Figure: why the same row's result changes with the batch: the split changes, and so does the order of additions](../assets/figures/float-order.svg){.aig-svg}

## Batch invariance: fixed splits that don't change with load {#与-batch-无关固定切分不随负载变}

The fix is simple to state: **each request's reduction order depends only on itself, regardless of who else is in the batch**. Operator by operator:

- **Matrix multiplication**: no split-K, or a fixed split; the accumulation order along K for each output element is independent of M (the number of tokens in the batch). On Ampere (SM80), vLLM replaces PyTorch's `mm`, `addmm`, `matmul` and `linear` with a persistent matmul written in Triton; on Hopper and Blackwell, the only batch-dependent part of cuBLAS is split-K, so it sets cuBLAS's workspace to the minimum, leaving no room for split-K;
- **Attention**: split-KV splits by a fixed length (independent of the number of requests). SGLang's deterministic mode has FlashInfer split prefill every 4096 tokens and decode every 2048, and FlashAttention 3 simply does not split (`num_splits=1`);
- **Normalization, softmax, means**: one thread block reduces each row in a fixed order, never splitting a row across several blocks just because there are few rows. vLLM registers batch-invariant implementations for `softmax`, `log_softmax` and `mean`;
- **Chunked prefill and prefix caching**: a request's KV must be bitwise identical whether computed in one prefill, in several chunks, or taken from the prefix cache, so attention kernels must also be invariant to "query chunking"; for MLA models SGLang restricts itself to a few deterministic attention backends that support prefix caching;
- **Communication**: all-reduce's reduction order must be fixed too. vLLM's batch-invariant mode fixes NCCL to the tree algorithm, a single channel and the Simple protocol, and turns off NVLS and symmetric-memory all-reduce; SGLang turns off the fusion of all-reduce with RMSNorm;
- **Sampling**: the same seed and the same distribution must sample the same token. SGLang forces PyTorch's sampling implementation, with random numbers computed from the request's seed and the token's position (`multinomial_with_seed`), independent of other requests in the batch.

What this achieves is **determinism within one deployment**: with the same model, the same parallel configuration, the same hardware and software versions, a request's result is independent of batch composition, arrival time, and whether it hits the prefix cache. Bitwise agreement across hardware or across TP sizes is a different, harder problem, generally not pursued.

## The cost {#代价}

- **Less parallelism**: split-K and split-KV exist precisely for "few requests, not enough parallelism". With fixed splits, small-batch matrix multiplies and long-context decode attention cannot fill the SMs, and latency grows; with large batches the effect is small;
- **Slower communication**: NCCL with a single channel and a fixed algorithm cannot saturate the bandwidth, and tensor-parallel all-reduce gets noticeably slower;
- **Limited optimizations**: some fused kernels, persistent kernels and autotuned configurations get turned off because "the result depends on the batch".

So it is usually turned on only where needed: RL rollouts, evaluations and debugging that must be reproducible, and regression tests.

## Why reinforcement learning needs it {#为什么强化学习需要它}

As the [inference in RL training](rl-rollout.md#问题二训练与推理的概率不一致) chapter showed, the inference and training sides compute noticeably different probabilities for the same sequence in BF16; algorithms like GRPO sample on the inference side and compute gradients with the training side's probabilities, so the mismatch makes training off-policy, requiring importance sampling to correct (TIS / MIS). Deterministic inference is another path: make the inference side's computation **batch-invariant and use the same operators as the training side**, and the two sides' probabilities become bitwise identical, making training truly on-policy. SGLang's `--rl-on-policy-target` exists for this: turning it on automatically enables deterministic inference and computes logprobs with `log_softmax`, consistent with the training side's algorithm.

## The switches in the frameworks {#框架里的开关}

| | vLLM | SGLang |
| --- | --- | --- |
| Switch | environment variable `VLLM_BATCH_INVARIANT=1` | `--enable-deterministic-inference`; `--rl-on-policy-target` for RL |
| Matrix multiplication | on SM80, a Triton persistent matmul replaces `mm` / `addmm` / `matmul` / `linear`; on SM90 and SM100, cuBLAS split-K is turned off; TF32 and reduced-precision reductions disabled | each backend picks a deterministic configuration (for example the MoE kernel configuration, with the fused finalize turned off) |
| Attention | attention backends take batch-invariant implementations | FlashInfer splits by fixed lengths (prefill 4096, decode 2048), FA3 does not split; MLA models only allow deterministic backends that support prefix caching |
| Communication | NCCL fixed to the tree algorithm, a single channel and the Simple protocol, with NVLS and symmetric-memory all-reduce turned off | all-reduce fusion turned off |
| Where it lives | `vllm/model_executor/determinism/batch_invariant.py` | `srt/batch_invariant_ops/`, plus `enable_deterministic_inference` branches throughout |

!!! interview "In an interview"
    First correct a common misconception: unstable results at temperature 0 come mainly not from "the randomness of GPU parallelism" but from **batch invariance**: kernels choose splits such as split-K and split-KV by the number of requests, the order of floating-point additions changes with them, and the same request's values depend on who is in its batch. The fix is to make each request's reduction order depend only on itself: no split-K or a fixed split in matrix multiplication, KV split by a fixed length in attention, one block per row for normalization and softmax, consistent results between chunked prefill and prefix caching, NCCL with a fixed algorithm and channel count, and fixed randomness in sampling. The cost is less parallelism for small batches and long-context decode, and slower communication. The use cases are on-policy RL rollouts (SGLang's `--rl-on-policy-target`), reproducible evaluations and regression tests.

## Exercises {#练习}

**1. With batch-invariant mode on, which requests slow down the most?**

??? success "Answer"
    Small-batch, long-context decode. In decode each request has a single query, so the matrix multiply's M is tiny and relies on split-K to fill the SMs; long-context attention relies on split-KV to spread one request's KV over many SMs scanning in parallel. With fixed splits, parallelism drops in both places. With large batches and short contexts there was little splitting to begin with, so the impact is small.

**2. If only matrix multiplication and attention are swapped for batch-invariant implementations, what else can make the same request's results differ?**

??? success "Answer"
    The reduction strategy of normalization and softmax (splitting rows across several blocks when there are few rows); the split points of chunked prefill and prefix caching (KV from different sources); the algorithm and channel count of tensor-parallel all-reduce (NCCL chooses by message size and topology); MoE expert grouping and token ordering, and fused-kernel configurations autotuned per batch; sampling randomness (which must come from the request's own seed and position, not drawn in turn from one global random stream); and CUDA Graphs choosing kernels by padded size. The deterministic modes of vLLM and SGLang handle exactly these, item by item.

**3. Why does deterministic inference make RL training more stable? Can it fully replace importance sampling correction?**

??? success "Answer"
    The gradients of algorithms like GRPO assume samples come from the current policy; when the inference and training sides' probabilities disagree, the samples actually come from a slightly different distribution, the gradients are biased, and at worst training becomes unstable. Deterministic inference plus "the same operators on the training and inference sides" can make the two sides' probabilities bitwise identical, removing this part of the bias. But it cannot replace every correction: in asynchronous RL samples come from older weights (policy lag), and partial rollouts span weight updates; these are other kinds of off-policy, still needing importance sampling or a cap on lag.

## Summary {#小结}

- [x] Results remain unstable at temperature 0 mainly because the batch changes: kernels choose splits by the number of requests, and the order of floating-point additions changes with them.
- [x] Batch invariance: no split-K or a fixed split in matrix multiplication, KV split by a fixed length in attention, normalization reduced in a fixed order, plus consistency across chunked prefill, prefix caching, communication and sampling.
- [x] The cost is less parallelism for small-batch and long-context decode and slower communication, so it is turned on only for RL rollouts, reproducible evaluations and regression tests.
- [x] vLLM uses `VLLM_BATCH_INVARIANT=1`, SGLang `--enable-deterministic-inference`, and for RL `--rl-on-policy-target` turns it on automatically and aligns with the training side using `log_softmax`.
