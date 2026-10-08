# Fine-grained FP8 quantization and grouped GEMM: DeepGEMM

<p class="lead">DeepSeek-V3 is trained in FP8, and inference also uses FP8 weights and FP8 activations directly (W8A8): weights get one scale per 128×128 block, and activations one per 128 channels of each token. This chapter covers why such fine-grained quantization is necessary and how a GEMM multiplies in the scales during accumulation, then looks at the problem specific to MoE: hundreds of experts each get only tens to thousands of tokens, so how does grouped GEMM organize the data, and why does decode's expert GEMM need a very large global batch to escape the memory bottleneck?</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How many mantissa bits does FP8's E4M3 format have? When does per-tensor scaling break down?
    2. In a fine-grained quantized GEMM, at which step are the scales multiplied in? Why promote partial sums to fp32 every 128 elements?
    3. MoE's grouped GEMM has "contiguous" and "masked" layouts. Which DeepEP mode does each pair with?
    4. In decode, how many tokens must an expert get per step for its GEMM to stop being limited by weight reads?
    5. Why does DeepGEMM JIT-compile its kernels at runtime?

??? success "Answers (try first, then expand to compare)"
    1. 3 mantissa bits (E4M3: 4 exponent bits, 3 mantissa bits, max 448). With extreme outliers, the per-tensor scale is inflated by the outlier, other values get squeezed into subnormals or even 0, and the error blows up quickly.
    2. Along K, do one FP8 matrix multiply (Tensor Core) per 128 elements, and in fp32 registers on CUDA Cores multiply the partial sum by the activation and weight scales, then accumulate. The H800's FP8 Tensor Cores have limited internal accumulation precision (about 14 bits), so accumulating a very large K in one go loses precision; promoting to fp32 every 128 elements multiplies in the scales along the way.
    3. The contiguous layout (experts' tokens placed end to end, each padded to the block size) pairs with DeepEP's high-throughput mode, for prefill; the masked layout (one fixed-size buffer per expert + the actual token count) pairs with the low-latency mode's fixed slots, with fixed shapes that can be captured in a CUDA Graph, for decode.
    4. The expert GEMM's arithmetic intensity is about the tokens per expert × 2 (per byte of FP8 weights); the H800's FP8 ridge point is about 591 FLOP/byte, so each expert needs at least about 295 tokens per step. Decode relies on DP Attention + large-scale EP to pool the global batch so each expert gets hundreds of tokens.
    5. Different expert counts, token counts and shapes need different kernel configurations (block sizes, pipeline stages); JIT-compiling for the actual shapes at runtime, with the shapes as compile-time constants, yields more specialized code without precompiling every combination.

## Scaling granularity {#缩放粒度}

![Figure: FP8 scaling granularity: activations 1×128, weights 128×128, and the GEMM accumulates once per 128 along K](../assets/figures/fp8-scaling.svg){.aig-svg}

FP8's E4M3 has only 3 mantissa bits and a maximum of 448. Quantizing first divides by a scale to fit values into this range. The coarser the granularity, the more one large outlier inflates the whole group's scale, squeezing the other values to tiny numbers, even into subnormals or 0. The [mixed precision and FP8](train://practice/mixed-precision/#fp8-训练) chapter of the distributed training handbook measured this from the training side; here we switch to the shape of an inference GEMM (input dim 7168) and also write out the kernel's real algorithm: along K, one FP8 matrix multiply per 128 elements, multiplied in fp32 by the activation and weight scales and then accumulated:

```python
import torch

torch.manual_seed(0)
M, K, N, G = 64, 7168, 256, 128                   # tokens, input dim (DeepSeek-V3's hidden), output dim, quantization group size
FP8, FP8_MAX = torch.float8_e4m3fn, 448.0
w = torch.randn(K, N) / K**0.5


def quant_act(a, fine):
    """返回 (FP8 值, 缩放)。fine=True：每个 token 每 128 个通道一个缩放；否则整个张量一个"""
    g = a.view(M, K // G, G)
    s = g.abs().amax(-1, keepdim=True) / FP8_MAX if fine else a.abs().max() / FP8_MAX * torch.ones(M, K // G, 1)
    return (g / s).to(FP8), s


def quant_w(fine):
    b = w.view(K // G, G, N // G, G)              # weights: one scale per 128×128 block
    s = b.abs().amax((1, 3), keepdim=True) / FP8_MAX if fine else w.abs().max() / FP8_MAX * torch.ones(K // G, 1, N // G, 1)
    return (b / s).to(FP8), s


def gemm(aq, sa, wq, sw):
    """DeepGEMM 的算法：沿 K 每 128 个元素做一次 FP8 矩阵乘，在 fp32 里乘上两个缩放后累加"""
    out = torch.zeros(M, N)
    for kb in range(K // G):
        part = aq[:, kb].float() @ wq[kb].float().reshape(G, N)
        out += part * sa[:, kb] * sw[kb, 0, :, 0].repeat_interleave(G)
    return out


a = torch.randn(M, K)
aq, sa = quant_act(a, True)
wq, sw = quant_w(True)
deq = (aq.float() * sa).view(M, K) @ (wq.float() * sw).view(K, N)
print("分块累加与先反量化再相乘一致：", torch.allclose(gemm(aq, sa, wq, sw), deq, rtol=1e-4, atol=1e-4))

outliers = torch.randperm(K)[:8]                  # 8 outlier channels
normal = torch.ones(K, dtype=torch.bool)
normal[outliers] = False
print("离群值倍数   逐张量缩放   分块缩放   （正常通道那部分输出的相对误差）")
for mag in (1e1, 1e3, 3e4, 1e5):
    a = torch.randn(M, K)
    a[:, outliers] *= mag
    ref = a[:, normal].double() @ w[normal].double()
    errs = []
    for fine in (False, True):
        aq, sa = quant_act(a, fine)
        wq, sw = quant_w(fine)
        a_dq = (aq.float() * sa).view(M, K)       # the scale is set by the whole row / whole tensor including the outliers
        out = a_dq[:, normal] @ (wq.float() * sw).view(K, N)[normal]
        errs.append(((out.double() - ref).norm() / ref.norm()).item())
    print(f"{mag:>8.0e}   {errs[0]:>9.4f}   {errs[1]:>8.4f}")
```

```text title="output"
分块累加与先反量化再相乘一致： True
离群值倍数   逐张量缩放   分块缩放   （正常通道那部分输出的相对误差）
   1e+01      0.0373     0.0370
   1e+03      0.0374     0.0367
   3e+04      0.1379     0.0388
   1e+05      0.4375     0.0592
```

Two observations:

- The roughly 3.7% error is the floor set by E4M3's 3 mantissa bits, which no scaling scheme can get around; FP8's exponent range is wide (the smallest normal number is about $2^{-6}$, subnormals reach $2^{-9}$), so per-tensor scaling copes with outliers only ten or a thousand times larger;
- Once outliers reach $3 \times 10^4$ and beyond, per-tensor scaling pushes the normal channels into subnormals or even 0 and the error blows up; block scaling affects only the group (128 channels) containing the outlier, leaving other groups' scales unchanged. Activations and gradients in training really do have outliers of this magnitude; a model trained with block scaling must be quantized at the same granularity for inference, or accuracy drops.

**Where the scales are multiplied in.** Each output element is $y_{mn} = \sum_k a_{mk} w_{kn}$; splitting K into blocks of 128, the activations in block $b$ share the scale $s^a_{m,b}$ and the weights share $s^w_{b,n'}$ (where $n'$ is the 128-column block containing $n$), so

$$
y_{mn} = \sum_b s^a_{m,b}\, s^w_{b,n'} \sum_{k \in b} \hat a_{mk}\, \hat w_{kn}
$$

The inner sum is done in FP8 on Tensor Cores; the outer scaling and accumulation in fp32 registers on CUDA Cores. This happens to solve another problem too: the internal accumulator of the H800's FP8 Tensor Cores has limited precision (the DeepSeek-V3 paper measured only about 14 bits), so accumulating all 7168 dims of a large K directly in the Tensor Core loses precision; "promoting" the partial sum to fp32 every 128 elements makes the loss negligible. Multiplying the scales is therefore almost free: the partial sum has to be moved out at this step anyway.

## Grouped GEMM in MoE {#moe-的分组-gemm}

In an MoE layer, each of a GPU's few experts does one GEMM with the tokens assigned to it. The token counts change every step and differ greatly between experts. Two common ways to organize this:

- **Contiguous layout**: sort the tokens assigned to each expert by expert and place them end to end, pad each expert's segment to a multiple of the kernel's block size (BLOCK_M, say 64 or 128), and add an index of "which expert each row belongs to". Suited to prefill: with many tokens, padding wastes a small fraction. The output of DeepEP's high-throughput mode can be arranged exactly this way;
- **Masked layout**: one fixed-size buffer per expert, `[experts, max tokens, hidden]`, plus an array of "how many tokens each expert actually has"; the kernel reads this array on the GPU and skips empty blocks. Shapes are fixed, the CPU need not know the real token counts, and it can be captured in a CUDA Graph. It matches the fixed-slot output of DeepEP's low-latency mode exactly (see [NVSHMEM and DeepEP](../comm/nvshmem-deepep.md#低延迟模式固定槽位没有-cpu-同步)), for decode.

The efficiency of expert GEMMs depends on how many tokens each expert gets. Simulate 256 experts with 4 per GPU (EP=64), comparing decode and prefill:

```python
import numpy as np

rng = np.random.default_rng(0)
EXPERTS, LOCAL, TOPK, HIDDEN, INTER = 256, 4, 8, 7168, 2048   # 4 experts per GPU (EP=64)
w_bytes = 3 * HIDDEN * INTER                                   # FP8 weight bytes of one expert's gate/up/down
ridge = 1979e12 / 3.35e12                                      # H800 FP8 ridge point (FLOP/byte)
print(f"一个专家的权重 {w_bytes / 2**20:.0f} MiB；每个 token 过一个专家 {2 * w_bytes / 1e6:.0f} MFLOPs")
print(f"H800 FP8 的屋脊点约 {ridge:.0f} FLOP/字节：每个专家每步至少要 {ridge / 2:.0f} 个 token，专家 GEMM 才不再受权重读取限制")

popularity = rng.dirichlet(np.full(EXPERTS, 2.0))              # how hot or cold each expert is
for stage, global_tokens in (("decode，全局每步 4K token", 4096), ("decode，全局每步 32K token", 32768), ("prefill，全局 256K token", 262144)):
    counts = rng.multinomial(global_tokens * TOPK, popularity)[:LOCAL]   # tokens assigned to each of this GPU's 4 experts
    for block_m in (64, 128):
        padded = int(sum(-(-c // block_m) * block_m for c in counts))    # contiguous layout: pad each expert's token count to a multiple of BLOCK_M
        print(f"{stage}，BLOCK_M={block_m}：每专家 {counts.min()}～{counts.max()} 个 token，"
              f"补齐后有效行 {counts.sum() / padded:.0%}，算术强度 {2 * counts.mean():.0f} FLOP/字节")
```

```text title="output"
一个专家的权重 42 MiB；每个 token 过一个专家 88 MFLOPs
H800 FP8 的屋脊点约 591 FLOP/字节：每个专家每步至少要 295 个 token，专家 GEMM 才不再受权重读取限制
decode，全局每步 4K token，BLOCK_M=64：每专家 69～248 个 token，补齐后有效行 88%，算术强度 308 FLOP/字节
decode，全局每步 4K token，BLOCK_M=128：每专家 69～248 个 token，补齐后有效行 80%，算术强度 308 FLOP/字节
decode，全局每步 32K token，BLOCK_M=64：每专家 552～2052 个 token，补齐后有效行 97%，算术强度 2440 FLOP/字节
decode，全局每步 32K token，BLOCK_M=128：每专家 552～2052 个 token，补齐后有效行 93%，算术强度 2440 FLOP/字节
prefill，全局 256K token，BLOCK_M=64：每专家 4467～16722 个 token，补齐后有效行 100%，算术强度 20004 FLOP/字节
prefill，全局 256K token，BLOCK_M=128：每专家 4467～16722 个 token，补齐后有效行 100%，算术强度 20004 FLOP/字节
```

This table explains a core design of large-scale MoE inference:

- An expert GEMM's arithmetic intensity is "tokens per expert × 2". With only 4K tokens per step globally, each expert averages 128 tokens, far below FP8's ridge point (about 590), so **the expert GEMM's time is almost exactly the time to read the expert's weights once**, and fewer tokens save little;
- So decode must make the **global** batch large: DP Attention has each GPU handle different requests, and the MoE layer pools all GPUs' tokens to each expert; only by adding up the batches of tens or hundreds of GPUs does each expert get hundreds of tokens. This is one reason DeepSeek forms an EP group of over a hundred GPUs in the decode phase: not just to fit the weights, but to "feed" every expert enough;
- The larger the block, the more padding waste; in decode, with few tokens, use a smaller BLOCK_M or the masked layout. In prefill each expert has thousands to tens of thousands of tokens, padding wastes almost nothing, and it is purely a compute problem.

## DeepGEMM {#deepgemm}

DeepGEMM is DeepSeek's open-source FP8 GEMM library, supporting the fine-grained scaling above, plus ordinary GEMM and grouped GEMM in the contiguous and masked layouts. A few design points stand out:

- **JIT compilation at runtime**: no kernels are compiled at install time; the first time a shape (N, K, number of groups and so on) appears, the corresponding kernel is generated and compiled with the shape as compile-time constants, so the compiler can fully unroll loops and pick the best tiling; results are cached on disk. Inference frameworks usually warm up with common shapes at startup;
- **The full set of Hopper features**: TMA asynchronous copies, warp specialization (some warps only move data, others only compute), WGMMA; these are covered in the [Hopper asynchronous programming](cuda://advanced/async-hopper/) chapter of the CUDA handbook;
- **Little code**: the core kernel is only a few hundred lines, easy to study and modify, and the official release reported over 1350 TFLOPS of FP8 compute on H800;
- **New hardware**: Blackwell natively supports a block-scaled format with "one scale per 32 elements" (MXFP8), with power-of-2 scales (UE8M0); DeepGEMM and later DeepSeek models followed this format, where scales are powers of 2 and multiplication becomes exponent addition.

In inference frameworks: the DeepSeek models in both SGLang and vLLM can use DeepGEMM for FP8 linear layers and MoE grouped GEMM (or implementations from CUTLASS, Triton or FlashInfer); with DeepEP, prefill goes "high-throughput dispatch → contiguous-layout grouped GEMM" and decode goes "low-latency dispatch → masked grouped GEMM", with fixed shapes so the whole layer can be captured in a CUDA Graph.

!!! interview "How to explain it"
    When discussing FP8 inference, start with format and granularity: E4M3 has three mantissa bits, per-tensor scaling fails with extreme outliers, hence block scaling with 1×128 for activations and 128×128 for weights; then the GEMM: one FP8 matrix multiply per 128 elements, with partial sums promoted to fp32 and multiplied by the two scales, which also fixes the Tensor Cores' limited accumulation precision. For MoE, always point out that "an expert GEMM's arithmetic intensity = tokens per expert × 2", and that decode relies on DP Attention + large EP to pool the global batch so experts are fed enough; for grouped GEMM, the contiguous layout goes with prefill, and the masked layout with decode and CUDA Graphs.

!!! info "Related chapters"
    - [Quantization in deployment](../perf/quantization-deploy.md) (this book: choices at the deployment level)
    - [Quantization and GEMV](cuda://advanced/quantization/), [Tensor Cores](cuda://advanced/tensor-core/) (Advanced CUDA: the kernel level)

## Exercises {#练习}

**1. How many tokens does an expert need to be fed?** If expert weights were stored in BF16 (all else equal), how many tokens per step would each expert need to escape the memory bottleneck? And in FP4?

??? success "Answer"
    BF16: 2 bytes per parameter with the same FLOPs per token, so the arithmetic intensity becomes "tokens per expert" (not ×2); the ridge point at BF16 peak is about $989 / 3.35 \approx 295$ FLOP/byte, so about 295 tokens are needed, about the same as FP8, since FP8 doubles both the peak and the bytes. FP4 (such as Blackwell's NVFP4, doubling the peak again and halving the bytes again): the arithmetic intensity is 4 × tokens, and the ridge point doubles too, so it still takes hundreds of tokens. Conclusion: lower precision saves memory and bandwidth, fitting more experts and KV at the same latency, but "each expert needs hundreds of tokens" does not change with lower precision.

**2. Padding in the contiguous layout.** In prefill one GPU has 32 experts (smallish EP), averaging 100 tokens per expert, with BLOCK_M = 128. Roughly how much does padding waste? How can it be reduced?

??? success "Answer"
    Each expert's token count is padded to a multiple of 128, so experts averaging 100 tokens mostly pad to 128, wasting about $1 - 100/128 \approx 22\%$ (more or less with uneven experts). Ways to reduce it: a smaller BLOCK_M (64); kernels that support unpadded variable-length groups (tiling by each expert's real length, with a mask on the last tile); or a larger prefill batch so each expert has more tokens.

## Summary {#小结}

- [x] E4M3 has only 3 mantissa bits, setting an error floor of about 3.7%; extreme outliers break per-tensor scaling, hence block scaling with 1×128 for activations and 128×128 for weights.
- [x] A block-scaled GEMM: one FP8 matrix multiply per 128 K elements, with partial sums promoted to fp32 and multiplied by the two scales; this also fixes the Tensor Cores' limited accumulation precision.
- [x] An expert GEMM's arithmetic intensity = tokens per expert × 2 (FP8); decode relies on DP Attention + large-scale EP to pool the global batch so each expert gets hundreds of tokens.
- [x] Grouped GEMM's contiguous layout (padded to BLOCK_M, paired with high-throughput dispatch) is for prefill, and the masked layout (fixed shapes, capturable in CUDA Graphs, paired with low-latency dispatch) for decode.
- [x] DeepGEMM JIT-compiles per shape at runtime using TMA, warp specialization and WGMMA, and on Blackwell follows the hardware's native block-scaled formats.
