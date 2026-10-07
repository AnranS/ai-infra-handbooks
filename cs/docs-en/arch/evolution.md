# The architectures: Volta to Blackwell, and the other roads

<p class="lead">Inference's hot topics of recent years, FP8, FP4, MoE, speculative decoding, prefill-decode disaggregation, MLA, look like innovations in algorithms and systems and rest on one hardware fact: **compute grows faster than bandwidth**. This chapter walks through Volta, Turing, Ampere, Ada, Hopper and Blackwell, what each generation added and why, then works out with two small models what "compute and bandwidth growing at different rates" means for inference, and finally sets AMD and Ascend beside them to see how others solve the same problem.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. From a V100 to a B200, how many times did compute at one precision grow, and bandwidth? How did the ridge point change?
    2. Why does every generation add a lower precision? How can FP4, with 16 representable values, possibly do matrix multiplication?
    3. In decode, how large does the batch have to be to go from bandwidth-bound to compute-bound? Can a larger batch make attention compute-bound too?
    4. What are Hopper's three new features that matter most to inference, against Ampere? And Blackwell's?
    5. What most distinguishes AMD's MI300X and Huawei's Ascend from NVIDIA architecturally?

??? success "Answers (try it yourself first, then expand)"
    1. Dense BF16/FP16 compute went from 125 to 2250 TFLOPS, 18 times; bandwidth from 0.9 to 8 TB/s, about 9 times. The ridge point (compute ÷ bandwidth) went from 139 to 281, which is to say that for every byte read from memory, more than twice as much arithmetic is now needed to keep the compute fed. Counting each generation's new low precision, the lowest precision's ridge point went from 139 to 1125.
    2. Because widening the datapath and adding compute units costs more and more, while halving the width nearly freely doubles the throughput and halves the memory and bandwidth the weights and the KV take. FP4 works through **block scaling**: every 16 or 32 elements share a scale factor (NVFP4's is FP8, MXFP4's an 8-bit exponent), so the dynamic range within a block is small and 4 bits is enough; the accumulation is still done in FP32.
    3. The weights are shared by the whole batch, so the arithmetic intensity = 2 × batch ÷ the bytes per parameter, and reaching an H100's ridge point of 295 with BF16 weights takes a batch of 295; FP8 weights halve it to 148 and 4-bit weights to 74. Attention is different: the KV belongs to each request, and a larger batch does not reuse any one request's KV, so its arithmetic intensity equals "how many query heads share one KV head", 1 for MHA, 8 for 8-group GQA, about 128 for an absorbed MLA, all far below the ridge point. So decode's attention is always bandwidth-bound and can only be sped up by fewer KV bytes (GQA, MLA, quantized KV).
    4. Hopper: FP8 Tensor Cores (double the compute, half the KV and weights), the TMA + thread block clusters (asynchronous movement, which supports warp-specialized kernels like FlashAttention-3), and NVLink 4 with a 900 GB/s NVSwitch (which supports large tensor parallelism and EP). Blackwell: FP4 and the block-scaled formats, Tensor Memory and the single-thread tcgen05 (which makes larger tiles feasible), and NVLink 5 with the NVL72 rack (72 cards in one NVLink domain, so large-scale expert parallelism no longer crosses the network).
    5. AMD's MI300X: one GPU built from several compute chiplets (XCDs) plus I/O chiplets, with 192 GB at 5.3 TB/s (against a contemporary H100's 80 GB / 3.35 TB/s) and 256 MB of Infinity Cache in front of the memory; its software stack is ROCm/HIP, close to CUDA at the source level with a weaker ecosystem. Ascend is the Da Vinci architecture: an AI Core has a Cube unit for matrix multiplication, a Vector unit and a Scalar unit, and the programming model is not SIMT but an explicit multi-level buffered pipeline (Ascend C), with CANN + HCCL as the stack.

## One table: what each generation added {#一张表每一代加了什么}

| Architecture | Compute capability | Representative card | The hardware added | What it means for inference |
| --- | --- | --- | --- | --- |
| Volta | 7.0 | V100 | the first Tensor Cores (FP16), independent thread scheduling | matrix multiplication got a dedicated unit; a warp can no longer be assumed implicitly synchronized |
| Turing | 7.5 | T4 | INT8 / INT4 Tensor Cores | the beginning of quantized inference; the T4 is still the lowest bar for deployment |
| Ampere | 8.0 | A100 | BF16, TF32, 2:4 sparsity, `cp.async`, 164 KB of shared memory, MIG | BF16 became the mainstream format for training and inference; asynchronous copies made multi-stage pipelines possible |
| Ada | 8.9 | L4, L40S | FP8 Tensor Cores, a large L2 (72 MB) | cheap inference cards gained FP8, though without HBM the bandwidth is the weak point |
| Hopper | 9.0 | H100, H200 | FP8 + the Transformer Engine, the TMA, thread block clusters and distributed shared memory, `wgmma`, NVLink 4 + NVSwitch, DPX | FlashAttention-3 style warp specialization; FP8 inference everywhere; 8 fully connected cards in a machine supporting TP and EP |
| Blackwell | 10.0 | B200, B300, GB200 | FP6 / FP4 and the block-scaled formats, Tensor Memory, `tcgen05` (single-thread issue, two-SM cooperation), two-chip packaging, NVLink 5 and NVL72, the second-generation Transformer Engine | FP4 inference; 72 cards in one NVLink domain, so large-scale EP no longer crosses the network |

(The RTX 50 series' compute capability is 12.0 and is also called Blackwell, but it has no tcgen05 and no Tensor Memory, with a programming model closer to Ada's.)

Three threads run through it:

1. **precision keeps going down**: FP16 → INT8 → BF16/TF32 → FP8 → FP6/FP4. Every step down doubles the Tensor Cores' throughput and halves the memory and bandwidth the weights and KV take;
2. **moving data keeps becoming asynchronous**: ordinary loads → `cp.async` (bypassing the registers) → the TMA (a whole tile in one instruction); and the synchronization moved from `__syncthreads()` to mbarriers and thread block clusters;
3. **the scale keeps going up**: one chip → two chips packaged together → 8 cards on an NVSwitch in a machine → a 72-card NVLink domain in a rack.

## Compute grows faster than bandwidth {#算力涨得比带宽快}

Putting several generations of SXM cards' compute, bandwidth and ridge point together:

First a look at each generation's ridge point and the batch decode has to accumulate:

<div class="aig-widget" data-widget="ridge-gen"></div>

```python title="ridge.py"
# how much each generation of data centre card (the SXM form) raised the compute and the bandwidth, and how the ridge point (compute / bandwidth) changed.
# what the ridge point means: how many operations per byte read from memory keep the compute fed
gens = [
    # name, year, dense FP16/BF16 TFLOPS, dense TFLOPS at that generation's lowest precision, the precision's name, bandwidth GB/s, memory GB
    ("V100", 2017, 125, 125, "FP16", 900, 32),
    ("A100", 2020, 312, 312, "BF16", 2039, 80),
    ("H100", 2022, 989, 1979, "FP8", 3350, 80),
    ("B200", 2024, 2250, 9000, "FP4", 8000, 180),
]
print("GPU   年份  BF16算力 较上代    带宽   较上代  BF16屋脊点   最低精度算力  屋脊点")
prev = None
for name, year, bf16, low, dtype, bw, mem in gens:
    up = (f"{bf16 / prev[0]:5.1f}x", f"{bw / prev[1]:5.1f}x") if prev else ("    -", "    -")
    print(f"{name:5s} {year} {bf16:7.0f} {up[0]} {bw / 1000:6.2f} TB/s {up[1]} "
          f"{bf16 * 1e12 / (bw * 1e9):9.0f} {low:11.0f} {dtype:4s} {low * 1e12 / (bw * 1e9):7.0f}")
    prev = (bf16, bw)
print()
print("同一个 8B 模型（BF16 权重 16 GB，KV 每 token 128 KB）在各代上的理论下限：")
print("GPU   decode 一步(batch=1)  prefill 4K(BF16，算力用满一半)  能放的 KV")
for name, year, bf16, low, dtype, bw, mem in gens:
    decode = 16e9 / (bw * 1e9) * 1e3                       # read the weights once
    prefill = 2 * 8e9 * 4096 / (bf16 * 1e12 * 0.5) * 1e3
    kv = (mem * 1e9 - 16e9) / (128 * 1024) / 1e3
    print(f"{name:5s} {decode:12.1f} ms {prefill:26.0f} ms {kv:9.0f}K token")
```

```text title="output"
GPU   年份  BF16算力 较上代    带宽   较上代  BF16屋脊点   最低精度算力  屋脊点
V100  2017     125     -   0.90 TB/s     -       139         125 FP16     139
A100  2020     312   2.5x   2.04 TB/s   2.3x       153         312 BF16     153
H100  2022     989   3.2x   3.35 TB/s   1.6x       295        1979 FP8      591
B200  2024    2250   2.3x   8.00 TB/s   2.4x       281        9000 FP4     1125

同一个 8B 模型（BF16 权重 16 GB，KV 每 token 128 KB）在各代上的理论下限：
GPU   decode 一步(batch=1)  prefill 4K(BF16，算力用满一半)  能放的 KV
V100          17.8 ms                       1049 ms       122K token
A100           7.8 ms                        420 ms       488K token
H100           4.8 ms                        133 ms       488K token
B200           2.0 ms                         58 ms      1251K token
```

Two things to see:

- **at one precision (BF16), compute grew 18 times and bandwidth 9**, and the ridge point went from 139 to 281. In the V100's day, 139 operations per byte read from memory kept the compute fed; now it takes 281. Counting each generation's new lowest precision, an H100's FP8 is 591 and a B200's FP4 is 1125, a wider gap still.
- **decode and prefill did not improve together**: one batch-1 decode step of an 8B model went from the V100's 17.8 ms to the B200's 2.0, 9 times faster (exactly the bandwidth's factor); prefilling 4K for the same model went from 1049 ms to 58, 18 times faster. Every generation of hardware makes "computing" much cheaper while "getting the data in" only becomes a little cheaper.

Nearly all of the inference systems work of recent years reads as "make the bandwidth's bill smaller and the compute's bill fuller":

| Direction | What it does | Which number above |
| --- | --- | --- |
| weight quantization (FP8, AWQ, GPTQ, FP4) | fewer bytes per parameter | directly fewer bytes read per decode step |
| GQA, MLA, KV quantization | smaller KV per token | the same, and a larger share with a long context |
| continuous batching, large batches | the weights read in are shared by more tokens | raises the arithmetic intensity towards the ridge point |
| MoE | each token activates only some experts | the parameter count grows while the bytes read per step do not grow in proportion |
| speculative decoding | several tokens verified per step | read the weights once and produce several tokens |
| prefill-decode disaggregation | prefill (compute-bound) and decode (bandwidth-bound) deployed apart | each kind of card used for its strength |
| CUDA Graphs, overlap scheduling, megakernels | less GPU idling | keeps the already expensive compute from waiting on the CPU |

### How large the batch has to be {#batch-要多大才够}

In one decode step, the weights are shared by all this batch's tokens: 1 byte of weight read does 2 × batch operations. How large does the batch have to be to reach the ridge point?

```python title="breakeven.py"
# how large a batch does decode need to go from bandwidth-bound to compute-bound?
# per parameter per step: w bytes read and 2 x batch operations (the batch's tokens share this one copy of the weights)
# so the arithmetic intensity = 2 x batch / w (FLOP/byte), which has to reach the ridge point, compute / bandwidth
gens = [("A100", 312, 2039), ("H100", 989, 3350), ("H100 FP8", 1979, 3350),
        ("B200", 2250, 8000), ("B200 FP4", 9000, 8000)]
weights = [("BF16 权重", 2), ("FP8 权重", 1), ("4 比特权重", 0.5)]
print("GPU       屋脊点   " + "  ".join(f"{n}" for n, _ in weights))
for name, tf, bw in gens:
    ridge = tf * 1e12 / (bw * 1e9)
    row = "".join(f"{ridge * w / 2:11.0f}" for _, w in weights)
    print(f"{name:9s} {ridge:6.0f}{row}")
print()
print("权重之外，decode 还要读 KV。KV 是每个请求自己的，加大 batch 不会让它被复用：")
print("每个 KV 元素被组里的 g 个查询头各用一次乘加，所以注意力部分的算术强度 = g（BF16 KV，FLOP/字节）")
print("每组查询头数  算术强度  和 H100 BF16 屋脊点 295 相比  注意力方案")
for name, g in [("MHA（每个头一份 KV）", 1), ("GQA 8 组", 8), ("MLA（吸收后，128 个头共用）", 128)]:
    print(f"{g:11d} {g:9d} {295 / g:20.0f} 倍之差  {name}")
```

```text title="output"
GPU       屋脊点   BF16 权重  FP8 权重  4 比特权重
A100         153        153         77         38
H100         295        295        148         74
H100 FP8     591        591        295        148
B200         281        281        141         70
B200 FP4    1125       1125        562        281

权重之外，decode 还要读 KV。KV 是每个请求自己的，加大 batch 不会让它被复用：
每个 KV 元素被组里的 g 个查询头各用一次乘加，所以注意力部分的算术强度 = g（BF16 KV，FLOP/字节）
每组查询头数  算术强度  和 H100 BF16 屋脊点 295 相比  注意力方案
          1         1                  295 倍之差  MHA（每个头一份 KV）
          8         8                   37 倍之差  GQA 8 组
        128       128                    2 倍之差  MLA（吸收后，128 个头共用）
```

On an H100, BF16 weights need a batch of 295 to become compute-bound, FP8 weights 148 and 4-bit weights 74: **quantization not only saves memory but lowers the bar to becoming compute-bound**. Conversely, a B200's FP4 ridge point of 1125 needs a batch of 281 even with 4-bit weights, which is hard to reach on one card in practice, so Blackwell depends more on large batches, MoE and speculative decoding to put the compute to work.

Attention is entirely different: the KV belongs to each request and a larger batch does not reuse it. Its arithmetic intensity depends only on "how many query heads share one KV head", 1 for MHA, 8 for 8-group GQA, about 128 for an absorbed MLA, all far below the ridge point. **Decode's attention is always bandwidth-bound**, and the only way out is smaller KV (GQA, MLA, quantized KV, sparse attention), which is why every model family keeps reworking its attention structure (see the Inference Systems handbook's [MLA inference](serving://moe/mla/) and [sparse attention](serving://moe/mtp-sparse/)).

## Why low precision works {#低精度为什么可行}

FP4 has 16 representable values; how can it do matrix multiplication? The key is **block scaling**: a group of elements shares a scale factor, and the real value is "a 4-bit mantissa × that block's scale".

| Format | Width | Elements per block | The scale | Where it is used |
| --- | --- | --- | --- | --- |
| FP8 E4M3 / E5M2 | 8 | the whole tensor or per row | FP32 | inference and training from Hopper |
| block FP8 | 8 | 128 × 128 or 1 × 128 | FP32 | DeepSeek-V3's training and inference |
| MXFP8 / MXFP6 / MXFP4 | 8 / 6 / 4 | 32 | an 8-bit exponent (a power of two) | an open compute standard, native on Blackwell |
| NVFP4 | 4 | 16 | FP8 E4M3, plus a tensor-level FP32 | the format Blackwell's inference pushes |

The smaller the block the narrower its dynamic range and the better a low width serves; the cost is the scale factors' own space (NVFP4's one FP8 per 16 elements adds 0.5 bits per element) and the extra handling. Blackwell's Tensor Cores take these formats directly and apply the scales in hardware during the multiply-accumulate, with no dequantization in software.

The accumulation is still FP32: a matrix multiply sums K products and low-precision accumulation's error grows quickly with K. So "FP4 inference" means the inputs and the weights are FP4 while the accumulation and the output stay at higher precision. How the accuracy is kept is in the Inference Systems handbook's [quantized deployment in practice](serving://perf/quantization-deploy/) and [low-bit inference for very large MoE](serving://frontier/low-bit/).

## How others solve the same problem {#别家怎么解同一道题}

**AMD Instinct (the CDNA architecture)**. The MI300X is built from chiplets: 8 compute chiplets (XCDs, 38 compute units each) plus 4 I/O chiplets, with 8 HBM3 stacks giving 192 GB at 5.3 TB/s against a contemporary H100's 80 GB / 3.35 TB/s, and the large memory is its great selling point (one card holds a 70B model's BF16 weights). The I/O chiplets also carry 256 MB of Infinity Cache in front of the memory. The Matrix Cores in the compute units correspond to Tensor Cores, and the MI300X's dense BF16 is 1307 TFLOPS and its FP8 2615. The stack is ROCm: HIP is highly similar to CUDA at the source level (`hipify` converts most code automatically), RCCL corresponds to NCCL, and vLLM and SGLang both support it. The gap is mostly in the ecosystem: the operator libraries' coverage and maturity, day-one support for new models, and the debugging and profiling tools. The later MI355X added FP6 / FP4 and more HBM3e.

**Huawei Ascend (the Da Vinci architecture)**. It is not SIMT: an AI Core has three kinds of unit, Cube (matrix multiplication, 16×16×16 per instruction), Vector and Scalar (scalars and control), with the data moved explicitly between Global Memory, L1 and the L0A/L0B/L0C buffers. Kernels are written in Ascend C, arranging the "bring it in, compute, move it out" pipeline yourself, more like writing for a DSP than like writing CUDA. The stack is CANN (corresponding to CUDA), the collectives are HCCL (corresponding to NCCL), the inference engine is MindIE, and vLLM and SGLang both have hardware back-end plugins for it (see [the multi-hardware platform abstraction](serving://ops/platforms/)). Inference roles in China often ask for familiarity with this.

**Google TPU**. It uses a systolic array: a large multiply-accumulate array where the data flows in from one side like a wave and the results flow out of the other, with very little control logic, which suits large matrix multiplies extremely well; there is no SIMT and no shared memory level, and the programming goes through the XLA compiler rather than hand-written kernels. A TPU's interconnect is a 3D torus, which forms a very large cluster without switches.

What the three have in common: **they all move in the same direction**, larger matrix units, lower precision, larger on-chip buffers and wider interconnects, because the ridge point's rise is the same for everybody.

## How to read a newly announced specification sheet {#怎么读一张新发布的规格表}

When a new card is announced, reading in this order quickly says what it means for inference:

1. **memory capacity and bandwidth** → decode's ceiling (how long reading it all takes), how large a model and how much KV fit;
2. **dense compute at each precision** (ignore the sparse numbers) → prefill's ceiling, the ridge point;
3. **the ridge point = compute ÷ bandwidth** → how large a batch becomes compute-bound, and whether the card suits prefill or decode;
4. **the interconnect**: NVLink / Ethernet bandwidth, how many cards in one domain → how large TP and EP can be and whether crossing machines means the network;
5. **the new hardware units** (new precisions, new movement or synchronization) → a newer CUDA and newer operator libraries, and roughly how long until vLLM / SGLang support it;
6. **the form factor and the power**: SXM or PCIe, the system's power and cooling, which decide whether it fits the existing data centre.

!!! interview "Answering in an interview"
    Asked about the architectures or "why is everybody doing quantization and MoE now": the core fact is that compute grows faster than bandwidth. From a V100 to a B200, compute at one precision rose 18 times and bandwidth 9, and the ridge point went from 139 to 281, or 1125 counting the new low precisions. So decode (bandwidth-bound) improves far more slowly than prefill (compute-bound), and inference systems' main work is making the bandwidth's bill smaller and the compute's bill fuller: quantizing the weights and the KV, GQA/MLA, large batches and continuous batching, MoE, speculative decoding, prefill-decode disaggregation. Quantitatively: BF16 weights need a batch of 295 to become compute-bound on an H100, FP8 148 and 4-bit 74; while attention's arithmetic intensity equals the query heads per group (1 for MHA, 8 for GQA, about 128 for MLA), so a larger batch does nothing and only fewer KV bytes help. Then go generation by generation: Ampere's BF16 and cp.async, Hopper's FP8 and TMA and clusters, Blackwell's FP4 block scaling and Tensor Memory and NVL72. Finish with the others: AMD on large memory and ROCm, Ascend's Da Vinci with Cube/Vector units and an explicit pipeline, and the TPU's systolic array with XLA.

## Exercises {#练习}

**1. The ridge point and choosing a card.** An H20 has 148 TFLOPS of dense BF16 at 4.0 TB/s; an H100 SXM has 989 TFLOPS at 3.35 TB/s. What is each one's ridge point? Decoding a BF16 model on each, how large does the batch have to be to become compute-bound? Why is the H20 said to suit decode and not prefill?

??? success "Answer"
    The H20's ridge point is 148 / 4.0 = 37 and the H100's is 295. With BF16 weights the batches become compute-bound at 37 and 295. The H20's bandwidth is 20% higher than the H100's, so decode (reading the weights once) is faster than on an H100; but its compute is 15% of the H100's, so prefill is over 6 times slower. In heterogeneous prefill-decode disaggregation, using the compute-heavy card for prefill and the bandwidth-heavy one for decode divides the work along exactly these two numbers.

**2. Quantization's two bills.** A 70B model's BF16 weights are 140 GB. Quantizing them to 4 bits: how much fewer bytes does a decode step read? Does prefill shorten in proportion? Why?

??? success "Answer"
    The weights go from 140 GB to 35, so a decode step reads three quarters fewer bytes and is 4 times faster by bandwidth (the KV part is unchanged, which dilutes the gain with a long context). Prefill does not shorten in proportion: it is compute-bound, the weights are read once, and the time goes to the matrix multiplies; if the operator "dequantizes to BF16 and multiplies", the compute is unchanged and prefill barely moves; only when the hardware's Tensor Cores support that low precision natively (FP8, FP4) does the compute really double and prefill speed up. That is why "weight quantization mainly accelerates decode, and quantizing the activations too is what accelerates prefill".

**3. One generation's arithmetic.** Suppose the next card doubles the compute and raises the bandwidth only 30%, with everything else unchanged. What happens to (a) batch-1 decode latency; (b) prefill throughput; (c) the batch needed to become compute-bound? How would you adjust the deployment?

??? success "Answer"
    (a) Decode's latency improves only with the bandwidth, about 23% faster; (b) prefill's throughput doubles; (c) the ridge point becomes 2 / 1.3 ≈ 1.54 times, and the batch needed rises by the same factor.

    The adjustments: quantize the weights and the KV more aggressively (making the bandwidth's bill smaller), run larger batches (more concurrency and a longer queueing window, at the cost of latency), find speculative decoding more worthwhile (read the weights once and produce several tokens), and deploy prefill and decode apart with higher-bandwidth cards for decode.

**4. Judging a claim.** "We switched the model from BF16 to FP8 and the compute doubled, so inference will be twice as fast." When does that hold and when does it not?

??? success "Answer"
    It holds for prefill (or any compute-bound phase with a large batch): FP8 Tensor Cores have twice BF16's throughput, and if the operators really compute in FP8 at a comparable utilization, the time nearly halves.

    It does not hold for small-batch decode: that is bandwidth-bound and the speed depends on the bytes read per step. If only the computation is FP8 while the weights are still stored as BF16, the bytes are unchanged and nothing speeds up; if the weights are stored as FP8 too, the bytes halve and it is twice as fast, and that gain comes from "smaller weights" rather than "double the compute". With a long context the KV's share grows and the weights' gain is diluted further.

## Summary {#小结}

- [x] Three threads: precision keeps going down (FP16 → FP8 → FP4, through block scaling), moving data keeps becoming asynchronous (cp.async → the TMA), and the scale keeps going up (one chip → two chips → NVSwitch → NVL72).
- [x] From a V100 to a B200, compute at one precision rose 18 times and bandwidth 9, with the ridge point going from 139 to 281 (1125 counting the low precisions): prefill improved far faster than decode.
- [x] Quantization, GQA/MLA, large batches, MoE, speculative decoding and prefill-decode disaggregation are all "make the bandwidth's bill smaller and the compute's bill fuller".
- [x] BF16 weights need a batch of 295 to be compute-bound on an H100, FP8 148 and 4-bit 74; attention's arithmetic intensity equals the query heads per group, so decode's attention is always bandwidth-bound.
- [x] AMD relies on chiplets, large memory and ROCm; Ascend's Da Vinci has Cube/Vector units and an explicit pipeline; the TPU is a systolic array with XLA. Everybody moves in the same direction.
- [x] The order for reading a specification sheet: capacity and bandwidth → dense compute → the ridge point → the interconnect → the new units → the form factor and the power.
