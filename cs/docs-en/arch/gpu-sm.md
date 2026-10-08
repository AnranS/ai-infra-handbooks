# A GPU's SMs and Tensor Cores: what is inside the chip

<p class="lead">Where does an H100's 989 TFLOPS come from? Why does a Tensor Core's instruction grow every generation, from one warp issuing it to four warps together, and on Blackwell to one thread? Why did the accumulator move out of the registers into a dedicated Tensor Memory? Why does an unremarkable exp inside attention hold a whole kernel back? This chapter starts from the chip's hierarchy, takes an SM apart to see the warp schedulers, the register file and the Tensor Cores, and works the peak, the latency hiding and the operand bandwidth out with small models. The CUDA handbook covers how to program this hardware; this chapter covers why the hardware looks like this.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which three numbers multiply to an H100 SXM's 989 TFLOPS of dense BF16?
    2. One warp scheduler issues one instruction per cycle. A warp issues a load, waits 500 cycles, and then has 16 compute instructions. How many warps does it take to keep the scheduler busy? A scheduler can hold only 16 warps; what now?
    3. Why is a GPU's register file so large (256 KB per SM)?
    4. From Ampere's `mma.sync` to Hopper's `wgmma` to Blackwell's `tcgen05.mma`, one matrix multiply instruction computes a larger and larger tile. Why?
    5. Why does FlashAttention need special handling for softmax's exp on Hopper and Blackwell?

??? success "Answers (try it yourself first, then expand)"
    1. The SM count × the Tensor Core work per SM per cycle × the clock: 132 SMs × 4096 FLOP (2048 multiply-adds) × about 1.83 GHz ≈ 989 TFLOPS.
    2. A warp's round takes 1 + 500 + 16 cycles, of which only 17 issue instructions, so about 517 / 17 ≈ 30 warps taking turns are needed to fill it, past a scheduler's limit of 16. The way out is more instruction-level parallelism: have each thread issue several independent loads at once (handling 4 elements with vectorized reads), and with 4 loads and 64 compute instructions per round, 8 warps suffice.
    3. A GPU hides latency by switching between warps, and that switch has to be free, so every resident warp's registers stay in the register file at once and cannot be spilled as a CPU's would be. An SM holds dozens of warps and thousands of threads, each with dozens or hundreds of registers, so the file is naturally large; conversely, the more registers a thread uses, the fewer warps can be resident.
    4. Each generation doubles the Tensor Cores' throughput while the shared memory's bandwidth (128 bytes per SM per cycle) stays the same. A larger tile per instruction means fewer operand bytes per multiply-add, which is what keeps the Tensor Cores fed within the shared memory's bandwidth; and a larger tile also amortizes the cost of issuing instructions and computing addresses. Past a point the accumulator grows too large for the registers, so Blackwell moved it into Tensor Memory with one thread issuing the whole tile's computation.
    5. Every score in attention takes one exp, and exp is computed by the special function units (SFUs), of which an SM does only 16 per cycle, far below the Tensor Cores' throughput. At head dimension 128 on an H100, exp takes half the SFU's capacity; with FP8 and on Blackwell it takes all of it or more, so exp becomes the bottleneck. FlashAttention-3 interleaves softmax and the matrix multiplies across warpgroups, the Blackwell implementation also offloads some exp to a polynomial approximation on the FMA units, and Blackwell Ultra simply doubled the SFU's exponential throughput.

## From the whole chip to an SM {#从整块芯片到-sm}

A GPU chip is a hierarchy of hierarchies. Taking Hopper:

<!-- i18n:diagram 72a72491e4 -->
```text
the GH100 chip (over 800 square millimetres, 80 billion transistors)
├── 8 GPCs (graphics processing clusters)
│   └── 9 TPCs (texture processing clusters) each
│       └── 2 SMs each
├── the L2 cache (50 MB, in two halves)
├── the memory controllers (to HBM)
└── NVLink, PCIe, the copy engines and the rest
```

A complete GH100 has 8 × 9 × 2 = 144 SMs, while an H100 SXM enables only 132 and an H100 PCIe only 114. The reason is **yield**: a chip this large nearly always has a few manufacturing defects, so redundancy is designed in, the defective units are disabled at the factory, and the parts are sold by how many SMs work. The A100 is the same: a complete GA100 has 128 SMs and the A100 enables 108.

For somebody writing kernels, the GPC level matters in one practical way: Hopper's [thread block clusters](cuda://advanced/async-hopper/#线程块集群与分布式共享内存-sm_90) can only be scheduled within one GPC, which is what lets a cluster's blocks reach each other's shared memory.

Blackwell's B200 goes further: one chip is already at the largest area a lithography machine can expose, so **two chips** are packaged together and joined into one GPU by a 10 TB/s inter-chip link, 208 billion transistors and 148 SMs in all. To software it is still one device, though reaching the other chip's L2 and memory takes a longer road.

## Taking an SM apart {#拆开一个-sm}

![Figure: an SM taken apart - 4 partitions each with a scheduler, registers, CUDA cores and a Tensor Core](../assets/figures/sm-anatomy.svg){.aig-svg}

An SM is divided into 4 **processing partitions** (SMSPs), each like a small core with its own:

| Part | Per partition on an H100 | What it does |
| --- | --- | --- |
| a warp scheduler + issue unit | 1 | picks a ready warp from this partition's and issues one instruction per cycle |
| a register file | 16384 32-bit registers (64 KB) | every resident thread's registers in this partition, held there throughout |
| FP32 units | 32 | one warp's FP32 instruction per cycle (an A100's 16 per partition take two) |
| INT32 / FP64 units | 16 each | integer arithmetic (addressing) and double precision |
| a Tensor Core | 1 | matrix multiply-accumulate |
| special function units (SFUs) | 4 | the transcendentals: exp, log, sin, rsqrt |
| load/store units (LD/ST) | several | computing addresses and issuing memory requests |

The four partitions share the SM-level resources: 256 KB of L1 / shared memory (at most 228 KB of it shared memory), the TMA unit and the texture units. An SM holds at most 64 warps (16 per partition), 2048 threads and 32 blocks.

A few numbers worth remembering:

- **the register file is larger than the shared memory**: 256 KB per SM, and 132 × 256 KB ≈ 33 MB across an H100, as much as all its L1 / shared memory and close to its 50 MB of L2. It is large because a GPU saves and restores nothing when switching between warps: every resident warp's registers stay in the file at once, which is what makes the switch free. The cost is that the more registers a thread uses the fewer warps can be resident, which is [occupancy](cuda://basics/execution/#占用率occupancy).
- **each partition issues one instruction per cycle**. An SM issues at most 4 warp instructions per cycle, each driving 32 threads. The compute is stacked up by making each instruction do a great deal (a Tensor Core instruction does thousands to over a million multiply-adds) rather than by issuing more instructions.
- **no out-of-order execution and no branch prediction**. Issue is in order: a warp whose next instruction depends on a result not yet computed waits, and the scheduler issues another warp's.

## How the peak compute is computed {#峰值算力是怎么算出来的}

The peak = the SM count × the work per SM per cycle × the clock. The work per cycle comes from the hardware: FP32 is "the FP32 unit count × 2" (a multiply-add is 2 operations); the Tensor Cores are "the multiply-adds per SM per cycle × 2". Working the clock back from the specification sheet's peak checks the decomposition:

```python title="peaks.py"
# the peak = the SM count x the work per SM per cycle x the clock. Working the clock back from the sheet's peak checks the decomposition
gpus = [
    # name, SM count, FP32 units per SM, the sheet's FP32 TFLOPS, dense FP16 Tensor Core multiply-adds per SM per cycle, the sheet's FP16/BF16 TFLOPS
    ("V100", 80, 64, 15.7, 512, 125),
    ("A100", 108, 64, 19.5, 1024, 312),
    ("H100 SXM", 132, 128, 67, 2048, 989.4),
    ("B200", 148, None, None, 4096, 2250),
]
print("GPU       SM数 FP32单元/SM 反推频率  Tensor FLOP/周期/SM 反推频率")
for name, sm, lanes, fp32, fma, tc in gpus:
    f_fp32 = f"{fp32 * 1e12 / (sm * lanes * 2) / 1e9:.2f} GHz" if fp32 else "-"
    f_tc = f"{tc * 1e12 / (sm * fma * 2) / 1e9:.2f} GHz"
    print(f"{name:9s} {sm:4d} {lanes or '-':>11} {f_fp32:>9} {fma * 2:>20} {f_tc:>9}")
```

```text title="output"
GPU       SM数 FP32单元/SM 反推频率  Tensor FLOP/周期/SM 反推频率
V100        80          64  1.53 GHz                 1024  1.53 GHz
A100       108          64  1.41 GHz                 2048  1.41 GHz
H100 SXM   132         128  1.98 GHz                 4096  1.83 GHz
B200       148           -         -                 8192  1.86 GHz
```

The V100 and the A100 give the same clock both ways, exactly their boost clocks (1.53 GHz, 1.41 GHz), which says the decomposition is right. The H100 is interesting: its FP32 peak implies 1.98 GHz while its Tensor Core peak corresponds to 1.83. A specification sheet's peaks for different units are not computed at one clock. At full load the Tensor Cores draw the most power and the power wall pulls the clock down, and a real large matrix multiply often runs lower still, which is one reason a measured matrix multiply usually reaches only seventy or eighty per cent of the peak.

The table also shows each generation's pattern: the Tensor Core work per SM per cycle doubles from Volta to Blackwell every generation (the B200 row worked back from 148 SMs and the specification), while the SM count and the clock barely rise. Nearly all the growth in compute comes from the Tensor Cores themselves.

A few things to watch on a specification sheet:

- **sparse and dense**: many sheets put the 2:4 structured sparse number in the most prominent place, twice the dense one. Estimating inference performance uses the dense number;
- **the form factor**: a model's SXM and PCIe versions differ in SM count, clock, power limit and memory bandwidth (an H100 PCIe's dense BF16 is about 756 TFLOPS against the SXM's 989);
- **the precision**: FP8 is twice BF16 and FP4 twice again; TF32 is half BF16, and FP32 (off the Tensor Cores) several times lower.

Each model's parameters are in the Inference Systems handbook's [hardware and ecosystem reference](serving://career/hardware/).

## Hiding latency: how many warps {#延迟掩盖要多少个-warp}

Which resource caps the resident warps first, moved along with the CUDA handbook's occupancy tool:

<div class="aig-widget" data-widget="occupancy"></div>

Memory takes hundreds of cycles and a GPU does not execute out of order, relying on **taking turns**: while one warp waits for data, the scheduler issues another's instructions. How many warps are enough? Modelling one scheduler, with each warp repeatedly "issuing a load, waiting 500 cycles, issuing a batch of compute instructions", and seeing what fraction of cycles the scheduler issues anything:

```python title="latency_hiding.py"
# a warp scheduler issues at most one instruction per cycle. Each warp repeatedly issues ilp independent loads in a row,
# waits for the data (latency cycles), then issues compute instructions. This models what fraction of cycles the scheduler issues anything.
# the policy is greedy-then-oldest (GTO): keep issuing one warp until it has to wait, then take the lowest-numbered ready warp.
# only the steady state is counted: stop once any warp has finished iters rounds
def simulate(warps, ilp, compute, latency=500, iters=100):
    phase = ["load"] * warps                 # load: issue the memory access; wait: wait for the data; compute: issue the computation
    left = [ilp] * warps                     # how many instructions are left in this phase
    ready_at = [0] * warps                   # the cycle the data arrives
    done = [0] * warps                       # how many rounds are done
    cycle = busy = 0
    cur = 0
    while max(done) < iters:
        def ready(w):
            return phase[w] != "wait" or cycle >= ready_at[w]
        if not ready(cur):
            cur = next((w for w in range(warps) if ready(w)), None)
        if cur is None:                      # every warp is waiting for data: this cycle idles
            cycle, cur = cycle + 1, 0
            continue
        w = cur
        if phase[w] == "wait":
            phase[w], left[w] = "compute", compute
        left[w] -= 1                         # issue one instruction
        busy += 1
        if phase[w] == "load" and left[w] == 0:
            phase[w], ready_at[w] = "wait", cycle + latency
        elif phase[w] == "compute" and left[w] == 0:
            phase[w], left[w], done[w] = "load", ilp, done[w] + 1
        cycle += 1
    return busy / cycle


print("每个调度器上的 warp 数       1     2     4     8    16")
for ilp, compute in [(1, 16), (4, 64)]:
    row = [f"{simulate(w, ilp, compute):5.0%}" for w in (1, 2, 4, 8, 16)]
    print(f"每轮 {ilp} 条访存、{compute:2d} 条计算：" + " ".join(row))
```

```text title="output"
每个调度器上的 warp 数       1     2     4     8    16
每轮 1 条访存、16 条计算：   3%    7%   13%   26%   52%
每轮 4 条访存、64 条计算：  12%   24%   48%   95%   99%
```

The first row, 1 load and 16 compute instructions per round: a warp's round takes 517 cycles of which only 17 issue, so about 517 / 17 ≈ 30 warps are needed to fill the scheduler. But a scheduler holds at most 16 warps (64 per SM), so **full occupancy still only reaches 52%**.

The second row issues 4 independent loads in a row and then 4 times the computation, keeping the ratio of loads to computation the same: 8 warps nearly saturate it. That is **instruction-level parallelism** at work, the same thing as the previous chapter's several accumulators on a CPU. It explains why nearly every high-performance kernel has each thread handle several elements and read 16 bytes at a time with `float4`: warps alone cannot hide the memory latency. The general formula for the estimate is Little's law: the work in flight = latency × throughput (see the CUDA handbook's [hiding latency](cuda://basics/execution/#延迟掩盖)).

The scheduling policy in the model is greedy-then-oldest (GTO): keep issuing one warp until it has to wait for data, then switch. It is better than round-robin, which puts every warp in lockstep so they all issue loads and all wait together, with nobody hiding anybody.

## SIMT and independent thread scheduling {#simt-与独立线程调度}

A programmer writes **one thread's** code and the hardware packs 32 threads into a warp, with one instruction driving all 32, which is SIMT (single instruction, multiple threads). It is the same thing as a CPU's SIMD in essence and differs in the programming model: SIMD has the programmer (or the compiler) use vector registers explicitly, while SIMT makes every thread look independent, with branches and loops written as usual and the hardware masking off the threads not on that path when they diverge.

From Volta onwards every thread in a warp has its own program counter, so after a divergence the two paths can interleave and "threads of one warp are implicitly synchronized" no longer holds. So exchanging data within a warp takes the `_sync` functions, and synchronizing takes an explicit `__syncwarp()` (see the CUDA handbook's [warp programming](cuda://basics/sync-warp/)).

## Tensor Cores: why they grow every generation {#tensor-core每一代为什么越做越大}

A Tensor Core does one thing, the matrix multiply-accumulate D = A × B + C. What changed each generation:

| Gen | Architecture | FP16 multiply-adds per SM per cycle | Who issues it | Where A and B come from | Where the accumulator is | New precisions |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Volta (V100) | 512 | one warp | registers | registers | FP16 |
| 2 | Turing (T4) | 512 | one warp | registers | registers | INT8, INT4 |
| 3 | Ampere (A100) | 1024 | one warp (`mma.sync`) | registers (loaded from shared memory with `ldmatrix`) | registers | BF16, TF32, 2:4 sparsity |
| 4 | Hopper (H100) | 2048 | one warpgroup, 4 warps (`wgmma`, asynchronous) | B straight from shared memory, A from shared memory or registers | registers | FP8 |
| 5 | Blackwell (B200) | 4096 | one thread (`tcgen05.mma`), with two SMs able to cooperate | shared memory or Tensor Memory | Tensor Memory | FP6, FP4, block-scaled formats |

Each generation doubles the throughput while the channel feeding the data does not: shared memory is still 32 banks of 4 bytes per cycle, 128 bytes per SM per cycle. One instruction computing an M × N tile and stepping once along K reads (M + N) × K 16-bit operands and does M × N × K multiply-adds, so the bytes per multiply-add are 2(M + N) / (MN): **the larger the tile, the less data per multiply-add**. Estimating how many bytes each generation's Tensor Cores need from shared memory per SM per cycle at full speed:

```python title="operand_bw.py"
# an estimate: at full Tensor Core speed, how many bytes of A and B operands (16-bit) each SM reads from shared memory per cycle,
# and how large one accumulator (FP32) is. Shared memory is taken as 128 bytes per SM per cycle (32 banks x 4 bytes)
SMEM = 128
cases = [
    # scheme, multiply-adds per SM per cycle, the M and N one SM computes (per K step), the rows of A and columns of B this SM reads from its own shared memory
    ("Ampere mma.sync（每个 warp 算 64x64）", 1024, 64, 64, 64, 64),
    ("Hopper wgmma m64n64k16", 2048, 64, 64, 64, 64),
    ("Hopper wgmma m64n256k16", 2048, 64, 256, 64, 256),
    ("Blackwell tcgen05 m128n256k16", 4096, 128, 256, 128, 256),
    ("Blackwell 双 SM m256n256k16", 4096, 128, 256, 128, 128),   # each SM holds only half of B
]
print("字节/乘加  共享内存 B/周期  占带宽  累加器  方案")
for name, fma, m, n, a_rows, b_cols in cases:
    per_fma = 2 * (a_rows + b_cols) / (m * n)       # each K step reads (a_rows + b_cols) x 2 bytes and does m x n multiply-adds
    need = fma * per_fma
    print(f"{per_fma:8.4f} {need:12.0f} {need / SMEM:11.0%} {m * n * 4 // 1024:5d} KB  {name}")
```

```text title="output"
字节/乘加  共享内存 B/周期  占带宽  累加器  方案
  0.0625           64         50%    16 KB  Ampere mma.sync（每个 warp 算 64x64）
  0.0625          128        100%    16 KB  Hopper wgmma m64n64k16
  0.0391           80         62%    64 KB  Hopper wgmma m64n256k16
  0.0234           96         75%   128 KB  Blackwell tcgen05 m128n256k16
  0.0156           64         50%   128 KB  Blackwell 双 SM m256n256k16
```

- on Ampere, a warp computes a 64 × 64 tile with several `mma.sync`s, loading the operands into registers with `ldmatrix` and reusing them there, which takes half the shared memory's bandwidth;
- Hopper doubles the throughput, and the same 64 × 64 tile would take all of the shared memory's bandwidth with nothing left for any other access. So `wgmma` grows the instruction to 4 warps computing 64 × N (N up to 256), and N = 256 brings it to 62%;
- Blackwell doubles it again, and even a 128 × 256 tile on one SM takes 75%; with two SMs cooperating on 256 × 256, each SM holds only half of B and it returns to 50%. That is why two-SM matrix multiplies (a CTA pair) exist.

A larger tile means a larger accumulator. A 128 × 256 FP32 accumulator is 128 KB: in registers, each thread of a 128-thread warpgroup would need 256 registers, past the limit of 255 per thread, let alone leaving registers for anything else. Every matrix multiply instruction also reads the whole accumulator, adds and writes it back, and doubling the throughput doubles that register bandwidth too. So Blackwell gave each SM 256 KB of **Tensor Memory** (128 rows × 512 columns × 32 bits), where the accumulator lives, which holds exactly two 128 × 256 accumulators: one being multiplied into while the other is read out for the epilogue (adding a bias, converting the precision, writing to memory), the two overlapping.

How it is issued changed accordingly: the larger the tile, the more work in one instruction, and a whole warp is not needed to issue it. On Blackwell one thread issues `tcgen05.mma`, the TMA moves the data into shared memory, the result sits in Tensor Memory, and the computing threads are left only "issue" and "epilogue". The structure of high-performance GEMM and attention thus moved from Ampere's "every warp moves data and computes" to Hopper's and Blackwell's [warp specialization](cuda://advanced/async-hopper/#wgmma-与-warp-专门化-sm_90a): some warps only move data, some only issue matrix multiplies, some only do the epilogue.

## The special function units and attention's exp {#特殊函数单元与注意力里的-exp}

exp, log and rsqrt are computed by the special function units (SFUs), of which an SM does only 16 per cycle. On an H100 that is 132 × 16 × 1.83 GHz ≈ 3.9 T per second against the Tensor Cores' 989 TFLOPS, 250 times apart.

Ordinarily that is no problem, but every score in attention takes one exp. A score has a matrix multiply on each side (QK<sup>T</sup> and PV), 4d FLOP in all at head dimension d. At full Tensor Core speed, how many exps per SM per cycle, and what fraction of the SFU's capacity?

```python title="exp_bound.py"
# every score s = q·k in attention takes one exp, with a matrix multiply on each side, QK^T and PV, 2d FLOP each per score.
# At full Tensor Core speed, how many exps per SM per cycle? The special function units (SFUs) do only 16 per SM per cycle
cases = [
    # GPU/precision, Tensor Core FLOP per SM per cycle, SFU exps per SM per cycle
    ("H100 BF16", 4096, 16),
    ("H100 FP8", 8192, 16),
    ("B200 BF16", 8192, 16),
    ("B200 BF16，SFU 翻倍", 8192, 32),
]
dims = [(64, 64), (128, 128), (576, 512)]            # (the dimension of q·k, the dimension of v); the last is MLA's decode
print("SFU 要达到的利用率（超过 100% 就是 exp 拖了 Tensor Core 的后腿）")
print("  d=64  d=128    MLA  GPU 与精度")
for name, tc, sfu in cases:
    need = [tc / (2 * dqk + 2 * dv) / sfu for dqk, dv in dims]   # the exps needed per cycle divided by the SFU's capacity
    print(f"{need[0]:>6.0%}{need[1]:>7.0%}{need[2]:>7.0%}  {name}")
```

```text title="output"
SFU 要达到的利用率（超过 100% 就是 exp 拖了 Tensor Core 的后腿）
  d=64  d=128    MLA  GPU 与精度
  100%    50%    12%  H100 BF16
  200%   100%    24%  H100 FP8
  200%   100%    24%  B200 BF16
  100%    50%    12%  B200 BF16，SFU 翻倍
```

BF16 attention at head dimension 128 on an H100 spends half the SFU's capacity on exp, which can still hide behind the matrix multiplies; in FP8, or on a B200 with twice the Tensor Core throughput, it takes all of it, exp and the matrix multiplies become equals and neither hides the other; head dimension 64 is worse still. Hence:

- **FlashAttention-3** (Hopper) interleaves two warpgroups: while one does the softmax the other does a matrix multiply, keeping the SFUs and the Tensor Cores busy at once;
- the Blackwell implementation computes some of the exps with a polynomial approximation on the FMA units, sharing the SFU's load;
- **Blackwell Ultra** (B300) simply doubled the SFU's exponential throughput, which NVIDIA calls "attention layer acceleration" and which is the effect of the table's last row.

MLA's decode does not have this problem: its "head dimensions" are 576 and 512, so each score corresponds to far more matrix multiplication and exp takes only a small part of the SFU. Every unit in the hardware has its own throughput, and a kernel's bottleneck is whichever unit is fullest, not necessarily the Tensor Cores.

!!! interview "How to explain it"
    To explain a GPU's hardware: the chip → GPCs → TPCs → SMs; an SM has 4 partitions, each with a warp scheduler (one instruction per cycle), 64 KB of registers, FP32 units, a Tensor Core and SFUs; the SM shares the L1 / shared memory and the TMA. The peak = the SM count × the work per SM per cycle × the clock (an H100: 132 × 4096 × about 1.83 GHz ≈ 989 TFLOPS), and the Tensor Cores doubling every generation is where the growth comes from. A GPU is in-order and does not predict branches, hiding latency by switching warps, with a register file large enough to hold every resident warp's state at once; occupancy alone cannot hide hundreds of cycles of memory latency, so ILP is needed (Little's law). A Tensor Core instruction grew from the warp-level `mma.sync` to the warpgroup-level `wgmma` to the single-thread `tcgen05` because the throughput doubles while the shared memory's bandwidth does not, so only a larger tile lowers the operand bytes per multiply-add; and the larger accumulator no longer fits in registers, which is where Tensor Memory and two-SM matrix multiplies come from. Finish with the SFUs: attention's exp becomes the bottleneck in FP8 and on Blackwell, hence FlashAttention-3's interleaving and Blackwell Ultra's doubled SFU.

## Exercises {#练习}

**1. Decompose a peak.** An A100's TF32 Tensor Core peak is 156 TFLOPS and its INT8 is 624 TOPS. By this chapter's decomposition, how many TF32 and INT8 multiply-adds per SM per cycle? An H100 SXM's dense FP8 peak is 1979 TFLOPS; which clock does that correspond to?

??? success "Answer"
    The A100 with 108 SMs at 1.41 GHz: TF32 is 156e12 / (108 × 1.41e9) ≈ 1024 FLOP/cycle/SM, 512 multiply-adds, half of FP16; INT8 is 624e12 / (108 × 1.41e9) ≈ 4096 OP/cycle/SM, 2048 multiply-adds, twice FP16. The narrower the data, the more multiplies the same hardware does per cycle.

    The H100's FP8 is 4096 multiply-adds (8192 FLOP) per SM per cycle, and 1979e12 / (132 × 8192) ≈ 1.83 GHz, the same as BF16, so FP8 is exactly twice BF16.

**2. Hiding latency.** An elementwise kernel has each thread read one float, run 8 compute instructions and write one float, with a memory latency of 600 cycles. How many warps does each scheduler need at minimum? And with each thread handling 4 elements and issuing all 4 reads first? What is the ultimate bottleneck in each case?

??? success "Answer"
    By this chapter's model, a warp's round is 1 read + 600 cycles of waiting + 8 compute instructions + 1 write, issuing 10 instructions over about 610 cycles, which needs about 61 warps, far past the limit of 16, so the scheduler idles most of the time. With 4 elements per thread and the 4 reads issued together, a round issues about 40 instructions and still waits 600 cycles once, needing about 16 warps, which exactly fills it.

    But for this kind of kernel the real ceiling is memory bandwidth and not issue: each element takes 8 operations over 8 bytes moved, an arithmetic intensity of 1 FLOP/byte, far below the ridge point. The point of more ILP is to keep enough memory requests in flight to saturate the bandwidth (Little's law: the bytes in flight = bandwidth × latency).

**3. Tile size and shared memory bandwidth.** On Hopper, a kernel computes a 64 × 128 tile with `wgmma` (m64n128k16). By this chapter's model, what fraction of the shared memory's bandwidth does keeping the Tensor Cores at full speed take? And with A read from registers instead (which `wgmma` allows)?

??? success "Answer"
    The bytes per multiply-add = 2 × (64 + 128) / (64 × 128) = 0.047, and 2048 multiply-adds per SM per cycle need 96 bytes per cycle, 75% of the 128.

    With A from registers, only B comes from shared memory: 2 × 128 / (64 × 128) = 0.031, 64 bytes per cycle, 50%. FlashAttention-3 does exactly this when computing PV: P has just come out of the softmax in registers and goes straight in as the A operand, never written back to shared memory.

**4. Tensor Memory's capacity.** Blackwell has 256 KB of Tensor Memory per SM. One FP32 accumulator is M × N × 4 bytes, so with a 128 × 256 tile through `tcgen05.mma`, how many accumulators fit at once? Why is more than one needed?

??? success "Answer"
    128 × 256 × 4 = 128 KB, so 256 KB holds two. Once one accumulator is finished, it has to be read back into registers for the epilogue (scaling, adding a bias, converting to BF16, writing to memory), which takes a while; with only one, the Tensor Cores would wait for the epilogue before starting the next tile. With two in turn, one is in its epilogue while the other is already being multiplied into, and the epilogue hides behind the matrix multiply. That is the same reasoning as a multi-stage pipeline in shared memory, applied to the accumulator.

## Summary {#小结}

- [x] The chip → GPCs → TPCs → SMs → 4 partitions; for yield, a shipped part has fewer SMs than the complete chip (an H100 SXM's 132 of 144); a B200 is two chips joined into one GPU.
- [x] Each partition issues one instruction per cycle, in order and without branch prediction; the register file is large enough to hold every resident warp, which makes switching free.
- [x] The peak = the SM count × the work per SM per cycle × the clock; the Tensor Cores doubling every generation is where the growth comes from; a specification sheet distinguishes dense from sparse and SXM from PCIe.
- [x] Occupancy alone cannot hide the memory latency, so instruction-level parallelism is needed, with the work in flight estimated by Little's law.
- [x] The throughput doubles while the shared memory's bandwidth does not, which is why Tensor Core instructions keep growing; the accumulator grows with them, hence Tensor Memory, single-thread issue and two-SM matrix multiplies.
- [x] The SFUs do only 16 exps per SM per cycle, so attention is held back by exp in FP8 and on Blackwell.
