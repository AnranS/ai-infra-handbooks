# A GPU's memory system: from registers to HBM

<p class="lead">Every token decode generates reads the weights and the KV cache out of device memory once, so inference's speed is first of all the memory bandwidth's speed. This chapter walks a GPU's memory system from the bottom up: where HBM's bandwidth comes from and why it is far wider than GDDR's; what the ratio of capacity to bandwidth sets as decode's ceiling; how the levels differ in capacity, bandwidth and latency; how much data has to be in flight to saturate the bandwidth and how that forced cp.async and the TMA into existence; and finally a GPU's own virtual memory and the paths between CPU and GPU memory.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Which two numbers multiply to an H100 SXM's 3.35 TB/s of memory bandwidth? Why can HBM reach a bandwidth GDDR cannot?
    2. With the memory full, how long does reading all of it once take? What does that mean for decode? What does it mean that a B300 has half again the memory of a B200 at the same bandwidth?
    3. How much on-chip storage (registers, shared memory, L2) does an H100 have in total? How many times smaller is that than the device memory?
    4. How much data has to be in flight to saturate an H100's bandwidth? Per SM and per thread? Why does that force the TMA into existence?
    5. What is the smallest granularity of a memory access? Why does DRAM dislike small random accesses internally?

??? success "Answers (try it yourself first, then expand)"
    1. The bus width × the per-pin rate: 5 HBM3 stacks of 1024 bits each, 5120 bits, at about 5.2 Gb/s per pin, 5120 × 5.23 / 8 ≈ 3350 GB/s. HBM stacks DRAM dies vertically and places them with the GPU on a silicon interposer, with short dense wiring giving each stack a 1024-bit interface; GDDR chips are soldered on a board with long traces, so a card's bus is only 384 to 512 bits and has to compensate with a higher per-pin rate (over 20 Gb/s).
    2. The capacity ÷ the bandwidth: an H100's 80 GB / 3.35 TB/s ≈ 24 ms. Every decode step reads all the weights and all of this batch's KV, so with the memory full a step takes at least that long and each request generates at most about 40 tokens a second. A B300's 288 GB holds half again as much KV, giving larger batches and more throughput, but reading it all takes 36 ms when full and each request is slower: capacity buys throughput and bandwidth decides speed.
    3. Registers 132 × 256 KB ≈ 33 MB, L1 / shared memory likewise about 33 MB, L2 50 MB, over 100 MB in all; against 80 GB of device memory, about 700 times smaller. The weights do not fit on chip, so every decode step streams them out of device memory.
    4. The bandwidth × the latency: estimating the access latency at 600 ns, 3.35 TB/s × 600 ns ≈ 2 MB, about 15 KB per SM, and with all 2048 threads per SM resident that is about 7.4 bytes in flight per thread, about 16 on a B200. An ordinary load occupies a register to receive the data and the thread count is capped, so "a few more reads per thread" already strains; cp.async lets the data bypass the registers straight into shared memory, and the TMA goes further, one thread issuing one instruction that moves a whole tile (a few to tens of KB), so the data in flight is no longer bounded by threads and registers.
    5. The smallest granularity is a 32-byte sector (L2's cache line is 128 bytes, four sectors). DRAM is organized in rows, and reading one activates it into a row buffer first, so contiguous data within a row is fast while changing rows takes closing the current one and activating the new one, tens of nanoseconds extra. Small random accesses change rows nearly every time and the effective bandwidth falls sharply.

## HBM: stacking memory beside the GPU {#hbm把内存堆在-gpu-旁边}

The memory bandwidth = the bus width × the per-pin rate. The two kinds of memory took two roads:

- **GDDR** (consumer cards, the L40S and others): memory chips soldered on the board around the GPU, 32 bits each. The traces are long and the bus cannot be made wide (384 to 512 bits), so the per-pin rate has to be pushed past 20 Gb/s;
- **HBM** (data centre cards): 8 to 12 DRAM dies stacked vertically and joined by through-silicon vias, each stack with a 1024-bit interface; the stacks and the GPU sit on one silicon interposer (TSMC's CoWoS packaging) with short dense wiring. A few stacks make a bus thousands of bits wide, so the per-pin rate need not be as high and the energy per bit is lower.

Working the per-pin rate back from the specifications, and computing a number that is very useful in inference, how long reading all of the memory once takes:

```python title="hbm.py"
# the memory bandwidth = the bus width x the per-pin rate. Working the per-pin rate back from the specifications, then how long reading all of the memory takes
gpus = [
    # name, memory type, bus width (bits), capacity GB, the sheet's bandwidth GB/s
    ("A100 80GB", "HBM2e x5", 5 * 1024, 80, 2039),
    ("H100 SXM", "HBM3 x5", 5 * 1024, 80, 3350),
    ("H200", "HBM3e x6", 6 * 1024, 141, 4800),
    ("B200", "HBM3e x8", 8 * 1024, 180, 8000),
    ("B300", "HBM3e x8", 8 * 1024, 288, 8000),
    ("MI300X", "HBM3 x8", 8 * 1024, 192, 5300),
    ("RTX 4090", "GDDR6X", 384, 24, 1008),
    ("RTX 5090", "GDDR7", 512, 32, 1792),
]
print("GPU        显存        总线宽度  每引脚速率    带宽    读满一遍")
for name, kind, bits, cap, bw in gpus:
    pin = bw * 8 / bits                              # GB/s x 8 / the bus width = Gb per second per pin
    print(f"{name:10s} {kind:10s} {bits:7d} 位 {pin:6.2f} Gb/s {bw / 1000:6.2f} TB/s {cap / bw * 1000:6.0f} ms")
```

```text title="output"
GPU        显存        总线宽度  每引脚速率    带宽    读满一遍
A100 80GB  HBM2e x5      5120 位   3.19 Gb/s   2.04 TB/s     39 ms
H100 SXM   HBM3 x5       5120 位   5.23 Gb/s   3.35 TB/s     24 ms
H200       HBM3e x6      6144 位   6.25 Gb/s   4.80 TB/s     29 ms
B200       HBM3e x8      8192 位   7.81 Gb/s   8.00 TB/s     22 ms
B300       HBM3e x8      8192 位   7.81 Gb/s   8.00 TB/s     36 ms
MI300X     HBM3 x8       8192 位   5.18 Gb/s   5.30 TB/s     36 ms
RTX 4090   GDDR6X         384 位  21.00 Gb/s   1.01 TB/s     24 ms
RTX 5090   GDDR7          512 位  28.00 Gb/s   1.79 TB/s     18 ms
```

(The H200's and the B200's usable capacity is slightly below the stacks' physical capacity, and the table uses the usable figure.)

"Reading it all once" is a hard boundary for decode. Every decode step reads all the weights and all of this batch's KV cache. The fuller the memory (a larger batch, a longer context), the more each step reads; with it full, a step takes at least the time to read it all, so each request generates at most 1 / 24 ms ≈ 40 tokens a second (an H100). This number is between 20 and 40 ms for several generations of data centre card, a balance between capacity and bandwidth struck when the hardware was designed.

The B200 to the B300 is a good example: the stacks went from 8 layers to 12, the capacity from 180 GB to 288, and the bandwidth stayed at 8 TB/s. The extra capacity holds more requests' KV, giving larger batches and more throughput; but with the memory full, reading it all takes 36 ms and each request generates more slowly. **Capacity buys throughput and bandwidth decides speed**, and how full to run is decided at deployment by the latency requirement (see the Inference Systems handbook's [load testing, SLOs and capacity planning](serving://perf/benchmark/)).

HBM's evolution these past years has been the stack height, the per-pin rate and the number of stacks all rising together: HBM2e → HBM3 → HBM3e, with the next generation, HBM4, widening each stack's interface from 1024 bits to 2048.

### Inside DRAM: why it dislikes random access {#dram-内部为什么怕随机访问}

A DRAM die is divided into banks, each an array of rows and columns. Reading takes three steps: **activate** (read a whole row, about 1 KB, into that bank's row buffer), **read a column** (take the part wanted out of the row buffer) and **precharge** (close that row to prepare the next activation). Reading on within the same row takes only the second step; moving to another row takes all three and tens of nanoseconds more.

So DRAM likes large contiguous accesses: the memory controller reorders requests to group those landing in one row; and a GPU accesses device memory in 32-byte sectors, so a warp's 32 threads reading contiguous addresses need only 4 sectors (128 bytes), which is [coalesced access](cuda://basics/memory/#全局内存合并访问). Small random accesses (reading KV scattered by token, looking up an enormous embedding table) both waste the unused bytes of a sector and make DRAM change rows constantly, leaving a fraction of the peak bandwidth. That is one reason PagedAttention's blocks cannot be too small.

## The levels: capacity, bandwidth and latency {#各级存储容量带宽与延迟}

![Figure: a GPU's memory hierarchy - registers, shared memory, L2, HBM](../assets/figures/memory-hierarchy.svg){.aig-svg}

The whole hierarchy on an H100:

| Level | Capacity (a whole H100 SXM) | Bandwidth | Latency | Managed by |
| --- | --- | --- | --- | --- |
| registers | 256 KB per SM, about 33 MB | the highest | a cycle | the compiler |
| L1 / shared memory | 256 KB per SM, about 33 MB | 128 bytes per SM per cycle, about 30 TB/s per card | tens of cycles | shared memory by the programmer, L1 by the hardware |
| L2 | 50 MB in two partitions | several times the memory bandwidth | about 200 cycles, more for the far partition | the hardware; part can be made persistent |
| HBM device memory | 80 GB | 3.35 TB/s | about 400 to 800 cycles | `cudaMalloc`, PyTorch's caching allocator |
| host memory (PCIe 5.0 x16) | hundreds of GB to a few TB | about 64 GB/s one way | microseconds | pinned memory + DMA |
| another GPU (NVLink 4) | — | 450 GB/s one way | microseconds | peer access, NCCL |

A few ratios decide how an inference kernel is written:

- **the device memory is about 700 times the on-chip storage**. An 8B model's BF16 weights alone are 16 GB, which does not fit on chip, so every decode step streams the weights out of device memory. Once a weight is read in, it has to be used as many times as possible on chip: with B requests in the batch each weight is used B times and the arithmetic intensity grows linearly with B, which is why batching matters so much for decode and what [the ridge point](serving://career/hardware/#由参数推出的推理特性) means physically.
- **shared memory's bandwidth is about 10 times the device memory's**, and the registers' higher still. So a high-performance kernel's formula is always: read from device memory once, put it in shared memory and registers, and use it repeatedly on chip.
- **L2 cannot hold the weights but holds a great deal else**: a decode step's activations per layer, small lookup tables, KV blocks being read by several blocks at once. From the A100, an access policy window (`cudaAccessPolicyWindow`) can mark a range "persisting" so it stays in L2 preferentially. L2 is split into two partitions and each SM reaches the nearer one, with the other across a crossbar at higher latency; where data sits in L2 is beyond an ordinary program's control.
- **leaving the card drops the bandwidth by an order of magnitude**: NVLink is about 7 times slower than device memory and PCIe about 7 times slower again. Offloading KV to host memory and loading weights from the host are estimated at these bandwidths.

## Data in flight: Little's law and the TMA {#在途的数据little-定律与-tma}

Saturating the memory bandwidth needs enough requests in flight at once. By Little's law, the data in flight = the bandwidth × the latency. Estimating the access latency at 600 ns:

Move the bandwidth and the latency:

<div class="aig-widget" data-widget="little-law"></div>

```python title="in_flight.py"
# Little's law: sustaining a bandwidth B needs data in flight (issued and not yet returned) = B x the latency. The access latency is estimated at 600 ns (an order of magnitude)
LATENCY = 600e-9
gpus = [("A100", 2039e9, 108), ("H100 SXM", 3350e9, 132), ("B200", 8000e9, 148)]
print("GPU        整卡在途   每个 SM   每个线程（每 SM 2048 个线程全满）")
for name, bw, sms in gpus:
    total = bw * LATENCY
    per_sm = total / sms
    print(f"{name:9s} {total / 1e6:6.2f} MB {per_sm / 1024:7.1f} KB {per_sm / 2048:9.1f} 字节")
```

```text title="output"
GPU        整卡在途   每个 SM   每个线程（每 SM 2048 个线程全满）
A100        1.22 MB    11.1 KB       5.5 字节
H100 SXM    2.01 MB    14.9 KB       7.4 字节
B200        4.80 MB    31.7 KB      15.8 字节
```

On an A100, each SM has to keep about 11 KB in flight, and with 2048 threads resident that is about 5.5 bytes per thread, more than one float, so a kernel reading one float per thread cannot saturate an A100's bandwidth (the CUDA handbook's [hiding latency](cuda://basics/execution/#延迟掩盖) works this out). By the B200, the bandwidth has risen nearly fourfold against 40% more SMs, so each thread needs about 16 bytes in flight, which amounts to **full occupancy with a `float4` load outstanding in every thread at all times**.

An ordinary load has a hidden cost too: the data lands in registers and every byte in flight occupies one, while the registers are also needed for computing. The hardware's answer is to let the data bypass the registers:

- **Ampere's `cp.async`**: an asynchronous copy from device memory straight into shared memory, bypassing the registers, after which the thread can do something else;
- **Hopper's TMA**: one thread issues one instruction that moves a whole multidimensional tile (a 128 × 64 BF16 tile, 16 KB), with the address arithmetic and the bounds handled by the hardware. Two or three such instructions keep tens of KB in flight for an SM, entirely free of the thread and register limits.

That is why high-performance kernels on Hopper and Blackwell have all become "one or two warps moving data with the TMA while the others compute" (see the CUDA handbook's [asynchronous copies and the TMA](cuda://advanced/async-hopper/)). The bandwidth rising faster than the SM count is the fundamental reason these new hardware units exist.

## A GPU's virtual memory {#gpu-的虚拟内存}

A GPU has an MMU, page tables and a TLB of its own. Every CUDA context has its own virtual address space, `cudaMalloc` returns a virtual address, and a GPU's accesses are translated too. Device memory is generally mapped in 2 MB huge pages, so the TLB reaches far further than a CPU's 4 KB pages; but random access spanning tens of GB (reading scattered through an enormous KV pool) still misses the TLB and slows the accesses.

As on a CPU, virtual memory separates "allocating an address" from "allocating physical memory", which has a direct use in an inference system: CUDA's virtual memory management interface (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`) reserves a large contiguous range of virtual addresses and maps physical memory into it on demand at a 2 MB granularity. That is a different layer from PagedAttention: PagedAttention is a block table in software that the attention kernel consults itself, while the virtual memory interface changes the hardware's page tables and the kernel sees contiguous addresses. The two are compared in [virtual memory, page tables and huge pages](../os/virtual-memory.md).

**Unified memory** (`cudaMallocManaged`) gives the CPU and the GPU one address space, with an access to a page that is not local faulting and the driver migrating it. It is convenient to write and the faults and migrations are very slow, so an inference system's hot path generally avoids it.

## The paths between the CPU and the GPU {#cpu-与-gpu-之间的通路}

PCIe is the usual path between a CPU and a GPU: PCIe 5.0 x16 is about 64 GB/s one way, 50 times slower than device memory. That decides how fast the KV cache can offload to host memory and how long a cold start's weight loading takes (the estimate is in [pinned memory, DMA and NUMA](../os/pinned-numa.md)).

NVIDIA's Grace Hopper (GH200) and Grace Blackwell (GB200) join the CPU and the GPU with NVLink-C2C at 900 GB/s (bidirectional in total), about 7 times PCIe 5.0, and the hardware keeps CPU memory and device memory coherent: the GPU can read and write the hundreds of GB of LPDDR5X on the CPU's side directly and efficiently. For inference that amounts to another large, not-too-slow tier of "secondary device memory", where the KV cache, a very long context or part of a large model's weights can live.

!!! interview "Answering in an interview"
    Asked about memory and bandwidth: the bandwidth = the bus width × the per-pin rate, and HBM gets a 1024-bit interface per stack from stacking and a silicon interposer, so an H100's 5 stacks of 5120 bits at about 5.2 Gb/s per pin give 3.35 TB/s. The capacity ÷ the bandwidth is decode's hard boundary: reading an H100's memory once takes about 24 ms, and the fuller it is the slower each step; a B300's half-again capacity at the same bandwidth trades speed for throughput. In the hierarchy, the registers, the shared memory and L2 together are over 100 MB against 80 GB of device memory, about 700 times smaller, so a kernel reads from device memory once and reuses it on chip, and decode raises the weights' reuse through batching. Saturating the bandwidth requires Little's law: the data in flight = the bandwidth × the latency, about 2 MB on an H100 and 15 KB per SM, doubled on a B200; the bandwidth rising faster than the SM count leaves the registers unable to hold it, hence cp.async bypassing them and the TMA moving a whole tile in one instruction. DRAM is organized in rows and dislikes small random accesses, and a GPU's 32-byte sectors call for coalescing.

## Exercises {#练习}

**1. Working it back and estimating.** An H20 has 96 GB of HBM3 at 4.0 TB/s. With 1024 bits per stack at about 5.2 Gb/s per pin, roughly how many stacks does it have? How long does reading it all once take? Deploying a model with 16 GB of BF16 weights on it and giving the rest to KV, how long does one decode step take at minimum with the memory full?

??? success "Answer"
    4000 GB/s × 8 / 5.2 Gb/s ≈ 6150 bits, about 6 stacks (6 × 16 GB = 96 GB, which matches the capacity too). Reading it all is 96 / 4000 ≈ 24 ms. With the memory full, a step reads 16 GB of weights plus about 80 GB of KV (less the activations and the runtime's small share), close to reading it all, so twenty-odd milliseconds. Whatever batch fits in that step shares those twenty-odd milliseconds, so the throughput can be high while each request gets at most about 40 tokens a second.

**2. How much the on-chip storage helps.** An H100's L2 is 50 MB. In one decode step, an 8B model's weights per layer are a few hundred MB (16 GB over 32 layers). Can L2 save the next decode step from reading device memory? Why? What does L2 help with in decode?

??? success "Answer"
    It cannot. Each step reads 16 GB of weights and L2 is 50 MB, so after streaming through once, what L2 holds is the little that was read last, long since replaced by the time the next step starts at the first layer, and the next step reads it all from device memory again. What L2 helps with is reuse within one step: the same weights read by several blocks at once (a matrix multiply split along N across blocks reads the same input activations), and the same KV block read by several query heads (a group of query heads sharing one KV head under GQA), where the repeated reads hit in L2 rather than going to device memory every time.

**3. Data in flight.** An elementwise kernel on an H100 has 1024 threads resident per SM (50% occupancy) and each thread reads one BF16 at a time. By this chapter's estimate (about 15 KB in flight per SM), what fraction of the peak bandwidth can it reach? And reading 16 bytes (8 BF16) per thread?

??? success "Answer"
    About 1024 × 2 = 2 KB in flight per SM, about 13% of the 15 KB needed, so at most about 13% of the peak bandwidth (Little's law: the bandwidth = the data in flight / the latency). Reading 16 bytes per thread puts 16 KB in flight, past the 15 KB, so it can saturate it in principle (eighty or ninety per cent of the peak in practice). That is the quantitative reason an elementwise kernel reads vectorized; and it also shows that 50% occupancy is no obstacle to saturating the bandwidth as long as each thread has enough in flight.

**4. Choosing a path.** A request's KV cache is 4 GB and has to be moved into an H100's device memory from elsewhere. Estimate the time from local host memory (PCIe 5.0 x16), from the CPU's side on a GH200 (NVLink-C2C) and from another GPU in the same machine (NVLink 4).

??? success "Answer"
    By one-way bandwidth: PCIe 5.0 x16 is about 64 GB/s (about 50 in practice), so 4 GB takes 60 to 80 ms; NVLink-C2C's 900 GB/s is bidirectional in total, about 450 one way, so about 9 ms; NVLink 4 is 450 GB/s one way, likewise about 9 ms. The same request's KV takes an order of magnitude longer to restore from the PCIe side than from the NVLink side, which decides whether offloading KV is worth it: how long recomputing that request's prefill takes is the ceiling offloading has to beat.

## Summary {#小结}

- [x] The bandwidth = the bus width × the per-pin rate: HBM gets a 1024-bit interface per stack from stacking and a silicon interposer while GDDR relies on a high per-pin rate; HBM4 widens the interface to 2048 bits.
- [x] The capacity ÷ the bandwidth is decode's hard boundary (about 24 ms on an H100); capacity buys throughput and bandwidth decides speed.
- [x] DRAM is organized in rows and dislikes small random accesses; a GPU's granularity is a 32-byte sector, which calls for coalescing.
- [x] The on-chip storage is about 100 MB and the device memory 700 times that: read from device memory once and reuse on chip; decode raises the weights' reuse through batching.
- [x] The data in flight = the bandwidth × the latency, tens of KB per SM; the bandwidth rising faster than the SM count is what brought cp.async and the TMA.
- [x] A GPU has page tables and a TLB of its own, mapped in 2 MB huge pages; the virtual memory interface changes the hardware's page tables while PagedAttention is a software block table. PCIe is about 50 times slower than device memory, and NVLink-C2C makes CPU memory a large, not-too-slow second tier.
