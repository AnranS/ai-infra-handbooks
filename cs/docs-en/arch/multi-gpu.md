# Multi-GPU systems: NVLink, topology, partitioning and power

<p class="lead">A model that does not fit on one card has to be split across several, and once it is split every layer has to communicate. The bandwidth inside a machine (NVLink) and between machines (the network) differ by nearly an order of magnitude, and that gap is what dictates the division of labour: tensor parallelism inside the machine, pipeline or expert parallelism between machines. How many cards one NVLink domain can connect decides whether a large mixture of experts can avoid the network entirely. This chapter covers the insides of an 8-card server and of an NVL72 rack: NVLink and NVSwitch, the topology of PCIe and the network cards, how to estimate the time a collective takes, how one card is divided among several tasks (MIG and MPS), and why power and cooling end up in the performance account.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How are the GPUs connected inside an 8-card server? Why is tensor parallelism across machines all but unworkable?
    2. How many steps does a ring all-reduce take? For decode's small messages, where does the time go?
    3. What do `NV18`, `PIX`, `NODE` and `SYS` mean in `nvidia-smi topo -m`? Why does the position of a network card relative to a GPU matter?
    4. What is the difference between MIG and MPS? If an H100 is cut into three, what does each slice's decode speed become?
    5. Why is one card slower when all 8 cards in a machine are loaded at once than when it runs alone?

??? success "Answers for the self-test (answer first, then open this)"
    1. The 8 SXM cards are fully connected through NVSwitch chips, so any two cards have the full NVLink bandwidth between them (450 GB/s one way on H100). Across machines there is only the network card: one 400 Gb/s card per GPU is about 50 GB/s, roughly 9 times lower than NVLink, and tensor parallelism does two all-reduces per layer with the communication on the critical path. So tensor parallelism stays inside the machine.
    2. 2(p-1) steps: p-1 for the reduce-scatter and p-1 for the all-gather. During decode each transfer is only a few hundred kilobytes, so the transfer time is tiny, and 28 steps on 8 cards with a few microseconds of fixed latency each become the dominant cost. That is why every engine implements its own one-shot or two-shot all-reduce.
    3. `NV18` means 18 NVLinks between the two cards; `PIX` means under the same PCIe switch chip (the closest); `NODE` means through a PCIe root bridge within one NUMA node; `SYS` means across CPU sockets (the furthest). GPUDirect RDMA has the network card read and write device memory directly, which is fastest when the card and the GPU hang off the same PCIe switch chip (`PIX`); a socket apart, the data has to cross the inter-socket link and both the bandwidth and the latency get worse.
    4. MIG cuts the SMs, the L2, the memory channels and the memory controllers into non-interfering slices in hardware, so each slice is an independent small GPU, isolated and with predictable performance. MPS lets several processes' kernels run on the whole card at once, sharing everything: weak isolation and mutual interference, but flexible and with nothing wasted. Cut into 3g.40gb (two slices per card), each slice gets about 45% of the SMs and 3/7 of the memory bandwidth, and a decode step goes from 4.2 ms to 9.8 ms.
    5. Power and cooling: one H100 SXM is capped at 700 W, 8 cards plus the CPUs exceed 10 kW, and a machine's power delivery and cooling are designed for a typical load, so at full load the GPUs clock down (`SW Power Cap` or `HW Thermal Slowdown` in `nvidia-smi`). On top of that the shared resources are contended: PCIe lanes, host memory bandwidth, and the NVSwitch's forwarding capacity.

## Inside an 8-card server {#一台-8-卡服务器的内部}

![Figure: the data paths inside an 8-card H100 server](../assets/figures/server-topology.svg){.aig-svg}

A typical HGX server:

<!-- i18n:diagram 25cfcca080 -->
```text
CPU socket 0 ──── memory                  CPU socket 1 ──── memory
   │                                          │
 PCIe switch chip x2                       PCIe switch chip x2
   │       │                                  │       │
 GPU0..3  NIC0..3                          GPU4..7  NIC4..7
   └────────── NVSwitch x4 (all-to-all between GPUs) ───────┘
```

Two distinct paths:

- **GPU to GPU over NVLink**. Each H100 has 18 NVLink 4 links, 450 GB/s one way in total (900 GB/s both ways). The 8 cards are not wired pairwise (that would take 28 links); instead each card takes its 18 links to 4 NVSwitch chips, and the switches forward. Any two cards then have the full bandwidth between them and no communication pattern collides. That is the precondition for all-to-all, the everyone-sends-to-everyone pattern of expert parallelism, to run fast.
- **GPU to the outside over PCIe**. The GPU connects to the CPU through a PCIe switch chip, and the network card hangs off the same switch. This is for **GPUDirect RDMA**: the network card reads and writes device memory directly, the data never enters host memory, and it takes a single hop inside the switch. A GPU and its partner network card have to be under the same switch chip, otherwise the data detours through the CPU or even across sockets (see [Pinned memory, DMA and NUMA](../os/pinned-numa.md)).

Use `nvidia-smi topo -m` to see the topology. The marks in its output:

| Mark | Meaning | Speed |
| --- | --- | --- |
| `NV#` | # NVLinks between the two | fastest |
| `PIX` | under the same PCIe switch chip | fast, the ideal spot for GPUDirect RDMA |
| `PXB` | through several PCIe switch chips | middling |
| `PHB` / `NODE` | through a PCIe root bridge / within one NUMA node | slow |
| `SYS` | across CPU sockets (over the inter-socket link) | slowest |

NCCL probes this table itself at startup and decides which links to use and what ring or tree to form from them; `NCCL_TOPO_DUMP_FILE` dumps the topology it found. The first step in debugging communication performance is to look at this table and at the `NCCL_DEBUG=INFO` log.

## From 8 cards to 72 {#从-8-卡到-72-卡}

NVLink's reach grows with each generation:

| Generation | Per-card one-way bandwidth | Size of one NVLink domain | Typical form |
| --- | --- | --- | --- |
| NVLink 3 (A100) | 300 GB/s | 8 cards | NVSwitch inside the machine |
| NVLink 4 (H100) | 450 GB/s | 8 cards (a few 256-card designs) | NVSwitch inside the machine |
| NVLink 5 (B200) | 900 GB/s | 72 cards (GB200/GB300 NVL72) | one domain per rack |

NVL72 puts 18 compute trays (2 Grace CPUs and 4 Blackwell GPUs each) and 9 switch trays into one rack, wired through a copper backplane, so 72 GPUs sit in one NVLink domain with NVLink bandwidth between any two. For inference this changes the deployment directly: when a large mixture of experts with hundreds of experts is spread out with expert parallelism, the all-to-all that dispatches tokens used to cross the inter-machine network and now stays on NVLink, an order of magnitude faster; the KV of a very long context can also be moved quickly between cards in the domain. The price is a power density (over 120 kW per rack) that requires liquid cooling.

## How to estimate the time a collective takes {#集合通信的时间怎么估}

Comparing several ways to do an all-reduce with the alpha-beta model (the same tool as in the inference-systems handbook):

<div class="aig-widget" data-widget="collective-cost"></div>

Estimate a collective with two terms, latency and transfer:

```python title="collective.py"
# the time a collective takes = the per-step latency x the steps + the volume / the bandwidth. Comparing NVLink inside a machine with a 400 Gb/s network between machines
LINKS = {  # name: (one-way bandwidth GB/s, per-step latency us)
    "NVLink 4": (450, 2),
    "NVLink 5": (900, 2),
    "IB NDR 400G": (50, 8),
}


def ring_allreduce(nbytes, n, bw, lat):
    """环形 all-reduce：2(n-1) 步，每步每张卡发出 nbytes/n 字节"""
    steps = 2 * (n - 1)
    return steps * lat * 1e-6 + steps * (nbytes / n) / (bw * 1e9)


print("8 卡 all-reduce 的耗时（ms）")
print(f"{'消息大小':28s}" + "".join(f"{k:>14s}" for k in LINKS))
for name, size in [("decode 一层的激活 256 KB", 256 << 10), ("prefill 一层的激活 64 MB", 64 << 20),
                   ("梯度同步 1 GB", 1 << 30)]:
    print(f"{name:26s}" + "".join(f"{ring_allreduce(size, 8, bw, lat) * 1e3:14.3f}"
                                  for bw, lat in LINKS.values()))
print()
print("同一条 256 KB 的消息，卡数从 8 张涨到 72 张（NVLink 5）：")
for n in (8, 16, 32, 72):
    t = ring_allreduce(256 << 10, n, 900, 2) * 1e3
    print(f"  {n:2d} 卡：{t:6.3f} ms，其中固定延迟 {2 * (n - 1) * 2 / 1e3:5.3f} ms（占 {2 * (n - 1) * 2 / 1e3 / t:.0%}）")
```

```text title="output"
8 卡 all-reduce 的耗时（ms）
消息大小                              NVLink 4      NVLink 5   IB NDR 400G
decode 一层的激活 256 KB                0.029         0.029         0.121
prefill 一层的激活 64 MB                0.289         0.158         2.461
梯度同步 1 GB                          4.204         2.116        37.693

同一条 256 KB 的消息，卡数从 8 张涨到 72 张（NVLink 5）：
   8 卡： 0.029 ms，其中固定延迟 0.028 ms（占 98%）
  16 卡： 0.061 ms，其中固定延迟 0.060 ms（占 99%）
  32 卡： 0.125 ms，其中固定延迟 0.124 ms（占 100%）
  72 卡： 0.285 ms，其中固定延迟 0.284 ms（占 100%）
```

Three conclusions:

- **Decode's small messages are dominated by latency**. Tensor parallelism does two all-reduces per layer, a few hundred kilobytes each, which takes only 0.03 ms inside a machine, but an 80-layer model does 160 of them per step, 4 to 5 ms in total, the same order as the compute time. That is why every engine implements its own small-message all-reduce: a one-shot algorithm (each card reads and sums the others' buffers directly) or a two-shot one (reduce-scatter then all-gather) uses NVLink's direct peer-to-peer access to compress 28 steps into one or two, bringing the latency down to a few microseconds, and fuses it with the RMSNorm that follows (see [Tensor parallelism](serving://distributed/tensor-parallel/#通信的代价)).
- **Prefill's large messages are dominated by bandwidth**: 0.16 to 0.29 ms inside a machine against 2.5 ms between machines, a factor of nearly 10. That is the root reason tensor parallelism does not cross machines.
- **The ring algorithm's fixed latency grows linearly with the card count**. A ring over 72 cards takes 142 steps, and a small message is then almost entirely latency. So large-scale communication does not use a ring: NVSwitch can reduce inside the switch chip (NVLS) and finish in one step, and NCCL also switches to a tree algorithm for small messages.

Two more terms worth remembering: `nccl-tests` reports **algbw** (the data volume divided by the time) and **busbw** (converted to the bandwidth each link actually carries; for an all-reduce busbw = algbw x 2(p-1)/p). Compare against the hardware peak using busbw.

## Dividing one card among several tasks {#一张卡切给多个任务}

Sometimes a model is small and several instances fit on one card. Two ways to do it:

**MIG (Multi-Instance GPU)**, supported from Ampere on. It cuts the GPU into up to 7 slices in hardware: the SMs, the paths into the L2, the memory controllers and the memory are all divided by slice, and each slice is an independent small GPU with its own device id, fully isolated from the others, so a fault in one does not affect the rest. The price is a fixed granularity (only the preset profiles), a reconfiguration to change it, and **a share of the bandwidth to match**:

```python title="sharing.py"
# several model instances on one card: the accounts for MIG (a hardware partition) and MPS (sharing)
# MIG divides the SMs, the L2 and the memory controllers by slice; MPS puts several processes' kernels on one whole card at once
SM, L2, MEM, BW = 132, 50, 80, 3350        # H100: SM count, L2 (MB), memory (GB), bandwidth (GB/s)
print("MIG 切分（H100 的 7 种档位里挑 3 种）：")
for name, slices, sms, mem in [("1g.10gb", 1, 16, 10), ("2g.20gb", 2, 32, 20), ("3g.40gb", 3, 60, 40)]:
    print(f"  {name:8s} {sms:3d} 个 SM（{sms / SM:4.0%}） 显存 {mem} GB  带宽约 {BW * slices / 7 / 1000:.2f} TB/s  "
          f"一张卡能切 {7 // slices} 个")
print()
print("同一个 7B 模型（BF16 权重 14 GB，decode 读一遍权重）：")
for name, sms, mem, share in [("整卡", SM, MEM, 1.0), ("3g.40gb", 60, 40, 3 / 7), ("2g.20gb", 32, 20, 2 / 7)]:
    bw = BW * share
    rest = f"还剩 {mem - 14:.0f} GB 放 KV" if mem > 14 else "放不下权重"
    print(f"  {name:8s} 带宽 {bw / 1000:.2f} TB/s  decode 一步下限 {14 / bw * 1000:5.1f} ms  {rest}")
```

```text title="output"
MIG 切分（H100 的 7 种档位里挑 3 种）：
  1g.10gb   16 个 SM（ 12%） 显存 10 GB  带宽约 0.48 TB/s  一张卡能切 7 个
  2g.20gb   32 个 SM（ 24%） 显存 20 GB  带宽约 0.96 TB/s  一张卡能切 3 个
  3g.40gb   60 个 SM（ 45%） 显存 40 GB  带宽约 1.44 TB/s  一张卡能切 2 个

同一个 7B 模型（BF16 权重 14 GB，decode 读一遍权重）：
  整卡       带宽 3.35 TB/s  decode 一步下限   4.2 ms  还剩 66 GB 放 KV
  3g.40gb  带宽 1.44 TB/s  decode 一步下限   9.8 ms  还剩 26 GB 放 KV
  2g.20gb  带宽 0.96 TB/s  decode 一步下限  14.6 ms  还剩 6 GB 放 KV
```

Decode is bandwidth-bound, and after cutting to 3g.40gb only 43% of the bandwidth is left, so a step goes from 4.2 ms to 9.8 ms. MIG therefore suits many unrelated small tasks that have to be isolated from each other (multi-tenant small-model serving, CI) and does not suit large-model inference where single-request latency matters.

**MPS (Multi-Process Service)** is the other route: several processes' kernels are submitted to the same hardware scheduler and can genuinely occupy the SMs at the same time (without MPS, different processes' kernels are time-sliced). It does not partition resources, so there is no isolation: one process can fill the memory or take all of the SMs, and one crashing process can affect the others. The upside is flexibility with nothing wasted, and `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` roughly caps the share of SMs a process may use.

Inference engines have a third route internally: **dividing the SMs within one process using separate streams and a [green context](serving://frontier/pd-multiplex/)**, so prefill and decode run in parallel on one card and fill each other's gaps. This is more flexible than MIG and more controllable than MPS.

## Power, cooling and clocks {#功耗散热与频率}

The performance account finally comes down to electricity and heat:

- **The per-card power cap**: 400 W for A100 SXM, 700 W for H100 SXM, about 1000 W for B200, 2700 W for one GB200 superchip. `nvidia-smi -q -d POWER` shows the current draw and the cap, and `nvidia-smi --query-gpu=clocks_throttle_reasons.active` shows why it clocked down.
- **Full load clocks down**. A matrix multiply that keeps the Tensor Cores busy draws the most power, and the clock drops below the boost clock. This is one reason measured compute is usually only 70 to 80 percent of the sheet's peak (we saw it in the previous chapter when working the clock back from the specifications: H100's FP32 peak implies 1.98 GHz, but its Tensor Core peak only corresponds to 1.83 GHz).
- **8 cards are not 8 times one card**. The machine's power delivery, its cooling, the NVSwitch's forwarding capacity, the PCIe lanes and the host memory bandwidth are all shared. A load test has to load the whole machine; you cannot take a single card's number and multiply by 8.
- **Liquid cooling is not only about energy**. An air-cooled rack cannot hold a power density like NVL72's; liquid cooling also keeps the GPUs at a higher clock, so the actual throughput is higher.

In capacity planning these turn into cost directly: a data centre's per-rack power cap is often lower than what a full rack of cards would draw, and how many cards fit is decided by power and cooling, not by rack space.

!!! interview "How to answer in an interview"
    On multi-GPU systems: inside an 8-card machine the GPUs are fully connected through NVSwitch (450 GB/s one way on H100), and a GPU and its network card hang off the same PCIe switch chip so that GPUDirect RDMA works; across machines there is only about 50 GB/s, a factor of about 9, so tensor parallelism stays inside the machine and pipeline or expert parallelism goes across. Estimate communication as "latency x steps + volume / bandwidth": a ring all-reduce is 2(p-1) steps, decode's small messages are latency-dominated (160 all-reduces in one step reach 4 to 5 ms), hence the custom one-shot/two-shot all-reduces and reduction inside the NVSwitch; prefill's large messages are bandwidth-dominated. The NVLink domain grew from 8 cards to NVL72's 72, so large-scale expert parallelism's all-to-all no longer crosses the network. On partitioning: MIG isolates in hardware and divides the bandwidth with it (cut an H100 into three and decode goes from 4.2 ms to 9.8 ms), MPS shares without isolating, and inside an engine a green context can run prefill and decode in parallel. Finish with power: 700 W to 1000 W per card, clocking down at full load, a machine that is not 8 times one card, and capacity planning limited by power and cooling.

!!! info "Related chapters"
    - [Multi-GPU and NCCL](cuda://tools/multi-gpu/) (CUDA: writing multi-GPU programs with NCCL)
    - [GPU interconnect and networking](serving://comm/interconnect/), [Collective communication: NCCL's algorithms and protocols](serving://comm/nccl/) (inference systems)
    - [Collective primitives](train://basics/collectives/) (distributed training)

## Exercises {#练习}

**1. The communication account for tensor parallelism.** A 70B model (80 layers, hidden 8192) is deployed with TP=8 in BF16. During decode the batch is 64 and each layer does two all-reduces. Using a ring algorithm over NVLink 4 inside the machine (450 GB/s one way, 2 us per step), how long does the communication in one decode step take? And with a one-shot algorithm at 5 us?

??? success "Answer"
    Each all-reduce moves 64 x 8192 x 2 bytes = 1 MB. The ring: 28 steps x 2 us = 56 us of latency plus 28 x (1 MB / 8) / 450 GB/s, about 8 us of transfer, so about 64 us; twice per layer over 80 layers is 160 of them, about 10 ms. One-shot is about 5 us plus the transfer each time, so 160 of them is a little over 1 ms.

    The compute in a decode step is itself only a few milliseconds, so 10 ms of communication is completely unacceptable and 1 ms is tolerable. That is why a custom all-reduce is indispensable.

**2. Debugging the topology.** A cross-machine expert-parallel deployment measures only half the expected all-to-all bandwidth. `nvidia-smi topo -m` shows `SYS` between GPU4 and the network card it uses. What could be wrong, and how would you fix it?

??? success "Answer"
    `SYS` means the GPU and the network card belong to two different CPU sockets, so the data has to cross the inter-socket link: the bandwidth is limited, the latency higher, and the benefit of GPUDirect RDMA much reduced. The fix: have each GPU use the network card under its own PCIe switch chip (NCCL's `NCCL_IB_HCA` names the card, or set the affinity per rank), and bind the process to the matching NUMA node with `numactl`. If the machine simply does not have enough network cards, either accept it or change the rank-to-GPU mapping so that the ranks with the most traffic get the well-placed cards.

**3. Whether to partition.** A 1.5B small-model service has a latency requirement of 30 ms per token, modest QPS, and a requirement that tenants be isolated from each other. You have an H100. Is MIG into 7 slices workable (16 SMs, 10 GB and about 0.48 TB/s each)? Do the arithmetic.

??? success "Answer"
    A 1.5B model's BF16 weights are 3 GB, which fits in 10 GB with 7 GB left for KV. A decode step has to read the weights at least once: 3 GB / 0.48 TB/s is about 6.3 ms, well under the 30 ms requirement, so yes (with attention and kernel-launch overhead on top, the measured figure is on the order of ten-odd milliseconds). This is exactly MIG's scenario: a small model, a strong isolation requirement, and headroom in the single-request latency. Conversely, if the model were 70B, no MIG slice could hold the weights at all and you would need a whole card or several.

**4. Why it is not 8 times.** One model instance on one card measures 1000 token/s. Starting 8 independent instances on the same 8-card machine (one per card, no parallelism) will most likely not reach 8000. List at least three reasons.

??? success "Answer"
    (1) Power and cooling: 8 cards at full load hit the machine's power and thermal caps and the GPUs clock down. (2) Shared host-side resources: 8 processes' scheduling, tokenisation and HTTP handling share the CPUs and the memory bandwidth, and the CPU may become the bottleneck first (especially obvious for a small model, where a GPU step takes only a few milliseconds). (3) PCIe and host memory: weight loading, KV offload, logging and monitoring all go over PCIe and the instances contend. (4) NUMA: the 8 processes are not necessarily each bound to the socket their GPU is on, and cross-socket memory access is slower. (5) If the instances also share a front end or a router, that may saturate first. To locate it, look at the utilisation and throttle reasons in `nvidia-smi` first, then at the CPU utilisation and `numastat`.

## Summary {#小结}

- [x] Inside an 8-card machine the GPUs are fully connected through NVSwitch (450 GB/s one way on H100), and a GPU and its network card share a PCIe switch chip so GPUDirect RDMA works; across machines it is about 50 GB/s, a factor of about 9.
- [x] `nvidia-smi topo -m`: `NV#` > `PIX` > `PXB` > `NODE` > `SYS`. NCCL builds its rings from this table, so start debugging with it.
- [x] Communication time = latency x steps + volume / bandwidth. A ring all-reduce is 2(p-1) steps, decode's small messages are latency-dominated and need one-shot/two-shot or reduction inside the NVSwitch.
- [x] The NVLink domain grew from 8 cards to NVL72's 72, so large-scale expert parallelism's all-to-all no longer crosses the network, at the price of a power density that requires liquid cooling.
- [x] MIG partitions in hardware (and divides the bandwidth with it: cut an H100 into three and decode is more than twice as slow), MPS shares without isolating, and inside an engine a green context can run prefill and decode in parallel.
- [x] 700 to 1000 W per card, clocking down at full load, and a machine's throughput is not 8 times one card's; capacity planning is often limited by power and cooling.
