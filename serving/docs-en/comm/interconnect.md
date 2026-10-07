# GPU interconnects and networks: NVLink, PCIe and RDMA networks

<p class="lead">Tensor parallelism, expert parallelism and PD disaggregation from the previous chapters all come down to "moving data from one GPU to another". How fast depends on the path taken: NVLink or PCIe within one machine, or an InfiniBand / RoCE network between machines. This chapter lays out the data paths in an 8-GPU server and in a cluster, uses the α-β model to explain "why small messages are always slow", then looks at the rail topology of cluster networks and GPUDirect, which lets NICs read and write GPU memory directly.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. In an 8-GPU H100 server, what are the orders of magnitude of bandwidth between GPUs, between a GPU and a NIC, and between machines?
    2. Why does tensor parallelism's all-reduce in the decode phase often "fail to use the bandwidth"?
    3. What is a rail-optimized topology? Why is cross-node communication best sent to the GPU "in the same position"?
    4. What problem does GPUDirect RDMA solve? What does it require of the hardware topology?
    5. Which trade-offs in parallelism does a supernode like NVL72 change?

??? success "Answers (try first, then expand to compare)"
    1. Per GPU, one way: NVLink between GPUs is about 450 GB/s; between GPU and NIC it is PCIe (Gen5, about 55 GB/s); between machines each NIC is 400 Gb/s, about 50 GB/s (a machine usually has 8 NICs). GPU memory itself is about 3.35 TB/s.
    2. In decode each all-reduce message is only a few hundred KB, close to or even below the "half-bandwidth point" (the message size where fixed overhead equals transfer time), so the time goes mostly to fixed latency rather than moving bytes.
    3. GPU i of every machine (and its NIC) connects to the same group of switches (rail i), so GPUs on the same rail are only one switch hop apart, the closest. Cross-node traffic is first sent to the GPU in the same position on the target node (same rail), then forwarded within the node over NVLink; this is faster than crossing rails directly and does not crowd the upper network layers.
    4. It lets the NIC read and write GPU memory directly, so data need not be copied to CPU memory before sending (saving the copy and CPU involvement). It requires the NIC and GPU to sit under the same PCIe switch (close in the topology), and the program must pick the NIC paired with its GPU.
    5. With 72 GPUs in one NVLink domain, large-scale EP and larger TP can go over NVLink instead of the cross-machine network, and the cost of all-to-all drops sharply; designs once forced by expensive cross-machine communication (such as capping the EP size, or some PD-disaggregation configurations) can be reconsidered.

<!-- comic ../assets/comics/interconnect.webp is in Chinese; put it back once the English version exists -->

## Data paths in an 8-GPU server {#一台-8-卡服务器里的数据通路}

Take an 8-GPU H100 SXM server (DGX / HGX H100), with bandwidth per GPU in one direction:

| Path | Bandwidth (per GPU, one way) | Connects | Who uses it in inference |
| --- | --- | --- | --- |
| HBM3 | about 3.35 TB/s | a GPU and its own memory | every kernel; the decode bottleneck |
| NVLink 4 + NVSwitch | 450 GB/s (900 GB/s both ways) | all 8 GPUs in the machine, any pair at full speed | TP's all-reduce, in-machine EP's all-to-all |
| PCIe Gen5 x16 | about 64 GB/s (50–55 in practice) | GPU and CPU memory, GPU and NIC | offloading KV to CPU, loading weights |
| NIC (ConnectX-7, 400 Gb/s) | 50 GB/s | between machines, one per GPU | cross-machine EP, KV transfer in PD disaggregation, multi-machine TP / PP |

A few numbers worth remembering: NVLink is about 9 times faster than a NIC and 8 times faster than PCIe, while HBM is another 7 times faster than NVLink. So the first principle of parallelism is "put the splits that communicate most within reach of NVLink": tensor parallelism generally stays within a machine, and cross-machine EP does everything it can to cut cross-machine traffic ([expert parallelism](../distributed/expert-parallel.md), and DeepEP below).

![Figure: data paths in an 8-GPU H100 server](../assets/figures/server-topology.svg){.aig-svg}

H100's NVLink consists of 18 links, and the 8 GPUs are joined by 4 NVSwitch chips in the machine, so any two GPUs talk at full bandwidth, with no "near GPUs fast, far GPUs slow". On the PCIe side, each GPU and its NIC hang off the same PCIe switch, which is crucial for GPUDirect RDMA later.

## Latency and bandwidth: the α-β model {#延迟与带宽α-β-模型}

A transfer's time can be roughly written as $T(n) = \alpha + n / \beta$: $\alpha$ is the fixed cost (launching the transfer, synchronization, network round trips), and $\beta$ is the link bandwidth. With small messages $\alpha$ dominates, and the achieved bandwidth is far below the nominal one:

```python
LINKS = {                          # one-way bandwidth β (bytes/s) and fixed cost per transfer α (s), orders of magnitude only
    "NVLink": (450e9, 1e-6),       # H100 through NVSwitch
    "PCIe 5": (55e9, 2e-6),        # x16, minus protocol overhead
    "IB 400G": (50e9, 3e-6),       # one ConnectX-7, across nodes
}


def fmt(n):
    return f"{n / 2**20:.0f} MiB" if n >= 2**20 else f"{n / 2**10:.0f} KiB"


print("消息大小  " + "".join(f"{k:>12}" for k in LINKS) + "   （实际达到的带宽，GB/s）")
for n in (4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20):
    print(f"{fmt(n):>8}  " + "".join(f"{n / (a + n / b) / 1e9:>12.1f}" for b, a in LINKS.values()))
for name, (b, a) in LINKS.items():
    print(f"{name}：半带宽点 n½ = α·β = {fmt(a * b)}")

# two typical messages in inference: one tensor-parallel all-reduce (hidden=8192, bf16)
b, a = LINKS["NVLink"]
for stage, tokens in (("decode，batch 32", 32), ("prefill，8K token", 8192)):
    n = tokens * 8192 * 2
    print(f"{stage}：每次 {fmt(n)}，NVLink 上固定开销占 {a / (a + n / b):.0%}")
```

```text title="output"
消息大小        NVLink      PCIe 5     IB 400G   （实际达到的带宽，GB/s）
   4 KiB           4.1         2.0         1.3
  64 KiB          57.2        20.5        15.2
   1 MiB         314.9        49.8        43.7
  16 MiB         438.2        54.6        49.6
 256 MiB         449.2        55.0        50.0
NVLink：半带宽点 n½ = α·β = 439 KiB
PCIe 5：半带宽点 n½ = α·β = 107 KiB
IB 400G：半带宽点 n½ = α·β = 146 KiB
decode，batch 32：每次 512 KiB，NVLink 上固定开销占 46%
prefill，8K token：每次 128 MiB，NVLink 上固定开销占 0%
```

Drag the message size and watch the "achieved / nominal bandwidth" curves of three links:

<div class="aig-widget" data-widget="alphabeta"></div>

The values of $\alpha$ are only orders of magnitude (real values depend on the software stack; an in-machine NCCL all-reduce usually has a latency of a few to a dozen-odd microseconds), but the conclusion holds: **for messages below the half-bandwidth point $n_{1/2} = \alpha\beta$, time goes mostly to fixed costs**. Communication in prefill is a bandwidth problem; in decode it is a latency problem. This explains many seemingly odd designs in inference frameworks:

- vLLM and SGLang both have their own **custom all-reduce**: for small messages, each GPU reads the other GPUs' buffers directly over NVLink and sums in one step (one-shot), skipping the multi-step round trips of NCCL's ring algorithm; the next chapter computes where it crosses over with ring;
- decode uses **CUDA Graphs** to capture dozens of communications together with compute, saving the CPU launch cost each time;
- **fuse** communication with compute (for example all-reduce + RMSNorm in one kernel) to remove a kernel boundary;
- decode with small batches may even choose **smaller TP**: the number of communications stays the same, but each one's fixed cost is very real.

## Between nodes: InfiniBand, RoCE and rail topology {#节点之间infinibandroce-与轨道拓扑}

Machines are connected by networks that support **RDMA** (remote direct memory access): the NIC reads and writes the remote machine's memory directly, without going through that machine's CPU and operating system (the programming model is in the [RDMA programming model](rdma.md) chapter). The two mainstream implementations:

- **InfiniBand**: a dedicated network stack and switches, naturally lossless with credit-based flow control, plus features such as adaptive routing and in-switch computing (SHARP); the first choice for training clusters;
- **RoCEv2** (RDMA over Converged Ethernet): runs RDMA over Ethernet (UDP/IP), with cheap equipment unified with the data center's existing network. Ethernet itself drops packets and RDMA is very sensitive to loss, so it relies on PFC (priority-based pause frames) to be made "lossless", plus ECN + DCQCN for congestion control; tuned badly, it suffers congestion spreading, PFC storms and other problems.

Large GPU clusters generally use a **rail-optimized** topology: NIC $i$ of every machine connects to leaf switch $i$, and this group of switches is called rail $i$. So two GPUs **in the same position** on different machines are only one switch apart; GPUs in different positions must go up to a spine switch and back down, with more hops and competition with other traffic for uplinks. Count where the traffic of a 32-GPU all-to-all goes:

```python
from collections import Counter

NODES, GPN = 4, 8                         # 4 nodes, 8 GPUs each, one NIC per GPU (GPU i's NIC connects to leaf switch i)


def path(src, dst, pxn=False):
    (sn, sg), (dn, dg) = divmod(src, GPN), divmod(dst, GPN)
    if sn == dn:
        return "节点内 NVLink"
    if sg == dg:
        return "同轨：网卡 → 叶交换机 → 网卡"
    if pxn:
        return "PXN：先 NVLink 转给同轨的卡，再同轨发出"
    return "跨轨：还要经过脊交换机"


for pxn in (False, True):
    c = Counter(path(s, d, pxn) for s in range(NODES * GPN) for d in range(NODES * GPN) if s != d)
    print("开启 PXN" if pxn else "不开 PXN", "，all-to-all 的 992 对收发：", sep="")
    for k, v in c.items():
        print(f"  {k}：{v} 对（{v / sum(c.values()):.0%}）")
```

```text title="output"
不开 PXN，all-to-all 的 992 对收发：
  节点内 NVLink：224 对（23%）
  同轨：网卡 → 叶交换机 → 网卡：96 对（10%）
  跨轨：还要经过脊交换机：672 对（68%）
开启 PXN，all-to-all 的 992 对收发：
  节点内 NVLink：224 对（23%）
  同轨：网卡 → 叶交换机 → 网卡：96 对（10%）
  PXN：先 NVLink 转给同轨的卡，再同轨发出：672 对（68%）
```

Two thirds of the all-to-all traffic crosses rails. NCCL's **PXN** (PCI × NVLink) first forwards that traffic over NVLink to the GPU on the same machine that is on the target's rail, then sends it out through that GPU's NIC, turning all cross-machine traffic into "same rail". DeepEP's high-throughput mode follows the same idea: across machines, first send over RDMA to the GPU **in the same position** on the target machine, then forward over NVLink within the target machine to the real destination (see [NVSHMEM and DeepEP](nvshmem-deepep.md)).

## GPUDirect: keeping data off the CPU detour {#gpudirect让数据不绕道-cpu}

Without GPUDirect, data in GPU memory bound for another machine must first be copied to CPU memory, then read from CPU memory by the NIC; the receiver copies once more in reverse. GPUDirect is a set of technologies that removes these hops:

| Technology | What it does | Use in inference |
| --- | --- | --- |
| **GPUDirect P2P** | GPUs in the same machine read and write each other's memory directly (NVLink or PCIe) | custom all-reduce, in-machine EP |
| **GPUDirect RDMA** | the NIC reads and writes GPU memory directly over PCIe | cross-machine KV transfer, cross-machine EP, NCCL across machines |
| **GPUDirect Storage** | direct DMA between NVMe drives and GPU memory | loading weights from SSD, spilling KV to disk |
| **GDRCopy** | maps GPU memory for the CPU to access with ordinary reads and writes (low-latency small copies) | passing flags and small metadata; internals of libraries such as NVSHMEM |

GPUDirect RDMA has topology requirements: the NIC and GPU should hang off **the same PCIe switch**, so the data path goes only through that switch; if it must cross the CPU's root complex (or even CPU sockets), bandwidth and latency get noticeably worse. `nvidia-smi topo -m` shows the connection type between each pair of devices (below is an illustration keeping only two GPUs and two NICs):

```text
        GPU0    GPU1    NIC0    NIC1
GPU0     X      NV18    PIX     SYS
GPU1    NV18     X      SYS     PIX
NIC0    PIX     SYS      X      SYS
NIC1    SYS     PIX     SYS      X
```

`NV18` means 18 NVLinks between the two GPUs; `PIX` means passing through only one PCIe switch (best); `NODE` means going through the same CPU's root complex; `SYS` means also crossing CPU sockets (worst). Each GPU should use the NIC it has `PIX` with. NCCL chooses this automatically, but when you write your own RDMA program (such as a KV transfer engine) you must handle it yourself; this is what "topology-aware" means. On the software side, GPUDirect RDMA needs the NIC driver to access GPU memory: the old way is the `nvidia-peermem` kernel module, the new way is Linux's dma-buf.

## Supernodes: making the NVLink domain bigger {#超节点把-nvlink-域做大}

GB200 NVL72 joins 72 Blackwell GPUs into one domain with NVLink 5, with 1.8 TB/s of NVLink bandwidth per GPU in both directions combined, and any two of the 72 GPUs talk at full speed. It changes several trade-offs:

- **Large-scale EP no longer needs the cross-machine network**: for a 256-expert model like DeepSeek-V3, all-to-all with EP up to 72 stays entirely on NVLink, and the tricks for cutting cross-machine traffic (capping the nodes per token, two-hop forwarding) become less necessary;
- **TP can be larger**: TP used to be limited to 8 GPUs because the NVLink domain had only 8;
- **KV transfer in PD disaggregation** can stay within the same NVLink domain.

The costs: such systems are expensive with demanding power and cooling, and beyond 72 GPUs the RDMA network is still needed. When designing an inference system, first ask "how big is the NVLink domain"; many parallelism choices follow from that.

!!! interview "In an interview"
    When asked "why doesn't tensor parallelism cross machines" or "where is the bottleneck of cross-machine EP", start with numbers: NVLink at 450 GB/s per GPU one way, a NIC at 50 GB/s, a 9× gap; then latency: decode's communication messages are only a few hundred KB, close to or below the half-bandwidth point, with fixed costs taking half the time, hence custom all-reduce, CUDA Graphs and fusing communication with compute. When discussing cross-machine communication, mention the rail topology and "send on the same rail first, then forward within the machine" (PXN, DeepEP); it shows you understand the physical structure of a cluster.

!!! info "Related chapters"
    - [Collective communication: NCCL's algorithms and protocols](nccl.md) (the next chapter of this book)
    - [Multi-GPU systems: NVLink, topology, partitioning and power](cs://arch/multi-gpu/) (CS Fundamentals: the basics of topology and NUMA)
    - [Multi-GPU and NCCL](cuda://tools/multi-gpu/) (Advanced CUDA), [collective communication primitives](train://basics/collectives/) (Distributed Training)

## Exercises {#练习}

**1. The half-bandwidth point.** A link has 200 GB/s of bandwidth and a fixed cost of 5 µs per transfer. How large must a message be to reach 100 GB/s? And 180 GB/s?

??? success "Answer"
    The achieved bandwidth is $n / (\alpha + n/\beta) = \beta / (1 + \alpha\beta / n)$. Reaching $\beta/2$ needs $n = \alpha\beta = 5\,\mu s \times 200\,\text{GB/s} = 1\,\text{MB}$; reaching $0.9\beta$ needs $\alpha\beta / n = 1/9$, i.e. $n = 9\alpha\beta = 9$ MB. To approach full bandwidth, messages must be an order of magnitude larger than the half-bandwidth point, which is why NCCL pipelines large messages in chunks, while small messages can only be helped by lowering $\alpha$.

**2. Picking a NIC.** On an 8-GPU server, GPU3 is `PIX` with NIC3 and `SYS` with NIC7. Your KV transfer program hands every GPU's data to NIC0 to send. What goes wrong?

??? success "Answer"
    First, bandwidth: 8 GPUs share one 50 GB/s NIC, 1/8 of the total bandwidth there should be. Second, the path: except for GPU0, every GPU's data must cross the PCIe root complex, or even CPU sockets, to reach NIC0, so GPUDirect RDMA performance drops sharply and competes with other PCIe traffic. The right approach is for each GPU to use the NIC it has `PIX` with (topology-aware), sending over several NICs in parallel; transfer engines such as the Mooncake Transfer Engine read the topology and do this automatically.

## Summary {#小结}

- [x] Per-GPU one-way bandwidth: HBM about 3.35 TB/s, NVLink 450 GB/s, PCIe about 55 GB/s, a 400G NIC 50 GB/s; put the splits that communicate most within the NVLink domain.
- [x] The α-β model: below $\alpha\beta$, fixed costs dominate. Decode's communication is a latency problem, prefill's a bandwidth problem; custom all-reduce, CUDA Graphs and fusing communication with compute all fight $\alpha$.
- [x] Cluster networks use InfiniBand or RoCEv2 (made lossless with PFC and congestion control); in a rail-optimized topology GPUs in the same position are closest, and PXN and DeepEP both send on the same rail first, then forward within the machine.
- [x] GPUDirect P2P / RDMA / Storage keep data off the CPU detour; the NIC and GPU should sit under the same PCIe switch, and programs must pick the right NIC.
- [x] Supernodes like NVL72 grow the NVLink domain to 72 GPUs, freeing large-scale EP and larger TP from the cross-machine network.
