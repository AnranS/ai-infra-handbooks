# Collective communication: NCCL's algorithms and protocols, and custom all-reduce in inference frameworks

<p class="lead">The <a href="cuda://tools/multi-gpu/">multi-GPU and NCCL</a> chapter of the CUDA handbook covered how to use NCCL and ring all-reduce. This chapter goes one level down: which algorithms and protocols NCCL has inside, how it chooses, and at which message sizes each wins; why vLLM and SGLang still write their own all-reduce; and how to read nccl-tests numbers and debug hung communication.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What are NCCL's ring and tree algorithms each suited for? And what is NVLS?
    2. How do the three protocols LL, LL128 and Simple differ?
    3. How are the algbw and busbw reported by nccl-tests related?
    4. Why is an inference framework's custom all-reduce faster than NCCL for small messages? How do one-shot and two-shot differ?
    5. A multi-GPU inference service hangs. How do you tell whether communication is the problem?

??? success "Answers (try first, then expand to compare)"
    1. ring: bandwidth-optimal, each GPU sends and receives about 2S bytes, but takes $2(n-1)$ steps, suited to large messages within a machine; tree: about $\log n$ steps, low latency, suited to small messages across machines; NVLS: uses NVSwitch to reduce inside the switch, halving traffic without using SMs.
    2. LL (low latency): every 8 bytes carry a 4-byte flag, and the receiver polls the flags to know data has arrived; lowest latency, but only half the bandwidth efficiency. LL128: every 128 bytes carry 8 bytes of flags, a compromise between latency and bandwidth. Simple: large-block transfers + explicit synchronization, the highest bandwidth and the highest latency.
    3. busbw = algbw × a correction factor, which for all-reduce is $2(n-1)/n$. busbw reflects the actual transfer rate on the links and can be compared directly with the links' peak bandwidth.
    4. For small messages, NCCL's ring takes $2(n-1)$ steps, each with synchronization and launch overhead; the custom implementation reads and writes peers' memory directly through CUDA IPC, with far fewer steps and synchronizations. one-shot: each GPU reads all peers' data directly and reduces in one step (for small messages); two-shot: reduce-scatter so each reduces one slice, then all-gather, in two steps with less data moved (for medium messages).
    5. First check which collective each rank is stuck in, whether the call order matches across ranks (did some rank make one call fewer, or with different arguments), and whether some rank has already crashed; then check NIC and network configuration. Tools: `NCCL_DEBUG=INFO` for initialization and topology, PyTorch's flight recorder for each rank's recent collectives, and py-spy for the stuck call stacks.

## Algorithms: the ways to run one all-reduce {#算法一次-all-reduce-有几种走法}

For each collective, NCCL picks an **algorithm** (how data flows between GPUs) and a **protocol** (how each step transfers, and how the other side learns the data has arrived). The main all-reduce algorithms:

- **ring**: $n-1$ steps each of reduce-scatter and all-gather, each GPU sending and receiving $2\frac{n-1}{n}S$; bandwidth-optimal, but the step count grows linearly with the number of GPUs;
- **tree** (double binary tree): reduce up the tree, then broadcast down it, in $O(\log n)$ steps; with large messages pipelined in chunks its bandwidth is also close to optimal. It is used mainly **across machines**: chains within a machine, trees between machines;
- **NVLS** (NVLink SHARP): H100's NVSwitch can add inside the switch. Each GPU sends its data to the switch once and fetches the summed result once, cutting traffic from $2\frac{n-1}{n}S$ to about $S$, while using almost no SMs;
- **CollNet**: similarly uses the SHARP of InfiniBand switches for reductions between machines.

Inference frameworks also bypass NCCL, mapping other GPUs' memory directly with CUDA IPC and finishing the communication inside one kernel:

- **one-shot**: each GPU reads the **full** data of the other $n-1$ GPUs directly and sums locally. Only one synchronization, but each GPU reads $(n-1)S$;
- **two-shot**: in the first step each GPU reads only $1/n$ of the other GPUs' data and sums it (a reduce-scatter), and in the second step reads the sums back (an all-gather). Two synchronizations, with each GPU reading $2\frac{n-1}{n}S$.

Compare them with the α-β model ([previous chapter](interconnect.md)), 8 GPUs on NVLink:

```python
import math

N, BETA, ALPHA = 8, 450e9, 1.5e-6        # 8 GPUs on NVLink: one-way bandwidth per GPU, fixed cost per step (sync + launch)

ALGOS = {                                 # α-β time of one all-reduce (S bytes per GPU)
    "ring":     lambda S: 2 * (N - 1) * ALPHA + 2 * (N - 1) / N * S / BETA,
    "tree":     lambda S: 2 * math.log2(N) * ALPHA + 2 * S / BETA,
    "one-shot": lambda S: 2 * ALPHA + (N - 1) * S / BETA,               # each GPU reads the other 7 GPUs' full data directly and sums locally
    "two-shot": lambda S: 4 * ALPHA + 2 * (N - 1) / N * S / BETA,       # reduce-scatter + all-gather done with direct reads and writes
    "NVLS":     lambda S: 2 * ALPHA + S / BETA,                         # NVSwitch adds inside the switch: each GPU sends one copy and receives one
}

print(f"{'每卡数据':>8}" + "".join(f"{k:>10}" for k in ALGOS) + "   不用 NVLS 时最快（单位 µs）")
for S in (16 << 10, 256 << 10, 1 << 20, 8 << 20, 64 << 20):
    t = {k: f(S) * 1e6 for k, f in ALGOS.items()}
    size = f"{S >> 20} MiB" if S >= 1 << 20 else f"{S >> 10} KiB"
    best = min((k for k in t if k != "NVLS"), key=t.get)
    print(f"{size:>8}" + "".join(f"{v:>10.1f}" for v in t.values()) + f"   {best}")

S = 1 << 30                               # the two bandwidths of nccl-tests: algbw = S / t, busbw = algbw × 2(n-1)/n
t = ALGOS["ring"](S)
print(f"1 GiB 的 ring all-reduce：{t * 1e3:.2f} ms，algbw {S / t / 1e9:.0f} GB/s，busbw {S / t * 2 * (N - 1) / N / 1e9:.0f} GB/s")
```

```text title="输出"
    每卡数据      ring      tree  one-shot  two-shot      NVLS   不用 NVLS 时最快（单位 µs）
  16 KiB      21.1       9.1       3.3       6.1       3.0   one-shot
 256 KiB      22.0      10.2       7.1       7.0       3.6   two-shot
   1 MiB      25.1      13.7      19.3      10.1       5.3   two-shot
   8 MiB      53.6      46.3     133.5      38.6      21.6   two-shot
  64 MiB     282.0     307.3    1046.9     267.0     152.1   two-shot
1 GiB 的 ring all-reduce：4.20 ms，algbw 256 GB/s，busbw 448 GB/s
```

Change the GPU count, the link and the message size to see which way is fastest:

<div class="aig-widget" data-widget="collective-cost"></div>

Reading the table:

- **Small messages (decode)**: ring takes 14 steps, 21 µs in fixed costs alone; one-shot gets there in one step, 3 µs. With two all-reduces per layer in decode over dozens of layers, that is a gap of milliseconds. This is exactly where vLLM's and SGLang's custom all-reduce comes from: one-shot for small messages, two-shot for medium ones;
- **Large messages (prefill)**: one-shot has each GPU read 7 copies of the data and is the slowest; two-shot moves the same amount as ring. In the model two-shot stays slightly faster than ring, but in practice the custom implementation must first copy the input into a pre-registered shared buffer, uses more SMs, and has a capped buffer size, so inference frameworks hand messages above a threshold (about 8 MB by default in vLLM) back to NCCL;
- **NVLS** wins at every size (the model is optimistic, but the direction is right), and NCCL enables it automatically on supported hardware; SGLang and vLLM can also call NCCL's NVLS directly, or an all-reduce built on PyTorch's symmetric memory (SymmetricMemory);
- The model's α of 1.5 µs is only illustrative; real crossover points must be measured on your own machines.

## Protocols: how do you know the data has arrived {#协议怎么知道数据到了}

The receiver needs to know "the data has fully arrived" before it can use it. NCCL has three protocols:

| Protocol | How | Bandwidth efficiency | Latency |
| --- | --- | --- | --- |
| **LL** (low latency) | every 8 bytes hold 4 bytes of data and 4 bytes of flag, and the receiver polls the flag; relies on 8-byte writes being atomic, with no memory fence needed | about 50% | lowest |
| **LL128** | every 128 bytes hold 120 bytes of data and 8 bytes of flag; relies on NVLink's guarantees for 128-byte writes | about 94% | low |
| **Simple** | write a large block, add a memory fence, then write a flag | close to 100% | high (one fence per block) |

NCCL picks automatically by message size and topology: LL for small messages, LL128 for medium ones, Simple for large ones. A hand-written one-shot kernel faces the same problem: after each GPU writes its data, it must notify the other GPUs with a "semaphore", and this synchronization is the main source of α.

Another dimension is **channels**: NCCL splits a communication across several channels in parallel, each executed by one CUDA block. In other words, **NCCL's communication kernels occupy SMs**: running alongside compute kernels, they compete for SMs. This is why DeepEP lets users specify how many SMs communication may use, why DeepSeek-V3 set aside 20 SMs for communication in training, and why transfer methods that use no SMs, like NVLS and the copy engine, are popular.

## Reading nccl-tests {#读懂-nccl-tests}

The standard tool for measuring multi-GPU communication performance is `nccl-tests` (`all_reduce_perf -b 8 -e 1G -f 2 -g 8`). Its output has two bandwidths:

- **algbw** (algorithm bandwidth) $= S / t$: the user's view, "an all-reduce of $S$ bytes per GPU took $t$ seconds";
- **busbw** (bus bandwidth) $= \text{algbw} \times$ a correction factor: converted to "how fast each GPU's links actually ran", directly comparable with the hardware's link bandwidth. The factor is $2\frac{n-1}{n}$ for all-reduce, $\frac{n-1}{n}$ for all-gather and reduce-scatter, and 1 for broadcast.

In the output above, the busbw of a 1 GiB ring all-reduce is 448 GB/s, close to NVLink's 450 GB/s, showing the links are saturated. In measurements, an 8-GPU H100 all-reduce usually reaches a busbw of 350–480 GB/s (the latter relying on NVLS); across machines it is bounded by the NICs, and a network with 400 Gb/s per GPU gets a busbw of 40–48 GB/s. **If busbw is far below these values, check the topology and configuration first, rather than suspecting the algorithm.**

`judge.py bench` in the practice problem bank measures multi-GPU all-reduce busbw on your machine (see [practice problems](root://practice/)).

## Common environment variables and debugging {#常用的环境变量与排查}

| Variable | Effect |
| --- | --- |
| `NCCL_DEBUG=INFO` (with `NCCL_DEBUG_SUBSYS=INIT,GRAPH`) | prints the topology NCCL detected, the NICs chosen, and the rings / trees built; the first step in debugging |
| `NCCL_ALGO` / `NCCL_PROTO` | force an algorithm (`Ring`, `Tree`, `NVLS`) or protocol (`LL`, `LL128`, `Simple`), for comparison experiments |
| `NCCL_NVLS_ENABLE` | turns NVLS on or off |
| `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` | which RDMA NICs to use, and which network interface for the initial handshake |
| `NCCL_NET_GDR_LEVEL` / `NCCL_P2P_LEVEL` | within what topology distance GPUDirect RDMA and P2P are enabled |
| `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS` | the number of channels: affects bandwidth and the SMs used |

Common reasons a multi-GPU inference service "hangs":

- **Ranks call collectives in a different order or with different arguments**: for example one rank does one extra all-reduce because of an if branch, or tensor shapes differ. Collectives require every rank to call in the same order, or they wait forever. In inference engines, take special care that "things only rank 0 does" (sampling, scheduling) never sit in the middle of communication;
- **A rank has already crashed**: the other ranks wait for it inside a collective. Look at every rank's logs and find the one that failed first;
- **Network configuration problems**: the wrong NIC, firewalls, or a misconfigured RoCE PFC causing packet loss and retransmission. `NCCL_DEBUG=INFO` shows the chosen NIC and transport (`NET/IB`, `NET/Socket`);
- PyTorch's **flight recorder** (`TORCH_NCCL_TRACE_BUFFER_SIZE`) records each rank's recent collectives; dump and compare them when hung to see which rank made one call too few or too many.

!!! source "Source code"
    - **vLLM**: `vllm/distributed/device_communicators/` has `custom_all_reduce.py` (CUDA IPC one-shot / two-shot, with the kernel in `csrc/custom_all_reduce.cuh`), `pynccl.py` (calls NCCL directly through ctypes, convenient inside CUDA Graphs), `symm_mem.py` (PyTorch symmetric memory), `flashinfer_all_reduce.py` and more; `cuda_communicator.py` picks the path by message size and environment.
    - **SGLang**: `srt/distributed/device_communicators/` has the corresponding `custom_all_reduce.py`, `pynccl.py` and `torch_symm_mem.py`, and `srt/layers/flashinfer_comm_fusion.py` integrates FlashInfer's fused all-reduce + RMSNorm kernel.

!!! interview "In an interview"
    "Why do inference frameworks write their own all-reduce" is a frequent question. The skeleton of an answer: decode's messages are only a few hundred KB, NCCL's ring takes $2(n-1)$ steps, and the time is almost all fixed cost; the custom implementation reads peers' memory directly through CUDA IPC, one-shot finishing in one step (small messages), two-shot in two (medium messages), and large messages going back to NCCL; then add that it must be capturable by CUDA Graphs (buffers pre-registered, addresses fixed), and that hardware reduction like NVLS gives the problem a new answer on new hardware.

## Exercises {#练习}

**1. Working back from busbw to time.** An 8-GPU machine measures an all-reduce busbw of 360 GB/s. How long does a 256 MiB all-reduce take? And an all-gather (each GPU ends up with 256 MiB)?

??? success "Answer"
    all-reduce: algbw $= 360 / (2 \times 7/8) = 205.7$ GB/s, $t = 268\,\text{MB} / 205.7\,\text{GB/s} \approx 1.3$ ms.
    all-gather's factor is $(n-1)/n$; assuming it reaches the same busbw: algbw $= 360 / (7/8) = 411$ GB/s, $t \approx 0.65$ ms. All-gather moves only half as much as all-reduce. This is also why sequence parallelism keeps the total communication unchanged after splitting all-reduce into reduce-scatter + all-gather.

**2. The one-shot crossover.** Use this chapter's model to derive the formula for the crossover between one-shot and two-shot. How does it change when the GPU count goes from 8 to 2?

??? success "Answer"
    Setting $2\alpha + (n-1)S/\beta = 4\alpha + 2\frac{n-1}{n}S/\beta$ gives $S^* = \frac{2\alpha\beta}{(n-1)(1 - 2/n)}$. At $n = 8$, $S^* = 2\alpha\beta / 5.25 \approx 257$ KB (in the table the two are nearly tied at 256 KiB, right at the crossover). At $n = 2$ the denominator is 0: with two GPUs, one-shot and two-shot move the same amount (both $S$), and one-shot has one fewer synchronization, so it is never worse than two-shot at any size. The fewer the GPUs, the better one-shot does; vLLM's custom all-reduce also sets different thresholds by GPU count.

**3. Why does custom all-reduce pre-register its buffers?**

??? success "Answer"
    CUDA IPC first exchanges a memory handle with other processes, which open it to get a mapped address; this is an expensive operation that involves the CPU and cannot be done for every communication. Moreover, decode is captured with CUDA Graphs, and the kernel arguments in a graph (including peers' buffer addresses) are fixed. So custom all-reduce allocates and exchanges a fixed buffer at startup and copies the input into it on each communication (or registers the addresses of tensors used in the graph directly while capturing the CUDA Graph), which also caps the largest message it can handle.

## Summary {#小结}

- [x] All-reduce algorithms: ring (bandwidth-optimal, many steps), tree ($\log n$ steps, used across machines), NVLS (reduction inside the switch, half the traffic, no SMs); inference frameworks add CUDA IPC one-shot / two-shot.
- [x] For small messages, count steps and synchronizations; for large ones, bytes sent and received per GPU; decode uses one-shot / two-shot, prefill goes to NCCL.
- [x] The LL / LL128 / Simple protocols trade latency against bandwidth efficiency; NCCL's channels are executed by CUDA blocks and compete with compute for SMs.
- [x] busbw = algbw × a correction factor ($2\frac{n-1}{n}$ for all-reduce), directly comparable with link bandwidth.
- [x] When communication hangs, check first: whether the call order matches across ranks, whether a rank has crashed, and NIC and network configuration; the tools are `NCCL_DEBUG=INFO` and PyTorch's flight recorder.
