# Practising without multiple GPUs: real multiple processes on the CPU

<p class="lead">Every chapter after this one is about how to partition and communicate across several cards, but learning that does not require actually having several cards. The only part of multi-GPU work that genuinely needs the hardware is performance and operations, a small slice; the design and the correctness, which are the bulk of it, can all be verified on one laptop with real multiple processes under <code>torchrun</code> and gloo. This is not a faked simulation: it is the same <code>torch.distributed</code> API and the same partitioning logic, with NCCL swapped for gloo. This chapter sets out how to run it, how far it verifies, what genuinely needs real cards, and the checklist to follow once you have rented some.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Running 4 processes with gloo on the CPU, which collective primitives are available? Which are not?
    2. Your tensor-parallel implementation is wrong. How do you find out without a GPU?
    3. Which conclusions can only be reached on real multiple cards? Why can the CPU not measure them?
    4. What is the difference between `torchrun --standalone --nproc-per-node=4` and setting `MASTER_ADDR` and `RANK` by hand?
    5. You have 8 cards for 4 hours. In what order would you run your tests?

??? success "Answers for the self-test (answer first, then open this)"
    1. In today's PyTorch, gloo already supports `all_reduce`, `all_gather`, `all_gather_into_tensor`, `reduce_scatter_tensor`, `all_to_all_single`, `broadcast`, `barrier` and the point-to-point `send` / `recv` / `isend` / `irecv`, which covers everything the multi-process examples in this book use. What is genuinely missing is the GPU-only capability: NCCL's communicators, asynchronous communication on a CUDA stream, GPUDirect and direct NVLink, and the GPU fast path for `reduce_scatter`.
    2. Write **a single-process reference implementation** as the golden comparison: the same weights and the same input, computed in full by one process and by the partitioned multi-process version, then compared elementwise with `allclose`. A partitioning mistake (the wrong dimension, a missing all-reduce, RoPE positions that do not follow the partition) will almost certainly show up there.
    3. The gap between measured and theoretical bandwidth, the difference between NVLink and PCIe, the effect of the topology (NUMA, PIX/PXB/SYS), the real benefit of overlapping communication with computation, where the scaling curve bends, and the behaviour after one rank dies. These depend on real links and on NCCL's implementation; gloo on the CPU goes over the loopback, so its numbers mean nothing.
    4. `--standalone` has torchrun bring up its own rendezvous locally (a random port, several processes on one machine), so `MASTER_ADDR`, `MASTER_PORT`, `RANK` and `WORLD_SIZE` are none of your concern and no port clashes with anyone else's. Setting the environment variables by hand is only needed across machines, or when the port has to be controlled exactly.
    5. The baselines first (nccl-tests' bus bandwidth across message sizes, plus the topology), then the experiments that have a control (the same model's scaling at TP=1/2/4/8, overlap on against off), and exploratory tuning last. Every script and control is written and debugged on the CPU first, so the card time is spent only running and recording.

## The conclusion first: what can be verified on a CPU {#先说结论哪些能在-cpu-上验证}

![Figure: what can be learned without multiple GPUs, and what needs real cards](../assets/figures/cpu-verifiable.svg){.aig-svg}

| What you want to learn | Can you practise it on a CPU | How |
| --- | --- | --- |
| The semantics of the collective primitives | ✅ entirely | the gloo backend, 4 processes |
| Implementing a ring all-reduce and its volume | ✅ | write it by hand with point-to-point `isend` / `irecv` and count the bytes |
| DDP's bucketing and overlap **logic** | ✅ | watch how the buckets are formed and when the communication fires |
| ZeRO-1/2/3's partitioning and the timing of the all-gather | ✅ | line it up elementwise against single-process Adam |
| How tensor and sequence parallelism partition | ✅ | line it up elementwise against the single-process forward pass |
| Pipeline parallelism's 1F1B schedule and the bubble | ✅ | count the bubbles and check against the formula |
| Context parallelism (Ulysses, Ring Attention) | ✅ | line it up elementwise against single-process attention |
| Expert parallelism's all-to-all and load imbalance | ✅ | count how many tokens each expert receives |
| The volume and time model (alpha-beta) | ✅ | pure arithmetic, nothing to run |
| **Measured bandwidth, topology, overlap benefit, scaling curves** | ❌ | real cards required |
| **NCCL tuning, multi-machine RDMA, fault recovery** | ❌ | real cards required (and several machines) |

In one sentence: **the logic and the numerical correctness can be 100% verified on a CPU, and not one performance number can be believed.**

## How to run it: torchrun with gloo {#怎么跑torchrun--gloo}

```bash
# 4 processes on one machine, with torchrun handling the rendezvous itself
torchrun --standalone --nproc-per-node=4 your_script.py

# on macOS, gloo also has to be pointed at the loopback interface, or the hostname resolves to an external address and cannot connect
GLOO_SOCKET_IFNAME=lo0 torchrun --standalone --nproc-per-node=4 your_script.py
# on Linux it is lo
```

The only change in a script is the backend from `nccl` to `gloo`; nothing else moves:

```python title="probe.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, n = dist.get_rank(), dist.get_world_size()
checks = []


def try_op(name, fn):
    try:
        fn()
        checks.append((name, "可用"))
    except Exception as e:
        checks.append((name, type(e).__name__))


x = torch.ones(4 * n) * (rank + 1)
try_op("all_reduce", lambda: dist.all_reduce(x.clone()))
try_op("all_gather_into_tensor", lambda: dist.all_gather_into_tensor(torch.zeros(4 * n * n), x.clone()))
try_op("reduce_scatter_tensor", lambda: dist.reduce_scatter_tensor(torch.zeros(4), x.clone()))
try_op("all_to_all_single", lambda: dist.all_to_all_single(torch.zeros(4 * n), x.clone()))
try_op("broadcast", lambda: dist.broadcast(x.clone(), 0))
try_op("barrier", dist.barrier)
if rank == 0:
    print(f"后端 {dist.get_backend()}，{n} 个进程")
    for name, ok in checks:
        print(f"  {name:24s} {ok}")
dist.destroy_process_group()
```

```text title="output"
后端 gloo，4 个进程
  all_reduce               可用
  all_gather_into_tensor   可用
  reduce_scatter_tensor    可用
  all_to_all_single        可用
  broadcast                可用
  barrier                  可用
```

Every example in this book marked with `torchrun` uses these primitives, so all of them run directly on any machine, which is also how they are verified in CI.

!!! note "Two things to watch when running several processes on a CPU"
    - **Cap the thread count**: `OMP_NUM_THREADS=2`. Without it every process opens as many threads as there are cores, 4 processes fight over them, and it gets so slow it looks hung.
    - **Keep the scale small**: what you are doing on a CPU is verifying logic, not producing performance, so a hidden dimension in the tens and a single-digit batch are plenty. What you want to verify is whether the partitioning is right, not whether it is fast.

## The most important technique: a single-process reference as the golden comparison {#最重要的一招用单进程参考实现做黄金对照}

The commonest bug in multi-GPU code is not a crash but **quietly computing the wrong thing**: the wrong dimension partitioned, a missing all-reduce, RoPE's position offset not following the partition, normalisation along the wrong axis. Bugs like this are hard to spot on real cards too, because the loss still goes down, just a little more slowly.

The remedy is very simple: the same weights and the same input, **computed in full by a single process** and by the partitioned multi-process version, then compared elementwise. Here is a two-layer MLP demonstrating column parallelism followed by row parallelism:

```python title="tp_check.py" torchrun="4"
import torch
import torch.distributed as dist

dist.init_process_group("gloo")
rank, P = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)                                   # every rank builds the same complete weights, modelling a partition from one checkpoint

B, d, h = 8, 64, 256
x = torch.randn(B, d)
W1 = torch.randn(d, h) / d ** 0.5                      # the first layer: partitioned by column (column parallel)
W2 = torch.randn(h, d) / h ** 0.5                      # the second layer: partitioned by row (row parallel)

ref = torch.relu(x @ W1) @ W2                          # the single-process reference implementation: the golden comparison

W1_local = W1.chunk(P, dim=1)[rank]                    # each rank takes only its own slice
W2_local = W2.chunk(P, dim=0)[rank]
y = torch.relu(x @ W1_local) @ W2_local                # column parallelism needs no communication afterwards; row parallelism needs an all-reduce
dist.all_reduce(y)

err = (y - ref).abs().max().item()
if rank == 0:
    print(f"张量并行度 {P}，每个 rank 只持有 {W1_local.shape[1]} / {h} 个隐藏维")
    print(f"和单进程结果逐元素一致：{torch.allclose(y, ref, atol=1e-5)}（最大绝对误差 < 1e-5：{err < 1e-5}）")
    print("整个 MLP 只在最后 all-reduce 一次：这就是张量并行的通信量")
dist.destroy_process_group()
```

```text title="output"
张量并行度 4，每个 rank 只持有 64 / 256 个隐藏维
和单进程结果逐元素一致：True（最大绝对误差 < 1e-5：True）
整个 MLP 只在最后 all-reduce 一次：这就是张量并行的通信量
```

Comment out `dist.all_reduce(y)`, or change `chunk(P, dim=1)` to `dim=0`, and the error jumps to the order of $10^{-1}$. That is what this comparison is worth. **Every multi-process example in the chapters that follow comes with a single-process reference**, and it is worth paying attention to how it is written.

This method covers more than you might expect:

- **ZeRO**: compare the parameters after the update against single-process AdamW.
- **Sequence and context parallelism**: compare the output and the log-sum-exp against single-process full attention.
- **Pipeline parallelism**: compare each stage's activations against a single-process layer-by-layer forward pass, then count the bubbles and check.
- **Mixture of experts**: compare the set of tokens each expert receives against a single-process implementation that dispatches by the router's output.

## Computing instead of measuring {#用算的代替用试的}

Performance cannot be measured on a CPU, but many performance **conclusions** can be computed. The alpha-beta model has only two parameters, the fixed cost per step $\alpha$ and the link bandwidth $\beta$, and one collective takes $\alpha + S / \beta$.

```python title="commtime.py"
import math

ALPHA = 5e-6            # the fixed cost per communication step (handshake, synchronisation), in seconds
BETA = 200e9            # the one-way link bandwidth, bytes per second (estimated at NVLink's 200 GB/s one way)


def ring(nbytes, n):    # the ring all-reduce: 2(n-1) steps, each sending only 1/n
    return 2 * (n - 1) * (ALPHA + nbytes / n / BETA)


def tree(nbytes, n):    # the tree: 2*log2(n) steps, each sending the whole thing
    return 2 * math.ceil(math.log2(n)) * (ALPHA + nbytes / BETA)


print("一次 all-reduce 要多久（α-β 模型，α = 5 μs，β = 200 GB/s）")
print(f"{'卡数':>4} {'消息大小':>10} {'环形':>12} {'树形':>12}  更快的")
for n in (4, 8, 16, 64):
    for name, nbytes in (("4 KB", 4 << 10), ("4 MB", 4 << 20), ("1 GB", 1 << 30)):
        t_ring, t_tree = ring(nbytes, n), tree(nbytes, n)
        print(f"{n:>4} {name:>10} {t_ring * 1e6:>9.1f} μs {t_tree * 1e6:>9.1f} μs  "
              f"{'环形' if t_ring < t_tree else '树形'}")

print()
print("通信占一步训练的多少（7B 模型，bf16 梯度 14 GB，一步前反向按 300 ms 算）")
grad = 14 * (1 << 30)
for n in (8, 16, 64, 256):
    t = ring(grad, n)
    print(f"  {n:>3} 卡数据并行：all-reduce {t * 1e3:>6.1f} ms，占一步的 {t / (0.3 + t):>5.1%}")
```

```text title="output"
一次 all-reduce 要多久（α-β 模型，α = 5 μs，β = 200 GB/s）
  卡数       消息大小           环形           树形  更快的
   4       4 KB      30.0 μs      20.1 μs  树形
   4       4 MB      61.5 μs     103.9 μs  环形
   4       1 GB    8083.1 μs   21494.8 μs  环形
   8       4 KB      70.0 μs      30.1 μs  树形
   8       4 MB     106.7 μs     155.8 μs  环形
   8       1 GB    9465.2 μs   32242.3 μs  环形
  16       4 KB     150.0 μs      40.2 μs  树形
  16       4 MB     189.3 μs     207.8 μs  环形
  16       1 GB   10216.3 μs   42989.7 μs  环形
  64       4 KB     630.0 μs      60.2 μs  树形
  64       4 MB     671.3 μs     311.7 μs  树形
  64       1 GB   11199.6 μs   64484.5 μs  环形

通信占一步训练的多少（7B 模型，bf16 梯度 14 GB，一步前反向按 300 ms 算）
    8 卡数据并行：all-reduce  131.6 ms，占一步的 30.5%
   16 卡数据并行：all-reduce  141.1 ms，占一步的 32.0%
   64 卡数据并行：all-reduce  148.6 ms，占一步的 33.1%
  256 卡数据并行：all-reduce  152.3 ms，占一步的 33.7%
```

Two conclusions jump out, and neither needed any hardware:

- **A tree for small messages and a ring for large ones**, with the crossover moving right as the card count grows. This is exactly the choice NCCL makes through `NCCL_ALGO`.
- **A ring all-reduce's time barely grows with the card count** (8 cards to 256 adds only 16%), which is why data parallelism scales; but that 30% communication share will not vanish by itself and has to be hidden by **overlapping communication with the backward pass** (see [Data parallelism and DDP](../data/ddp.md)).

## What one card adds {#有一张卡能多补什么}

If you have one consumer card to hand, even in a laptop, you can add these:

- **2 to 4 ranks running NCCL on the one card**. The performance numbers mean nothing (they all contend for the same SMs), but NCCL's API semantics, communicator creation, communication groups (`new_group`) and asynchrony on a stream are all real, and they catch the class of problem that does not error on a CPU but does under NCCL.
- **Pinned memory, host-to-device and device-to-host overlapped with computation, and CUDA IPC.** Transferring KV under prefill-decode disaggregation in inference is essentially this machinery (see [Prefill-decode disaggregation](serving://distributed/pd-disagg/) in the inference-systems handbook).
- **The memory budget and out-of-memory on one card.** How much the activations take, how much recomputation saves, what fragmentation looks like: all measurable on one card (see [The memory budget](overview.md)).
- **Mixed precision and numerical stability.** bf16 and fp16 overflow, loss scaling and gradient clipping are all reproducible on one card (see [Mixed precision and FP8 training](../practice/mixed-precision.md)).

## The real-card checklist: what to run once you have rented some {#必须真卡的清单租到卡照着跑}

The rest can only be done on real machines. The good news is that this list is short and **one session of 8 cards over 4 to 6 hours gets through it**, which is cheap by the hour. There is only one rule of discipline: **the scripts, the controls and the metrics to record are all written and debugged on the CPU first, so the card time is spent only running and recording.**

| What to measure | How | What to look at | What wrong looks like |
| --- | --- | --- | --- |
| 1. The topology baseline | `nvidia-smi topo -m` | whether a pair of cards is NV#, PIX, PXB or SYS | SYS where NVLink was expected, which means a slot or virtualisation problem |
| 2. The communication baseline | `all_reduce_perf` from `nccl-tests`, sweeping from 8 B to 8 GB | the bus bandwidth curve | large messages below 70% of the link's peak, or a step in the curve |
| 3. The algorithm choice | the same, pinning `NCCL_ALGO=Ring` and then `Tree` | at which message size the crossover falls | an order of magnitude away from where the alpha-beta model above put it |
| 4. The single-card baseline | a fixed model at TP=1, measuring tokens/s and memory | the denominator for the scaling | — |
| 5. Tensor-parallel scaling | the same model at TP=2/4/8 | the speedup against one card, and the communication share | the speedup stalling already at TP=4, usually unoverlapped communication or a topology that crosses NUMA |
| 6. The overlap benefit | one run with communication overlap on and one with it off | the difference in time per step | slower with it on: the buckets are too small and the communication is fragmented |
| 7. Data-parallel scaling | DDP over 2/4/8 cards with a fixed global batch | how tokens/s per card decays | decaying more than the alpha-beta model predicts, so check whether the gradients are bucketed |
| 8. The ZeRO stages | one run each of ZeRO-1/2/3 | memory per card, time per step | ZeRO-3 more than 1.5 times slower, usually the parameter all-gather not overlapping with the forward pass |
| 9. Fault behaviour | `kill` one rank mid-training | how long until the timeout, what error, how to recover | hanging without an error: `NCCL_ASYNC_ERROR_HANDLING` is off |
| 10. Profile one step | capture 10 steps with Nsight Systems | whether the communication and computation kernels overlap | communication and computation entirely serial on the timeline |

The first four are the baselines and have to be run first; 5 to 8 are the experiments with a control and are what is genuinely worth putting on a CV; 9 and 10 depend on time.

!!! tip "The checklist before renting"
    - Everything `git push`ed, so the card session only `git clone`s; do not write code on the card.
    - The dataset and the weights uploaded to object storage first, or a mirror you can reach directly, so the download does not eat the machine time.
    - A `run_all.sh` chaining every experiment, with each one appending its result to the same CSV.
    - **Expected values** prepared: compute with the alpha-beta model what each one should be before running it, check immediately afterwards, and investigate on the spot if it is far off, rather than leaving it until you are off the machine.
    - Before finishing, save the raw output of `nvidia-smi -q` and `nccl-tests` together; you will need it for the review.

## The common traps {#常见的坑}

- **The port is taken, or last run's zombie processes are still there**: `--standalone` picks a random port, but processes left over from the last round still hold device memory and file locks. Run `pkill -f torch.distributed` first.
- **gloo cannot connect**: the hostname resolved to an external address. Set `GLOO_SOCKET_IFNAME=lo` (`lo0` on macOS).
- **So slow it looks hung**: `OMP_NUM_THREADS` is unset, so every process opens all the threads it can and they fight over the cores.
- **Deadlock**: some rank took a different branch and made one fewer collective call; or `send` before `recv` lines up into a cycle. A collective has to be **called by every rank in the same order**; point-to-point either uses `isend` / `irecv` with a single `wait` afterwards, or alternates by parity.
- **Inconsistent randomness**: different `torch.manual_seed` per rank gives different initial weights, so the results do not line up. Either use one seed, or `broadcast` one copy from rank 0.
- **Print only on rank 0**: without `if rank == 0` the output interleaves into a mess and you cannot tell right from wrong.

!!! interview "How to explain it"
    If multi-GPU experience comes up, say no plainly if you do not, but follow it immediately with **what you have verified and how**: implementing and verifying a ring all-reduce, ZeRO-2, 1F1B and Ring Attention from scratch with gloo on a CPU, all lined up elementwise against a single-process reference; computing the volume and the time with the alpha-beta model, so you know why a small message switches to a tree algorithm and why the ring's time barely grows with the card count; and, on real cards, measuring NCCL's bus bandwidth curve and tensor-parallel scaling, what you found and why. This carries far more weight than "I have used DeepSpeed", because it shows you understand the mechanism rather than the command line.

## Exercises {#练习}

1. Remove `dist.all_reduce(y)` from `tp_check.py` and see what the error becomes; then change `W1`'s partition from `dim=1` to `dim=0` and see what it is then. Explain what each of these two mistakes gets wrong mathematically.

??? success "Answer"
    Removing the all-reduce: each rank has computed only a partial sum (its own 64 hidden dimensions' contribution), so the result is about $1/P$ of the correct value and the error is the same order as the output itself.
    Partitioning `W1` by row: the shapes of `x @ W1_local` simply do not match (`x` has 64 columns while `W1_local` now has 16 rows), so it raises a shape error straight away. That kind of mistake is a good one, because it surfaces immediately. The genuinely dangerous case is when the shapes happen to match but the result is wrong, such as partitioning `W2` by column as well: the shapes are fine and the mathematics is not.

2. Use `commtime.py`'s model to answer: 64 cards under ZeRO-3 all-gather the parameters twice per step (once forward, once backward) and reduce-scatter the gradients once. A 7B model has 14 GB of bf16 parameters. How long is the communication in total? How much more is that than plain data parallelism?

??? success "Answer"
    Each of ZeRO-3's three collectives is on the order of half an all-reduce: the reduce-scatter and each all-gather are $(n-1)/n \cdot S$, so the three together are about $1.5 \times$ one all-reduce. With the 64-card, 14 GB numbers above, one all-reduce is about 149 ms and ZeRO-3 about 223 ms, roughly 50% more. This is exactly why you use the lowest stage that works; what the cost buys is memory per card dropping from $16\Psi$ to $16\Psi/N$.

3. Design an experiment you can do on a CPU that verifies the pipeline bubble fraction $(P-1)/(M+P-1)$, where $P$ is the stage count and $M$ the micro-batch count. What do you have to record?

??? success "Answer"
    You do not have to compute a model at all: have each rank simulate a stage of fixed duration with `time.sleep`, send and receive activations and gradients in 1F1B order, and record the **total idle time** of each rank between first starting work and finishing last. The bubble fraction is the idle time divided by the total. Sweep $M$ from 1 to 16 and the plot should land on the formula. This experiment needs no GPU at all, because the bubble comes from **scheduling**, not from compute.

## Summary {#小结}

- [x] Multi-GPU design and correctness can be 100% verified on a CPU: `torchrun --standalone` with gloo, the same `torch.distributed` API, which is how every multi-process example in this book runs.
- [x] The most effective method is **a single-process reference as the golden comparison**, compared elementwise. A partitioning mistake will almost certainly show up there.
- [x] Many performance conclusions can be **computed**: the alpha-beta model explains why small messages use a tree and large ones a ring, and why a ring all-reduce barely slows down as cards are added.
- [x] Only ten items genuinely need real cards, and one session of 8 cards over a few hours covers them; the discipline is to debug the scripts on a CPU first and spend the card time only running and recording.
