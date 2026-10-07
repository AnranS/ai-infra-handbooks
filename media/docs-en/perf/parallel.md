# Multi-GPU parallelism: spreading one step over several cards

<p class="lead">One 720p video takes over twenty minutes on a single card and nobody will wait for that; multi-GPU parallelism here is not a way to raise throughput but to bring <strong>one generation's latency</strong> down to something usable. Diffusion's parallelism is both like and unlike an LLM's: sequence parallelism (Ulysses, Ring) carries over almost unchanged, tensor parallelism works but does not pay, and the two methods an LLM does not have, guidance parallelism and PipeFusion, exploit exactly what is peculiar to diffusion: the two forward paths are independent of each other and neighbouring steps' activations are highly similar. This chapter works out each kind's communication volume, verifies Ulysses's all-to-all in one process on a CPU, and shows how a framework like xDiT combines them and at how many cards each pays off.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why does a diffusion model need several cards? How does the purpose differ from an LLM service's?
    2. What is guidance parallelism? How much does it communicate? Why is it capped at 2 cards?
    3. When do Ulysses and Ring sequence parallelism each communicate, and how is the volume computed? Which suits a video DiT better?
    4. What property does PipeFusion exploit? How does it relate to feature caching?
    5. Why is tensor parallelism not the first choice on a DiT?

??? success "Answers for the self-test (answer first, then open this)"
    1. For latency: a step is compute-bound, so however fast one card is it takes tens of seconds to tens of minutes, and several cards split one step's computation to do it at once. An LLM service uses several cards mainly to hold the model and raise throughput (tensor and pipeline parallelism), where a single request's latency is not the first objective.
    2. Guidance's conditional and unconditional paths differ only in the input condition and are independent, so two cards compute one each and exchange their outputs once at the end of each step (a tensor the size of one latent, a few megabytes), getting close to 2 times for almost no communication; but there are only two paths, so 2 cards is the ceiling and more has to combine with another kind.
    3. Ulysses: partition by token, do an all-to-all before attention to repartition by head (each card holding some heads of every token), and another all-to-all afterwards to partition back, so two all-to-alls per layer each moving on the order of $N d / P$. Ring: the same token partition, but attention passes the K and V blocks once around a ring, so each layer passes $(P-1)/P$ of the K and V, overlapped with the computation. With enough heads, Ulysses communicates less and is simpler; with few heads or more cards, Ring, and the two can be stacked (USP). A video DiT usually starts with Ulysses (cards at most the head count) and adds Ring when that is not enough.
    4. That neighbouring steps' activations are highly similar: cut the image into patches across the cards (like a pipeline), and when a card computes its own patch's attention it uses **the previous step's** K and V for the other patches, so the cards do not wait for each other and the whole thing pipelines asynchronously; the only communication is sending its own patch's K and V once per step, which also overlaps with the computation. It has the same root as feature caching: both trade the time dimension's redundancy for computation or communication.
    5. Tensor parallelism does two all-reduces per layer (after attention and after the MLP) with a volume proportional to the token count, so a video model's hundred thousand tokens means hundreds of megabytes per layer, and an all-reduce cannot overlap the computation; sequence parallelism's volume relates only to $N d / P$ and does overlap. Tensor parallelism's advantage is partitioning the weights to save memory, which a 12 to 14B DiT does not particularly need.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/media-parallel.webp is in Chinese; put it back once the English version exists -->

## Why parallelise: latency, not throughput {#为什么要并行延迟不是吞吐}

[The accounting chapter](accounting.md)'s conclusion: a step is compute-bound and a batch cannot amortise the cost. So several cards have one purpose, bringing one generation's latency down. The target is clear: video from 25 minutes to 3, an image from 5 seconds to 1.

There are five dimensions to cut along:

| Parallelism | What it cuts | Communication | Ceiling | Notes |
| --- | --- | --- | --- | --- |
| Data parallelism | different requests | none | the card count | raises throughput only, never latency |
| Guidance parallelism | the conditional and unconditional paths | one exchange of outputs per step | 2 | nearly free, use it first |
| Sequence parallelism (Ulysses, Ring) | the tokens | an all-to-all per layer, or K and V round a ring | the head count / unlimited | the mainstay for video models |
| PipeFusion | patches plus the timestep | its own patch's K and V per step | the patch count | uses stale activations, pipelines asynchronously |
| Tensor parallelism | the weights' rows and columns | two all-reduces per layer | the head count | saves memory; communication-heavy on a DiT |

There are also two parallelisms outside the denoising network: decoding the VAE in parallel by patch (mandatory for video), and putting the text encoder on a card of its own.

## Guidance parallelism: the cheapest 2 times {#cfg-并行最便宜的-2-倍}

Guidance's two forward passes are identical except for the condition and **do not depend on each other**: only the extrapolation at the end of each step needs both outputs. Two cards compute one each and exchange once per step:

```python
import torch

def cfg_parallel_cost(latent_shape, steps, dtype_bytes=2):
    numel = 1
    for s in latent_shape:
        numel *= s
    per_step = numel * dtype_bytes                  # each card sends its own path's prediction each step
    return per_step, per_step * steps

for name, shape, steps in [("FLUX 1024²", (16, 128, 128), 28), ("Wan 720p 81 帧", (16, 21, 90, 160), 50)]:
    per, total = cfg_parallel_cost(shape, steps)
    print(f"{name:<16} 每步交换 {per / 2 ** 20:>6.1f} MB，整次生成 {total / 2 ** 20:>7.1f} MB——NVLink 上几乎可以忽略")
```

```text title="output"
FLUX 1024²       每步交换    0.5 MB，整次生成    14.0 MB——NVLink 上几乎可以忽略
Wan 720p 81 帧    每步交换    9.2 MB，整次生成   461.4 MB——NVLink 上几乎可以忽略
```

The price is both cards holding the complete model, the gain is close to 2 times, and the implementation is a few dozen lines. Its ceiling is equally clear: there are only two paths, so it **always combines with another kind**: 8 cards = guidance 2 x sequence parallelism 4.

## Sequence parallelism: Ulysses and Ring {#序列并行ulysses-与-ring}

![Figure: the two kinds of sequence parallelism, with Ulysses converting between partitioning the sequence and the heads through an all-to-all, and Ring passing the K and V blocks round a ring](../assets/figures/ulysses-ring.svg){.aig-svg}

A DiT's body is the same as an LLM's Transformer block, so the Ulysses and Ring Attention from [Context parallelism](train://model/context/) apply unchanged. The tokens are split across cards, the linear layers and the MLP naturally compute independently (they act per token), and only attention has to see every token. Here is what each method does at that step:

**Ulysses**: an all-to-all before attention turns "every head of a segment of tokens on each card" into "some heads of every token on each card", each card computes complete attention for its own heads independently, and another all-to-all turns it back. Simulating 4 cards in one process and verifying the result matches the unpartitioned one exactly:

```python
import torch.nn.functional as F

torch.manual_seed(0)
B, H, N, D, P = 1, 8, 256, 32, 4                      # 8 heads, 256 tokens, 4 cards
q, k, v = (torch.randn(B, H, N, D) for _ in range(3))
ref = F.scaled_dot_product_attention(q, k, v)

# partition by token: card r holds every head of tokens [r*N/P, (r+1)*N/P)
shards = [(q[:, :, r * N // P:(r + 1) * N // P], k[:, :, r * N // P:(r + 1) * N // P], v[:, :, r * N // P:(r + 1) * N // P]) for r in range(P)]

def all_to_all_heads(tensors):
    """all-to-all：输入每卡 [B, H, N/P, D]（按 token 切），输出每卡 [B, H/P, N, D]（按头切）"""
    out = []
    for r in range(P):                                 # card r collects every card's r-th group of heads and concatenates them by token
        out.append(torch.cat([t[:, r * H // P:(r + 1) * H // P] for t in tensors], dim=2))
    return out

qh, kh, vh = (all_to_all_heads([s[i] for s in shards]) for i in range(3))
local_out = [F.scaled_dot_product_attention(qh[r], kh[r], vh[r]) for r in range(P)]   # each card: its own heads, every token

def all_to_all_tokens(tensors):                        # turn it back: each card goes from [B, H/P, N, D] to [B, H, N/P, D]
    return [torch.cat([t[:, :, r * N // P:(r + 1) * N // P] for t in tensors], dim=1) for r in range(P)]

back = all_to_all_tokens(local_out)
full = torch.cat(back, dim=2)
print(f"Ulysses（4 卡模拟）和不切的结果一致：{torch.allclose(full, ref, atol=1e-5)}")
sent = sum(t.numel() for t in [s[i] for s in shards for i in range(3)]) * 2 * (P - 1) / P   # the (P-1)/P share each card sends out
print(f"每层每卡发送约 {sent / P / 2 ** 10:.0f} KB（QKV 三个张量各 (P−1)/P 份），注意力后再发一次输出")
```

```text title="output"
Ulysses（4 卡模拟）和不切的结果一致：True
每层每卡发送约 72 KB（QKV 三个张量各 (P−1)/P 份），注意力后再发一次输出
```

**Ring**: the same token partition, but instead of exchanging heads, attention passes the K and V blocks round a ring, computing a local attention for each block received and merging with a log-sum-exp (the same formula as FlashAttention's online softmax; the training handbook's [Context parallelism](train://model/context/) has a runnable implementation). Its communication overlaps with the computation and is not limited by the head count.

Writing out each one's per-layer volume and substituting FLUX and Wan:

```python
def seq_parallel_comm(N, d, P, dtype=2):
    """每层每卡要发送的字节数"""
    ulysses = 2 * 4 * (N * d * dtype) * (P - 1) / P / P   # two all-to-alls, each moving its own share of QKV (3) plus the output (1)
    ring = 2 * (N * d * dtype) * (P - 1) / P              # one copy each of K and V, passed round the ring P−1 times, 1/P at a time
    return ulysses, ring

NVLINK = 300e9                                            # an H100's NVLink is about 900 GB/s both ways per card; estimated here at an effective 300 GB/s one way
print(f"{'模型 / 卡数':<22} {'Ulysses 每层':>12} {'Ring 每层':>10} {'Ulysses 每步(57/40 层)':>20} {'通信时间':>8}")
for name, N, d, L in [("FLUX 1024²", 4608, 3072, 57), ("Wan 720p 81 帧", 76112, 5120, 40)]:
    for P in (2, 4, 8):
        u, r = seq_parallel_comm(N, d, P)
        print(f"{name + ' / ' + str(P) + ' 卡':<22} {u / 2 ** 20:>9.1f} MB {r / 2 ** 20:>7.1f} MB {u * L / 2 ** 20:>17.0f} MB {u * L / NVLINK * 1e3:>6.1f} ms")
```

```text title="output"
模型 / 卡数                  Ulysses 每层    Ring 每层  Ulysses 每步(57/40 层)     通信时间
FLUX 1024² / 2 卡            54.0 MB    27.0 MB              3078 MB   10.8 ms
FLUX 1024² / 4 卡            40.5 MB    40.5 MB              2308 MB    8.1 ms
FLUX 1024² / 8 卡            23.6 MB    47.2 MB              1347 MB    4.7 ms
Wan 720p 81 帧 / 2 卡       1486.6 MB   743.3 MB             59462 MB  207.8 ms
Wan 720p 81 帧 / 4 卡       1114.9 MB  1114.9 MB             44597 MB  155.9 ms
Wan 720p 81 帧 / 8 卡        650.4 MB  1300.7 MB             26015 MB   90.9 ms
```

Against one step's computation (about 170 ms for FLUX on an H100, about 15 s for Wan): Ulysses's per-step communication is a few milliseconds for FLUX and on the order of a hundred for Wan, both far below the computation. **Sequence parallelism pays within NVLink.** Ring's volume looks larger, but it overlaps the computation entirely so its real cost is often lower; the price is a more complex implementation and slightly lower efficiency on small attention blocks.

The rule of thumb: with enough heads (FLUX's 24, Wan's 40) and cards no more than the head count, use Ulysses first; across machines or on more cards, use Ring, or stack the two (Ulysses within a machine and Ring between, which xDiT calls USP).

## PipeFusion: trading stale activations for the synchronisation {#pipefusion用旧激活换掉同步}

Sequence parallelism synchronises at every layer. PipeFusion takes another route: cut the image into $P$ patches across $P$ cards and pass each along like a pipeline when a card is done; the crucial step is that **when computing one patch's attention, the other patches' K and V come from the previous timestep**. The cards then do not wait for each other and the whole denoising pipelines asynchronously, with the only communication being its own patch's K and V, which also overlaps the computation.

Its premise is exactly [feature caching](caching.md)'s: neighbouring steps' activations are highly similar. So its error behaves the same way: the first and last few steps have to warm up (computed fully synchronously) with stale values in the middle. One number shows its communication advantage:

```python
def pipefusion_comm(N, d, L, P, dtype=2):
    """每步每卡：把自己 patch 的 K、V 发给其他卡，一共 L 层"""
    return L * 2 * (N / P) * d * dtype

for name, N, d, L in [("FLUX 1024²", 4608, 3072, 57), ("Wan 720p 81 帧", 76112, 5120, 40)]:
    for P in (4, 8):
        u, _ = seq_parallel_comm(N, d, P)
        print(f"{name:<16} {P} 卡：PipeFusion 每步每卡发 {pipefusion_comm(N, d, L, P) / 2 ** 20:>7.0f} MB，"
              f"Ulysses 每步 {u * L / 2 ** 20:>6.0f} MB，但 PipeFusion 不需要每层同步、可以跨 PCIe 机器")
```

```text title="output"
FLUX 1024²       4 卡：PipeFusion 每步每卡发     770 MB，Ulysses 每步   2308 MB，但 PipeFusion 不需要每层同步、可以跨 PCIe 机器
FLUX 1024²       8 卡：PipeFusion 每步每卡发     385 MB，Ulysses 每步   1347 MB，但 PipeFusion 不需要每层同步、可以跨 PCIe 机器
Wan 720p 81 帧    4 卡：PipeFusion 每步每卡发   14866 MB，Ulysses 每步  44597 MB，但 PipeFusion 不需要每层同步、可以跨 PCIe 机器
Wan 720p 81 帧    8 卡：PipeFusion 每步每卡发    7433 MB，Ulysses 每步  26015 MB，但 PipeFusion 不需要每层同步、可以跨 PCIe 机器
```

PipeFusion communicates two or three times less than Ulysses, but its greater value is **having no per-layer synchronisation point**: across PCIe, across machines or on poor bandwidth, sequence parallelism stalls on each layer's all-to-all while PipeFusion keeps flowing. The price is the error from the stale activations (to be controlled as with caching) and the pipeline's warm-up.

## Tensor parallelism: workable, but not first {#张量并行能用但不是首选}

Tensor parallelism splits each linear layer's weights by column or row across the cards (see [Tensor parallelism](serving://distributed/tensor-parallel/)), with an all-reduce after attention and another after the MLP. The volume:

```python
def tp_comm(N, d, L, P, dtype=2):
    return L * 2 * 2 * (N * d * dtype) * (P - 1) / P      # two all-reduces per layer, with a ring implementation sending 2(P−1)/P per card

for name, N, d, L in [("FLUX 1024²", 4608, 3072, 57), ("Wan 720p 81 帧", 76112, 5120, 40)]:
    u, _ = seq_parallel_comm(N, d, 4)
    print(f"{name:<16} 4 卡：张量并行每步每卡 {tp_comm(N, d, L, 4) / 2 ** 20:>7.0f} MB（不能与计算重叠），Ulysses {u * L / 2 ** 20:>6.0f} MB")
```

```text title="output"
FLUX 1024²       4 卡：张量并行每步每卡    4617 MB（不能与计算重叠），Ulysses   2308 MB
Wan 720p 81 帧    4 卡：张量并行每步每卡   89194 MB（不能与计算重叠），Ulysses  44597 MB
```

Tensor parallelism's communication is proportional to the token count, happens twice per layer and cannot overlap the computation, which is twice Ulysses's volume and a larger gap in practice. Its advantage is **partitioning the weights**: a 14B model across 4 cards is 7 GB each. But a DiT's weights fit anyway (twenty or thirty gigabytes at most), so the advantage goes unused. That is why a framework like xDiT puts tensor parallelism last: only when sequence and guidance parallelism are exhausted, cards remain, or one card genuinely cannot hold the weights.

## Combining: what xDiT does {#组合xdit-的做法}

In a real deployment these stack. Taking 8 H100s running Wan 2.1-14B at 720p:

| Configuration | Description | One video's latency (estimated) |
| --- | --- | --- |
| 1 card | 25 minutes | the baseline |
| Guidance 2 | the two paths separated | about 13 minutes |
| Guidance 2 x Ulysses 4 | 8 cards, a quarter of the tokens and one guidance path each | about 3.5 minutes |
| Guidance 2 x Ulysses 4 + TeaCache | caching added | about 2 minutes |
| Guidance 2 x Ulysses 4 + TeaCache + FP8 | low precision added | about 1.5 minutes |

The speedup is not linear: parallelism carries communication and load imbalance (the text tokens are awkward to split, and so are the patch boundaries) and caching costs quality. The published figures are roughly 6 to 7 times on 8 cards, and over 10 with caching and quantization on top.

xDiT (xDiT / xFuser) is the framework that turns these combinations into configuration: `--ulysses_degree 4 --ring_degree 2 --use_cfg_parallel --pipefusion_parallel_degree 2`, with the parallel groups' product equal to the card count. Its design is exactly this chapter's order: guidance first, then sequence parallelism, Ring or PipeFusion between machines, and tensor parallelism at the bottom. Later frameworks like SGLang Diffusion and vLLM-Omni use the same combination.

!!! interview "How to answer in an interview"
    Asked how video generation uses several cards, say the purpose first: lowering latency rather than raising throughput, since 25 minutes on one card is unusable. Then go by communication volume: guidance parallelism exchanges one latent per step and is nearly free but caps at 2 cards; sequence parallelism splits the tokens so that the linear layers compute independently and only attention communicates, with Ulysses doing two all-to-alls per layer (on the order of $Nd/P$) and Ring passing the K and V round a ring overlapped with the computation, so within NVLink the per-step communication is far below the computation; PipeFusion trades the previous step's K and V for the per-layer synchronisation and suits crossing machines; tensor parallelism does two non-overlapping all-reduces per layer at twice sequence parallelism's volume and is only for when the weights have to be split. Finish with the combination: 8 cards = guidance 2 x Ulysses 4, plus caching and FP8, taking Wan from 25 minutes to under 2.

## Exercises {#练习}

1. What happens if `P` in the Ulysses simulation becomes 16 (more than the 8 heads)? How does a real system handle it?

??? success "Answer"
    `H // P` is 0 and the heads cannot be partitioned: Ulysses's degree cannot exceed the head count (more precisely, the head count has to be divisible by it). Beyond that, either use Ring (unlimited by the head count) or stack Ulysses with Ring (USP), so 16 cards = Ulysses 8 x Ring 2.

2. Using this chapter's functions: on 8 cards connected by PCIe 4.0 (about 25 GB/s each), how long is Ulysses's per-step communication for Wan at 720p? How does it compare with one step's computation (about 15 s / 8)? Would PipeFusion be better?

??? success "Answer"
    Ulysses on 8 cards is about 1.6 GB per step, about 65 ms on PCIe 4.0, against about 2 seconds of computation per card, so 3% in communication: acceptable even over PCIe, because a video model's computation is so heavy. PipeFusion's advantage here is not bandwidth but the absence of synchronisation points: an all-to-all over 8 cards waits for the slowest, and an uneven PCIe topology amplifies the wait; but it introduces the stale-activation error. With enough bandwidth, prefer sequence parallelism.

3. A team runs FLUX with tensor parallelism on 4 cards and finds it only 1.8 times faster than one. By this chapter's accounting, where is the bottleneck? What configuration would be better?

??? success "Answer"
    Tensor parallelism does two all-reduces per layer (FLUX has 57 layers, so over a hundred per step), each a synchronisation point that cannot overlap the computation, and the smaller kernels lose efficiency too, so 1.8 times on 4 cards is quite common. Switch to guidance parallelism x2 (not applicable to FLUX.1-dev, which has no guidance) or Ulysses 4: two all-to-alls per layer moving less data, usually over 3 times; and add CUDA graphs and compilation to remove the small kernels' overhead.

## Summary {#小结}

- [x] A diffusion model uses several cards to lower latency: a step is compute-bound and a batch cannot amortise the cost; one card's 25 minutes for a video makes splitting mandatory.
- [x] Guidance parallelism is nearly free but caps at 2 cards; sequence parallelism (Ulysses's two all-to-alls per layer, Ring's K and V round a ring overlapped with the computation) is the mainstay for video models, with communication far below the computation within NVLink.
- [x] PipeFusion trades the previous step's K and V for the per-layer synchronisation, suits crossing machines, and shares feature caching's error behaviour; tensor parallelism communicates twice as much without overlapping and is only for splitting the weights.
- [x] A real deployment combines them: 8 cards = guidance 2 x Ulysses 4, plus caching and FP8; the published speedups are 6 to 7 times on 8 cards and over 10 with everything stacked.
