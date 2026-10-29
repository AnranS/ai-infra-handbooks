# 推理服务核心概念

<p class="lead">前面讲的都是"一个请求怎么算"。真实的推理服务要同时处理成百上千个请求，它们长短不一、随时到达、随时结束。这一章把 vLLM、SGLang 这类推理引擎的核心设计串起来：指标、连续批处理、KV Cache 管理与前缀缓存、分块 prefill、投机解码、PD 分离和并行策略。每个概念都配一个可运行的小实验。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. TTFT、TPOT、吞吐量、goodput 分别是什么？它们之间有什么矛盾？
    2. 连续批处理和静态批处理的区别？为什么前者吞吐高得多？
    3. 前缀缓存为什么能省计算？它和 RoPE、KV Cache 有什么关系？
    4. 投机解码为什么能在不改变输出分布的前提下加速？什么时候效果好？
    5. 什么是 PD 分离？它解决了什么问题？
    6. 负载接近容量时，为什么 TTFT 会突然暴涨，吞吐却几乎不再增加？
    7. 分块 prefill 的块是不是越小越好？

??? success "自测参考答案（先自己答，再展开对照）"
    1. TTFT：首 token 延迟（排队 + prefill）；TPOT：之后平均每个 token 的时间；吞吐：单位时间生成的 token 数；goodput：满足 SLO 的吞吐。batch 越大吞吐越高，但每步越慢、排队越长，延迟变差——两者矛盾。
    2. 静态批处理要等整批里最长的请求结束，短请求占着位置空转；连续批处理每一步重新组批，结束的立即离开、新来的立即加入，GPU 一直满载，吞吐高得多。
    3. 相同前缀的 KV 已经算过，直接复用就省掉了这部分 prefill。从第 1 层起，位置 i 的 K、V 取决于位置 ≤ i 的全部 token，K 里还带着按位置旋转的 RoPE，所以只有从开头起逐个 token 完全相同的前缀才能复用——这就是前缀缓存只能从头匹配的原因。
    4. 便宜的草稿模型先猜 k 个 token，目标模型一次前向并行验证；贪心验证时输出与逐个生成完全相同，采样时用拒绝采样保证分布不变。草稿接受率高（输出可预测）、batch 小（算力有富余）时效果好。
    5. 把 prefill 和 decode 放到不同的实例（甚至不同的硬件）上，中间传 KV：两个阶段不再互相干扰（长 prefill 不会卡住 decode），可以各自选择并行方式和机器配比；代价是 KV 传输和配比的调度。
    6. 请求到达的速度接近处理的速度时，处理不过来的请求只能排队，利用率越接近 100%，排队时间越趋于无穷，TTFT 就暴涨；吞吐已经到了服务的容量，再多的请求也只是在排队。本章的模拟里，吞吐一直涨到每秒 5～6 个请求，goodput（TTFT < 1 s 且 TPOT < 50 ms 的请求）却在每秒 3 个请求时就最高。
    7. 不是。块越小，长 prefill 对 decode 的阻塞越短（本章模拟里停顿的 p99 从 1.46 s 降到 84 ms），但长提示词要更多步才能 prefill 完，TTFT 变长；块太小时 prefill 跟不上请求到达的速度，队列积压，TTFT 反而暴涨。要按 SLO 调。

## 指标

| 指标 | 含义 | 主要受什么影响 |
| --- | --- | --- |
| **TTFT**（Time To First Token） | 从请求到达到收到第一个 token 的时间 | 排队时间 + prefill 时间 |
| **TPOT** / **ITL**（Time Per Output Token / Inter-Token Latency） | 之后每个 token 的间隔 | decode 每步的时间，受 batch 大小、上下文长度影响 |
| **端到端延迟** | TTFT + TPOT × (输出长度 − 1) | 两者之和 |
| **吞吐量** | 单位时间生成的 token 数（或处理的请求数） | batch 越大越高 |
| **goodput** | 满足延迟要求（SLO，比如 TTFT < 1 s 且 TPOT < 50 ms）的请求的吞吐 | 吞吐和延迟的平衡 |

吞吐和延迟天然矛盾：[batch 越大](kv-cache.md#批处理让多个请求分摊权重读取)，吞吐越高，但每个请求的 TPOT 也越长；新请求的 prefill 插进来，会让正在 decode 的请求卡顿一下。推理引擎的调度，就是在给定的 SLO 下最大化 goodput。

## 连续批处理

**静态批处理**：凑齐一批请求，一起跑到**全部**结束，再换下一批。问题在于输出长度差异很大：一个请求生成 20 个 token 就结束了，却要陪着同批次生成 500 个 token 的请求空转。

**连续批处理（continuous batching，也叫迭代级调度）**（Orca，2022）：**每一步**都重新决定哪些请求参与计算。结束的请求立即离开，等待的请求立即补上。GPU 上的 batch 始终尽量保持满载。

先看一个动画版的对比（静态批、连续批、分块 prefill 三种调度方式，推理系统手册里的同一个工具）：

<div class="aig-widget" data-widget="contbatch"></div>

用一个简化的模拟看看差距（decode 一步的耗时模型：固定开销 20 ms + 每个请求 0.2 ms，符合"访存瓶颈、batch 增大耗时几乎不变"的特点）：

```python
import random

def simulate(policy, n_requests=300, max_batch=32, seed=0):
    rng = random.Random(seed)
    lengths = [rng.randint(10, 400) for _ in range(n_requests)]      # 每个请求要生成的 token 数
    waiting = list(range(n_requests))
    running = {}                                                     # 请求 id -> 剩余 token 数
    t, finish = 0.0, {}
    while waiting or running:
        if policy == "continuous" or not running:                    # 静态批处理：只有整批跑完才补充
            while waiting and len(running) < max_batch:
                r = waiting.pop(0)
                running[r] = lengths[r]
        t += 20 + 0.2 * len(running)                                 # 一步 decode 的耗时（毫秒）
        for r in list(running):
            running[r] -= 1
            if running[r] == 0:
                finish[r] = t
                del running[r]
    return sum(lengths) / (t / 1000), sum(finish.values()) / n_requests / 1000

for policy in ("static", "continuous"):
    tput, latency = simulate(policy)
    print(f"{policy:10s} 吞吐 {tput:6.0f} token/s，平均完成时间 {latency:6.1f} s")
static_tput, _ = simulate("static")
cont_tput, _ = simulate("continuous")
assert cont_tput > 1.5 * static_tput
```

```text title="输出"
static     吞吐    696 token/s，平均完成时间   43.1 s
continuous 吞吐   1124 token/s，平均完成时间   27.4 s
```

连续批处理的吞吐高出约 60%，平均完成时间也缩短了三分之一以上。代价是实现复杂：batch 中每个请求的上下文长度不同、KV Cache 存放位置不同、处于不同阶段（有的在 prefill，有的在 decode），这要求注意力 kernel 支持"变长的、分页的"KV，调度器要逐步管理显存。这正是 vLLM 的 PagedAttention 和调度器要解决的问题。

### 实验：请求不断到来时，各个指标怎么变

上面的模拟是一次性给 300 个请求。真实的服务里，请求是陆续到来的。把模拟做得更像一点：请求按泊松过程到达，一成是 4000～8000 个 token 的长提示词；prefill 也算进每一步的耗时（每个 token 0.1 ms），新请求做完 prefill 才出第一个 token。然后把每秒到达的请求数从 1 加到 6：

```python
import random

def serve(rate, chunk=None, n=2000, max_batch=64, seed=0):
    """模拟一个用连续批处理的推理服务，返回各项指标。
    rate：每秒到达几个请求；chunk：每步最多 prefill 多少个 token，None 表示不分块。
    一步的耗时（毫秒）= 20 + 0.2 × 本步 decode 的请求数 + 0.1 × 本步 prefill 的 token 数"""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)                                           # 泊松到达
        prompt = rng.randint(4000, 8000) if rng.random() < 0.1 else rng.randint(100, 1000)   # 一成是长提示词
        reqs.append({"arrive": t, "todo": prompt, "out": rng.randint(50, 300), "times": []})
    now, nxt, active = 0.0, 0, []
    while nxt < n or active:
        while nxt < n and reqs[nxt]["arrive"] <= now and len(active) < max_batch:
            active.append(reqs[nxt])                                         # 有空位，排队的请求就进来
            nxt += 1
        if not active:
            now = reqs[nxt]["arrive"]
            continue
        budget = float("inf") if chunk is None else chunk
        n_decode, prefill, emit = 0, 0, []
        for r in active:
            if r["todo"] == 0:                                               # 在 decode：本步出一个 token
                n_decode += 1
                emit.append(r)
            elif budget > 0:                                                 # 在 prefill：吃掉一块提示词
                take = min(r["todo"], budget)
                r["todo"], budget, prefill = r["todo"] - take, budget - take, prefill + take
                if r["todo"] == 0:
                    emit.append(r)                                           # prefill 做完，出第一个 token
        now += (20 + 0.2 * n_decode + 0.1 * prefill) / 1000
        for r in emit:
            r["times"].append(now)
        active = [r for r in active if len(r["times"]) < r["out"]]

    def pct(xs, q):
        return sorted(xs)[int(q * (len(xs) - 1))]
    ttft = [r["times"][0] - r["arrive"] for r in reqs]
    tpot = [(r["times"][-1] - r["times"][0]) / (len(r["times"]) - 1) for r in reqs]
    stall = [max(b - a for a, b in zip(r["times"], r["times"][1:])) for r in reqs]   # 每个请求最长的一次停顿
    ok = sum(a < 1 and b < 0.05 for a, b in zip(ttft, tpot)) / n             # SLO：TTFT < 1 s 且 TPOT < 50 ms
    return {"tput": sum(r["out"] for r in reqs) / max(r["times"][-1] for r in reqs), "ok": ok,
            "ttft50": pct(ttft, 0.5), "ttft99": pct(ttft, 0.99), "tpot50": pct(tpot, 0.5), "stall99": pct(stall, 0.99)}

for rate in (1, 2, 3, 4, 5, 6):
    m = serve(rate)
    print(f"每秒 {rate} 个请求：吞吐 {m['tput']:3.0f} token/s，TTFT 中位数 {m['ttft50']:5.2f} s、p99 {m['ttft99']:5.2f} s，"
          f"TPOT 中位数 {m['tpot50'] * 1000:2.0f} ms，达标 {m['ok']:4.0%}（goodput {rate * m['ok']:.1f} 个/s）")
```

```text title="输出"
每秒 1 个请求：吞吐 178 token/s，TTFT 中位数  0.09 s、p99  0.81 s，TPOT 中位数 23 ms，达标 100%（goodput 1.0 个/s）
每秒 2 个请求：吞吐 354 token/s，TTFT 中位数  0.10 s、p99  0.97 s，TPOT 中位数 27 ms，达标  99%（goodput 2.0 个/s）
每秒 3 个请求：吞吐 529 token/s，TTFT 中位数  0.12 s、p99  1.28 s，TPOT 中位数 34 ms，达标  92%（goodput 2.8 个/s）
每秒 4 个请求：吞吐 703 token/s，TTFT 中位数  0.13 s、p99  1.65 s，TPOT 中位数 45 ms，达标  64%（goodput 2.6 个/s）
每秒 5 个请求：吞吐 874 token/s，TTFT 中位数  1.56 s、p99  5.89 s，TPOT 中位数 67 ms，达标   6%（goodput 0.3 个/s）
每秒 6 个请求：吞吐 893 token/s，TTFT 中位数 32.13 s、p99 56.75 s，TPOT 中位数 69 ms，达标   0%（goodput 0.0 个/s）
```

- 负载不高时（每秒 1～4 个请求），吞吐跟着请求数线性增长，TTFT 中位数一直在 0.1 s 左右：来一个处理一个，几乎不用排队；
- TPOT 随负载上升（23 → 45 ms）：batch 越大，每一步越慢，新请求的 prefill 也越来越频繁地插进来；
- 每秒 5 个请求时已经接近这台"服务器"的容量（约 890 token/s），吞吐只多了一点，TTFT 中位数却从 0.13 s 跳到 1.6 s；到每秒 6 个，请求来得比处理得快，队列越排越长，TTFT 涨到几十秒（模拟得越久越长）。这是排队论的结论：利用率接近 100% 时，等待时间趋于无穷；
- goodput 在每秒 3 个请求时最高，之后吞吐还在涨，goodput 却往下掉：越来越多的请求 TPOT 或 TTFT 超标，不再算数。所以线上服务不会把负载推到容量的极限，而是按 SLO 限流或扩容，见[性能与服务中的数学](math://performance-math/)里的排队论。

## KV Cache 管理与前缀缓存

[KV Cache](kv-cache.md#kv-cache-的显存管理) 一章讲过 PagedAttention 的块表设计。在此之上最重要的优化是**前缀缓存**：很多请求有相同的开头（系统提示、多轮对话的历史、few-shot 示例、同一份文档），它们的 KV Cache 完全一样（相同的 token 在相同的位置，[RoPE 旋转](../transformer/position.md)也相同），只需计算一次。

在真实模型上验证：先为一段较长的系统提示计算 KV Cache，然后两个不同的问题都直接复用它：

```python
import copy
import time
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer, generate

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

system = "你是一个推理优化方面的专家助手。" + "回答时要简洁、准确，必要时给出数字依据。" * 10
questions = ["什么是 KV Cache？", "什么是连续批处理？"]
full_ids = [tok(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids
            for q in questions]
# 两个请求的公共前缀
n_prefix = 0
while full_ids[0][0, n_prefix] == full_ids[1][0, n_prefix]:
    n_prefix += 1

with torch.no_grad():
    prefix_cache = KVCache(model.cfg.num_hidden_layers)
    model(full_ids[0][:, :n_prefix], prefix_cache)                   # 前缀只算一次
    for ids in full_ids:
        cache = copy.copy(prefix_cache)                              # 共享前缀的 KV（张量不会被原地修改）
        cache.k, cache.v = list(prefix_cache.k), list(prefix_cache.v)
        reused = model(ids[:, n_prefix:], cache)[:, -1]              # 只计算各自不同的后缀
        full = model(ids)[:, -1]                                     # 对照：从头完整计算
        assert (reused - full).abs().max() < 1e-3
print(f"提示词共 {full_ids[0].shape[1]} 个 token，其中公共前缀 {n_prefix} 个；复用前缀后，每个请求只需 prefill "
      f"{full_ids[0].shape[1] - n_prefix} 个 token，结果与完整计算一致")
```

```text title="输出"
提示词共 159 个 token，其中公共前缀 147 个；复用前缀后，每个请求只需 prefill 12 个 token，结果与完整计算一致
```

请求一多，缓存的前缀就长成一棵树：公共的部分只存一份，分叉的地方各存各的。换几种请求序列，看这棵树怎么长、每个新请求能命中多少：

<div class="aig-widget" data-widget="radixcache"></div>

前缀必须从开头起逐个 token 完全相同才能复用，中间改一个 token，后面的 KV 就全变了，[KV Cache](kv-cache.md#实验历史-token-的-kv-会不会变) 一章做过这个实验。

!!! inference "推理视角"
    vLLM 按块对 token 内容做哈希来识别可复用的前缀；SGLang 的 **RadixAttention** 用一棵基数树组织所有缓存的 token 序列，自然地支持多个请求共享任意长度的公共前缀，并配合 LRU 淘汰。在多轮对话、Agent（反复调用、上下文不断增长）、批量评测（同一套 few-shot 示例）等场景下，前缀缓存的命中率可以非常高，TTFT 大幅降低。调度器还可以优先调度能命中缓存的请求（cache-aware scheduling）。

## 分块 prefill

一个很长的提示词（比如 32K token）的 prefill 可能要几百毫秒到几秒。如果一次做完，这段时间内所有正在 decode 的请求都得等着，TPOT 出现尖峰。**分块 prefill（chunked prefill）** 把长 prefill 切成若干块（比如每块 512 或 2048 个 token），每一步只做一块，和其他请求的 decode 放在同一个 batch 里一起计算：

- decode 请求不再被长 prefill 阻塞，TPOT 更平稳；
- prefill 块（计算密集）和 decode（访存密集）混在一起，硬件利用更均衡。

分块 prefill 在数学上与一次性 prefill 完全等价，因为每一块都能通过 KV Cache 看到之前所有块：

```python ci="loose"
ids = full_ids[0]
with torch.no_grad():
    one_shot = model(ids)[:, -1]
    cache = KVCache(model.cfg.num_hidden_layers)
    for start in range(0, ids.shape[1], 32):                         # 每次 prefill 32 个 token
        chunk_logits = model(ids[:, start:start + 32], cache)
diff = (chunk_logits[:, -1] - one_shot).abs().max().item()
print(f"一次 prefill {ids.shape[1]} 个 token，和每次 32 个、分 {(ids.shape[1] + 31) // 32} 次比：最后一个位置的 logits 最大差异 {diff:.1e}")
assert diff < 1e-3
```

```text title="输出"
一次 prefill 159 个 token，和每次 32 个、分 5 次比：最后一个位置的 logits 最大差异 3.2e-05
```

### 实验：块大小怎么选

用上面的服务模拟，固定每秒 4 个请求，只改每一步最多 prefill 多少个 token，看每个请求最长的那次停顿（相邻两个输出 token 的最大间隔）和 TTFT：

```python
for chunk in (None, 2048, 512, 256):
    m = serve(4, chunk)
    print(f"块大小 {chunk or '不分块'}：最长停顿 p99 {m['stall99'] * 1000:4.0f} ms，TTFT 中位数 {m['ttft50']:.2f} s、p99 {m['ttft99']:.2f} s")
```

```text title="输出"
块大小 不分块：最长停顿 p99 1460 ms，TTFT 中位数 0.13 s、p99 1.65 s
块大小 2048：最长停顿 p99  237 ms，TTFT 中位数 0.14 s、p99 1.70 s
块大小 512：最长停顿 p99   84 ms，TTFT 中位数 0.27 s、p99 2.94 s
块大小 256：最长停顿 p99   57 ms，TTFT 中位数 1.40 s、p99 7.02 s
```

- 不分块时，一个 8000 token 的提示词要 prefill 800 ms，这期间所有正在 decode 的请求都停着；几个长提示词赶到同一步，停得更久。最长停顿的 p99 是 1.46 s，用户看到的就是输出卡住了一下；
- 块越小，停顿越短（2048 → 237 ms，512 → 84 ms）。代价是长提示词要分好几步才能 prefill 完，TTFT 变长；
- 块太小（256）就适得其反：每一步能 prefill 的 token 太少，prefill 跟不上请求到达的速度，队列排了起来，TTFT 中位数涨到 1.4 s。所以块大小是一个要按 SLO 调的参数，vLLM 里是 `max_num_batched_tokens`，SGLang 里是 `chunked_prefill_size`。

## 投机解码

decode 慢在"每步只算一个 token，却要读一遍全部权重"。**投机解码（speculative decoding）** 的思路是：先用一个便宜的方法**猜**出接下来的 k 个 token，再让大模型**一次前向**同时验证这 k 个位置。因为验证 k 个 token 和生成 1 个 token 读的权重一样多（多出的计算在访存瓶颈下几乎免费），只要猜中的比例足够高，就能在一次前向里前进多个 token。

**贪心验证**：大模型在每个位置取 argmax，与草稿逐个比较，接受最长的匹配前缀，再加上大模型在第一个不匹配位置给出的 token。所以**每次验证至少前进 1 个 token，输出与普通贪心解码完全相同**。

草稿的接受率、宽度和深度怎么决定加速比，用这个工具拨一拨（推理系统手册里的同一个工具）：

<div class="aig-widget" data-widget="spectree"></div>

"猜"的方法有很多：一个小的草稿模型（同系列的小模型）、模型自带的额外预测头（Medusa、EAGLE、DeepSeek-V3 的 [MTP](../training/pretraining.md#多-token-预测)），或者最简单的 **n-gram 查找（prompt lookup）**：如果最后几个 token 在上文中出现过，就把上文中紧随其后的 token 当作草稿。在复述、改写、代码编辑等输出大量重复输入内容的任务里，它的命中率非常高：

```python
def truncate(cache, n):
    """丢掉被拒绝的草稿 token 的 KV。"""
    for layer in range(len(cache.k)):
        cache.k[layer] = cache.k[layer][:, :, :n]
        cache.v[layer] = cache.v[layer][:, :, :n]

def ngram_draft(seq, k, n=3):
    """在上文中查找最后 n 个 token 上一次出现的位置，把紧随其后的 k 个 token 作为草稿。"""
    tail = seq[-n:]
    for start in range(len(seq) - n - 1, -1, -1):
        if seq[start:start + n] == tail:
            return seq[start + n:start + n + k]
    return []

@torch.no_grad()
def speculative_generate(model, ids, max_new, k=6, eos=None):
    cache = KVCache(model.cfg.num_hidden_layers)
    nxt = model(ids, cache)[0, -1].argmax().item()
    seq, out, target_calls = ids[0].tolist(), [], 1
    while len(out) < max_new:
        out.append(nxt)
        seq.append(nxt)
        if nxt == eos:
            break
        draft = ngram_draft(seq, k)
        base = cache.length
        preds = model(torch.tensor([[nxt] + draft]), cache)[0].argmax(-1).tolist()   # 一次前向验证全部草稿
        target_calls += 1
        n_ok = 0
        while n_ok < len(draft) and preds[n_ok] == draft[n_ok] and len(out) < max_new:
            out.append(draft[n_ok])
            seq.append(draft[n_ok])
            n_ok += 1
            if out[-1] == eos:
                break
        if out[-1] == eos:
            break
        truncate(cache, base + 1 + n_ok)                              # 回滚未被接受的草稿
        nxt = preds[n_ok]                                             # 大模型在第一个不匹配处的预测
    return out, target_calls

para = "推理服务的调度器在每一步决定哪些请求参与计算。连续批处理允许新请求在任意一步加入，已完成的请求立即离开，从而保持较高的硬件利用率。"
msgs = [{"role": "user", "content": "请把下面这段话原样重复一遍，不要做任何修改：\n" + para}]
ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids

t0 = time.perf_counter()
greedy = generate(model, ids, 80, eos_token_id=tok.eos_token_id)[0].tolist()
t_greedy = time.perf_counter() - t0
t0 = time.perf_counter()
spec, calls = speculative_generate(model, ids, 80, eos=tok.eos_token_id)
t_spec = time.perf_counter() - t0
assert spec == greedy                                                 # 输出与普通贪心逐 token 相同
print(f"生成 {len(greedy)} 个 token：普通解码 {len(greedy)} 次前向（{t_greedy:.2f} s），"
      f"投机解码 {calls} 次前向（{t_spec:.2f} s）")
```

在本手册的环境里：

```text
生成 38 个 token：普通解码 38 次前向（1.83 s），投机解码 8 次前向（0.58 s）
```

38 个 token 只用了 8 次大模型前向，输出逐字相同。

### 随机采样时的投机解码

采样时不能简单地比较 argmax。标准的**投机采样**（Leviathan 等、Chen 等，2023）用拒绝采样保证输出分布与大模型完全一致：草稿模型的分布为 q、大模型为 p，对草稿给出的 token x：

- 以概率 $\min(1, p(x)/q(x))$ 接受；
- 若拒绝，从修正分布 $\text{norm}(\max(0, p - q))$ 中重新采样一个 token，本轮结束。

用蒙特卡洛模拟验证"输出分布等于 p"：

```python
torch.manual_seed(0)
p = torch.tensor([0.5, 0.3, 0.15, 0.05])          # 大模型的分布
q = torch.tensor([0.25, 0.25, 0.25, 0.25])        # 草稿模型的分布（很差的草稿）

def speculative_sample(p, q):
    x = torch.multinomial(q, 1).item()                               # 草稿模型提议
    if torch.rand(()) < min(1.0, (p[x] / q[x]).item()):
        return x, True                                               # 接受
    residual = (p - q).clamp(min=0)
    return torch.multinomial(residual / residual.sum(), 1).item(), False

n = 200_000
samples, accepted = zip(*(speculative_sample(p, q) for _ in range(n)))
freq = torch.bincount(torch.tensor(samples), minlength=4).float() / n
print("输出分布", [round(v, 3) for v in freq.tolist()], " 目标分布", [round(v, 3) for v in p.tolist()],
      f" 接受率 {sum(accepted) / n:.3f}（理论值 Σmin(p,q) = {torch.minimum(p, q).sum().item():.3f}）")
assert (freq - p).abs().max() < 0.01
```

```text title="输出"
输出分布 [0.499, 0.3, 0.151, 0.05]  目标分布 [0.5, 0.3, 0.15, 0.05]  接受率 0.699（理论值 Σmin(p,q) = 0.700）
```

即使草稿很差，输出分布也与大模型分毫不差，只是接受率低、加速少。**接受率** $\sum_x \min(p(x), q(x))$ 衡量草稿与大模型的接近程度，直接决定加速比。

!!! inference "推理视角"
    投机解码在**小 batch、低延迟**的场景收益最大：此时 decode 严重访存受限，验证多个 token 几乎不增加耗时。batch 很大时，GPU 已经接近计算瓶颈，额外验证的草稿 token 不再"免费"，收益会下降甚至变负。此外它需要回滚 KV Cache（被拒绝的草稿 token 的 KV 要作废）、与 CUDA Graphs 和分页 KV 配合，工程上相当复杂。EAGLE 系列、MTP 是目前推理引擎中最常用的方案。

## PD 分离

prefill 是计算密集的，decode 是访存密集的，把它们放在同一组 GPU 上会互相干扰：新请求的 prefill 让 decode 卡顿（TPOT 抖动），decode 的大量小步又拖慢 prefill（TTFT 变长）。**PD 分离（prefill-decode disaggregation）** 把两个阶段放到不同的 GPU（甚至不同的机器）上：

1. prefill 实例处理提示词，生成 KV Cache；
2. KV Cache 通过高速网络（NVLink、RDMA）传给 decode 实例；
3. decode 实例继续生成。

两类实例可以使用不同的并行策略、batch 大小，甚至不同型号的 GPU，分别优化 TTFT 和 TPOT。代价是 KV Cache 的传输开销和系统复杂度。DistServe、Splitwise、Mooncake（Kimi）等工作推动了这一方向，vLLM、SGLang、NVIDIA Dynamo 都已支持。

传 KV 要花多久？用[估算](estimation.md)一章的工具算一笔账：

```python
from estimate import H100, kv_bytes_per_token, prefill_ms

llama3_8b = dict(model_type="llama", vocab_size=128256, hidden_size=4096, intermediate_size=14336,
                 num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8)
T, link = 4096, 50e9                                                  # 4K 提示词；400 Gb/s 的 RDMA 网卡约 50 GB/s
size = T * kv_bytes_per_token(llama3_8b)
print(f"LLaMA-3-8B，{T} 个 token：KV 共 {size / 1e6:.0f} MB，传过去 {size / link * 1000:.1f} ms；"
      f"一张 H100 上 prefill 这些 token 约 {prefill_ms(llama3_8b, T, H100):.0f} ms")
```

```text title="输出"
LLaMA-3-8B，4096 个 token：KV 共 537 MB，传过去 10.7 ms；一张 H100 上 prefill 这些 token 约 133 ms
```

传 KV 的时间不到 prefill 的一成，而且可以一层算完就传一层，和后面的层的计算重叠起来。所以 PD 分离的难点通常不在带宽上，而在调度：prefill 和 decode 实例的数量怎么配、KV 传过去之前 decode 端的显存怎么预留、某一端成了瓶颈怎么办。

## 推理中的并行

| 并行方式 | 切分 | 通信 | 适合 |
| --- | --- | --- | --- |
| 张量并行（TP） | 每层的矩阵按头 / 按维度切到多卡 | 每层 2 次 all-reduce | 单机内（NVLink），降低单请求延迟、容纳大模型 |
| 流水线并行（PP） | 按层切分 | 相邻阶段点对点 | 跨机，通信少，但单请求延迟不降 |
| 数据并行（DP） | 每张卡（或每组卡）一个完整的副本 | 无 | 扩展吞吐 |
| 专家并行（EP） | MoE 的专家分到不同卡 | 每个 MoE 层 2 次 all-to-all | MoE 模型 |
| DP Attention | 注意力部分按请求做数据并行，MoE 部分做专家并行 | all-to-all / all-gather | MLA 这类 KV 无法按头切分的模型（DeepSeek） |

### 实验：张量并行，切开算再加起来

为什么张量并行每层只要两次 all-reduce？拿 Qwen3-0.6B 第 0 层真实的 SwiGLU，模拟两张卡：

```python ci="loose"
import torch.nn.functional as F

torch.manual_seed(0)
mlp, tp = model.layers[0].mlp, 2
x = torch.randn(4, model.cfg.hidden_size)                             # 4 个 token 的输入，每张卡都有一份
with torch.no_grad():
    full = mlp(x)
    partial = []
    for rank in range(tp):
        part = slice(rank * mlp.up_proj.out_features // tp, (rank + 1) * mlp.up_proj.out_features // tp)
        h = F.silu(x @ mlp.gate_proj.weight[part].T) * (x @ mlp.up_proj.weight[part].T)   # 列切：各算一半的中间维度
        partial.append(h @ mlp.down_proj.weight[:, part].T)           # 行切：得到形状完整的部分和
    out = sum(partial)                                                # all-reduce：把各卡的部分和加起来
n_full = sum(w.numel() for w in mlp.parameters())
print(f"单卡和 {tp} 卡张量并行的最大差异 {(out - full).abs().max():.1e}；每张卡存 {n_full // tp / 1e6:.1f}M 个参数（单卡 {n_full / 1e6:.1f}M），"
      f"中间的 {mlp.up_proj.out_features // tp} 维各算各的，最后 all-reduce 一次，传 {out.numel()} 个数")
```

```text title="输出"
单卡和 2 卡张量并行的最大差异 7.2e-07；每张卡存 4.7M 个参数（单卡 9.4M），中间的 1536 维各算各的，最后 all-reduce 一次，传 4096 个数
```

- gate、up 按输出维度切（列切），每张卡算出中间向量的一段；SwiGLU 是逐元素的，各算各的就行，不需要通信；down 按输入维度切（行切），正好用上自己那一段，得到的是完整形状的部分和，最后加一次。这就是 Megatron-LM 的切法。注意力也一样：q、k、v 按头切，o 按行切，所以每层两次 all-reduce；
- 每张卡只存、只读一半的权重。decode 读权重是瓶颈，卡数翻倍，每张卡要读的字节减半，这就是[估算](estimation.md#延迟的下限)一章里 70B 做 8 卡张量并行后，单请求延迟和单卡 8B 差不多的原因；
- 代价是通信：每次 all-reduce 只传 batch × d 个数，量很小，但次数多（28 层 × 2 = 56 次），每次都要等最慢的那张卡，对延迟很敏感。所以张量并行一般只在用 NVLink 连着的单机内做，跨机用流水线并行或数据并行。

## 把它们串起来：一个推理引擎的组成

```text
API 服务（OpenAI 兼容接口）
  │ 分词、应用对话模板
  ▼
调度器 ──────────────────────────────── 每一步决定：哪些请求 decode，哪些新请求 prefill（分块），
  │                                      是否抢占；为它们分配 KV Cache 块、查找前缀缓存
  ▼
模型执行器（每张 GPU 一个 worker）
  │ 构造批次输入（变长序列、块表、位置）
  │ 前向：融合算子 + 注意力后端（FlashAttention / FlashInfer）+ 量化 GEMM + TP/EP 通信
  │ decode 使用 CUDA Graphs
  ▼
采样器（温度、top-p、惩罚、约束解码、投机解码的验证）
  ▼
反分词与流式返回
```

vLLM 和 SGLang 的主要源码目录几乎可以一一对应到这张图上。SGLang 还把"调度下一批"的 CPU 工作与"执行当前批"的 GPU 工作重叠起来（overlap scheduler），进一步减少 GPU 的空闲。

!!! interview "怎么讲清楚"
    讲推理服务的核心指标和手段：先定义 TTFT（排队 + prefill）、TPOT（decode 每个 token）、goodput（满足 SLO 的吞吐），说明吞吐和延迟是矛盾的，调度的目标是 SLO 下的 goodput 最大；再列手段并说清各自解决哪个指标：连续批处理（逐步调度，提吞吐）、前缀缓存与分块 prefill（数学上等价，降 TTFT、稳 TPOT）、投机解码（贪心验证不改输出、拒绝采样不改分布，降 TPOT）、PD 分离（两个阶段放到不同的硬件上）以及 TP、PP、DP、EP、DP Attention 这几种并行。

!!! note "延伸阅读"
    - [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)（Anyscale）：讲清静态批处理和连续批处理的区别，再在真实 GPU 上测吞吐和延迟，"23 倍吞吐"的说法就来自这篇；
    - [Continuous batching from first principles](https://huggingface.co/blog/continuous_batching)（Hugging Face）：从注意力和 KV Cache 出发，一步步推到分块 prefill 和连续批处理；
    - [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)（OSDI 2022）：迭代级调度的原始论文；[Sarathi-Serve](https://arxiv.org/abs/2403.02310)：分块 prefill 怎么在吞吐和延迟之间取舍；
    - [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/)（LMSYS）：用基数树做前缀缓存；
    - [Throughput is Not All You Need](https://hao-ai-lab.github.io/blogs/distserve/)（DistServe）：goodput 的定义和 PD 分离的动机，讲得很清楚；[Mooncake](https://arxiv.org/abs/2407.00079)：Kimi 以 KV Cache 为中心的分离式架构；
    - 投机解码：[Leviathan 等](https://arxiv.org/abs/2211.17192)的原始论文、[EAGLE](https://arxiv.org/abs/2401.15077)；张量并行：[Megatron-LM](https://arxiv.org/abs/1909.08053)。

## 练习

**1. 读懂 SLO。** 一个服务的 SLO 是 TTFT < 500 ms、TPOT < 40 ms。在高峰期，你发现 TPOT 经常超标但 TTFT 正常，可能的原因和对策是什么？

??? success "参考思路"
    TPOT 超标通常说明 decode 每一步太慢：batch 太大（每步要读的 KV 太多）、上下文太长，或者大的 prefill 插进来造成卡顿。对策：限制最大 batch（牺牲吞吐换延迟）；启用或调小分块 prefill 的块大小，避免长 prefill 阻塞 decode；使用 KV Cache 量化减少读取量；启用投机解码降低单 token 延迟；如果 prefill 与 decode 干扰严重，考虑 PD 分离。然后用监控确认改动的效果。

**2. 估算接受率与加速。** 投机解码每轮提议 k = 4 个草稿 token，每个草稿被接受的概率都是 α = 0.7（且相互独立，前一个被拒绝则后面全部作废）。每轮平均前进多少个 token？

??? success "参考答案"
    每轮前进的 token 数 = 被接受的草稿数 + 1（大模型在第一个拒绝处给出的 token，或者全部接受后的下一个 token）。期望值为 $\sum_{i=0}^{k} \alpha^i = (1 - \alpha^{k+1}) / (1 - \alpha)$：

    ```python
    alpha, k = 0.7, 4
    print(f"{(1 - alpha ** (k + 1)) / (1 - alpha):.2f}")   # 约 2.77
    ```

    每次大模型前向平均前进约 2.77 个 token。实际加速比还要扣除生成草稿的开销和验证多个 token 带来的额外计算。

## 小结

- [x] TTFT 看 prefill 和排队，TPOT 看 decode；吞吐与延迟矛盾，调度目标是在 SLO 下最大化 goodput。
- [x] 连续批处理逐步调度，结束的请求立即离开、新请求立即加入，吞吐远高于静态批处理。
- [x] 前缀缓存复用相同前缀的 KV Cache；分块 prefill 让长 prefill 不阻塞 decode，二者都在数学上与原计算等价。
- [x] 投机解码用便宜的草稿加一次验证前进多个 token；贪心验证输出不变，拒绝采样保证采样分布不变。
- [x] 负载接近容量时 TTFT 暴涨，goodput 比吞吐更早见顶：线上服务要按 SLO 限流和扩容。
- [x] 分块 prefill 的块大小在"decode 停顿"和"TTFT"之间取舍，太小反而让 prefill 跟不上。
- [x] PD 分离把两个性质不同的阶段放到不同的硬件上，传 KV 的时间远小于 prefill 本身；推理并行有 TP、PP、DP、EP 和 DP Attention。
- [x] 张量并行：列切加行切，中间不用通信，每层两次 all-reduce；每张卡只读一部分权重，所以能降低 decode 延迟。

相关的数学：投机解码接受率 = 1 − 总变差距离的证明见[概率与采样](math://probability/)；Little 定律、排队论与 P99 的置信区间见[性能与服务中的数学](math://performance-math/)。
