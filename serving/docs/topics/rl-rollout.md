# RL 训练中的推理

<p class="lead">推理模型（o1、DeepSeek-R1 这一类）靠大规模强化学习训练出来，而强化学习的每一步都要先让当前的模型生成大量回答（rollout），再根据奖励更新模型。rollout 通常占据 RL 训练一半以上的时间，推理引擎因此成了训练系统的核心部件，也带来了在线服务中不存在的新问题：长尾生成拖慢整批、训练与推理算出的概率不一致、每一步都要把新权重同步给推理引擎、训练和推理争抢同一批 GPU 的显存。这一章用模拟和实测把这些问题讲清楚。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么 rollout 的 GPU 利用率往往很低？有哪些缓解办法？
    2. 用推理引擎采样出的 token，在训练端重新计算 log 概率，结果会一样吗？差多少？这对 RL 有什么影响？
    3. 训练和推理共用 GPU 时，怎样在两者之间切换显存？权重怎样同步？
    4. 同步 RL 与异步 RL 的区别是什么？异步的代价是什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 回答长度是长尾分布：每一批都要等最长的几条写完，大部分时间只剩少数请求在跑，batch 很小、GPU 利用率低。缓解：partial rollout（把没写完的留到下一轮继续）、异步 RL、过量采样（多发一些请求，够数就停）。
    2. 不会完全一样：推理端和训练端的 kernel、批处理方式、并行方式不同，BF16 下概率可以相差百分之几十。这让 RL 实际上变成了 off-policy，梯度有偏；要用重要性采样修正（TIS、MIS），或者用 batch 不变、和训练端一致的 kernel。
    3. 共置时，推理引擎在训练阶段进入休眠：释放 KV 和权重占用的物理显存、保留虚拟地址（sleep），训练完再唤醒（wake）。权重用 CUDA IPC（同一台机器）、NCCL 广播或按层的流水线同步给推理引擎，训练端和推理端的切分不同时要重新切分。
    4. 同步 RL 每一步都用最新的权重生成、生成完再训练；异步 RL 让生成和训练重叠进行（样本来自稍旧的权重）。异步的代价是策略陈旧：样本不是当前策略生成的，需要重要性修正或者限制陈旧度。

## 一个 RL 训练步

以 GRPO 为例（大模型手册的[后训练](llm://training/post-training/#推理模型用可验证的奖励做强化学习)一章讲了算法）：

1. **rollout**：对一批问题，每个问题采样 G 个回答（例如 512 个问题 × 8 个回答）——这是纯推理，由 vLLM 或 SGLang 完成；
2. **打分**：用规则或奖励模型给每个回答打分，计算组内的相对优势；
3. **计算概率**：训练框架（FSDP、Megatron）对所有回答做一次前向，得到当前策略（和参考模型）的 log 概率；
4. **更新**：反向传播、优化器更新；
5. **同步权重**：把新权重传给推理引擎，进入下一步。

训练框架与推理引擎可以**共置**（同一批 GPU 上轮流运行，verl 的默认方式）或**分离**（各用一批 GPU，slime、AReaL 等常见）。训练端的视角——一个完整的 GRPO 最小循环、训练端与推理端的权重重新切分、共置与分离的取舍——见分布式训练手册的[训练框架与 RL 训练系统](train://practice/frameworks-rl/)。

## 问题一：长尾

推理模型的回答长度差别极大：多数几百到几千个 token，少数会写到上万个。rollout 要等**最长的那个**结束才能进入下一步。模拟一个推理引擎（7B 模型，单卡，最多 512 个并发）处理 1024 个回答，长度服从对数正态分布：

```python
import math
import random

random.seed(0)
params, weights, kv_per_token, bw, peak, prompt_len = 7.6e9, 15.2e9, 57344, 3.35e12, 989e12, 512

def step_time(batch, context_tokens):              # decode 一步：屋顶线估计
    return max(2 * params * batch / (peak * 0.5), (weights + kv_per_token * context_tokens) / bw) + 0.4e-3

lengths = [min(16384, max(64, int(random.lognormvariate(math.log(1500), 0.9)))) for _ in range(1024)]

def rollout(lengths, max_seqs=512, cap=None):
    queue, running, elapsed, timeline, tokens = list(range(len(lengths))), {}, 0.0, [], 0
    target = [l if cap is None else min(l, cap) for l in lengths]
    while queue or running:
        while queue and len(running) < max_seqs:
            running[queue.pop(0)] = 0
        dt = step_time(len(running), sum(prompt_len + g for g in running.values()))
        elapsed += dt
        timeline.append((dt, len(running)))
        tokens += len(running)
        for i in list(running):
            running[i] += 1
            if running[i] >= target[i]:
                del running[i]
    return elapsed, timeline, tokens

elapsed, timeline, tokens = rollout(lengths)
full_speed = 512 / step_time(512, 512 * (prompt_len + 1500))              # 满批次时每秒生成的 token
low = sum(dt for dt, batch in timeline if batch < 128) / elapsed
print(f"回答长度：中位数 {sorted(lengths)[512]}，最长 {max(lengths)}")
print(f"rollout 用时 {elapsed:.0f} s；如果全程保持满批次，只需 {tokens / full_speed:.0f} s")
print(f"有 {low:.0%} 的时间，批次中的请求不到 128 个（容量的四分之一）")
capped, _, capped_tokens = rollout(lengths, cap=4096)
print(f"每轮最多生成 4096 个 token（partial rollout）：本轮用时 {capped:.0f} s，"
      f"{sum(l > 4096 for l in lengths)} 个回答留到下一轮继续")
```

```text title="输出"
回答长度：中位数 1549，最长 16384
rollout 用时 203 s；如果全程保持满批次，只需 102 s
有 54% 的时间，批次中的请求不到 128 个（容量的四分之一）
每轮最多生成 4096 个 token（partial rollout）：本轮用时 96 s，145 个回答留到下一轮继续
```

一半以上的时间花在"批次里只剩少数几个长回答"的阶段：此时 decode 严重访存受限，GPU 的大部分能力被浪费。常见的缓解办法：

- **partial rollout**：每一轮给生成设一个上限，没写完的回答保存状态，下一轮接着写（用的是稍旧的策略，引入一点 off-policy）；
- **异步 RL**：推理引擎不停地生成，训练端有足够的样本就开始更新，两者流水线化（AReaL、slime 的异步模式）；代价同样是策略的滞后，需要重要性采样等方式修正；
- **过量采样**：发出比需要更多的请求，够数之后丢弃（或留到下一轮）剩下的长尾；
- **更大的推理规模、PD 分离、投机解码**：长尾阶段 batch 很小，正是投机解码最有效的区间。

## 问题二：训练与推理的概率不一致

RL 的损失用到了"采样时的概率"。理论上，推理引擎采样时的概率，应该与训练框架对同一个序列重新计算的概率完全相同；但两者的计算方式不同：推理用 KV Cache 逐个 token 计算，训练对整段序列一次前向；推理可能用了不同的 kernel、不同的批次组成，甚至量化。在同一个模型上实测：用 KV Cache 逐步采样 120 个 token，记录每个 token 的 log 概率，再对整段序列做一次前向重新计算：

```python
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
prompt = tok(tok.apply_chat_template([{"role": "user", "content": "1 到 100 的和是多少？请一步步推理。"}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)).input_ids
for dtype in (torch.float32, torch.bfloat16):
    model = Transformer.from_pretrained(path, dtype=dtype)
    gen, seq, rollout_logp = torch.Generator().manual_seed(0), list(prompt), []
    cache = KVCache(model.cfg.num_hidden_layers)
    with torch.no_grad():
        logits = model(torch.tensor([prompt]), cache)[0, -1].float()
        for _ in range(120):                                 # 推理端：逐 token 采样，记录采样时的 log 概率
            probs = logits.softmax(-1)
            t = torch.multinomial(probs, 1, generator=gen).item()
            rollout_logp.append(probs[t].log().item())
            seq.append(t)
            logits = model(torch.tensor([[t]]), cache)[0, -1].float()
        full = model(torch.tensor([seq]))[0].float().log_softmax(-1)   # 训练端：整段序列一次前向
    train_logp = torch.tensor([full[len(prompt) - 1 + i, seq[len(prompt) + i]].item() for i in range(120)])
    diff = train_logp - torch.tensor(rollout_logp)
    print(f"{str(dtype):15s} |Δ log p| 平均 {diff.abs().mean():.1e}、最大 {diff.abs().max():.1e}；"
          f"重要性比 p_train/p_rollout 的范围 [{diff.exp().min():.3f}, {diff.exp().max():.3f}]")
```

```text
torch.float32   |Δ log p| 平均 1.2e-06、最大 1.2e-05；重要性比 p_train/p_rollout 的范围 [1.000, 1.000]
torch.bfloat16  |Δ log p| 平均 9.7e-03、最大 1.0e-01；重要性比 p_train/p_rollout 的范围 [0.909, 1.107]
```

FP32 下两者几乎一致；**BF16 下，同一个 token 的概率在两种算法下可以相差 10%**。这不是 bug，而是 BF16 精度下不同的累加顺序造成的正常误差（参见[哪些优化会改变输出](llm://synthesis/token-journey/#哪些优化会改变输出)）。但对 RL 来说，这意味着"用来算损失的策略"和"实际采样的策略"不是同一个，训练变成了隐式的 off-policy，积累下去可能导致训练不稳定甚至崩溃。实际系统中的应对：

- **重要性采样修正**：用 $p_\text{train}/p_\text{rollout}$ 的比值对损失加权，并截断过大的比值（TIS），或者直接丢弃偏差过大的样本（MIS）。这要求推理引擎**返回采样时的 logprobs**；
- **让两边算得一样**：推理端使用"与 batch 无关"的 kernel（batch invariant），或者训练端与推理端使用相同的 kernel 与计算顺序；SGLang 与 vLLM 都提供了确定性/batch 不变的模式，代价是一定的性能；
- **提高精度**：关键部分（如 lm_head、logits 计算）用 FP32。

## 问题三：显存切换与权重同步

共置部署时，训练和推理轮流使用同一批 GPU：

- **显存切换**：rollout 结束后，推理引擎要把 KV Cache（以及权重，如果训练端有自己的一份）占用的显存释放出来给训练用，训练结束后再恢复。vLLM 的 sleep 模式（`enable_sleep_mode`，`llm.sleep()` / `llm.wake_up()`）和 SGLang 的 `release_memory_occupation` / `resume_memory_occupation`（基于 torch_memory_saver）都是为此设计的：释放物理显存但保留虚拟地址，恢复时不需要重新捕获 CUDA Graph；
- **权重同步**：每个训练步之后，新权重要进入推理引擎。共置时可以通过 CUDA IPC 句柄直接共享显存中的张量；分离部署时通过 NCCL 广播或专门的传输服务。大模型的权重有几百 GB，同步时间不容忽视，需要按层流水线化、只传发生变化的部分，并处理训练与推理的并行方式不同（TP、EP 切分不同）带来的重新切分。

!!! source "源码对照"
    - **vLLM**：RL 相关的示例在 `examples/rl/`（包括 HTTP、NCCL、IPC 三种权重同步方式），`vllm/distributed/weight_transfer/` 是权重传输的实现；Rust 前端也支持 RL 的权重同步生命周期。采样时的 logprobs 由 `SamplingParams(logprobs=...)` 返回，`logprobs_mode` 决定返回原始还是处理后的值。
    - **SGLang**：`srt/weight_sync/`、`srt/checkpoint_engine/`（权重更新），`srt/batch_invariant_ops/`（batch 不变的算子）；`update_weights_from_tensor`、`update_weights_from_distributed` 等接口供 verl、slime 等训练框架调用。

!!! interview "面试怎么答"
    RL 基础设施是 2025 年以来推理岗的热门方向。被问到"RL 训练中推理引擎要做什么特殊支持"时，按四个问题回答：**长尾**（partial rollout、异步 RL、过量采样，最好能给出"一半时间在处理长尾"这样的量化认识）、**概率不一致**（BF16 下不同计算顺序带来的差异、重要性采样修正、batch 不变 kernel）、**显存切换**（sleep/wake，释放物理显存保留虚拟地址）、**权重同步**（IPC、NCCL、按层流水线、重新切分）。能说出 verl、slime 或 AReaL 中任何一个的架构，会是很大的加分项。

## 练习

**1. 为什么共置时权重同步可以很快？** 训练端用 FSDP（参数按卡切分），推理端用 TP=2，两者在同一批 GPU 上。同步权重需要做什么？

??? success "参考思路"
    FSDP 下每张卡只有参数的一个分片，先要把一层的完整参数聚合出来（all-gather），再按推理端的 TP 方式切分，写进推理引擎对应的参数张量。因为在同一批 GPU 上，可以逐层进行：聚合一层、通过 CUDA IPC 把张量句柄交给推理进程（不需要拷贝），推理进程按自己的切分方式取用，再释放。整个过程只在 GPU 之间和 GPU 内部搬运数据，不经过 CPU，也不需要写磁盘。

**2. 异步 RL 的策略滞后。** 异步 RL 中，某个回答由 k 步之前的策略生成。这会带来什么问题？怎样控制？

??? success "参考答案"
    训练用的样本不再来自当前策略，策略梯度的估计有偏，滞后越大偏差越大，严重时训练不稳定。控制办法：限制最大滞后步数（例如最多比训练端落后 1～2 步）、用重要性采样修正并截断比值、对滞后太大的样本丢弃或降权。这与"训练与推理概率不一致"的修正手段是同一套工具，所以推理引擎返回准确的采样 logprobs 非常关键。

## 小结

- [x] RL 的每一步都要先 rollout，推理引擎是 RL 训练系统的核心部件；训练与推理可以共置或分离。
- [x] 回答长度的长尾让 rollout 大部分时间处在小批次、低利用率的状态；partial rollout、异步 RL、过量采样可以缓解。
- [x] BF16 下推理端与训练端的概率可以相差百分之几十，需要重要性采样修正或 batch 不变的 kernel。
- [x] 共置部署需要显存切换（sleep/wake）与高效的权重同步（IPC、NCCL、按层流水线）。
