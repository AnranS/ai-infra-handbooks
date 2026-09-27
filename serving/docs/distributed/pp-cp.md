# 流水线并行与上下文并行

<p class="lead">张量并行切的是每一层的矩阵，专家并行切的是专家。还有两个维度可以切：按**层**切（流水线并行，PP），按**序列**切（上下文并行，CP）。PP 通信量小，适合跨机器扩展模型规模；CP 专门对付超长上下文的 prefill 和 KV Cache。这一章用两个进程实现一个两段流水线，并用 ring attention 演示上下文并行如何在多张卡之间计算因果注意力。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. PP 能降低单个请求的延迟吗？它提高的是什么？
    2. 推理中的 PP 为什么需要"多个批次同时在流水线里"？
    3. ring attention 中，每张卡的部分注意力结果怎么合并？为什么需要 log-sum-exp？
    4. 因果注意力下，按连续区间切分序列会导致什么问题？怎么解决？

## 流水线并行

PP 把模型的层分成若干段（stage），每张卡（或每组卡）负责一段。前一段算完，把隐藏状态发给下一段。每个 stage 只存自己那些层的权重和 KV Cache。

```python title="pp.py"
"""pp.py —— 流水线并行的最小实现：两个进程各持有一半的层，点对点传递隐藏状态。

用法：torchrun --standalone --nproc-per-node 2 pp.py
stage 0：嵌入 + 前一半的层；stage 1：后一半的层 + 最终归一化 + 输出层。
每个 stage 只存自己那些层的权重和 KV Cache；每生成一个 token，隐藏状态从 stage 0 发到 stage 1，
采样出的 token 再发回 stage 0。
"""

import os

import torch
import torch.distributed as dist

from mini_llm import KVCache, Transformer, rope_cos_sin


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    full = Transformer.from_pretrained(os.environ.get("MODEL", "models/Qwen2.5-0.5B-Instruct"))
    cfg = full.cfg
    L = cfg.num_hidden_layers
    my_layers = range(0, L // 2) if rank == 0 else range(L // 2, L)
    cache = KVCache(L)                                   # 只会用到自己那些层的槽位
    prompt = [151644, 872, 198, 105043, 100165, 30, 151645, 198, 151644, 77091, 198]
    ids, generated, sent_bytes = torch.tensor([prompt]), [], 0

    with torch.no_grad():
        for step in range(20):
            T = ids.shape[1]
            first = cache.k[my_layers[0]]                # 本 stage 第一层的 KV Cache 长度就是已处理的 token 数
            start = 0 if first is None else first.shape[2]
            cos, sin = rope_cos_sin(torch.arange(start, start + T), cfg.hd, cfg.rope_theta)
            if rank == 0:
                x = full.embed_tokens(ids)
            else:
                x = torch.empty(1, T, cfg.hidden_size)
                dist.recv(x, src=0)                      # 收到上一个 stage 的隐藏状态
            for i in my_layers:
                x = full.layers[i](x, cos, sin, cache)
            if rank == 0:
                dist.send(x, dst=1)                      # 发给下一个 stage：[T, hidden]
                sent_bytes += x.numel() * 2              # 按 BF16 计算实际部署时的通信量
                nxt = torch.empty(1, dtype=torch.long)
                dist.recv(nxt, src=1)                    # 等最后一个 stage 采样出的 token
            else:
                nxt = full.lm_head(full.norm(x[:, -1])).argmax(-1)
                dist.send(nxt, dst=0)
            generated.append(nxt.item())
            ids = nxt.view(1, 1)

    if rank == 0:
        ref = []
        with torch.no_grad():
            ref_cache, x = KVCache(L), torch.tensor([prompt])
            for _ in range(20):
                nxt = full(x, ref_cache)[0, -1].argmax().item()
                ref.append(nxt)
                x = torch.tensor([[nxt]])
        stage_layers = [f"{r.start}～{r.stop - 1}" for r in (range(0, L // 2), range(L // 2, L))]
        print(f"PP=2：stage 0 负责第 {stage_layers[0]} 层，stage 1 负责第 {stage_layers[1]} 层")
        print(f"生成 20 个 token，stage 之间共传输 {sent_bytes / 1024:.0f} KB 隐藏状态")
        print(f"与单进程一致：{generated == ref}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```python
import os
import subprocess
import sys

result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2",
                         "build/code/pp.py"], capture_output=True, text=True,
                        env={**os.environ, "OMP_NUM_THREADS": "8"}, timeout=1200)
print(result.stdout.strip())
assert "一致：True" in result.stdout
```

```text
PP=2：stage 0 负责第 0～11 层，stage 1 负责第 12～23 层
生成 20 个 token，stage 之间共传输 52 KB 隐藏状态
与单进程一致：True
```

和张量并行比较：TP 每层两次 all-reduce，每次都是整个 `[token 数, hidden]`；PP 每个 stage 边界只有一次点对点发送，同样大小。24 层的模型切成 2 段，通信次数从 48 次 all-reduce 降到 1 次 send。所以 **PP 对互连带宽的要求低得多，适合跨机器**。

### 流水线的气泡

PP 的问题在于：一个批次必须依次经过所有 stage，同一时刻只有一个 stage 在工作。

```text
一个批次在流水线里（PP=4）：            四个批次同时在流水线里：
stage 0  ██░░░░░░██░░░░░░██               stage 0  ██████████████████
stage 1  ░░██░░░░░░██░░░░░░               stage 1  ░░████████████████
stage 2  ░░░░██░░░░░░██░░░░               stage 2  ░░░░██████████████
stage 3  ░░░░░░██░░░░░░██░░               stage 3  ░░░░░░████████████
        每张卡只有 1/4 的时间在工作               稳定后每张卡都在工作
```

因此：

- **PP 不降低单个请求的延迟**：一个 token 仍然要依次经过所有层，还多了 stage 之间的传输；
- **PP 提高的是吞吐**，前提是流水线里同时有足够多的批次。推理引擎会让调度器一次领先调度多个批次：vLLM 的 `step_with_batch_queue` 维护一个长度等于 PP 大小的批次队列（见 [vLLM 源码导读](../source/vllm.md#主线二enginecore-主循环)），SGLang 也有专门的 PP 事件循环（`scheduler_pp_mixin.py`）；
- decode 时，第 N 步的输入依赖第 N−1 步的输出，同一个请求不能同时出现在两个在途批次中。所以要把请求分成若干组，轮流进入流水线。

典型用法：模型大到一台机器放不下时，机内 TP、机间 PP（例如两台 8 卡机器部署一个 405B 模型，TP=8、PP=2）。

## 上下文并行

上下文长到几十万、上百万 token 时，出现两个新问题：prefill 的注意力计算量随长度平方增长，一张卡要算很久；一个请求的 KV Cache 本身就可能超过一张卡的显存。上下文并行把**序列**切开：每张卡负责一段 token 的 query，和对应那段的 K/V。

### ring attention

每张卡只有自己那段 K/V，但它的 query 需要看到所有前面的 K/V。ring attention 的做法是：各卡排成一个环，每一步把自己手里的 K/V 块传给下一张卡，同时用刚收到的块计算一部分注意力；转完一圈，每张卡的 query 就见过了所有 K/V。

关键是**怎样合并分块算出的注意力**。每块算出的是"只在这部分 key 上做 softmax"的结果，要合并成"在全部 key 上做 softmax"的结果，需要记录每行的 log-sum-exp（LSE），这和 FlashAttention 的在线 softmax 是同一个技巧：

$$
\text{LSE} = \log(e^{\text{LSE}_1} + e^{\text{LSE}_2}),\quad
O = O_1 e^{\text{LSE}_1 - \text{LSE}} + O_2 e^{\text{LSE}_2 - \text{LSE}}
$$

在单个进程里模拟 4 张卡的 ring attention，验证它与完整的因果注意力相同：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
heads, seq, dim, n_dev = 4, 64, 32, 4
q, k, v = (torch.randn(heads, seq, dim) for _ in range(3))
reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)

def partial_attention(q_idx, k_idx):
    """一张卡用自己的 query 和收到的一块 K/V 计算：返回部分输出和每行的 LSE。"""
    s = q[:, q_idx] @ k[:, k_idx].transpose(-1, -2) * dim ** -0.5
    s = s.masked_fill(q_idx[:, None] < k_idx[None, :], float("-inf"))   # 因果掩码用的是全局位置
    return torch.softmax(s, -1).nan_to_num() @ v[:, k_idx], torch.logsumexp(s, -1)

def merge(o1, lse1, o2, lse2):
    lse = torch.logaddexp(lse1, lse2)
    return o1 * torch.exp(lse1 - lse).nan_to_num()[..., None] + o2 * torch.exp(lse2 - lse).nan_to_num()[..., None], lse

def ring_attention(partition):
    out, work = torch.empty_like(q), []
    for dev, q_idx in enumerate(partition):
        o = lse = None
        for step in range(n_dev):                                  # 第 step 步，收到的是往前数第 step 张卡的 K/V
            k_idx = partition[(dev - step) % n_dev]
            if q_idx.max() < k_idx.min():                          # 整块都在未来：跳过
                continue
            po, pl = partial_attention(q_idx, k_idx)
            o, lse = (po, pl) if o is None else merge(o, lse, po, pl)
        out[:, q_idx] = o
        work.append(int((q_idx[:, None] >= torch.cat(partition)[None, :]).sum()))   # 需要计算的 (q, k) 对数
    return out, work

contiguous = [torch.arange(i * 16, (i + 1) * 16) for i in range(n_dev)]
zigzag = [torch.cat([torch.arange(i * 8, (i + 1) * 8), torch.arange((7 - i) * 8, (8 - i) * 8)]) for i in range(n_dev)]
for name, partition in [("连续切分", contiguous), ("之字形切分", zigzag)]:
    out, work = ring_attention(partition)
    print(f"{name}：与完整因果注意力的最大误差 {(out - reference).abs().max().item():.1e}，各卡的计算量 {work}")
```

```text
连续切分：与完整因果注意力的最大误差 6.0e-07，各卡的计算量 [136, 392, 648, 904]
之字形切分：与完整因果注意力的最大误差 4.2e-07，各卡的计算量 [520, 520, 520, 520]
```

两种切法结果都正确，但**计算量差别很大**。因果注意力中，靠后的 token 要看的 key 更多：按连续区间切分时，最后一张卡的计算量是第一张卡的 6 倍多，整体速度被它拖慢。解决办法是**之字形（zigzag）切分**：把序列切成 2n 段，第 i 张卡拿第 i 段和倒数第 i 段，一前一后，各卡的计算量完全相同。

### 推理中的上下文并行

- **prefill 的 CP**：如上所述，用于超长提示词。每张卡只算 1/n 的 query，KV 在环上传递，通信可以与计算重叠（每一步计算当前块的同时传下一块）。
- **decode 的 CP**：decode 时 query 只有一个 token，瓶颈是读 KV Cache。可以把一个请求的 KV Cache 按序列切到多张卡上，每张卡在自己那段 KV 上算部分注意力，再用 LSE 合并，读取 KV 的时间也降为 1/n。vLLM 称之为 DCP（decode context parallel），特别适合 MLA 这类 KV 无法按头切分的模型：TP 的各卡本来就要各存一份完整的 KV，改为按序列切分后，KV 不再冗余。

!!! source "源码对照"
    - **vLLM**：PP 由 `--pipeline-parallel-size` 开启，stage 之间传递 `IntermediateTensors`；调度侧在 `EngineCore.step_with_batch_queue`。DCP 由 `--decode-context-parallel-size` 开启，KV 块按 `dcp_world_size` 在各卡之间交错存放（`kv_cache_utils.py` 中的 `resolve_dcp_kv_block_size` 等），注意力后端在各卡上算部分结果后用 LSE 合并。prefill 的上下文并行（`--prefill-context-parallel-size`）也在逐步完善。
    - **SGLang**：`--pp-size` 开启 PP，调度逻辑在 `managers/scheduler_pp_mixin.py`；上下文并行相关代码在 `srt/layers/cp/`（其中 `zigzag.py` 就是本章的之字形切分）、`srt/layers/dcp/`，以及 DeepSeek 稀疏注意力的 CP 实现（`communicator_dsa_cp.py`）。

!!! interview "面试怎么答"
    被问到"TP、PP、DP、EP、CP 怎么选"时，按**通信模式**回答：TP 每层 all-reduce，要求机内高带宽，降低单请求延迟；PP 只在 stage 边界点对点传输，适合跨机、扩展模型规模，但不降低延迟、需要多批次填满流水线；DP 无通信，扩展吞吐；EP 每个 MoE 层两次 all-to-all，用于大规模 MoE；CP 按序列切分，用于超长上下文，需要 LSE 合并。实际部署往往是组合：机内 TP 或 EP，机间 PP 或 DP，长上下文再加 CP。

## 练习

**1. PP 的延迟。** 一个模型用 PP=4 部署，每个 stage 算一步 decode 需要 5 ms，stage 之间的传输需要 0.2 ms。单个请求的 TPOT 是多少？流水线填满时，每秒能完成多少步（所有批次合计）？

??? success "参考答案"
    单个请求每一步要依次经过 4 个 stage 和 3 次传输，再把采样结果传回第一个 stage：约 4 × 5 + 4 × 0.2 = 20.8 ms，TPOT 约 21 ms，比单卡（假设放得下）的 20 ms 还慢一点。流水线填满时，每个 stage 每 5 ms 完成一步，整体每秒约 200 步，是单个批次的 4 倍。PP 用延迟换来了放得下和吞吐。

**2. LSE 合并。** 为什么合并时不能直接对两个部分输出求平均？如果某一块对某一行完全被掩码（全是 −∞），合并公式会怎样？

??? success "参考答案"
    两块的 softmax 分母不同，直接平均相当于假设两块的 key 在注意力中占相同的总权重，这是错的；正确的权重是两块各自的 softmax 分母（即 $e^{\text{LSE}}$）在总分母中的占比。完全被掩码的块 LSE 为 −∞，$e^{\text{LSE}_i - \text{LSE}} = 0$，它的贡献自然为 0；实现时要注意 −∞ − (−∞) 会得到 NaN，需要单独处理（上面的代码用 `nan_to_num`，也可以直接跳过这样的块）。

## 小结

- [x] PP 按层切分，stage 之间只有点对点通信，适合跨机；它不降低单请求延迟，需要多个批次同时在流水线中才能提高吞吐。
- [x] CP 按序列切分，用于超长上下文；ring attention 在环上传递 K/V 块，用 LSE 合并部分结果。
- [x] 因果注意力下连续切分会让计算量严重不均，之字形切分让各卡负载相同。
- [x] decode 的 CP（DCP）把一个请求的 KV 按序列分到多卡，特别适合 MLA 模型。
