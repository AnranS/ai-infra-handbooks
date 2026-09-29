# 变长批处理与一次混合前向

<p class="lead">有了分页 KV Cache，下一步是让一次前向同时处理很多请求。这些请求长短不一，有的在做 prefill，有的在 decode，有的只算提示词的一段。推理引擎不把它们填充成矩形张量，而是把所有要计算的 token 首尾相接排成一维，再用几个"元数据"数组描述每个 token 属于谁、在哪个位置、K/V 写到哪里。这一章实现这种布局，并验证一次混合前向的结果与逐个请求单独计算完全一致。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么推理引擎不用 `[batch, max_len]` 的填充布局？
    2. `query_start_loc`（`cu_seqlens_q`）、`seq_lens`、`positions`、`slot_mapping`、`logits_indices` 各是什么？
    3. prefill、分块 prefill 的中间一段、decode 在一次前向中的唯一区别是什么？
    4. 为什么一次前向里除了注意力，其他层根本不需要知道 batch 的结构？

## 为什么不填充

把 batch 填充成 `[B, max_len]`，浪费的计算与长度差成正比。一个 decode 请求只有 1 个新 token，和一个 2000 token 的 prefill 放在一起，要被填充成 2000 个 token：

```python
query_lens = [2000, 512, 1, 1, 1, 1, 1, 1]             # 两个 prefill 和六个 decode
padded = len(query_lens) * max(query_lens)
print(f"填充布局要计算 {padded} 个 token，实际只需要 {sum(query_lens)} 个，浪费 {1 - sum(query_lens) / padded:.0%}")
```

```text title="输出"
填充布局要计算 16000 个 token，实际只需要 2518 个，浪费 84%
```

而 Transformer 中除了注意力，**所有层都是逐 token 独立计算的**（见大模型手册的[逐站解读](llm://synthesis/token-journey/#逐站解读)）：嵌入、QKV 投影、RoPE、MLP、归一化只需要一个 `[N, hidden]` 的矩阵，根本不关心这些 token 来自哪个请求。只有注意力需要知道边界：每个 token 只能看到**自己请求**的上下文。所以只要给注意力准备好元数据，其余部分就是一个大矩阵乘法。

## 布局与元数据

以三个请求为例：请求 A 做完整 prefill（7 个 token），请求 B 做 decode（上下文已有 5 个 token，新算 1 个），请求 C 是分块 prefill 的第二段（前 10 个 token 已算过，本步算第 10～21 个）：

```text
input_ids        [A0 A1 A2 A3 A4 A5 A6 | B5 | C10 C11 ... C21]   N = 7 + 1 + 12 = 20
positions        [ 0  1  2  3  4  5  6 |  5 |  10  11 ...  21]   每个 token 在自己序列中的位置
query_start_loc  [0, 7, 8, 20]                                    第 i 个请求的 token 是 [qsl[i], qsl[i+1])
seq_lens         [7, 6, 22]                                       本步算完后各请求的上下文长度
slot_mapping     每个 token 的 K/V 写到哪个 slot（由块表算出）
block_tables     各请求的块表
logits_indices   [6, 7, 19]                                       只有这些位置需要算输出层、做采样
```

prefill、decode、分块 prefill 在这个布局里**没有任何区别**：都只是"某个请求本步有 n 个新 token，它们前面已经有 k 个 token 在缓存里"。decode 是 n = 1，prefill 是 k = 0，分块 prefill 是两者都不为零。vLLM 的调度器正是基于这一点设计的（下一章）。

输出层只需要为每个请求的**最后一个** token 计算：分块 prefill 的中间段不需要采样，可以不算输出层，prefill 也只要最后一个位置的 logits（大模型手册[算过](llm://synthesis/token-journey/#prefill-时只需要最后一个位置的-logits)，这能省掉很可观的计算）。

## 实现

`build_batch` 把每个请求本步的新 token、已计算的长度和块表拼成上面的元数据；`ModelRunner` 直接复用 `mini_llm` 的权重，逐层计算：

```python title="runner.py"
"""runner.py —— 把多个请求的新 token 拼成一维，一次前向完成：prefill、分块 prefill 与 decode 可以混在同一个批次里。

权重直接复用 mini_llm.Transformer 的模块，只是把"每个请求一个 [T, d] 张量"换成"所有 token 拼成 [N, d]"。
"""

import math
from dataclasses import dataclass

import torch

from mini_llm import Transformer, apply_rope, rope_cos_sin
from paged import PagedKVCache, paged_attention, slot_mapping_for


@dataclass
class BatchInput:
    input_ids: torch.Tensor        # [N] 所有请求本步要计算的 token，首尾相接
    positions: torch.Tensor        # [N] 每个 token 在自己序列中的位置
    slot_mapping: torch.Tensor     # [N] 每个 token 的 K/V 写到哪个 slot
    query_start_loc: list[int]     # [B+1] 第 i 个请求的 token 是 input_ids[qsl[i]:qsl[i+1]]（cu_seqlens_q）
    seq_lens: list[int]            # [B] 计算完本步后，每个请求的上下文长度
    block_tables: list[list[int]]  # [B] 每个请求的块表
    logits_indices: torch.Tensor   # [M] 需要采样的 token 在 N 中的下标（通常是各请求的最后一个 token）


def build_batch(items, block_size: int) -> BatchInput:
    """items: [(token_ids, num_computed, block_table, need_logits)]，token_ids 是本步要算的新 token。"""
    input_ids, positions, slots, qsl, seq_lens, tables, logits_idx = [], [], [], [0], [], [], []
    for token_ids, num_computed, table, need_logits in items:
        pos = list(range(num_computed, num_computed + len(token_ids)))
        input_ids += token_ids
        positions += pos
        slots += slot_mapping_for(table, pos, block_size)
        qsl.append(qsl[-1] + len(token_ids))
        seq_lens.append(num_computed + len(token_ids))
        tables.append(table)
        if need_logits:
            logits_idx.append(qsl[-1] - 1)
    return BatchInput(torch.tensor(input_ids), torch.tensor(positions), torch.tensor(slots), qsl, seq_lens,
                      tables, torch.tensor(logits_idx, dtype=torch.long))


class ModelRunner:
    def __init__(self, model: Transformer, kv: PagedKVCache):
        self.model, self.kv = model, kv
        cfg = model.cfg
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hd
        self.scale = 1 / math.sqrt(cfg.hd)

    @torch.no_grad()
    def forward(self, b: BatchInput) -> torch.Tensor:
        """返回 [M, vocab]：只为 logits_indices 指向的 token 计算输出层。"""
        m, N = self.model, b.input_ids.shape[0]
        x = m.embed_tokens(b.input_ids)                                        # [N, d]
        cos, sin = rope_cos_sin(b.positions, self.hd, m.cfg.rope_theta)       # [N, hd]
        cos, sin = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)   # 对所有头广播
        for i, layer in enumerate(m.layers):
            attn, h = layer.self_attn, layer.input_layernorm(x)
            q = apply_rope(attn.q_norm(attn.q_proj(h).view(N, self.nh, self.hd)), cos, sin)     # QK-Norm 在 RoPE 之前
            k = apply_rope(attn.k_norm(attn.k_proj(h).view(N, self.nkv, self.hd)), cos, sin)
            v = attn.v_proj(h).view(N, self.nkv, self.hd)
            self.kv.write(i, b.slot_mapping, k, v)                            # 先写入，再读出整段上下文
            o = paged_attention(q, self.kv, i, b.block_tables, b.seq_lens, b.query_start_loc, self.scale)
            x = x + attn.o_proj(o.reshape(N, self.nh * self.hd))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return m.lm_head(m.norm(x[b.logits_indices]))
```

和 `mini_llm.Transformer.forward` 相比，只有两处不同：输入从 `[B, T]` 变成了 `[N]`，注意力从"拼接连续的 KV Cache"变成了"写入分页缓存，再按块表读出"。

## 验证：一次混合前向

构造上面的三个请求，第一步让 A、B 做完整 prefill，C 只算前 10 个 token；第二步把 A 的 decode、B 的 decode 和 C 的剩余部分放进**同一次前向**。每个结果都和 `mini_llm` 对完整序列单独计算的 logits 比较：

```python
import math
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer
from paged import BlockPool, PagedKVCache
from runner import ModelRunner, build_batch

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
cfg, block_size = model.cfg, 16
pool = BlockPool(64)
kv = PagedKVCache(cfg.num_hidden_layers, 64, block_size, cfg.num_key_value_heads, cfg.hd)
runner = ModelRunner(model, kv)

a, b, c = (tok(t).input_ids for t in ["请用一句话介绍 KV Cache。", "The capital of France is",
                                      "写一首关于大海的五言绝句，并解释其中的意象。然后再用英文翻译一遍。"])
tables = [pool.allocate(math.ceil(len(ids) / block_size) + 1) for ids in (a, b, c)]

def reference(ids):                                     # mini_llm 对完整序列计算，取最后一个位置
    with torch.no_grad():
        return model(torch.tensor([ids]))[0, -1]

# 第 1 步：A、B 完整 prefill，C 只算前 10 个 token（不需要 logits）
step1 = build_batch([(a, 0, tables[0], True), (b, 0, tables[1], True), (c[:10], 0, tables[2], False)], block_size)
logits = runner.forward(step1)
next_a, next_b = logits.argmax(-1).tolist()

# 第 2 步：A decode、B decode、C 的第 10～21 个 token，放在同一次前向里
step2 = build_batch([([next_a], len(a), tables[0], True), ([next_b], len(b), tables[1], True),
                     (c[10:], 10, tables[2], True)], block_size)
logits = runner.forward(step2)
print("query_start_loc", step2.query_start_loc, " seq_lens", step2.seq_lens,
      " logits_indices", step2.logits_indices.tolist())
print("positions", step2.positions.tolist())
for name, got, ids in [("A decode", logits[0], a + [next_a]), ("B decode", logits[1], b + [next_b]),
                       ("C 分块 prefill", logits[2], c)]:
    err = (got - reference(ids)).abs().max().item()
    print(f"{name:14s} 与单独计算的最大误差 {err:.1e}")
    assert err < 1e-3
```

```text
query_start_loc [0, 1, 2, 14]  seq_lens [8, 6, 22]  logits_indices [0, 1, 13]
positions [7, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
A decode       与单独计算的最大误差 1.8e-05
B decode       与单独计算的最大误差 1.5e-05
C 分块 prefill   与单独计算的最大误差 2.5e-05
```

三种不同阶段的请求在同一次前向中完成，每个结果都与单独计算一致（误差来自浮点累加顺序的不同）。

!!! source "源码对照"
    - **vLLM**：`GPUModelRunner._prepare_inputs`（`vllm/v1/worker/gpu_model_runner.py`）用 numpy 向量化地构造 `positions`、`query_start_loc`、`slot_mapping`，其中 `logits_indices = query_start_loc[1:] - 1`。为了不在每步重新构造全部状态，vLLM 维护一个跨步复用的 `InputBatch`（`gpu_input_batch.py`，"persistent batch"），每步只做增量更新（`_update_states`）。
    - 注意力元数据由各后端的 metadata builder 构造。`FlashAttentionMetadata`（`vllm/v1/attention/backends/flash_attn.py`）的注释给出了三个长度的定义：`context_len`（之前已缓存的）、`query_len`（本步新算的）、`seq_len`（两者之和）。
    - **SGLang**：一次前向的输入是 `ForwardBatch`（`srt/model_executor/forward_batch_info.py`）。`extend_prefix_lens` 对应"已缓存的长度"，`extend_seq_lens` 对应"本步新算的长度"，`out_cache_loc` 对应 slot mapping。`ForwardMode` 区分 `EXTEND`（prefill）、`DECODE` 和 `MIXED`（分块 prefill 与 decode 混合），因为 SGLang 会针对纯 decode 批次走更快的路径（例如 CUDA Graph）。

!!! inference "推理视角"
    构造这些元数据是**CPU 工作**。当模型很快（小模型、decode 批次）时，CPU 准备输入的时间可能和 GPU 计算时间相当，GPU 会空等。所以推理引擎非常在意这部分的效率：vLLM 用 numpy 和预分配的 pinned buffer 避免 Python 循环与重复分配，并把"调度下一步"与"执行当前步"重叠（async scheduling）；SGLang 的 overlap scheduler 做的是同一件事。本书的 `build_batch` 用 Python 循环写成，便于阅读，效率则远不如它们。

## 练习

**1. 元数据推导。** 一个批次有三个请求：请求 1 已缓存 30 个 token，本步 decode；请求 2 是新请求，提示词 6 个 token；请求 3 已缓存 16 个 token，本步算接下来的 4 个 token，但提示词还没算完（总长 40）。写出 `query_start_loc`、`seq_lens`、`positions` 和 `logits_indices`。

??? success "参考答案"
    - `query_start_loc = [0, 1, 7, 11]`；
    - `seq_lens = [31, 6, 20]`；
    - `positions = [30, 0, 1, 2, 3, 4, 5, 16, 17, 18, 19]`；
    - `logits_indices = [0, 6]`。请求 3 的提示词还没算完，不需要采样。

**2. 为什么 decode 批次特殊？** 既然所有阶段都能用同一种布局，为什么引擎还要区分"纯 decode 批次"？

??? success "参考答案"
    纯 decode 批次中每个请求恰好 1 个 query token，形状规整（N = B），可以用专门的 decode kernel（split-KV、Flash-Decoding），也可以录制成 CUDA Graph 重放（[CUDA Graphs 一章](graphs-compile.md)）。混合批次的形状每步都不同，难以用 CUDA Graph 覆盖。vLLM 的 `FULL_AND_PIECEWISE` 模式就是这样：纯 decode 批次用完整的 CUDA Graph，混合批次用"分段"的 CUDA Graph，注意力部分在图外单独执行。

## 小结

- [x] 推理引擎把所有请求的新 token 首尾相接排成一维，不做填充。
- [x] 注意力之外的层逐 token 计算，不关心 batch 结构；注意力靠 `query_start_loc`、`seq_lens`、块表和 `slot_mapping` 区分请求。
- [x] prefill、分块 prefill、decode 只是"已缓存长度"和"本步长度"的不同组合，可以混在同一次前向中。
- [x] 输出层只为需要采样的位置计算；构造元数据是 CPU 开销，推理引擎会想办法把它和 GPU 计算重叠。
