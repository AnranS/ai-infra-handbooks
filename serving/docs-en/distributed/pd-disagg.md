# PD disaggregation and KV transfer

<p class="lead">Prefill is compute-bound and decode is memory-bound; on the same GPUs they interfere with each other, and neither can pick its best parallelism and batch size. PD disaggregation puts them on different instances: the prefill instance computes the prompt's KV Cache and the first token, then passes the KV to a decode instance that continues generating. This chapter implements the flow with the mini engine (the two instances even have different block sizes) and checks that the output is unchanged, then discusses the cost of the transfer, converting KV layouts, and the implementations in vLLM and SGLang.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Which two problems does PD disaggregation solve? When is it not worth doing?
    2. How much data is the KV going from a prefill instance to a decode instance? Can it hide behind compute?
    3. When the two sides use different parallelism (for example prefill TP=4, decode TP=8), what extra work does the KV transfer need?
    4. Why does the decode instance usually allocate blocks first and then let the prefill instance write the KV into them?

??? success "Answers (try first, then expand to compare)"
    1. Two problems: prefill and decode together interfere with each other (a long prefill stalls decode, making TPOT jitter); and the two phases differ in nature yet must share one parallelism and batch size. Disaggregated, each picks its best configuration and scales separately. It is not worth it at small scale, with short prompts, or when TPOT jitter does not matter; chunked prefill is enough.
    2. KV per token × prompt length: for LLaMA-3-70B, a 4K prompt is about 1.34 GB, about 27 ms over a single 400 Gb/s link; DeepSeek-V3 (MLA) is only 0.29 GB, about 6 ms. That is roughly a fifth of the prefill time, and a machine has 8 links in parallel, so transferring layer by layer hides most of it behind compute.
    3. The two sides split KV differently (by heads into 4 and into 8), and the block size and layout may differ too, so the transfer must convert the layout: gather by the prefill side's block table, then redistribute by the decode side's split and write into the positions given by its block table.
    4. The decode instance first confirms it has room and allocates blocks, then tells the prefill instance the addresses, and the prefill instance writes the data straight in: this avoids an extra copy through an intermediate buffer, and avoids wasted prefill work when the decode instance is out of memory.

<!-- comic ../assets/comics/pd-disagg.webp is in Chinese; put it back once the English version exists -->

## Why disaggregate {#为什么要分离}

The [serving chapter](llm://inference/serving/#pd-分离) of the LLM handbook introduced the motivation; here is the deployer's view:

![Figure: the overall structure of PD disaggregation](../assets/figures/pd-disagg.svg){.aig-svg}

1. **Interference**: when a long prompt's prefill slips in, every decode request in the same batch waits for it that step (the [scheduler chapter](../engine/scheduler.md#token-预算的取舍) measured the TPOT spikes). Chunked prefill eases this but cannot remove it; splitting one GPU by SMs is covered in [PD multiplexing](../frontier/pd-multiplex.md).
2. **Different best configurations**: prefill wants a smaller TP or EP and large token batches to saturate compute; decode wants the largest possible batch to amortize weight reads, and MoE models like DeepSeek also want very large EP (see [expert parallelism](expert-parallel.md)) along with CUDA Graphs and low-latency communication. Mixed together, you can only compromise.
3. **Decoupled SLOs**: TTFT is set mainly by prefill instances and TPOT mainly by decode instances, and the two can scale separately. The ratio of prefill to decode instances (the familiar xPyD) is tuned to the load.

The costs are the KV transfer overhead, a more complex system, and possibly unbalanced resources (one side busy while the other idles). At small scale, with short prompts, or when TPOT jitter does not matter, not disaggregating and using chunked prefill is often enough.

## The flow {#流程}

<!-- i18n:diagram c94d7194b6 -->
```text
             ① request                        ② prefill: compute the prompt, get KV and the first token
client ─────▶ router ───────────────────────▶ ┌────────────────────────────────┐
                │                             │ Prefill instance (small TP/EP) │
                │ ③ pre-allocate KV blocks    └───────────────┬────────────────┘
                │   on the decode instance                    │ ④ write KV by block (or layer by layer)
                ▼                                             ▼   into the decode instance's memory (RDMA)
         ┌──────────────────────────────────┐  ◀───────────────┘
         │ Decode instance (large EP/batch) │ ⑤ on "transfer done": joins the running queue, keeps decoding
         └──────────────────────────────────┘ ⑥ streams back
```

Real systems differ in the details, but two points are nearly universal: **the decode instance allocates KV blocks first**, and the prefill instance writes the data straight into them (avoiding an intermediate buffer, and wasted work when the decode instance is out of memory); and **KV can be transferred layer by layer**: as soon as prefill finishes a layer, that layer starts transferring, overlapping with the compute of later layers.

## Implementation {#实现}

Two `LLMEngine` instances play the prefill and decode roles. The prefill instance's block size is 16 and the decode instance's is 32, so the "transfer" is not a block-for-block copy: first gather the KV by the prefill instance's block table, then write it by the decode instance's block table. This is a **layout conversion**:

```python title="pd.py"
"""pd.py —— PD 分离的最小实现：prefill 实例算出 KV 和第一个 token，把 KV 传给 decode 实例继续生成。

两个实例是两个独立的 LLMEngine（各自的块池、KV 张量，甚至块大小都可以不同），
"传输"就是按块表把 KV 从一边的物理块拷贝到另一边的物理块；真实系统中这一步由 RDMA 完成。
"""

import math

import torch

from nano_engine import LLMEngine, Request, SamplingParams, Status
from runner import build_batch


@torch.no_grad()
def prefill(engine: LLMEngine, prompt_ids: list[int]):
    """在 prefill 实例上计算整段提示词，返回第一个 token 和按层收集好的 K/V（[层][seq, kv_heads, head_dim]）。"""
    n_blocks = math.ceil(len(prompt_ids) / engine.block_size)
    blocks = engine.pool.allocate(n_blocks)
    logits = engine.runner.forward(build_batch([(prompt_ids, 0, blocks, True)], engine.block_size))
    first_token = logits.argmax(-1).item()
    kv = [engine.kv.gather(layer, blocks, len(prompt_ids)) for layer in range(len(engine.kv.k))]
    engine.pool.free(blocks)                        # once the transfer is done, the prefill instance frees these blocks right away
    return first_token, kv


def admit(engine: LLMEngine, prompt_ids: list[int], first_token: int, kv, params: SamplingParams) -> Request:
    """在 decode 实例上接收 KV：分配块、写入，然后让请求直接以"已计算完提示词"的状态进入运行队列。"""
    req = Request(f"pd-{engine._next_id}", prompt_ids, params)
    engine._next_id += 1
    n = len(prompt_ids)
    req.block_ids = engine.pool.allocate(math.ceil((n + 1) / engine.block_size))
    slots = torch.tensor([req.block_ids[p // engine.block_size] * engine.block_size + p % engine.block_size
                          for p in range(n)])
    for layer, (k, v) in enumerate(kv):
        engine.kv.write(layer, slots, k, v)
    req.output_ids, req.num_computed = [first_token], n  # the prompt's KV is in place, so the next step goes straight to decode
    req.status = Status.RUNNING
    engine.scheduler.running.append(req)
    return req
```

After `admit`, the decode instance sees this request as an ordinary running request whose "prompt is already computed and 1 token already generated": `num_computed` equals the prompt length and `num_tokens` is 1 more, so the scheduler gives it 1 token in the next step, which is decode.

```python
import torch
from transformers import AutoTokenizer
from mini_llm import Transformer
from nano_engine import LLMEngine, SamplingParams
from pd import admit, prefill

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

def chat(q):
    return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)).input_ids

prompts = [chat(q) for q in ["什么是 PD 分离？", "介绍一下 RDMA。", "写一句关于秋天的诗。", "Explain KV cache transfer."]]
params = SamplingParams(max_tokens=24)
reference = LLMEngine(model, eos_token_id=tok.eos_token_id).generate(prompts, params)   # not disaggregated

prefill_instance = LLMEngine(model, eos_token_id=tok.eos_token_id, block_size=16)
decode_instance = LLMEngine(model, eos_token_id=tok.eos_token_id, block_size=32)
requests, transferred = [], 0
for p in prompts:
    first_token, kv = prefill(prefill_instance, p)
    transferred += sum(k.numel() + v.numel() for k, v in kv) * 2        # counted in BF16
    requests.append(admit(decode_instance, p, first_token, kv, params))
while decode_instance.scheduler.has_unfinished():
    decode_instance.step()

print(f"4 个请求共 {sum(len(p) for p in prompts)} 个提示词 token，传输 KV {transferred / 1024:.0f} KB")
print("与不分离的结果一致：", [r.output_ids == o for r, o in zip(requests, reference)])
assert all(r.output_ids == o for r, o in zip(requests, reference))
```

```text title="output"
4 个请求共 70 个提示词 token，传输 KV 7840 KB
与不分离的结果一致： [True, True, True, True]
```

## The cost of transfer {#传输的代价}

The KV size per token is the "KV/token" [estimated](llm://synthesis/models/#十个模型的体检表) in the LLM handbook. How long does transferring one request's KV take?

```python
cases = {"LLaMA-3-70B（GQA，320 KB/token）": 320 * 1024, "DeepSeek-V3（MLA，69 KB/token）": 70272}
prompt_len, rdma = 4096, 50e9                                      # 4K prompt; 400 Gb/s RDMA ≈ 50 GB/s
prefill_ms = 2 * 70.55e9 * prompt_len / (8 * 989e12 * 0.5) * 1e3   # 70B prefill on 8 H100s (MFU 50%)
for name, per_token in cases.items():
    size = per_token * prompt_len
    print(f"{name}：4K 提示词的 KV {size / 1e9:.2f} GB，单条 400 Gb/s 链路传输 {size / rdma * 1e3:.0f} ms")
print(f"对比：70B 模型在 8 张 H100 上 prefill 4K token 约 {prefill_ms:.0f} ms")
```

```text title="output"
LLaMA-3-70B（GQA，320 KB/token）：4K 提示词的 KV 1.34 GB，单条 400 Gb/s 链路传输 27 ms
DeepSeek-V3（MLA，69 KB/token）：4K 提示词的 KV 0.29 GB，单条 400 Gb/s 链路传输 6 ms
对比：70B 模型在 8 张 H100 上 prefill 4K token 约 146 ms
```

The transfer takes about a fifth of the prefill time (and a machine usually has 8 such links, transferring in parallel per GPU), so transferring layer by layer and overlapping with compute hides most of it. MLA's KV is much smaller and puts less pressure on the transfer, one reason DeepSeek-style models suit PD disaggregation particularly well.

## Layout conversion {#布局转换}

In real systems the two sides' KV layouts often differ:

- **Different block sizes**: as in this chapter's example; remapping by token is enough;
- **Different TP**: at prefill TP=4 each GPU stores 1/4 of the KV heads, at decode TP=8 each stores 1/8. One prefill GPU's KV must be split between two decode GPUs, or merged the other way;
- **Different attention backends**: backends require different physical layouts (for example whether K and V are packed together, and the order of the head and token dimensions); vLLM's KV connector can declare the layout it needs (`get_required_kvcache_layout`);
- **Different data types**: BF16 on one side and FP8 KV on the other need conversion before or after the transfer.

These conversions are best done on the GPU, as part of the transfer, or they become a new bottleneck.

!!! source "Source code"
    - **vLLM**: PD disaggregation is implemented through the **KV connector** interface (`KVConnectorBase_V1` in `vllm/distributed/kv_transfer/kv_connector/v1/base.py`), which has two sides: on the scheduler side, `get_num_new_matched_tokens` (how many of this request's tokens can have their KV loaded from outside), `update_state_after_alloc`, `build_connector_meta` and `request_finished`; on the worker side, `start_load_kv`, `wait_for_layer_load`, `save_kv_layer` and `wait_for_save`, which naturally support loading and saving layer by layer. Implementations include `nixl/` (NVIDIA NIXL, RDMA transfer), `mooncake/`, `lmcache_connector.py`, `offloading_connector.py` and more, configured with `--kv-transfer-config`. The `WAITING_FOR_REMOTE_KVS` state in the scheduler means "the KV is still in transit".
    - **SGLang**: `--disaggregation-mode prefill|decode` starts the two kinds of instances, and `--disaggregation-transfer-backend` picks a transfer backend such as `mooncake` or `nixl`; the logic is in `srt/disaggregation/` (`prefill.py`, `decode.py`, and a subdirectory per backend), where the decode side pre-allocates KV and handshakes with the prefill side through a bootstrap service.
    - The ecosystem also has scheduling frameworks built around PD disaggregation and KV management, such as NVIDIA Dynamo, llm-d and Mooncake.

!!! interview "In an interview"
    When explaining PD disaggregation, cover at least four points: **motivation** (interference and conflicting configurations, decoupling TTFT/TPOT), **flow** (decode pre-allocates → prefill computes and pushes KV layer by layer → decode takes over), **cost** (an estimate of KV size × bandwidth, using this chapter's numbers to show "it can be hidden by overlap"), and **difficulties** (layout conversion, the xPyD ratio shifting with load, failures and retries, not worth it at small scale). Mentioning vLLM's KV connector interface or SGLang's disaggregation mode shows you have read the implementations.

## Exercises {#练习}

**1. The xPyD ratio.** A workload has an average prompt of 2000 tokens and an output of 300 tokens. One prefill instance handles 20000 prompt tokens per second, and one decode instance outputs 3000 tokens per second while meeting the TPOT SLO. To support 30 requests per second, how many prefill and decode instances are needed?

??? success "Answer"
    Prompt tokens: 30 × 2000 = 60000 per second, needing 3 prefill instances; output tokens: 30 × 300 = 9000 per second, needing 3 decode instances. That is 3P3D, plus some headroom on each. If the average prompt becomes 8000 tokens, prefill demand quadruples, needing 12 prefill instances, while the number of decode instances stays the same. This is why xPyD must adjust dynamically with load.

**2. What about prefix caching?** After PD disaggregation, should the prefix cache live on the prefill instances, the decode instances, or somewhere else?

??? success "Approach"
    Prefix caching exists to cut prefill compute, so first it must hit on the prefill instance, and the router should send requests sharing a prefix to the same prefill instance (cache-aware routing). The KV on a decode instance can also be kept after a request ends, and if the next turn of a multi-turn conversation is routed to the same decode instance it could in principle be reused; but the next turn still needs to prefill new content, so the more common approach is a **KV cache pool independent of instances** (CPU memory, SSD or distributed storage, such as Mooncake Store or LMCache), from which prefill instances load the prefixes that hit. That is the next chapter.

## Summary {#小结}

- [x] PD disaggregation removes the interference between prefill and decode, lets each pick its best configuration, and lets them scale separately.
- [x] The flow: decode pre-allocates KV blocks, prefill writes into them after computing (layer by layer), and decode takes over the request as "prompt already computed".
- [x] KV transfer volume = KV per token × prompt length, a fraction of the prefill time at RDMA bandwidth, and it can overlap with compute.
- [x] When block size, TP, attention backend or data type differ, the layout must be converted before or after the transfer.
