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
    engine.pool.free(blocks)                        # 传输完成后，prefill 实例立即释放这些块
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
    req.output_ids, req.num_computed = [first_token], n  # 提示词的 KV 已就位，下一步直接 decode
    req.status = Status.RUNNING
    engine.scheduler.running.append(req)
    return req
