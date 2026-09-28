"""第 17 章：FlashInfer 与 FlashAttention 后端拿到的元数据长什么样。

CPU 上没有这两个库，这里把 tests/fakes（同接口的 PyTorch 假实现）放到 import 路径上，
官方写法的后端代码原样运行，再与参考后端比较。
"""

import sys

sys.path.insert(0, "tests/fakes")

import torch  # noqa: E402
from minisgl.core import Batch, Req, SamplingParams, reset_global_ctx  # noqa: E402
from minisgl.distributed import DistributedInfo, reset_tp_info  # noqa: E402
from minisgl.engine import Engine, EngineConfig  # noqa: E402

PROMPTS = [[785, 6722, 315, 9625, 374], [16, 488, 220, 16, 284, 220]]


def build(backend: str, page_size: int) -> Engine:
    reset_global_ctx()
    reset_tp_info()
    e = Engine(EngineConfig(model_path="models/Qwen3-0.6B", tp_info=DistributedInfo(0, 1),
                            dtype=torch.float32, attention_backend=backend, page_size=page_size,
                            max_running_req=2, num_page_override=64 // page_size, max_seq_len_override=32))
    e.page_table[0, :16] = torch.arange(16, 32)  # 请求 0：第 16～31 个位置
    e.page_table[1, :16] = torch.arange(40, 56)  # 请求 1：第 40～55 个位置
    return e


def batch_of(engine: Engine, phase: str, reqs):
    b = Batch(reqs=reqs, phase=phase)
    b.padded_reqs = reqs
    b.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).int()
    b.input_ids = torch.cat([r.input_ids[r.cached_len:r.device_len] for r in reqs])
    b.out_loc = torch.cat([engine.page_table[r.table_idx, r.cached_len:r.device_len] for r in reqs])
    engine.attn_backend.prepare_metadata(b)
    return b


def reqs(cached):
    return [Req(input_ids=torch.tensor(p, dtype=torch.int32), table_idx=i, cached_len=c, output_len=4,
                uid=i, sampling_params=SamplingParams(), cache_handle=None)
            for i, (p, c) in enumerate(zip(PROMPTS, cached))]


results = {}
for backend, ps in (("torch", 1), ("fi", 1), ("fa", 4)):
    engine = build(backend, ps)
    with engine.ctx.forward_batch(b := batch_of(engine, "prefill", reqs([0, 0]))):
        engine.model.forward()  # 先把两个请求的 KV 都算好
    rs = reqs([3, 5])  # 请求 0 命中前 3 个 token，请求 1 只剩最后 1 个（像 decode）
    b = batch_of(engine, "prefill", rs)
    m = b.attn_metadata
    if backend == "fi":
        print("FlashInfer（按 page_size=1 使用 KV 池）:")
        print("  qo_indptr      =", m.cu_seqlens_q_cpu.tolist())
        print("  kv_indptr      =", m.cu_seqlens_k_cpu.tolist())
        print("  kv_indices     =", m.indices.tolist())
        print("  last_page_len  =", m.last_page_len_cpu.tolist())
    if backend == "fa":
        print("FlashAttention（page_size=4，page table 存页号）:")
        print("  cu_seqlens_q   =", m.cu_seqlens_q.tolist())
        print("  cache_seqlens  =", m.cache_seqlens.tolist())
        print("  page_table     =", m.page_table.tolist())
    with engine.ctx.forward_batch(b):
        results[backend] = engine.model.forward()
    if backend == "fi":
        print("  prefill wrapper 的 plan 次数:", engine.attn_backend.prefill_wrapper.plan_count,
              "（两个 batch 各一次；每个 batch 的 28 层共用一次 plan）")
    engine.shutdown()
for name in ("fi", "fa"):
    print(f"{name} 与参考后端 logits 的最大误差: {(results[name] - results['torch']).abs().max().item():.2e}")
