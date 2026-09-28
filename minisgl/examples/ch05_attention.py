"""第 5 章：一个混合 batch 的注意力元数据，以及参考后端与"不分页的完整注意力"的对比。

请求 A：6 个 token 从头 prefill；请求 B：前 4 个 token 命中缓存，再算 3 个；请求 C：decode 一步。
"""

import torch
import torch.nn.functional as F
from minisgl.attention.torch_backend import TorchAttnBackend
from minisgl.core import Batch, Context, Req, SamplingParams, set_global_ctx
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.models import ModelConfig
from minisgl.utils import cached_load_hf_config

set_tp_info(0, 1)
H, HKV, D = 16, 8, 128
ctx = Context(page_size=1)
set_global_ctx(ctx)
ctx.kv_cache = MHAKVCache(HKV, num_layers=1, head_dim=D, num_pages=64, page_size=1,
                          dtype=torch.float32, device=torch.device("cpu"))
ctx.page_table = torch.zeros((3, 16), dtype=torch.int32)
ctx.page_table[0, :6] = torch.arange(0, 6)
ctx.page_table[1, :7] = torch.tensor([40, 41, 42, 43, 20, 21, 22])  # 前 4 个位置来自前缀缓存
ctx.page_table[2, :9] = torch.arange(50, 59)
backend = TorchAttnBackend(ModelConfig.from_hf(cached_load_hf_config("models/Qwen3-0.6B")))


def req(n, cached, row):
    return Req(input_ids=torch.zeros(n, dtype=torch.int32), table_idx=row, cached_len=cached,
               output_len=4, uid=row, sampling_params=SamplingParams(), cache_handle=None)


torch.manual_seed(0)
lens, cached = [6, 7, 9], [0, 4, 8]
full = [(torch.randn(n, H, D), torch.randn(n, HKV, D), torch.randn(n, HKV, D)) for n in lens]
for i, c in enumerate(cached):  # 已缓存的部分预先写进 KV 池
    if c:
        ctx.kv_cache.store_kv(full[i][1][:c], full[i][2][:c], ctx.page_table[i, :c], 0)

batch = Batch(reqs=[req(n, c, i) for i, (n, c) in enumerate(zip(lens, cached))], phase="prefill")
batch.padded_reqs = batch.reqs
batch.out_loc = torch.cat([ctx.page_table[i, c:n] for i, (n, c) in enumerate(zip(lens, cached))])
backend.prepare_metadata(batch)
m = batch.attn_metadata
print("cu_seqlens_q :", m.cu_seqlens_q.tolist(), " (每个请求本轮的 query 起止)")
print("cache_seqlens:", m.cache_seqlens.tolist(), "  (每个请求的 KV 总长)")
print("page_table   :", m.page_table.tolist())
print("out_loc      :", batch.out_loc.tolist())
print("最后位置下标 :", m.get_last_indices(3).tolist())

q = torch.cat([f[0][c:] for f, c in zip(full, cached)])
k = torch.cat([f[1][c:] for f, c in zip(full, cached)])
v = torch.cat([f[2][c:] for f, c in zip(full, cached)])
out = backend.forward(q, k, v, layer_id=0, batch=batch)
ref = torch.cat([
    F.scaled_dot_product_attention(f[0].transpose(0, 1), f[1].transpose(0, 1), f[2].transpose(0, 1),
                                   is_causal=True, scale=D**-0.5, enable_gqa=True).transpose(0, 1)[c:]
    for f, c in zip(full, cached)])
print("与完整因果注意力的最大误差:", (out - ref).abs().max().item())
