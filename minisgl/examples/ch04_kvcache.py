"""第 4 章：KV 池的布局、容量估算，以及 page table / token pool 的样子。"""

import torch
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.models import ModelConfig
from minisgl.utils import cached_load_hf_config

set_tp_info(0, 1)
c = ModelConfig.from_hf(cached_load_hf_config("models/Qwen3-0.6B"))
per_token = 2 * c.num_layers * c.num_kv_heads * c.head_dim * 2  # K+V，bf16
print(f"Qwen3-0.6B 每个 token 的 KV：{per_token} 字节 = {per_token / 1024:.0f} KiB")
for gib in (4, 40):
    print(f"  {gib:>2} GiB 显存能放 {gib * 2**30 // per_token:>8} 个 token")

pool = MHAKVCache(num_kv_heads=c.num_kv_heads, num_layers=c.num_layers, head_dim=c.head_dim,
                  num_pages=64 + 1, page_size=16, dtype=torch.bfloat16, device=torch.device("cpu"))
print("KV 池形状 [K/V, 层, 页, 页内位置, 头, 维度]:", tuple(pool._kv_buffer.shape))
print("第 3 层的 K:", tuple(pool.k_cache(3).shape), " 按 token 展平:", pool._storage_shape)

# 两个请求的 page table（page_size = 16，按 token 存位置）
page_table = torch.zeros((3, 64), dtype=torch.int32)
page_table[0, :20] = torch.cat([torch.arange(32, 48), torch.arange(80, 84)])  # 第 2 页 + 第 5 页的前 4 个
page_table[1, :5] = torch.arange(160, 165)  # 第 10 页的前 5 个
print("请求 0 的前 20 个位置:", page_table[0, :20].tolist())
print("请求 1 的前 5 个位置: ", page_table[1, :5].tolist())
