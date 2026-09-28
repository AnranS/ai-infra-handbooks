import torch
import torch.nn.functional as F

from conftest import QWEN3
from helpers import build_engine, identity_page_table, new_req


def dense_causal(q, k, v, scale):
    """不分页、不变长的参考：单条序列的完整因果注意力。q/k/v: [T, H, D]。"""
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                       is_causal=True, scale=scale, enable_gqa=True)
    return o.transpose(0, 1)


def test_torch_backend_mixed_batch_with_prefix_hits():
    """两个请求：A 从头 prefill 6 个 token；B 前 4 个 token 已在缓存里，只算后 3 个。"""
    engine = build_engine(QWEN3)
    identity_page_table(engine, 2)
    backend, layer, H, Hkv, D = engine.attn_backend, 5, 16, 8, 128
    torch.manual_seed(0)
    full = {name: (torch.randn(n, H, D), torch.randn(n, Hkv, D), torch.randn(n, Hkv, D))
            for name, n in (("A", 6), ("B", 7))}
    # B 的前 4 个 token 预先写进 KV 池（模拟前缀缓存命中）
    loc_b = engine.page_table[1, :4]
    engine.kv_cache.store_kv(full["B"][1][:4], full["B"][2][:4], loc_b, layer)
    a, b = new_req(list(range(6)), 0), new_req(list(range(7)), 1, cached_len=4)
    from minisgl.core import Batch

    batch = Batch(reqs=[a, b], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.out_loc = torch.cat([engine.page_table[0, :6], engine.page_table[1, 4:7]])
    backend.prepare_metadata(batch)
    q = torch.cat([full["A"][0], full["B"][0][4:]])
    k = torch.cat([full["A"][1], full["B"][1][4:]])
    v = torch.cat([full["A"][2], full["B"][2][4:]])
    out = backend.forward(q, k, v, layer, batch)
    scale = D**-0.5
    ref_a = dense_causal(*full["A"], scale)
    ref_b = dense_causal(*full["B"], scale)[4:]
    assert torch.allclose(out[:6], ref_a, atol=1e-5) and torch.allclose(out[6:], ref_b, atol=1e-5)
    # LM head 只需要每个请求最后一个位置
    assert batch.attn_metadata.get_last_indices(2).tolist() == [5, 8]
    engine.shutdown()
