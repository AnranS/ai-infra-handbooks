import torch
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache


def test_kv_pool_layout_and_store():
    set_tp_info(0, 1)
    pool = MHAKVCache(num_kv_heads=2, num_layers=3, head_dim=4, num_pages=5, page_size=2,
                      dtype=torch.float32, device=torch.device("cpu"))
    assert pool._kv_buffer.shape == (2, 3, 5, 2, 2, 4)  # K/V, 层, 页, 页内位置, 头, 维度
    k = torch.arange(3 * 8, dtype=torch.float32).view(3, 8)  # 3 个 token，每个 2 头 x 4 维
    pool.store_kv(k, -k, out_loc=torch.tensor([9, 0, 4], dtype=torch.int32), layer_id=1)
    flat_k = pool.k_cache(1).reshape(10, 2, 4)  # 按 token 展平：第 9 个 token 在第 4 页的第 2 个位置
    assert torch.equal(flat_k[9].flatten(), k[0]) and torch.equal(flat_k[4].flatten(), k[2])
    assert torch.equal(pool.v_cache(1)[4, 1].flatten(), -k[0])


def test_kv_pool_replicates_heads_under_tp():
    set_tp_info(1, 4)  # 2 个 KV 头、4 个 rank：每个 rank 保存 1 个（复制的）头
    pool = MHAKVCache(num_kv_heads=2, num_layers=1, head_dim=8, num_pages=4, page_size=1,
                      dtype=torch.float32, device=torch.device("cpu"))
    assert pool.k_cache(0).shape == (4, 1, 1, 8)
