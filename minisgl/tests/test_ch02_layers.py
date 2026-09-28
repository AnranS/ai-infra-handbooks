import torch
from minisgl.distributed import set_tp_info
from minisgl.kernel import torch_ops
from minisgl.layers import BaseOP, OPList, RMSNorm
from minisgl.layers.rotary import get_rope


class Block(BaseOP):
    def __init__(self):
        self.norm = RMSNorm(4, 1e-6)
        self._scratch = torch.zeros(1)  # 下划线开头：不是权重


class Model(BaseOP):
    def __init__(self):
        self.layers = OPList([Block(), Block()])
        self.lm_head_weight = torch.zeros(2, 4)


def test_state_dict_names_follow_attribute_paths():
    names = sorted(Model().state_dict())
    assert names == ["layers.0.norm.weight", "layers.1.norm.weight", "lm_head_weight"]


def test_load_state_dict_replaces_tensors_and_rejects_extra_keys():
    m = Model()
    sd = {k: torch.full_like(v, 7.0) for k, v in m.state_dict().items()}
    m.load_state_dict(dict(sd))
    assert torch.equal(m.layers.op_list[1].norm.weight, torch.full((4,), 7.0))
    sd["unknown"] = torch.zeros(1)
    try:
        m.load_state_dict(sd)
        raise AssertionError("should fail")
    except RuntimeError as e:
        assert "unknown" in str(e)


def test_rmsnorm_matches_hf():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    torch.manual_seed(0)
    x, w = torch.randn(5, 64), torch.randn(64)
    ref = Qwen3RMSNorm(64, eps=1e-6)
    ref.weight.data.copy_(w)
    assert torch.allclose(torch_ops.rmsnorm(x, w, 1e-6), ref(x), atol=1e-6)
    residual = torch.randn(5, 64)
    x2, r2 = x.clone(), residual.clone()
    torch_ops.fused_add_rmsnorm(x2, r2, w, 1e-6)
    assert torch.allclose(r2, x + residual) and torch.allclose(x2, ref(x + residual), atol=1e-6)


def test_rope_matches_hf():
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    set_tp_info(0, 1)
    torch.manual_seed(0)
    T, H, D, base = 7, 4, 64, 10000.0
    rope = get_rope(D, D, 128, base)
    positions = torch.tensor([0, 1, 2, 10, 11, 50, 127])
    q, k = torch.randn(T, H * D), torch.randn(T, 2 * D)
    q_ref, k_ref = q.view(T, H, D).transpose(0, 1), k.view(T, 2, D).transpose(0, 1)
    inv_freq = 1.0 / base ** (torch.arange(0, D, 2).float() / D)
    freqs = positions.float()[:, None] * inv_freq[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    q_hf, k_hf = apply_rotary_pos_emb(q_ref[None], k_ref[None], emb.cos()[None], emb.sin()[None])
    rope.forward(positions, q, k)
    assert torch.allclose(q.view(T, H, D).transpose(0, 1), q_hf[0], atol=1e-5)
    assert torch.allclose(k.view(T, 2, D).transpose(0, 1), k_hf[0], atol=1e-5)


def test_vocab_parallel_indexing_masks_other_shards():
    w = torch.arange(20.0).view(10, 2)  # 词表 [5, 15) 这一段
    ids = torch.tensor([3, 5, 14, 15, 9])
    out = torch_ops.indexing(w, ids, vocab_range=(5, 10))
    assert out.tolist() == [[0, 0], [0, 1], [18, 19], [0, 0], [8, 9]]
