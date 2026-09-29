import torch

from lm.model import Config, TransformerLM

CFG = Config(vocab_size=100, context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=96)


def test_shape():
    torch.manual_seed(0)
    out = TransformerLM(CFG)(torch.randint(0, 100, (3, 17)))
    assert out.shape == (3, 17, 100)


def test_param_count():
    V, D, L, F = CFG.vocab_size, CFG.d_model, CFG.n_layers, CFG.d_ff
    expected = V * D + L * (4 * D * D + 3 * D * F + 2 * D) + D + D * V
    assert sum(p.numel() for p in TransformerLM(CFG).parameters()) == expected


def test_causal():
    torch.manual_seed(0)
    model = TransformerLM(CFG)
    x = torch.randint(0, 100, (1, 20))
    y = x.clone()
    y[0, 12:] = torch.randint(0, 100, (8,))
    a, b = model(x), model(y)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5), "改动后面的 token 不应影响前面的输出"
    assert not torch.allclose(a[0, 12:], b[0, 12:]), "改动的位置之后输出应该变化"


def test_full_context():
    torch.manual_seed(0)
    out = TransformerLM(CFG)(torch.randint(0, 100, (2, CFG.context_length)))
    assert torch.isfinite(out).all(), "满长度的输入输出应当有限"
