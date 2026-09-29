import numpy as np

from checker import check, check_close
from solution import absorb, decode_step, mla_reference


def make(seed, D=32, H=4, NOPE=8, ROPE=4, DV=6, DC=12):
    rng = np.random.default_rng(seed)
    return {
        "dims": (D, H, NOPE, ROPE, DV, DC),
        "W_q": rng.standard_normal((D, H * (NOPE + ROPE))) / 4,
        "W_dkv": rng.standard_normal((D, DC)) / 4,
        "W_kr": rng.standard_normal((D, ROPE)) / 4,
        "W_uk": rng.standard_normal((H, DC, NOPE)) / 3,
        "W_uv": rng.standard_normal((H, DC, DV)) / 3,
        "W_o": rng.standard_normal((H * DV, D)) / 4,
    }, rng


def run_decode(h, W):
    A = absorb(W)
    cache = {"c": [], "kr": []}
    return np.stack([decode_step(h[t], t, cache, A) for t in range(len(h))]), cache


def test_example():
    W, rng = make(0)
    h = rng.standard_normal((6, 32))
    out, _ = run_decode(h, W)
    check_close(out, mla_reference(h, W), rtol=1e-8, atol=1e-10, what="逐个位置 decode = 一次算完整个序列")


def test_absorbed_shapes():
    W, _ = make(1, D=20, H=3, NOPE=6, ROPE=4, DV=5, DC=10)
    A = absorb(W)
    check(A["W_q_lat"].shape, (3, 20, 10), "W_q_lat：(H, D, DC)")
    check(A["W_q_rope"].shape, (3, 20, 4), "W_q_rope：(H, D, ROPE)")
    check(A["W_o_lat"].shape, (3, 10, 20), "W_o_lat：(H, DC, D)")
    x = np.random.default_rng(2).standard_normal(20)
    q_n = (x @ W["W_q"]).reshape(3, 10)[:, :6]
    check_close(np.einsum("d,hdc->hc", x, A["W_q_lat"]), np.einsum("hn,hcn->hc", q_n, W["W_uk"]),
                rtol=1e-10, atol=1e-12, what="W_q_lat 把 q 的非位置部分变到潜空间：q^N W_UKᵀ")


def test_cache_contents():
    W, rng = make(3)
    h = rng.standard_normal((5, 32))
    _, cache = run_decode(h, W)
    check(len(cache["c"]), 5, "每个 token 往缓存里追加一个潜向量")
    check_close(np.stack(cache["c"]), h @ W["W_dkv"], rtol=1e-10, atol=1e-12, what="缓存的潜向量 c")
    D, H, NOPE, ROPE, DV, DC = W["dims"]
    check(np.stack(cache["kr"]).shape, (5, ROPE), "每个 token 只缓存一份所有头共享的 RoPE 键")


def test_longer_and_other_sizes():
    for seed, dims in ((4, dict()), (5, dict(D=48, H=8, NOPE=16, ROPE=8, DV=16, DC=32)), (6, dict(H=1, DC=3))):
        W, rng = make(seed, **dims)
        h = rng.standard_normal((40, W["dims"][0]))
        out, _ = run_decode(h, W)
        check_close(out, mla_reference(h, W), rtol=1e-8, atol=1e-10, what=f"尺寸 {W['dims']}、40 个位置")


def test_prefilled_cache():
    W, rng = make(7)
    h = rng.standard_normal((12, 32))
    ref = mla_reference(h, W)
    A = absorb(W)
    D, H, NOPE, ROPE, DV, DC = W["dims"]
    from solution import rope

    cache = {"c": list(h[:8] @ W["W_dkv"]), "kr": list(rope(h[:8] @ W["W_kr"], np.arange(8)))}   # 前 8 个 token 由 prefill 写好
    out = np.stack([decode_step(h[t], t, cache, A) for t in range(8, 12)])
    check_close(out, ref[8:], rtol=1e-8, atol=1e-10, what="接着 prefill 写好的缓存继续 decode")
