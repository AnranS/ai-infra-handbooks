from checker import check, check_close, need_torch, raises


def _m():
    need_torch()
    import torch

    from solution import attn_scores, broadcast_shape, merge_heads, split_heads

    return torch, attn_scores, merge_heads, split_heads, broadcast_shape


def test_example():
    torch, attn_scores, _, _, broadcast_shape = _m()
    torch.manual_seed(0)
    q, k = torch.randn(2, 4, 6, 8), torch.randn(2, 4, 10, 8)
    out = attn_scores(q, k)
    check(tuple(out.shape), (2, 4, 6, 10), "打分的形状")
    ref = q @ k.transpose(-1, -2) * 8**-0.5
    check_close(out, ref, rtol=1e-5, atol=1e-6, what="缩放点积")
    check(broadcast_shape((4, 1, 3), (2, 3)), (4, 2, 3), "广播形状")


def test_scores_scaling():
    torch, attn_scores, _, _, _ = _m()
    torch.manual_seed(1)
    for b, h, t, s, d in ((1, 1, 1, 1, 4), (3, 2, 5, 7, 16), (2, 8, 1, 9, 64)):
        q, k = torch.randn(b, h, t, d), torch.randn(b, h, s, d)
        check_close(attn_scores(q, k), q @ k.transpose(-1, -2) * d**-0.5,
                    rtol=1e-5, atol=1e-6, what=f"({b},{h},{t},{s},{d}) 的打分")


def test_merge_split_roundtrip():
    torch, _, merge_heads, split_heads, _ = _m()
    torch.manual_seed(2)
    x = torch.randn(3, 4, 5, 6)
    m = merge_heads(x)
    check(tuple(m.shape), (3, 5, 24), "合并后的形状")
    # 第 0 个 batch、第 2 个 token：拼接的顺序必须是「头在外、维度在内」
    check_close(m[0, 2], torch.cat([x[0, i, 2] for i in range(4)]), rtol=1e-6, atol=1e-7,
                what="合并的顺序（先头后维度）")
    back = split_heads(m, 4)
    check(tuple(back.shape), (3, 4, 5, 6), "拆开后的形状")
    check_close(back, x, rtol=1e-6, atol=1e-7, what="split_heads 是 merge_heads 的逆运算")


def test_broadcast_rules():
    _, _, _, _, broadcast_shape = _m()
    cases = [((), (3,), (3,)), ((5,), (), (5,)), ((1,), (7,), (7,)), ((2, 1), (1, 3), (2, 3)),
             ((4, 1, 3), (2, 3), (4, 2, 3)), ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),
             ((3, 4), (3, 4), (3, 4))]
    for sa, sb, want in cases:
        check(broadcast_shape(sa, sb), want, f"{sa} 和 {sb} 广播")
        check(broadcast_shape(sb, sa), want, f"{sb} 和 {sa} 广播（交换顺序）")
    for sa, sb in (((3,), (4,)), ((2, 3), (2, 4)), ((5, 2), (3,))):
        with raises(ValueError, what=f"{sa} 和 {sb} 不能广播"):
            broadcast_shape(sa, sb)
