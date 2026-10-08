from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import last_token, mask_scores, pad_mask, top_k_filter

    return torch, pad_mask, mask_scores, last_token, top_k_filter


def test_example():
    torch, pad_mask, _, _, _ = _m()
    m = pad_mask(torch.tensor([2, 3]), 4)
    check(m.dtype, torch.bool, "掩码的 dtype")
    check(m.tolist(), [[True, True, False, False], [True, True, True, False]], "掩码")


def test_pad_mask_shapes():
    torch, pad_mask, _, _, _ = _m()
    for lens, n in (([1], 1), ([0, 3, 5], 5), ([4, 4, 4, 4], 6)):
        m = pad_mask(torch.tensor(lens), n)
        check(tuple(m.shape), (len(lens), n), f"lengths={lens} 时掩码的形状")
        check(m.tolist(), [[j < L for j in range(n)] for L in lens], f"lengths={lens} 时的掩码")


def test_mask_scores():
    torch, _, mask_scores, _, _ = _m()
    torch.manual_seed(0)
    scores = torch.randn(3, 4, 5)
    keep = scores.clone()
    lengths = torch.tensor([5, 2, 0])
    out = mask_scores(scores, lengths)
    check(tuple(out.shape), (3, 4, 5), "输出形状")
    check_close(scores, keep, rtol=0, atol=0, what="不能修改输入（要返回新张量）")
    for b, L in enumerate(lengths.tolist()):
        check(bool(torch.isinf(out[b, :, L:]).all() and (out[b, :, L:] < 0).all()), True,
              f"第 {b} 条序列 {L} 之后的 key 要是 -inf")
        check_close(out[b, :, :L], scores[b, :, :L], rtol=0, atol=0, what=f"第 {b} 条序列的有效位置不能改")


def test_last_token():
    torch, _, _, last_token, _ = _m()
    torch.manual_seed(1)
    h = torch.randn(4, 7, 5)
    lengths = torch.tensor([1, 7, 3, 6])
    out = last_token(h, lengths)
    check(tuple(out.shape), (4, 5), "输出形状")
    for b, L in enumerate(lengths.tolist()):
        check_close(out[b], h[b, L - 1], rtol=0, atol=0, what=f"第 {b} 条序列的最后一个有效 token")


def test_top_k_filter():
    torch, _, _, _, top_k_filter = _m()
    torch.manual_seed(2)
    logits = torch.randn(5, 11)
    keep = logits.clone()
    for k in (1, 3, 11):
        out = top_k_filter(logits, k)
        check(tuple(out.shape), (5, 11), "输出形状")
        check_close(logits, keep, rtol=0, atol=0, what="不能修改输入")
        for b in range(5):
            finite = torch.isfinite(out[b])
            check(int(finite.sum()), k, f"k={k} 时第 {b} 行保留的个数")
            want = logits[b].topk(k).indices.sort().values
            check(finite.nonzero().flatten().tolist(), want.tolist(), f"k={k} 时第 {b} 行保留的位置")
            check_close(out[b][finite], logits[b][finite], rtol=0, atol=0, what="保留下来的值不能变")
        check(bool((out[~torch.isfinite(out)] < 0).all()), True, "被过滤的位置要是 -inf 而不是 +inf")
