from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import bucket_batches, collate, pad_fraction

    return torch, collate, bucket_batches, pad_fraction


def test_example():
    _, _, bucket_batches, _ = _m()
    check(bucket_batches([5, 1, 3, 2], 2), [[1, 3], [2, 0]], "按长度排序后切块")


def test_collate():
    torch, collate, _, _ = _m()
    batch = [(torch.tensor([1, 2, 3]), 0), (torch.tensor([4]), 1), (torch.tensor([5, 6]), 2)]
    out = collate(batch)
    check(sorted(out.keys()), ["ids", "labels", "mask"], "返回的 key")
    check(tuple(out["ids"].shape), (3, 3), "ids 要补齐到这一批的最长")
    check(out["ids"].dtype, torch.long, "ids 的 dtype")
    check(out["ids"].tolist(), [[1, 2, 3], [4, 0, 0], [5, 6, 0]], "右侧补 0")
    check(out["mask"].dtype, torch.bool, "mask 的 dtype")
    check(out["mask"].tolist(), [[True] * 3, [True, False, False], [True, True, False]], "mask")
    check(out["labels"].dtype, torch.long, "labels 的 dtype")
    check(out["labels"].tolist(), [0, 1, 2], "labels")


def test_collate_single_and_equal():
    torch, collate, _, _ = _m()
    out = collate([(torch.tensor([7, 8, 9, 10]), 5)])
    check(tuple(out["ids"].shape), (1, 4), "只有一条样本时")
    check(bool(out["mask"].all()), True, "长度相同时不该有 padding")
    out = collate([(torch.tensor([1, 1]), 0), (torch.tensor([2, 2]), 1)])
    check(tuple(out["ids"].shape), (2, 2), "两条等长样本不该被撑大")
    check(bool(out["mask"].all()), True, "等长时 mask 全 True")


def test_bucket_batches():
    _, _, bucket_batches, _ = _m()
    lengths = [9, 1, 7, 3, 5, 2, 8]
    for bs in (1, 2, 3, 7, 10):
        bs_list = bucket_batches(lengths, bs)
        flat = [i for b in bs_list for i in b]
        check(sorted(flat), list(range(len(lengths))), f"batch_size={bs} 时每个下标出现且只出现一次")
        check([len(b) for b in bs_list[:-1]], [bs] * (len(bs_list) - 1), f"batch_size={bs} 时除最后一批都要满")
        check([lengths[i] for i in flat], sorted(lengths), f"batch_size={bs} 时要按长度从短到长")


def test_pad_fraction():
    _, _, bucket_batches, pad_fraction = _m()
    lengths = [10, 2, 9, 3]
    check_close(pad_fraction(lengths, [[0, 1], [2, 3]]), (20 - 12 + 18 - 12) / 38,
                rtol=1e-9, atol=1e-12, what="padding 占比")
    check_close(pad_fraction(lengths, [[0, 2], [1, 3]]), (20 - 19 + 6 - 5) / 26,
                rtol=1e-9, atol=1e-12, what="分桶之后的 padding 占比")
    check_close(pad_fraction([4, 4, 4], [[0, 1, 2]]), 0.0, rtol=1e-9, atol=1e-12, what="等长时没有 padding")
    worse = pad_fraction(lengths, [[0, 1], [2, 3]])
    better = pad_fraction(lengths, bucket_batches(lengths, 2))
    check(better < worse, True, "分桶之后 padding 占比要更低")
