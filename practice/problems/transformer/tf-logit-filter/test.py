from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import filter_logits

    return torch, filter_logits


def _kept(x):
    return [i for i, v in enumerate(x.tolist()) if v != float("-inf")]


def test_example():
    torch, filter_logits = _m()
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
    check(_kept(filter_logits(logits, top_k=2)), [0, 1], "top-k=2 留两个")
    check(_kept(filter_logits(logits, top_p=0.5)), [0], "top-p=0.5 只留第一个")
    check_close(filter_logits(logits), logits, rtol=0, atol=1e-6, what="默认参数不改分数")


def test_temperature():
    torch, filter_logits = _m()
    logits = torch.tensor([1.0, -2.0, 0.5])
    check_close(filter_logits(logits, temperature=0.5), logits / 0.5, rtol=1e-6, atol=1e-6, what="除以温度")


def test_top_p_boundary():
    torch, filter_logits = _m()
    logits = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
    check(_kept(filter_logits(logits, top_p=0.65)), [0, 1], "0.4 + 0.3 = 0.7 刚好跨过 0.65：留两个")
    check(_kept(filter_logits(logits, top_p=0.71)), [0, 1, 2], "超过之后多留一个")
    check(_kept(filter_logits(logits, top_p=0.01)), [0], "p 很小也至少留一个")
    order = torch.tensor([2, 0, 3, 1])                                        # 位置打乱：按分数而不是按下标
    check(sorted(_kept(filter_logits(logits[order], top_p=0.65))), [1, 3], "乱序输入")


def test_penalties_and_order():
    torch, filter_logits = _m()
    logits = torch.tensor([2.0, -1.0, 0.5, 1.0])
    out = filter_logits(logits, repetition_penalty=2.0, generated=[0, 1])
    check_close(out[0].item(), 1.0, rtol=1e-6, atol=1e-6, what="正分除以惩罚")
    check_close(out[1].item(), -2.0, rtol=1e-6, atol=1e-6, what="负分乘以惩罚")
    check_close(out[2].item(), 0.5, rtol=1e-6, atol=1e-6, what="没出现过的不动")
    out = filter_logits(logits, presence_penalty=1.5, generated=[3, 3, 3])
    check_close(out[3].item(), -0.5, rtol=1e-6, atol=1e-6, what="出现惩罚是减法，出现几次都只减一次")
    # 顺序：惩罚在温度之前，截断在温度之后
    out = filter_logits(logits, temperature=2.0, top_k=1, repetition_penalty=4.0, generated=[0])
    check(_kept(out), [3], "惩罚之后原来的第一名掉下去，top-k 看的是惩罚和温度之后的分数")
    check_close(out[3].item(), 0.5, rtol=1e-6, atol=1e-6, what="留下的分数已经除过温度")
