from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import accumulated_grad, freeze_count, mid_grad

    return torch, mid_grad, accumulated_grad, freeze_count


def test_example():
    torch, mid_grad, _, _ = _m()
    x = torch.tensor([0.5, -1.0])
    dh, dx = mid_grad(x)
    h = x.sigmoid()
    check(dh is not None, True, "中间结果的梯度不能是 None（想想 retain_grad）")
    check_close(dh, 2 * h, rtol=1e-6, atol=1e-7, what="dy/dh")
    check_close(dx, 2 * h * h * (1 - h), rtol=1e-6, atol=1e-7, what="dy/dx")


def test_mid_grad_shapes():
    torch, mid_grad, _, _ = _m()
    torch.manual_seed(0)
    for shape in ((1,), (4,), (3, 5)):
        x = torch.randn(shape)
        keep = x.clone()
        dh, dx = mid_grad(x)
        h = x.sigmoid()
        check(tuple(dh.shape), shape, f"{shape} 时 dy/dh 的形状")
        check_close(dh, 2 * h, rtol=1e-6, atol=1e-7, what="dy/dh")
        check_close(dx, 2 * h * h * (1 - h), rtol=1e-6, atol=1e-7, what="dy/dx")
        check_close(x, keep, rtol=0, atol=0, what="不能修改传进来的 x")


def test_accumulated_grad():
    torch, _, accumulated_grad, _ = _m()
    torch.manual_seed(1)
    d = 6
    for sizes in ([4, 4, 4], [3], [5, 2, 7, 1]):
        xs = [torch.randn(m, d) for m in sizes]
        ys = [torch.randn(m) for m in sizes]
        w = torch.randn(d, requires_grad=True)
        got = accumulated_grad(w, xs, ys)
        big_w = w.detach().clone().requires_grad_(True)
        ((torch.cat(xs) @ big_w - torch.cat(ys)) ** 2).mean().backward()
        check(got is not None, True, "要返回 w.grad")
        check_close(got, big_w.grad, rtol=1e-5, atol=1e-6,
                    what=f"小批 {sizes} 累积出来的梯度要等于大批一次算的梯度")


def test_accumulated_grad_is_clean():
    torch, _, accumulated_grad, _ = _m()
    torch.manual_seed(2)
    d = 4
    xs, ys = [torch.randn(3, d), torch.randn(3, d)], [torch.randn(3), torch.randn(3)]
    w = torch.randn(d, requires_grad=True)
    w.grad = torch.full((d,), 100.0)                     # 上一步留下来的脏梯度
    got = accumulated_grad(w, xs, ys)
    big_w = w.detach().clone().requires_grad_(True)
    ((torch.cat(xs) @ big_w - torch.cat(ys)) ** 2).mean().backward()
    check_close(got, big_w.grad, rtol=1e-5, atol=1e-6, what="开始前要先清掉上一步的梯度")


def test_freeze_count():
    torch, _, _, freeze_count = _m()
    import torch.nn as nn

    layers = [nn.Linear(4, 4) for _ in range(5)]
    n = freeze_count(layers, "layer3")
    check(n, 2, "只有 layer3 可训练时，可训练的参数张量个数（weight + bias）")
    for i, layer in enumerate(layers):
        want = i == 3
        for p in layer.parameters():
            check(p.requires_grad, want, f"layer{i} 的 requires_grad")

    layers = [nn.Linear(3, 3, bias=False) for _ in range(4)]
    check(freeze_count(layers, "layer"), 4, "前缀匹配全部时：4 个 weight")
    check(freeze_count(layers, "nope"), 0, "前缀谁也不匹配时")
    check(all(not p.requires_grad for L in layers for p in L.parameters()), True, "全部冻结")
