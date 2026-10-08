from checker import check, check_close, need_torch


def _m():
    need_torch()
    import torch

    from solution import Stack, load_partial, state_keys

    return torch, Stack, state_keys, load_partial


def test_example():
    torch, Stack, state_keys, _ = _m()
    m = Stack(4, 3)
    check(len(list(m.parameters())), 6, "参数张量个数（子模块要放进 ModuleList）")
    check(state_keys(m)[:3], ["scale", "layers.0.weight", "layers.0.bias"], "state_dict 的前三个 key")


def test_parameters_and_buffer():
    torch, Stack, state_keys, _ = _m()
    for dim, n in ((4, 1), (8, 3), (16, 5)):
        m = Stack(dim, n)
        check(len(list(m.parameters())), 2 * n, f"Stack({dim}, {n}) 的参数张量个数")
        check(sum(p.numel() for p in m.parameters()), n * (dim * dim + dim), "参数元素总数")
        names = [k for k, _ in m.named_buffers()]
        check(names, ["scale"], "注册的 buffer")
        check_close(m.scale, torch.tensor(dim**-0.5), rtol=1e-6, atol=1e-7, what="scale 的值")
        check(any(p is m.scale for p in m.parameters()), False, "scale 不能是参数")
        check("scale" in state_keys(m), True, "scale 要进 state_dict")


def test_forward():
    torch, Stack, _, _ = _m()
    torch.manual_seed(0)
    m = Stack(5, 3)
    x = torch.randn(7, 5)
    out = m(x)
    check(tuple(out.shape), (7, 5), "前向的输出形状")
    ref = x
    for layer in m.layers:
        ref = layer(ref).relu()
    check_close(out, ref * m.scale, rtol=1e-5, atol=1e-6, what="逐层 linear + relu，最后乘 scale")


def test_to_moves_everything():
    torch, Stack, _, _ = _m()
    m = Stack(4, 2).to(torch.float64)
    check(all(p.dtype == torch.float64 for p in m.parameters()), True, ".to() 要能搬走所有参数")
    check(m.scale.dtype, torch.float64, ".to() 要能搬走 buffer")
    out = m(torch.randn(3, 4, dtype=torch.float64))
    check(out.dtype, torch.float64, "float64 下前向不该报 dtype 不匹配")


def test_load_partial():
    torch, Stack, _, load_partial = _m()
    torch.manual_seed(1)
    small, big = Stack(4, 2), Stack(4, 3)
    missing, unexpected = load_partial(big, small.state_dict())
    check(missing, ["layers.2.bias", "layers.2.weight"], "模型里有、文件里没有的 key")
    check(unexpected, [], "文件里有、模型里没有的 key")
    check_close(big.layers[0].weight, small.layers[0].weight, rtol=0, atol=0, what="对得上的权重要真的加载进去")

    missing, unexpected = load_partial(small, big.state_dict())
    check(missing, [], "反过来：没有缺的")
    check(unexpected, ["layers.2.bias", "layers.2.weight"], "反过来：多出来的 key")

    state = dict(small.state_dict())
    state["head.weight"] = torch.zeros(4, 4)
    state.pop("scale")
    missing, unexpected = load_partial(Stack(4, 2), state)
    check(missing, ["scale"], "buffer 也算在 missing 里")
    check(unexpected, ["head.weight"], "多出来的 head")
