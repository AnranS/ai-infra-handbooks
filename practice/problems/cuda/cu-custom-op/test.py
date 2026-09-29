from checker import check, check_close, need_torch


def _ops():
    need_torch()
    import torch

    import solution  # 导入时注册算子

    for name in ("silu_and_mul", "fused_add_rms_norm"):
        if type(getattr(solution, name, None)).__name__ != "CustomOpDef":
            raise AssertionError(f"{name} 要用 @torch.library.custom_op(\"practice::{name}\", ...) 注册（模块里的 {name} 应该是它返回的对象）")
    return torch


def test_example():
    torch = _ops()
    x = torch.randn(4, 16)
    out = torch.ops.practice.silu_and_mul(x)
    check(tuple(out.shape), (4, 8), "输出的最后一维减半")
    check_close(out, torch.nn.functional.silu(x[:, :8]) * x[:, 8:], rtol=1e-6, atol=1e-6, what="silu(gate) * up")


def test_fused_add_rms_norm_inplace():
    torch = _ops()
    torch.manual_seed(0)
    x, res, w = torch.randn(3, 32), torch.randn(3, 32), torch.rand(32) + 0.5
    x0, r0 = x.clone(), res.clone()
    ret = torch.ops.practice.fused_add_rms_norm(x, res, w, 1e-6)
    check(ret is None, True, "没有返回值")
    new_res = x0 + r0
    check_close(res, new_res, rtol=1e-6, atol=1e-6, what="residual 被原地加上 x")
    check_close(x, new_res * torch.rsqrt(new_res.pow(2).mean(-1, keepdim=True) + 1e-6) * w, rtol=1e-5, atol=1e-6,
                what="x 被原地改写成 rmsnorm(residual) * weight")


def test_opcheck():
    torch = _ops()
    torch.manual_seed(1)
    for shape in ((2, 8), (3, 5, 12)):
        x = torch.randn(*shape, requires_grad=True)
        torch.library.opcheck(torch.ops.practice.silu_and_mul.default, (x,))
    x, res, w = torch.randn(4, 16), torch.randn(4, 16), torch.randn(16)
    torch.library.opcheck(torch.ops.practice.fused_add_rms_norm.default, (x, res, w, 1e-5))
    schema = str(torch.ops.practice.fused_add_rms_norm.default._schema)
    check("x" in schema and schema.count("!") == 2, True, f"schema 要声明原地修改 x 和 residual：{schema}")


def test_meta_and_grad():
    torch = _ops()
    m = torch.ops.practice.silu_and_mul(torch.empty(7, 3, 2048, device="meta"))
    check((tuple(m.shape), m.device.type), ((7, 3, 1024), "meta"), "meta 设备上只推导形状")
    x = torch.randn(5, 10, dtype=torch.float64, requires_grad=True)
    check(torch.autograd.gradcheck(torch.ops.practice.silu_and_mul, (x,)), True, "gradcheck")


def test_compile():
    torch = _ops()

    def block(h, residual, w_up, w_down, norm_w):
        torch.ops.practice.fused_add_rms_norm(h, residual, norm_w, 1e-6)
        return torch.ops.practice.silu_and_mul(h @ w_up) @ w_down

    torch.manual_seed(2)
    args = (torch.randn(6, 32), torch.randn(6, 32), torch.randn(32, 128) / 6, torch.randn(64, 32) / 8, torch.rand(32) + 0.5)
    eager_args = [a.clone() for a in args]
    ref = block(*eager_args)
    compiled = torch.compile(block, fullgraph=True, backend="aot_eager")
    comp_args = [a.clone() for a in args]
    out = compiled(*comp_args)
    check_close(out, ref, rtol=1e-5, atol=1e-5, what="torch.compile(fullgraph=True) 的结果与 eager 一致")
    check_close(comp_args[0], eager_args[0], rtol=1e-5, atol=1e-5, what="编译后原地修改仍然生效（h）")
    check_close(comp_args[1], eager_args[1], rtol=1e-5, atol=1e-5, what="编译后原地修改仍然生效（residual）")
