import torch


class SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)          # 只保存输入，中间结果反向时重算
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        sig = torch.sigmoid(gate)
        silu = gate * sig
        d_gate = grad_out * up * sig * (1 + gate * (1 - sig))
        d_up = grad_out * silu
        return d_gate, d_up


torch.manual_seed(0)
g = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
u = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
print("gradcheck：", torch.autograd.gradcheck(SiluMul.apply, (g, u)))

y = SiluMul.apply(g, u)
ref = torch.nn.functional.silu(g) * u
print("前向与参考实现一致：", torch.allclose(y, ref))
print("反向节点：", type(y.grad_fn).__name__)
