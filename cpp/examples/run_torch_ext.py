import torch
from torch.utils.cpp_extension import load

ext = load(name="rmsnorm_ext", sources=["rmsnorm_ext.cpp"], extra_cflags=["-O2"], verbose=False)


def reference(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


torch.manual_seed(0)
x = torch.randn(4, 7, 4096)
w = torch.rand(4096)
torch.testing.assert_close(ext.rmsnorm(x, w, 1e-6), reference(x, w), rtol=1e-5, atol=1e-5)
print("pybind11 接口与参考实现一致")
torch.testing.assert_close(torch.ops.demo.rmsnorm(x, w, 1e-6), reference(x, w), rtol=1e-5, atol=1e-5)
print("torch.ops.demo.rmsnorm 与参考实现一致")
xt = x.transpose(0, 1)   # 非连续的输入
torch.testing.assert_close(ext.rmsnorm(xt, w, 1e-6), reference(xt, w), rtol=1e-5, atol=1e-5)
print("非连续的输入也正确")
try:
    ext.rmsnorm(x.double(), w, 1e-6)
except RuntimeError as e:
    print("类型检查：", str(e).splitlines()[0])
