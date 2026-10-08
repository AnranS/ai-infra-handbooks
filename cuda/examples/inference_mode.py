import torch

x = torch.randn(4, requires_grad=True)
with torch.no_grad():
    a = x * 2
with torch.inference_mode():
    b = x * 2
print("no_grad：requires_grad", a.requires_grad, "是 inference 张量", a.is_inference())
print("inference_mode：requires_grad", b.requires_grad, "是 inference 张量", b.is_inference())
try:
    (b * x).sum().backward()
except RuntimeError as e:
    print("把 inference 张量用在需要求导的计算里：", str(e).split(".")[0])
