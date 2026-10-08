import torch


def step(x, w):
    h = x @ w
    if h.sum().item() > 0:      # 依赖数据的 Python 控制流
        h = h * 2
    return torch.relu(h)


exp = torch._dynamo.explain(step)(torch.randn(4, 16), torch.randn(16, 16))
print("捕获了", exp.graph_count, "张图，断开", exp.graph_break_count, "次")
for r in exp.break_reasons:
    print("原因：", str(r.reason).splitlines()[0])
