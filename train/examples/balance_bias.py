import torch

torch.manual_seed(0)
E, K, T = 16, 2, 4096
skew = torch.linspace(1.5, -1.5, E)                      # 路由器天生偏爱前面几个专家
bias = torch.zeros(E)                                    # 只用于"选哪几个专家"，不参与门控权重
u = 0.02                                                 # 偏置的更新步长


def route(scores):
    _, idx = (scores + bias).topk(K, dim=-1)             # 选择时加偏置
    gate = torch.softmax(scores, -1).gather(-1, idx)     # 门控权重仍然用原始分数
    return idx, gate


for step in range(301):
    scores = torch.randn(T, E) + skew
    idx, gate = route(scores)
    load = torch.bincount(idx.flatten(), minlength=E).float()
    if step % 100 == 0:
        f = load / (T * K)                               # 每个专家分到的 token 比例
        p = torch.softmax(scores, -1).mean(0)            # 每个专家的平均路由概率
        aux = E * (f * p).sum().item()                   # Switch Transformer 的负载均衡损失（均衡时为 1）
        print(f"step {step:>3}：最忙 / 最闲专家的负载比 {load.max() / load.min():6.2f}，辅助损失 {aux:.3f}")
    bias += u * torch.sign(load.mean() - load)           # 负载高于平均就调低偏置，反之调高
