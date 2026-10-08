import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
T, d = 512, 64
x = torch.randn(T, 256)                                       # 某一层注意力的输入（已经过归一化）
Wq, Wk = torch.randn(256, d) / 16, torch.randn(256, d) / 16   # 初始化时 q、k 的每个分量约为 N(0, 1)


def attention_stats(scale, qk_norm):
    q, k = x @ (Wq * scale), x @ (Wk * scale)                  # 训练中 W_q、W_k 的范数一起变大
    if qk_norm:                                               # QK-Norm：点积之前，q、k 各做一次 RMSNorm
        q, k = F.rms_norm(q, (d,)), F.rms_norm(k, (d,))
    logits = (q @ k.T / math.sqrt(d)).tril() + torch.full((T, T), float("-inf")).triu(1)
    p = logits.softmax(-1)
    entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)[T // 2:].mean()
    return logits[torch.isfinite(logits)].abs().max().item(), entropy.item(), p.max(-1).values[T // 2:].mean().item()


print(f"均匀分布的熵：{math.log(T // 2):.2f}～{math.log(T):.2f}（后一半 query 各自能看到 256～512 个 key）")
for qk_norm in (False, True):
    for scale in (1, 2, 4, 8):
        m, h, top = attention_stats(scale, qk_norm)
        print(f"{'QK-Norm ' if qk_norm else '普通     '} 权重放大 {scale} 倍：最大 logit {m:6.1f}，平均熵 {h:4.2f}，最大注意力权重 {top:.2f}")
