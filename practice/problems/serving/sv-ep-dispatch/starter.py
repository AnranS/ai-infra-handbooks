import numpy as np


def ep_moe(x, topk_ids, topk_w, experts, n):
    # 单机版本：没有模拟 dispatch / combine，也没有按专家分组
    outputs = []
    for r in range(n):
        out = np.zeros_like(x[r])
        for i in range(len(x[r])):
            for e, w in zip(topk_ids[r][i], topk_w[r][i]):
                out[i] += w * experts[e](x[r][i:i + 1])[0]
        outputs.append(out)
    return outputs, [[0] * n for _ in range(n)]
