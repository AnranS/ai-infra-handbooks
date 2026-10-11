---
title: 采样前的分数处理：惩罚、温度、top-k、top-p
chapter: sample.md
difficulty: 简单
tags: [采样, top-k, top-p, 重复惩罚]
requires: [torch]
---
写 `filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, presence_penalty=0.0, generated=())`：输入 `[V]` 的分数，返回处理后的 `[V]` 分数（被砍掉的候选是 `-inf`），顺序必须是：

1. 对 `generated` 里出现过的 token：先做重复惩罚（正分除以 `repetition_penalty`、负分乘以它），再减去 `presence_penalty`；
2. 除以温度（`temperature > 0`）；
3. `top_k > 0` 时只留分数最高的 k 个（和第 k 名同分的也留）；
4. `top_p < 1` 时按概率从高到低累加，累计**超过** p 之后的候选去掉（刚好跨过线的那个留下）。

不要在函数里采样，只返回分数；调用方自己 `softmax` + `multinomial`。

```python
logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
filter_logits(logits, top_k=2)          # tensor([3., 2., -inf, -inf])
filter_logits(logits, top_p=0.5)        # 第一个的概率 0.64 已经超过 0.5：只留它
```

<!-- 题解 -->
```python
def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, presence_penalty=0.0, generated=()):
    logits = logits.float().clone()
    if generated:
        seen = torch.tensor(sorted(set(generated)))
        s = logits[seen]
        s = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)
        logits[seen] = s - presence_penalty
    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    return logits
```

top-p 的判断用 `cumsum - probs > p`：减掉自己之后的累计才是"前面的候选已经占了多少"，这样刚好跨过线的那个候选会留下，否则 p 很小时可能一个都不剩。重复惩罚是乘法，对负分 logit 的效果是"更负"，所以 Qwen 的模型卡推荐用减法的出现惩罚。温度要在惩罚之后、截断之前：截断看的是温度之后的分布。
