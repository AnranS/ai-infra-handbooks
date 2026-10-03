---
title: temperature、top-k、top-p 采样
chapter: probability.md
difficulty: 中等
tags: [采样, 概率分布, 逆变换采样]
---
实现一个确定性的采样函数 `sample(logits, u, temperature=1.0, top_k=-1, top_p=1.0)`：

- `logits`：一维 numpy 数组（词表上的打分）；`u` 是 $[0, 1)$ 里的一个随机数，由调用方给出，这样结果可以复现。
- 处理顺序：
  1. `temperature == 0` 时直接返回 `argmax`（贪心，忽略其他参数；并列时取下标最小的）；
  2. `logits / temperature` 后做 softmax 得到概率 $p$；
  3. `top_k > 0` 时只保留概率最大的 `top_k` 个（并列时保留下标小的）；
  4. `top_p < 1` 时，把剩下的 token 按概率**从大到小**排序（并列按下标从小到大），保留累计概率**第一次达到或超过** `top_p` 的最短前缀；
  5. 把保留下来的概率重新归一化；
- **逆变换采样**：按词表下标**从小到大**累加保留下来的概率，返回第一个累计和 $> u$ 的下标。

```python
logits = np.log(np.array([0.1, 0.2, 0.3, 0.4]))
sample(logits, u=0.05)                 # 0   （累计：0.1 0.3 0.6 1.0）
sample(logits, u=0.05, top_k=2)        # 2   （只剩下标 2、3，归一化后 3/7、4/7）
sample(logits, u=0.99, top_p=0.5)      # 3   （按概率排序：0.4 0.3 … 累计 0.4 < 0.5，0.7 ≥ 0.5，保留 {3, 2}）
sample(logits, u=0.5, temperature=0)   # 3
```

<!-- 题解 -->
top-k 和 top-p 都可以用"被过滤掉的位置概率置 0"来实现，最后统一归一化、累加。

- top-k：`order = np.lexsort((np.arange(n), -p))`（先按概率降序、再按下标升序），`p[order[k:]] = 0`；
- top-p：用同样的顺序求累计和 `c`，保留到第一个 `c >= top_p` 的位置（`np.searchsorted(c, top_p) + 1` 个），其余置 0；
- 逆变换采样：`np.searchsorted(np.cumsum(p), u, side="right")`，再防一下浮点误差导致的越界。
