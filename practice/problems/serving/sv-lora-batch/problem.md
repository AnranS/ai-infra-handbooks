---
title: 多 LoRA：分段计算与按适配器组 batch
chapter: ops/multi-lora.md
difficulty: 困难
tags: [LoRA, SGMV, 调度, 缓存]
requires: [numpy]
---
**一、分段计算。** `lora_delta(x, adapter_ids, A, B, scale)` 算出一个混合批次的 LoRA 增量（基座部分 $xW$ 不用管）：

- `x`：`(N, d_in)`；`adapter_ids`：长度 `N`，每个 token 用哪个适配器，`-1` 表示不用 LoRA（增量为 0）；
- `A[i]`：`(d_in, r_i)`，`B[i]`：`(r_i, d_out)`，**每个适配器的秩可以不同**；`scale[i]` 是它的缩放系数；
- token `t` 的增量是 `scale[a] * (x[t] @ A[a]) @ B[a]`（`a = adapter_ids[t]`）。

要求按 SGMV 的方式算：把用同一个适配器的 token 排到一起（稳定排序），每一段做一次矩阵乘，再按原来的顺序放回。

**二、组 batch。** `schedule(requests, max_loras, gpu_slots, max_batch)` 模拟 decode 调度。`requests` 是 `(rid, adapter, arrival, steps)` 的列表：请求在第 `arrival` 步到达、需要跑 `steps` 步，`adapter = -1` 表示只用基座。从第 0 步开始，每一步：

1. **准入**：等待队列按 `(arrival, rid)` 排序，依次检查：`running` 已有 `max_batch` 个请求就停止扫描；否则，适配器是 `-1`、或者已经被 `running` 里的请求使用、或者 `running` 使用的不同适配器还不到 `max_loras` 个时，这个请求加入 `running`（立刻计入，影响后面的判断），否则**跳过它、继续看后面的请求**；
2. **显存槽位**：`running` 用到的每个适配器（按编号从小到大）如果不在显存里就加载（加载次数 +1）；槽位（`gpu_slots` 个，保证不少于 `max_loras`）已满时，先换出一个**当前没被 `running` 使用、最久没用过**的适配器（"用过"指某一步的 batch 里有请求用它；都没用过或同样久时换出编号小的）；
3. **执行**：记录这一步的 batch（`running` 的 rid 从小到大），它用到的适配器的"最近使用"更新为这一步；每个请求剩余步数减一，减到 0 的离开。

所有请求都结束时停止。返回 `(batches, loads)`：`batches[t]` 是第 `t` 步的 batch（没有请求可跑的步是空列表），`loads` 是加载适配器的总次数。

```python
reqs = [(0, 1, 0, 2), (1, 2, 0, 1), (2, 3, 0, 1), (3, 1, 0, 1)]
schedule(reqs, max_loras=2, gpu_slots=2, max_batch=4)
# ([[0, 1, 3], [0, 2]], 3)：第 0 步请求 2 的适配器 3 超出上限，被跳过；请求 3 用的适配器 1 已经在跑，可以加入
```

<!-- 题解 -->
分段计算的要点是**稳定**排序和"放回原位"：`order = argsort(ids, kind="stable")`，每段 `xs[s:e] @ A[a] @ B[a]` 写进 `out[order[s:e]]`。每个适配器按自己的秩算，不用补齐到最大秩——真实的 kernel 往往按最大秩分配（见本章练习 2），这就是秩差别大时要分组调度的原因。

调度的几条规则都来自真实系统：`max_loras` 是一个 batch 里的适配器上限（vLLM 的 `--max-loras`、SGLang 的 `--max-loras-per-batch`）；"跳过而不是堵住"避免了队头阻塞，但被跳过的请求可能一直等下去，实际系统要加上等待时间的优先级；显存槽位按 LRU 换出、并且绝不换出正在用的适配器，加载次数就是 PCIe 上的换入次数。把同一个适配器的请求路由到同一个实例（路由亲和），能直接减少这里的 `loads`。
