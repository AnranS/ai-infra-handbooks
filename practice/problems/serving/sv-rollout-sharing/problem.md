---
title: RL rollout 的前缀共享与并发
chapter: topics/rl-rollout.md
difficulty: 简单
tags: [RL, rollout, 前缀共享]
---
GRPO 等算法对每个提示词采样 $n$ 个回答（一组）。同一组的 $n$ 个请求提示词完全相同：推理引擎只需要算一次提示词的 KV，其余 $n-1$ 个共享（前缀缓存或 fork）。实现：

1. `prefill_tokens(prompt_lens, n, share)`：所有组的 prefill 计算量（token 数）。`share=True` 时每组的提示词只算一次；
2. `kv_tokens_per_group(prompt_len, n, max_new, share, block_size)`：一组请求在最坏情况下（每个回答都生成 `max_new` 个 token）占用的 KV（以 token 计，按块向上取整）：
   - 不共享：每个请求 `ceil((prompt_len + max_new) / block_size) * block_size`；
   - 共享：提示词里**装满的块**只存一份（`prompt_len // block_size` 个块），每个请求各自的部分从提示词不满的那一块开始（写时复制）：`ceil((prompt_len % block_size + max_new) / block_size) * block_size`；
3. `max_concurrent_groups(prompt_lens, n, max_new, kv_budget, share, block_size)`：按给定顺序依次接纳组，直到下一组放不进 `kv_budget`（以 token 计）为止，返回能同时运行的组数。

<!-- 题解 -->
共享时一组的 KV = `(prompt_len // bs) * bs + n * ceil((prompt_len % bs + max_new) / bs) * bs`。
提示词很长（长的系统提示词、few-shot 示例、多轮 agent 轨迹）而 $n$ 较大（8、16）时，共享能把 KV 占用降低好几倍，同时运行的组数随之增加，rollout 吞吐大幅提升——这是 RL 训练框架普遍用 SGLang / vLLM 做 rollout 的原因之一。
