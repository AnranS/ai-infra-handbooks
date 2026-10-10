# 阶段二的里程碑：调度器接管

<p class="lead">阶段一结束时，batch 的字段还要自己填：哪些请求进这一步、它们的 KV 放在哪几页、位置从几开始。阶段二的五章把这几行变成一个调度器——连续批处理、准入控制、页的分配与回收、Radix Cache、分块 prefill、重叠调度——对外只剩一个离线接口 <code>LLM.generate</code>。这一页先把阶段结束时的样子跑给你看：三组数字，每组对应骨架做不到的一件事。</p>

**这一阶段结束时你手里有什么**

- `LLM.generate(prompts)`：一次吞下任意多个请求，长短不一也行，先完成的先退出 batch，后来的随时补进来；
- 同一个系统提示词只算一遍 KV，后面的请求直接复用；
- 长提示词被切成固定大小的块，别的请求可以插在块之间；
- CPU 上准备下一步的时间藏在计算后面（重叠调度，本页先关掉它，数字只反映调度器本身）。

## 先跑起来

```bash
cd minisgl && python examples/stage2_milestone.py
```

@@code examples/stage2_milestone.py@@

@@output stage2_milestone@@

## 读这几个数字

**8 个请求 1.23 秒，104.5 tokens/s。** 第 0 步 25.6，阶段一手工组 4 个请求的 batch 是 70。两件事叠在一起：batch 更大了（8 个请求一步算完，decode 读一次权重出 8 个 token），而且不用自己填字段了——`generate` 里的每一步都是调度器在做阶段一 `step()` 里那几行的事：准入、分配页、算 `positions` 和 `out_loc`、收回结束请求的页。真实服务里请求是陆续到的、长短不一的，连续批处理的意义是**batch 的成员随时在变**而计算从不停下。

**同一个系统提示词：prefill 从 231 个 token 降到 73 个，后 3 个请求 409 ms → 244 ms。** 四个问题共享一段 158 个 token 的系统提示词。naive 缓存下每个请求都把它重算一遍；Radix Cache 在第一个请求算完后把前缀留在树里，后面三个只算各自的问题部分。这就是多轮对话和共享系统提示词场景下吞吐差几倍的来源——少算的那 158 × 3 个 token，一个也没有白省。

**302 个 token 的提示词分了 4 步：96、96、96、14。** 没有分块 prefill，一个长提示词进来会独占引擎一整步，正在 decode 的所有请求都得等它；分块之后每步最多算 `max_extend_tokens` 个，decode 请求插在中间，首 token 延迟稍长一点，换来的是其他请求的每 token 延迟不抖。

## 手工的那几行变成了什么

| 阶段一 `step()` 里你手工做的 | 阶段二 | 在哪一章 |
| --- | --- | --- |
| 决定哪些请求进这一步的 batch | 准入控制、连续批处理的主循环 | [调度器骨架与离线接口](scheduler.md) |
| `page_table[row, :256] = arange(...)`——手工"分配"KV 位置 | `CacheManager`：按需分配、结束时回收、满了就等 | [CacheManager](cache-manager.md) |
| `cached_len=0`——每个请求从头算 | Radix Cache：匹配前缀、复用 KV、引用计数、LRU 淘汰 | [Radix Cache](radix-cache.md) |
| 整段提示词一步 prefill | 分块 prefill：每步最多 `max_extend_tokens` 个 token | [分块 prefill](chunked-prefill.md) |
| 算完这一步再准备下一步 | 重叠调度：准备下一步时 GPU 还在算这一步 | [重叠调度](overlap.md) |

## 怎么读这五章

[调度器骨架](scheduler.md)先读——它定义主循环，后面四章都是往这个循环里加东西。读到 [Radix Cache](radix-cache.md) 时把本页脚本的 `cache_type` 来回切几次，看 `prefill 实际计算` 那个数怎么变；读到[分块 prefill](chunked-prefill.md) 时改 `max_extend_tokens`，看步数和每步的 token 数。最后一章[重叠调度](overlap.md)读完，把本页第一行的 `ENV.DISABLE_OVERLAP_SCHEDULING.value = True` 删掉重跑：CPU 上这项收益不大（没有 GPU 在另一头并行算），但在真卡上它是吞吐的最后一截。

!!! abstract "验收：做到这些再进阶段三"
    - [ ] 画出一个请求从进入队列到释放 KV 页的状态变化，标出每一步调度器做的决定；
    - [ ] 解释 Radix Cache 的树为什么要分裂节点，引用计数防的是什么；
    - [ ] 说出分块 prefill 牺牲了什么、换来了什么；
    - [ ] 把本页脚本的 `max_running_req` 改成 2，解释吞吐为什么掉、掉到哪。

## 小结

- [x] 阶段二把阶段一手工填 batch 的那几行变成调度器，对外只剩 `LLM.generate`。
- [x] 吞吐从 70 到 104.5 tokens/s：batch 更大、成员随时变、计算不停。
- [x] Radix Cache 让共享前缀只算一遍：231 个 prefill token 降到 73 个。
- [x] 分块 prefill 让长提示词不再独占引擎：302 个 token 分 4 步，别的请求插在中间。
