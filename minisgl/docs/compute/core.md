# 核心数据结构：Req、Batch、Context

<p class="lead">整个系统里流动的只有几样东西：请求 <code>Req</code>、一批请求 <code>Batch</code>、以及保存"当前正在计算哪个 batch"的全局 <code>Context</code>。它们都定义在一个 136 行的文件里，却决定了后面每个模块的写法。这一章先把它们写出来，重点是 <code>Req</code> 上那几个长度字段——调度器的全部记账都建立在它们之上。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个提示词长 6、前 2 个 token 命中了前缀缓存、`max_tokens=3` 的请求，prefill 这一轮要算几个 token？prefill 结束后 `device_len` 是多少？
    2. 为什么 `Req` 用 `@dataclass(eq=False)`？
    3. `Batch` 的 `reqs` 和 `padded_reqs` 有什么区别？
    4. 模型的每一层怎样拿到当前 batch 的位置信息？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 算 4 个（`extend_len = device_len − cached_len = 6 − 2`）。这一轮的 `device_len` 是 6；前向结束后 `complete_one()` 把 `cached_len` 推进到 6、`device_len` 推进到 7，下一轮 decode 要算第 7 个位置。
    2. 让 `Req` 按对象身份比较和哈希：调度器要把请求放进 `set`，而两个不同的请求可能所有字段都相等（比如同时发来两个一样的提示词），按字段比较会被当成同一个。
    3. `reqs` 是真实的请求；`padded_reqs` 在 CUDA Graph 补齐 batch 大小时，还包括用来凑数的 dummy 请求。
    4. 从全局的 `Context` 里读当前 batch：`positions` 等张量由调度器在准备 batch 时填好，每一层按需取用。

**本章要写的文件**：`minisgl/core.py`。

@@tree@@

这一步的 main：`examples/ch01_req.py`——它只用到上面这些文件；`python tools/steps.py check` 会逐章搭出这棵树、跑这个 main。

## SamplingParams

@@code python/minisgl/core.py:SamplingParams@@

`is_greedy` 让采样器可以走快路径：整个 batch 都是贪心时直接 `argmax`，不用算 softmax。注意 `top_k == 1` 也算贪心，但 `top_p < 1` 不算——即使温度为 0，也要经过 top-p 的过滤逻辑（结果仍然确定，只是走的代码路径不同）。

## Req：请求的三个长度

一个请求在引擎里的状态，全部由三个长度描述：

@@diagram req-lengths 同一个请求在接纳时、prefill 后、第一轮 decode 后的三个长度@@

- `cached_len`：前多少个 token 的 KV **已经在缓存里**（算过了，或者命中了前缀缓存）；
- `device_len`：**本轮前向结束后**，缓存里会有多少个 token 的 KV；
- `max_device_len = 提示词长度 + max_tokens`：最多能长到多长。

由此派生出调度器需要的一切：

- `extend_len = device_len - cached_len`：本轮要送进模型的 token 数。prefill 时是没命中缓存的那部分提示词，decode 时是 1；
- `remain_len = max_device_len - device_len`：还能生成多少个；
- `can_decode = remain_len > 0`：能不能再进行一轮 decode。

每轮前向结束，`complete_one()` 推进状态：本轮的 token 都进了缓存（`cached_len = device_len`），下一轮要多算一个位置（`device_len += 1`）。采样出的新 token 由调度器用 `append_host` 追加到 CPU 上的 `input_ids`。

@@code python/minisgl/core.py:Req@@

跟踪一遍上面自测题里的请求：

@@code examples/ch01_req.py@@

@@output ch01_req@@

几个值得注意的细节：

- prefill 这一轮 `extend_len = 4`：命中的 2 个 token 不用再算。
- 第三个生成的 token（decode 2 采样出的那个）的 KV 永远不会被计算：`device_len = 9` 而 `cached_len = 8`。最后一个 token 只需要被输出，不需要作为输入。
- `input_ids` 在 CPU 上（`__post_init__` 里有断言）。GPU 上的输入另有一份，放在第 4 章的 `token_pool` 里；CPU 上这份用于前缀缓存的插入和判断结束。
- `@dataclass(eq=False)` 让 `Req` 按对象身份比较和哈希。调度器要把请求放进 `set`，而两个不同的请求可能所有字段都相等（比如同时发来两个一样的提示词）。

## Batch：谁来填哪个字段

@@code python/minisgl/core.py:Batch@@

`Batch` 只有 `reqs` 和 `phase` 两个构造参数，其余字段在调度的不同阶段由不同的模块填写：

| 字段 | 谁填 | 内容 |
| --- | --- | --- |
| `padded_reqs` | `GraphRunner.pad_batch`（第 18 章） | 为了使用 CUDA Graph，decode batch 会用 dummy 请求补齐到已录制的批大小；不用 graph 时等于 `reqs` |
| `positions` | 调度器 `_prepare_batch` | 本轮每个 token 的位置 `[cached_len, device_len)`，拼接成一维 |
| `out_loc` | 调度器 `_prepare_batch` | 本轮每个 token 的 KV 要写到 KV 池的哪个位置 |
| `input_ids` | 调度器 `_forward` | 本轮输入的 token，从 GPU 上的 `token_pool` 里按下标取 |
| `attn_metadata` | 注意力后端 `prepare_metadata` | 注意力 kernel 需要的变长、分页信息 |

所有张量都是把 batch 中各请求的 token **首尾相接拼成一维**（`[总 token 数]`），没有 padding。这是现代推理引擎的通用做法：prefill 和 decode 的请求长度差异很大，padding 会浪费大量计算。各请求的边界由注意力元数据里的 `cu_seqlens` 描述（第 5 章）。

## Context：全局的"当前 batch"

@@code python/minisgl/core.py:Context@@

`Context` 保存每个进程只有一份的东西：page table、注意力后端、MoE 后端、KV 池，以及"当前正在计算的 batch"。`forward_batch` 是一个上下文管理器，只在 `with` 块里设置当前 batch，退出时清空，并且不允许嵌套。

@@code python/minisgl/core.py:get_global_ctx@@

于是模型代码可以写成 `model.forward()`，不传任何参数。例如注意力层需要位置时读 `get_global_ctx().batch.positions`，LM head 需要知道是不是 prefill 时读 `batch.is_prefill`。好处是模型代码里没有层层传递的参数，所有层的签名都很干净；代价是隐式依赖：离开 `forward_batch` 上下文调用模型会直接断言失败。

!!! upstream "官方实现"
    @@upstream core.py@@ 整个文件。我们的版本字段与方法完全相同，只多了一个供测试使用的 `reset_global_ctx()`：官方的 `set_global_ctx` 不允许设置两次，测试里要在同一个进程中反复创建引擎，需要先清空。

## 测试

@@code tests/test_ch01_core.py:test_req_lengths_through_prefill_and_decode@@

`tests/test_ch01_core.py` 另外两个测试检查 `is_greedy` 的边界情况，以及 `forward_batch` 不能嵌套、退出后无法访问当前 batch。

!!! interview "怎么讲清楚"
    讲请求状态：mini-sglang 的 `Req` 只用三个长度描述状态——`cached_len`（已经在缓存里的）、`device_len`（本轮算完后在缓存里的）、`max_device_len`（上限），本轮要算的 `extend_len = device_len - cached_len`，prefill、分块 prefill、decode 都是它的特例；例如提示词 6、命中前缀 2，prefill 这一轮算 4 个 token。`Batch` 里的张量是各请求首尾相接的一维数组，不做 padding（CUDA Graph 用 `padded_reqs` 补齐批大小）；模型的每一层从全局 `Context` 取当前 batch 的位置和注意力元数据。`Req` 用 `eq=False`，按身份比较和哈希，才能放进集合、当字典的键。

## 练习

1. 一个请求提示词长 100、`max_tokens=50`，前 64 个 token 命中缓存。写出接纳时、prefill 后、第 10 轮 decode 后的 `cached_len`、`device_len`、`remain_len`。
2. 如果 `Req` 使用默认的 `eq=True`，调度器的 `running_reqs: Set[Req]` 会出什么问题？（提示：dataclass 的 `eq=True` 会让实例不可哈希。）
3. 为什么 `Req.__post_init__` 要求 `cached_len < device_len`，而不是 `<=`？什么情况下会违反它？

??? success "参考答案"
    1. 接纳时 `(64, 100, 50)`；prefill 后 `(100, 101, 49)`；第 10 轮 decode 后 `(110, 111, 39)`。
    2. `@dataclass` 在 `eq=True`（默认）且没有 `frozen=True` 时会把 `__hash__` 设为 `None`，`Req` 放不进 `set`；即使能哈希，两个内容相同的不同请求也会被当成同一个。`eq=False` 保留了 `object` 的按身份比较和哈希。
    3. 每一轮至少要算一个 token：即使整个提示词都在缓存里，也需要最后一个位置的隐状态来算 logits。所以调度器在匹配前缀时故意不匹配最后一个 token（第 8 章 `match_req` 里的 `input_len - 1`），保证 `cached_len < device_len`。

## 小结

- [x] `Req` 用三个长度描述状态：`cached_len`（已在缓存）、`device_len`（本轮后在缓存）、`max_device_len`（上限）；`extend_len`、`remain_len`、`can_decode` 都由它们派生。
- [x] `complete_one()` 在每轮前向后推进状态；新 token 由调度器 `append_host` 追加到 CPU 上的 `input_ids`。
- [x] `Batch` 里的张量都是各请求首尾相接的一维数组，不做 padding；各字段由调度器、注意力后端、GraphRunner 在不同阶段填写。
- [x] 模型从全局 `Context` 读取当前 batch，`forward()` 不需要参数。
