# Engine 与采样器

<p class="lead"><code>Engine</code> 是一个 TP rank 上的计算引擎：它按固定的顺序初始化模型、KV 池、page table、注意力后端、采样器和 CUDA Graph，然后对外只提供一个方法 <code>forward_batch</code>——给一个准备好的 batch，返回每个请求的下一个 token。这一章写完它，并在没有调度器的情况下手工驱动它完成 prefill 和 decode，结果与 Hugging Face 逐 token 一致。"算得对"这一部分到此完成。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 引擎怎样决定 KV 池有多少页？为什么要在加载模型**之前**先量一次可用显存？
    2. `forward_batch` 返回的 token 为什么有 GPU 和 CPU 两份？
    3. 一个 batch 里有的请求贪心、有的请求 `temperature=0.7, top_p=0.9`，采样器怎么在一次调用里处理？
    4. top-k 和 top-p 同时设置时，先做哪个？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 页数 =（`memory_ratio` × 加载模型前的空闲显存 − 模型占用）÷ 每页的字节数，TP 下各 rank 取最小值。先量一次是为了算出模型本身占了多少：加载前后空闲显存的差就是模型占用。
    2. GPU 上的一份直接写回 token pool，作为下一轮的输入，不需要经过 CPU；CPU 上的一份（异步拷贝）给调度器判断是否结束、交给反分词。
    3. 把每个请求的参数排成张量，一次批量处理：全是贪心时直接 argmax；混合时，贪心的请求用一个极小的温度（效果等同于 argmax），和其他请求一起走采样的路径。
    4. 先 top-k 再 top-p。

**本章要写的文件**：`engine/config.py`、`engine/engine.py`、`engine/sample.py`、`utils/device.py`（设备抽象）。

## 配置

@@code python/minisgl/engine/config.py:EngineConfig@@

大部分字段对应命令行参数。`device` 和 `cpu_kv_cache_bytes` 是我们加的；`model_config` 是 `cached_property`，第一次访问时从 `config.json` 解析（第 3 章）。

## 初始化的顺序

@@code python/minisgl/engine/engine.py:Engine.__init__@@

顺序是有讲究的：

1. **TP 信息和设备**。之后构建的每一层都要读 TP 信息决定自己的切分。
2. **stream 和全局 Context**。在 CUDA 上，引擎的所有计算都在自己的 stream 上（第 11 章的重叠调度需要调度器和引擎各用一条 stream）。
3. **通信**。TP>1 时建立进程组（第 16 章）。
4. **量一次可用显存，然后加载模型**。加载前后的差值就是模型占用的显存。
5. **KV 池**。用剩下的显存算出页数（见下一节），多分配一页给 dummy 请求。
6. **page table**，最后一行给 dummy 请求，并让它指向 dummy 页（`fill_(num_tokens)`：dummy 页的第一个位置）。
7. **注意力后端和 MoE 后端**。注意力后端在构造时要读 KV 池（`get_global_ctx().kv_cache`），所以必须在 KV 池之后。
8. **采样器和 CUDA Graph**。CUDA Graph 在捕获时会真的跑模型，必须放在最后。

## 显存规划

@@code python/minisgl/engine/engine.py:Engine._determine_num_pages@@

每页的字节数是 `2（K、V）× head_dim × 本 rank 的 KV 头数 × page_size × dtype 字节数 × 层数`。可用显存的算法：

```text
可用 = memory_ratio × 加载模型前的空闲显存 − 模型占用
模型占用 = 加载前空闲 − 加载后空闲
```

`memory_ratio` 默认 0.9，留出 10% 给激活、CUDA Graph 和各种工作区。张量并行时，各 rank 的可用显存取最小值（`_sync_get_memory` 用一次 all-reduce），保证所有 rank 的 KV 池一样大——调度器在每个 rank 上做完全相同的决策，前提是资源完全相同。

CPU 上"空闲内存"是整台机器的，按比例算会得到几十 GB 的 KV 池，所以我们额外加了一个上限 `cpu_kv_cache_bytes`（默认 2 GiB）；测试里通常直接用 `num_page_override` 指定页数。

## forward_batch

@@code python/minisgl/engine/engine.py:Engine.forward_batch@@

- 在 `forward_batch` 上下文里执行模型（或 replay CUDA Graph），得到 `[batch_size, vocab]` 的 logits；
- **推进所有请求的状态**：`complete_one()`。注意这发生在采样之前、且在 CPU 上立即完成——调度器此后就可以把这些请求当作"已经多了一个 token"来调度下一轮，而不必等 GPU 算完（第 11 章）；
- 采样得到 `next_tokens_gpu`，再 `non_blocking` 地拷一份到 CPU，并记录一个 event。调用方需要 CPU 上的值时，先 `copy_done_event.synchronize()`。

两份 token 各有用途：GPU 上的那份由调度器写回 token pool，作为下一轮的输入，全程不离开 GPU；CPU 上的那份用来判断是否结束、发给 detokenizer。

## 设备抽象

官方在这里直接用 `torch.cuda.Stream()`、`torch.cuda.Event()` 和 `pin_memory=True`。我们把它们换成几个小函数：

@@code python/minisgl/utils/device.py:create_stream@@

@@code python/minisgl/utils/device.py:NullEvent@@

CPU 上所有计算都是同步完成的：`record` 的时候结果早已算好，`synchronize` 当然不需要等。锁页内存只在 CUDA 上有意义（它让主机到设备的拷贝可以异步），CPU 上 `pin(device)` 返回 `False`。

## 采样器

@@code python/minisgl/engine/sample.py:Sampler@@

采样分两步：

- `prepare(batch)` 在调度器一侧、前向**之前**调用，把每个请求的采样参数整理成张量并异步拷到设备上。整个 batch 都是贪心时返回 `temperatures=None`，`sample` 直接 `argmax`，省掉 softmax。
- `sample(logits, args)` 在前向之后调用。

一个 batch 里的请求可以有各自的参数。混合 batch 里的贪心请求被赋予一个极小的温度（1e-6），softmax 之后退化成 one-hot，采样结果就是 argmax——这样整个 batch 可以走同一条代码路径。top-k 值为 -1（不限制）时换成词表大小；只有 batch 里至少一个请求真的设置了 top-k（或 top-p），才生成对应的张量。

PyTorch 参考实现：

@@code python/minisgl/engine/sample.py:sample_torch@@

与 FlashInfer 的默认行为一样，先做 top-k、再在剩下的概率上做 top-p。top-p 保留"排在它前面的概率之和小于 p"的 token，所以概率最大的那个永远保留。GPU 上改用 FlashInfer 的采样 kernel（`sample_flashinfer`），它用拒绝采样避免排序，结果的分布相同。

## 手工驱动引擎

没有调度器时，我们自己扮演调度器：给每个请求"分配"一段 KV 位置（直接写 page table），每一轮填好 batch 的四个字段，调用 `forward_batch`，再把新 token 追加到请求上。

@@code examples/ch06_engine.py@@

@@output ch06_engine@@

第一轮是 prefill：两个请求共 9 个 token 拼成一维送进模型；之后每轮 decode 两个请求各 1 个 token。8 个 token 与 Hugging Face 的 `generate` 完全相同。下一部分的调度器要做的，就是把这段手工代码里的"分配 KV 位置""填 batch 字段""追加 token"自动化，并且让请求可以随时加入和离开。

!!! upstream "官方实现"
    - 初始化：@@upstream engine/engine.py:Engine.__init__@@
    - 显存规划：@@upstream engine/engine.py:Engine._determine_num_pages@@
    - 前向：@@upstream engine/engine.py:Engine.forward_batch@@
    - 采样：@@upstream engine/sample.py:Sampler@@（只有 FlashInfer 实现）

!!! diff "与官方的差异"
    - 设备：官方只支持 CUDA；我们通过 `utils/device.py` 同时支持 CPU。
    - 通信：官方即使 TP=1 也会初始化一个 gloo 进程组；我们只在 TP>1 时初始化，测试里可以在同一个进程中反复创建引擎（`shutdown` 会清掉全局状态）。
    - CPU 线程数：我们在 CPU 上把 PyTorch 的线程数设为物理核数除以 TP 数，否则 decode 的小矩阵乘会被线程同步拖慢几十倍（见[导读](../overview/architecture.md)）。

## 测试

`tests/test_ch06_engine.py` 有三个测试：prefill logits 与 HF 的最大误差小于 1e-4；手工循环的贪心输出与 HF 逐 token 相同；采样器的 top-k、top-p、混合贪心的行为和概率（top-k=2 时第一名被抽中的频率接近 e⁵ / (e⁵ + e⁴) ≈ 0.731）。

@@code tests/test_ch06_engine.py:test_sampler_greedy_and_filters@@

!!! interview "怎么讲清楚"
    讲引擎：初始化顺序是 TP 信息 → stream 与上下文 → 通信 → 加载模型 → KV 池 → page table → 注意力后端 → 采样器 → CUDA Graph。KV 池的大小 = 显存比例 × 加载模型前的空闲显存 − 模型占用，换算成页数，TP 下各 rank 取最小值（要在加载前先量一次，才知道模型占了多少）。`forward_batch` 前向、推进请求状态、采样，返回 GPU 上的 token（直接写回 token pool）和异步拷贝到 CPU 的一份（给调度器判断是否结束）。采样器一次处理参数各不相同的请求：全贪心时直接 argmax，混合时贪心请求用极小的温度，先 top-k 再 top-p。

## 练习

1. 在一张 24 GB 的卡上跑 Qwen3-0.6B（bf16），模型占约 1.2 GB，`memory_ratio=0.9`，KV 池大约有多少个 token？
2. 给采样器加上 `min_p`：保留概率不低于"最大概率 × min_p"的 token。它应该放在 top-k、top-p 之前还是之后？
3. 如果 `forward_batch` 在采样**之后**才调用 `complete_one()`，对正确性有影响吗？对第 11 章的重叠调度呢？

??? success "参考答案"
    1. 可用约 0.9 × 24 − 1.2 ≈ 20.4 GB，每 token 112 KiB，约 18 万个 token（实际还要扣掉 CUDA Graph 等占用）。
    2. min_p 的阈值取决于最大概率，通常作用在温度缩放后的完整分布上：可以放在 top-k、top-p 之前计算掩码，再与它们的掩码取交集。
    3. 对正确性没有影响：`complete_one` 只改 CPU 上的计数。对重叠调度也没有本质影响，因为调度器在 `forward_batch` 返回之后才组下一轮；关键是它不依赖采样结果的值，所以 CPU 不需要等 GPU。

## 小结

- [x] `Engine` 的初始化顺序：TP 信息 → stream 与上下文 → 通信 → 加载模型 → KV 池 → page table → 后端 → 采样器 → CUDA Graph。
- [x] KV 池的大小 = `memory_ratio × 加载前空闲显存 − 模型占用`，按每页字节数换算成页数；TP 下各 rank 取最小值。
- [x] `forward_batch` 前向、推进请求状态、采样，返回 GPU 上和（异步拷贝中的）CPU 上两份 token。
- [x] 采样器支持每个请求不同的参数：全贪心时 argmax；混合时贪心请求用极小温度；先 top-k 后 top-p。
