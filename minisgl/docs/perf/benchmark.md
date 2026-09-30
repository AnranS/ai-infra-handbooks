# 基准测试与消融

<p class="lead">所有功能都写完了，最后要回答"每个优化到底值多少"。这一章给出两个压测工具——离线吞吐测试和在线的流式压测客户端——以及一组消融实验的做法：关掉某一项优化，看吞吐和延迟怎么变。CPU 上我们跑了两个趋势明显的消融；需要 GPU 的那些给出命令和要观察的指标。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 离线吞吐测试和在线压测分别回答什么问题？
    2. TTFT、TPOT 各受哪些因素影响？哪项优化主要改善 TTFT、哪项主要改善 TPOT？
    3. 为什么 CPU 上"批大小从 1 到 16"的吞吐提升远小于 GPU？
    4. 想量化重叠调度的收益，应该用大模型还是小模型？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 离线吞吐测试：一次性把所有请求都给引擎，衡量满负载下每秒能生成多少 token（引擎本身有多快）；在线压测：按某个请求速率发送请求，衡量这个负载下的 TTFT、TPOT 和尾延迟（能不能满足 SLO）。
    2. TTFT：排队时间、prefill 的计算量（提示词长度、前缀命中率）、调度策略；TPOT：每一步 decode 的时间（batch 大小、KV 长度、kernel 效率、CPU 开销）。前缀缓存、分块 prefill 主要改善 TTFT；CUDA Graph、重叠调度、更好的注意力 kernel 主要改善 TPOT。
    3. CPU 的算力和带宽之比与 GPU 不同：CPU 上 batch 为 1 时计算就已经占了相当大的比例，不像 GPU 那样几乎全在读权重，所以增大 batch 带来的"免费"吞吐提升小得多。
    4. 小模型：重叠调度藏的是固定的 CPU 开销，模型越小、每一步的 GPU 时间越短，这部分开销占比越大，收益越明显；大模型上 GPU 计算时间长，CPU 开销本来就被淹没了。

**本章要写的文件**：`benchmark/offline.py`、`benchmark/client.py`。

## 离线吞吐

@@code python/minisgl/benchmark/offline.py:run@@

与官方的 `benchmark/offline/bench.py`（源自 nano-vllm）相同的做法：随机生成若干请求，输入、输出长度各自在区间内均匀随机，`ignore_eos=True` 保证每个请求都生成满 `max_tokens`，统计输出 token 的吞吐。它衡量的是**满负载下引擎能跑多快**，不关心单个请求的延迟。

```bash
python -m minisgl.benchmark.offline --model Qwen/Qwen3-0.6B --num-seqs 256 \
    --max-input-len 1024 --max-output-len 1024 --page-size 256 --cuda-graph-max-bs 256
```

## 在线压测

@@code python/minisgl/benchmark/client.py:one_request@@

在线压测模拟真实的用户：请求按泊松过程到达（相邻请求的间隔服从指数分布），流式接收，记录每个请求的 TTFT（发出到收到第一个片段）、TPOT（之后每个片段的平均间隔）和端到端延迟，报告均值、中位数和 P99。它回答的是**在某个请求速率下，用户体验如何**。

```bash
python -m minisgl --model Qwen/Qwen3-0.6B &
python -m minisgl.benchmark.client --num-requests 256 --rate 8 --max-tokens 256
```

逐步提高 `--rate`，画出"请求速率 - P99 TTFT / P99 TPOT"曲线，找到满足延迟目标（SLO）的最大速率，就是这台机器的容量（方法见[推理系统手册的压测与容量规划一章](serving://perf/benchmark/)）。

## CPU 上的两个消融

@@code examples/ch21_benchmark.py@@

@@output ch21_benchmark@@

**连续批处理**：同样 16 个请求，一次只允许 1 个请求运行时，吞吐只有约 21 token/s；允许 16 个同时运行时约为两倍。在 GPU 上这个差距会大得多：decode 是访存受限的，批大小从 1 到 16，每步的时间几乎不变（读一遍权重的时间占主导），吞吐接近线性增长；而 CPU 的算力相对带宽弱得多，批大小一增加，计算就成了瓶颈，收益被压缩。

**前缀缓存**：16 个请求共享 400 个 token 的前缀时，naive 缓存要把前缀算 16 遍（6656 个 token）；Radix Cache 只算第一遍，之后每个请求只算自己的 16 个 token 加上最后一个前缀 token（656 个），总用时从约 18 秒降到约 4 秒。前缀缓存的收益与"共享前缀的长度 × 请求数"成正比，在多轮对话、Agent、少样本提示等场景里非常可观。

## GPU 上的消融清单

| 消融 | 命令 | 主要看什么 |
| --- | --- | --- |
| 重叠调度 | `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` | 离线吞吐；小模型、大批量时差距最明显 |
| CUDA Graph | `--cuda-graph-max-bs 0` | TPOT；批大小越小差距越大 |
| 前缀缓存 | `--cache-type naive` | 共享前缀负载下的 TTFT 与吞吐 |
| 注意力后端 | `--attn fi` / `--attn fa` / `--attn fa,fi` | prefill 吞吐（FA3 在 Hopper 上强）、decode TPOT（FlashInfer 强） |
| 分块大小 | `--max-prefill-length 2048` / `16384` | 长提示词下的 TTFT 与 decode 的 TPOT 尖峰 |
| page size | `--page-size 1` / `16` / `256` | 前缀命中率（页越大越粗）与管理开销 |

做消融时每次只改一项，其余保持不变；每个配置跑三次取中位数。想知道时间具体花在哪，用 Nsight Systems 抓一段时间线：CPU 调度与 GPU kernel 之间的空隙，就是重叠调度和 CUDA Graph 要消灭的东西（方法见[推理系统手册的 Profiling 一章](serving://perf/profiling/)）。

!!! upstream "官方实现"
    - 离线：[`benchmark/offline/bench.py`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/benchmark/offline/bench.py)（256 个请求、输入输出各 100～1024，`page_size=256`、`cuda_graph_max_bs=256`）
    - 在线：[`benchmark/online/bench_qwen.py`](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/benchmark/online/bench_qwen.py)，回放阿里云百炼公开的 Qwen 请求轨迹；客户端 @@upstream benchmark/client.py@@ 约 500 行，支持更多统计
    - 官方 README 给出了 H200 上的离线吞吐与在线延迟结果（在线测试与 SGLang 对比）

## 测试

@@code tests/test_ch21_benchmark.py:test_offline_benchmark_runs@@

!!! interview "面试怎么答"
    基准测试题：离线吞吐测满负载下引擎每秒能处理多少 token，在线压测测某个请求速率下的 TTFT、TPOT 和尾延迟（P99），两者回答的问题不同。TTFT 受排队、prefill 计算量（提示词长度、前缀命中）影响，分块 prefill、前缀缓存、PD 分离主要改善它；TPOT 受 decode 一步的时间影响，CUDA Graph、重叠调度、量化、投机解码主要改善它。做消融每次只改一项；重叠调度和 CUDA Graph 的收益在小模型、小 batch 时最大（CPU 开销占比高），要用小模型来量化它们。CPU 上批大小从 1 到 16 吞吐提升有限，因为 CPU 的算力早已跑满，不像 GPU 那样受访存限制。

## 练习

1. 用 `benchmark/client.py` 在 CPU 上压一个 `--max-running-requests 4` 的服务，把 `--rate` 从 0.5 逐步提高到 4，观察 TTFT 的变化，解释拐点出现的位置。
2. 设计一个"多轮对话"负载：每个用户连续发 5 轮，每轮的提示词包含之前所有轮次。比较 radix 与 naive 在这个负载下的 TTFT。
3. 在 GPU 上复现官方的离线测试，分别关闭重叠调度和 CUDA Graph，记录吞吐。哪一项对 Qwen3-0.6B 影响更大？对 Qwen3-14B 呢？为什么？

??? success "参考答案"
    1. 当到达速率超过服务能力（`max_running_req` 个并发请求能提供的吞吐）时，请求开始在等待队列里堆积，TTFT 从"prefill 时间"跳升为"排队时间 + prefill 时间"，并随时间持续增长——这是排队论里利用率接近 1 时延迟爆炸的现象。
    2. 多轮对话下，每轮的提示词前缀就是上一轮的完整对话，Radix Cache 能命中几乎全部前缀，TTFT 只取决于新增部分；naive 每轮都从头 prefill，TTFT 随轮次线性增长。
    3. 小模型每步 GPU 时间很短，CPU 开销和 kernel 发射开销占比大，两项优化的收益都明显；大模型每步 GPU 时间长，开销占比小，收益变小。具体哪项更大取决于批大小：批越小，kernel 发射开销（CUDA Graph 消除的）占比越大。

## 小结

- [x] 离线吞吐衡量满负载下的引擎速度，在线压测衡量某个请求速率下的 TTFT、TPOT 与尾延迟。
- [x] CPU 上可以观察到连续批处理和前缀缓存的收益；GPU 上前者更明显。
- [x] 消融每次只改一项；重叠调度与 CUDA Graph 的收益在小模型、小批量时最大。
