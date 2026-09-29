# 大作业：推理引擎的 GPU 性能门槛

<p class="lead">这本书带你在 CPU 上把 mini-sglang 的每个模块都实现并验证了一遍。大作业要求你把它搬到 GPU 上，跑真实的模型，并达到一组明确的性能门槛——这是"写过一个推理引擎"和"写过一个能用的推理引擎"之间的距离。</p>

说明在仓库的 [`assignments/a3-engine-gpu/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a3-engine-gpu)。需要一张 NVIDIA GPU。

## 门槛

| 项目 | 要求 |
| --- | --- |
| 正确性 | 贪心解码的输出与 Hugging Face transformers 逐 token 一致，本书的测试全部通过 |
| 吞吐 | 同一张卡、同一个模型、同一组请求，离线吞吐达到 SGLang 的 60% 以上 |
| 延迟 | 并发为 1 时的 TPOT 不超过 SGLang 的 1.5 倍（CUDA Graph 生效） |
| 消融 | 分别关掉重叠调度、CUDA Graph、Radix Cache、分块 prefill，报告各自的贡献 |

## 怎么测

- 两边用完全相同的请求：固定随机种子生成数据集（例如 `sglang.bench_serving` 的随机数据集或 ShareGPT），一次性提交全部请求，统计输出 token 数 / 总时间；
- 测量脚本和消融的方法直接复用[基准测试与消融](../perf/benchmark.md)一章；
- 达不到门槛时，用 profiler 的时间线找原因：GPU 上的 kernel 是否是瓶颈、CPU 调度是否跟不上（空隙）、CUDA Graph 是否命中，对照[与 SGLang 的差距](next-steps.md)一章逐项排查。

## 交付物

与官方同结构的代码仓库、一份性能报告（硬件、模型、请求集、每一项的数字和消融表）和一篇博客。这正是冲刺计划里"作品 A"的要求，写进简历时每个数字都要能解释来源。

另一个大作业不需要 GPU：[接入混合架构模型 Qwen3.5](assignment-hybrid.md)，练的是新模型接入和"每个请求带状态"的引擎改造。
