# 大作业三：推理引擎的 GPU 性能门槛

沿用[手写 mini-sglang](https://anrans.github.io/ai-infra-handbooks/minisgl/)的实现（先自己写，写完再对照），在一张 GPU 上跑通真实模型，并达到下面的门槛。这一项需要 NVIDIA GPU，没有测试脚本以外的骨架。

## 门槛

| 项目 | 要求 |
| --- | --- |
| 正确性 | 贪心解码的输出与 Hugging Face transformers 逐 token 一致（mini-sglang 的测试集全部通过） |
| 吞吐 | 同一张卡、同一个模型（例如 Qwen3-0.6B 或 Qwen2.5-7B）、同一组请求，离线吞吐达到 SGLang 的 60% 以上 |
| 延迟 | 并发 1 时的 TPOT 不超过 SGLang 的 1.5 倍（CUDA Graph 生效） |
| 消融 | 分别关掉重叠调度、CUDA Graph、Radix Cache、分块 prefill，报告各自的贡献 |

## 怎么测

- 请求集：用 SGLang 自带的 `python -m sglang.bench_serving` 生成的随机数据集或 ShareGPT 数据集，固定随机种子，两边用完全相同的请求；
- 离线吞吐：一次性提交全部请求，统计总的输出 token / 总时间；
- 手写 mini-sglang 的[基准测试与消融](https://anrans.github.io/ai-infra-handbooks/minisgl/perf/benchmark/)一章有完整的测量脚本和方法，可以直接复用。

## 交付物

一个与官方 mini-sglang 同结构的仓库、一份性能报告（硬件、模型、请求集、每一项的数字和消融表），以及一篇讲清楚"差距在哪里"的博客——用 profiler 的时间线说明剩下的差距来自哪里（kernel、调度开销、CPU 瓶颈等）。
