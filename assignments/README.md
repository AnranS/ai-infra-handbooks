# 大作业

参考 CS336 和 CMU DLSys 的做法：只给接口、测试和评分标准，不给骨架。每个大作业都对应站点上的一章说明。

| 大作业 | 内容 | 验证 | 需要 |
| --- | --- | --- | --- |
| [一：从零训练一个小语言模型](a1-lm/README.md) | BPE、Transformer、AdamW、训练循环 | 12 个单元测试 + 验证集损失达标线 | CPU |
| [二：训练系统](a2-trainsys/README.md) | 分桶 DDP、ZeRO-1、激活重计算，以及组合 | 4 项多进程检查 + 显存与吞吐报告 | CPU（gloo） |
| [三：推理引擎的 GPU 性能门槛](a3-engine-gpu/README.md) | 手写 mini-sglang 在 GPU 上跑到 SGLang 的 60% | 正确性 + 吞吐 / 延迟门槛 + 消融 | NVIDIA GPU |

参考实现不公开：测试和评分脚本就是标准。
