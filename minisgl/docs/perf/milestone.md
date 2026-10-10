# 阶段四的里程碑：同一份代码，更多的卡、更快的 kernel

<p class="lead">前三个阶段做出来的引擎是完整的，但只会用一张卡、只会用 PyTorch 的算子。阶段四的六章让同一份代码在多卡上切开跑（张量并行）、在 GPU 上换成 FlashInfer / FlashAttention、把 decode 录成 CUDA Graph 回放、换上自定义 kernel、接入 MoE 模型，最后用一张基准表把每项优化的收益量出来。这一页在 CPU 上先把"对"验证掉：切成两个 rank，或者录成 graph 回放，输出一个 token 都不变。"快"的数字要上真卡。</p>

**这一阶段结束时你手里有什么**

- 同一个 `Engine`，`tp_size=2` 时每个 rank 只拿一半权重、一半 KV 头，前向中间用 all-reduce 合并，输出和单卡一致；
- GPU 上注意力自动切到 FlashInfer / FlashAttention，decode 自动走 CUDA Graph，KV 写入和索引用自定义 kernel；
- 一个 MoE 模型（Qwen3-MoE）也能跑，fused MoE 的 Triton kernel 在 GPU 上替换掉逐专家的循环；
- 第 21 章一张基准表：每项优化单独开关，吞吐和延迟各多少。

## 先跑起来

```bash
cd minisgl && python examples/stage4_milestone.py
```

@@code examples/stage4_milestone.py@@

@@output stage4_milestone@@

## 读这几行

**TP=2 时 `qkv_proj` 从 (4096, 1024) 变成 (2048, 1024)，KV 池每层的头数从 8 变成 4。** 每个 rank 只加载、只保存自己那一半——这就是张量并行解决的问题：模型放不进一张卡时，把每个线性层按行或按列切开，两张卡各算一半，中间用一次 all-reduce 把结果合起来。4 个 token 和 TP=1 完全一样：切法正确的话，数学上就是同一个矩阵乘。CPU 上两个进程走 gloo，不会更快；真卡上两张卡走 NCCL，权重读取的带宽翻倍，decode 就快了。

**CUDA Graph 仿真：3 个请求补齐成 4，replay 了 5 轮，输出一致。** CUDA Graph 要求每次回放的张量形状和地址都固定，所以 batch 要补齐到录制过的大小、所有输入都要先拷进固定的缓冲区。CPU 上的仿真只做一件事：强制这套"固定缓冲区"的纪律——漏拷任何一个输入，输出立刻出错（第 18 章故意演示了这一点）。真卡上这套纪律换来的是 decode 每步少掉几十次 kernel 启动的开销。

**CPU 上到此为止。** 这一阶段的其他三章——FlashInfer / FlashAttention、自定义 CUDA kernel、fused MoE——在 CPU 上都只有"同接口的替身"：假实现、CPU 模拟器、Triton 解释器。它们能验证逻辑和接口，不能给出速度。有卡的话，仓库根目录的 `gpu_check.py minisgl` 会把这几章在真卡上跑一遍；第 21 章的基准表是这一阶段真正的里程碑。

## 每一章加了什么

| 优化 | 解决什么 | CPU 上怎么验证 | 在哪一章 |
| --- | --- | --- | --- |
| 张量并行 | 模型放不进一张卡；多卡一起读权重 | gloo 上两个进程，输出与单卡一致 | [张量并行](tensor-parallel.md) |
| FlashInfer / FlashAttention | 不物化注意力分数矩阵；分页 KV 直接读 | 同接口的 PyTorch 假实现 | [GPU 注意力后端](gpu-attention.md) |
| CUDA Graph | decode 每步几十次 kernel 启动的 CPU 开销 | `EmulatedGraph`，强制固定缓冲区 | [CUDA Graph](cuda-graph.md) |
| 自定义 kernel | KV 写入、词表索引这类小操作的启动开销和访存 | CUDA 手册的 CPU 模拟器 | [自定义 CUDA kernel](kernels.md) |
| fused MoE | 逐专家循环变成一个分组的矩阵乘 | Triton 解释器模式 | [MoE 与 fused MoE](moe.md) |
| 基准测试 | 每项优化到底值多少 | 只能上真卡 | [基准测试与消融](benchmark.md) |

## 怎么读这六章

[张量并行](tensor-parallel.md)和 [CUDA Graph](cuda-graph.md) 在 CPU 上就能完整验证，先读；[GPU 注意力后端](gpu-attention.md)和[自定义 kernel](kernels.md) 读的时候对照 [CUDA 手册](cuda://)里对应的 kernel 章节；[MoE](moe.md) 可以放到最后。读完六章如果手边有一张卡，按[大作业](../wrap/assignment.md)的门槛把引擎跑到正式版的 60%——那是这本书真正的终点。

!!! abstract "验收：读完这一阶段"
    - [ ] 说出四种线性层切法各适用于哪一层，为什么 `o_proj` 和 `down_proj` 后面要 all-reduce；
    - [ ] 解释 CUDA Graph 为什么要补齐 batch、为什么漏拷一个输入不会报错只会算错；
    - [ ] 讲清 FlashInfer 的 plan / run 两步各做什么；
    - [ ] 在真卡上跑过 `gpu_check.py minisgl`，并能解释第 21 章基准表里每一行的差异来自哪里。

## 小结

- [x] 张量并行：每个 rank 一半权重、一半 KV 头，all-reduce 合并，输出与单卡一致。
- [x] CUDA Graph：固定形状、固定缓冲区，CPU 仿真只验证这套纪律。
- [x] 注意力后端、自定义 kernel、fused MoE 在 CPU 上只能验证接口与逻辑；速度要上真卡。
- [x] 第 21 章的基准表是这一阶段的真正里程碑。
