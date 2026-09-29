# 分布式训练手册

<p class="lead">推理岗位同样要懂训练侧的并行：面试会问 ZeRO 和 Megatron，RL 训练把推理引擎和训练框架绑在一起，推理用的张量并行、专家并行、上下文并行也都源自训练。这本手册从"一张卡放不下什么"讲起，把数据并行、ZeRO、张量并行、流水线并行、上下文并行和专家并行逐个从零实现，并在 CPU 上用多进程与单进程逐项对齐。</p>

## 这份手册适合谁

- 懂 Transformer 的结构、会用 PyTorch 写训练循环，但没有做过多卡训练；
- 或者用过 DeepSpeed、Megatron-LM、FSDP，但说不清它们每一步在通信什么、显存省在哪里；
- 做推理系统，需要理解训练侧的并行（RL 训练、权重同步、推理与训练共用的 TP / EP 实现）。

读完并练完这份手册，你应该能做到：

- 对任意模型和集群配置算出每张卡的显存账本（参数、梯度、优化器状态、激活）和每一步的通信量；
- 讲清 DP、ZeRO-1/2/3、TP、SP、PP、CP、EP 各自切的是什么、通信什么、代价是什么；
- 从零写出 DDP（分桶 + 通信重叠）、ZeRO-1、张量并行的 MLP（含反向）、Ulysses 注意力、专家并行的 MoE 层，并验证与单卡一致；
- 为一个具体的模型和集群选出合理的并行组合，并解释理由。

## 学习路线

<div class="roadmap" markdown>

| 阶段 | 章节 | 学完能做什么 | 建议用时 |
| --- | --- | --- | --- |
| 一、基础 | [总论：显存账本与时间模型](basics/overview.md) · [集合通信原语](basics/collectives.md) | 算清显存和通信，知道瓶颈在哪 | 2～3 天 |
| 二、数据并行 | [DDP](data/ddp.md) · [ZeRO 与 FSDP](data/zero-fsdp.md) | 写出分桶重叠的 DDP 和 ZeRO-1 | 2～3 天 |
| 三、模型并行 | [张量并行与序列并行](model/tensor-sequence.md) · [流水线并行](model/pipeline.md) · [上下文并行](model/context.md) · [MoE 与专家并行](model/moe-ep.md) | 每种并行都能讲清切法和通信，并写出最小实现 | 1 周 |
| 四、精度与策略 | [混合精度与 FP8](practice/mixed-precision.md) · [3D / 5D 并行的组合](practice/strategy.md) · [训练框架与 RL 系统](practice/frameworks-rl.md) | 为具体场景选配置，读懂 Megatron / DeepSpeed / verl | 3～4 天 |

</div>

七本手册的逐章路线见[学习路线图](root://roadmap/)，求职冲刺的安排见[冲刺计划](root://plan/)；本书对应计划的第 9 周（与推理系统手册的[分布式推理](serving://distributed/tensor-parallel/)一起学）。

## 怎么验证的

- 所有标了文件名的脚本都在 **CPU 版 PyTorch** 上实际运行，页面上的输出与运行结果逐行一致；
- 多进程的脚本用 `torchrun --standalone --nproc-per-node N` 启动，通信后端是 **gloo**（CPU）；换到 GPU 上只要把后端改成 `nccl`、把张量放到 `cuda` 上；
- 每种并行都和单进程的计算逐项对齐：前向输出、损失，以及**梯度**。

```bash
# 需要 CPU 版 PyTorch（2.4 以上）
pip install torch --index-url https://download.pytorch.org/whl/cpu
torchrun --standalone --nproc-per-node 4 collectives.py
```
