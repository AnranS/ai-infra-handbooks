# 大作业：DDP、ZeRO-1 与激活重计算

<p class="lead">这本书的每一章都给出了一种并行的最小实现。大作业要求你合上书，只看接口，自己写出分桶的 DDP、ZeRO-1 和激活重计算，并让它们组合在一起训练时与单进程的结果逐元素一致；然后用它们训练一个模型，把显存和吞吐的实测与第一章的账本对比。</p>

代码与说明在仓库的 [`assignments/a2-trainsys/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a2-trainsys)。

## 要做什么

| 文件 | 内容 | 对应章节 | 检查（CPU 上多进程，gloo） |
| --- | --- | --- | --- |
| `trainsys/ddp.py` | `BucketedDDP`：构造时广播参数、按反向顺序分桶、桶满即异步 all-reduce | [DDP](../data/ddp.md) | 与单进程一致；反向过程中确实发起了异步 all-reduce |
| `trainsys/zero1.py` | `ZeRO1`：平铺参数、均分优化器状态、更新后 all-gather | [ZeRO 与 FSDP](../data/zero-fsdp.md) | 与单进程 AdamW 一致；每个 rank 的状态约为 1 / world |
| `trainsys/recompute.py` | `checkpoint(fn, *args)`：自定义 autograd 函数，反向时重算 | [混合精度](mixed-precision.md#省激活的其他手段) | 梯度一致；为反向保存的字节数至少减半 |
| 组合 | 三者一起训练带残差块的网络 | — | 与单进程一致 |

检查脚本不关心你的内部实现：DDP 的"重叠"是通过拦截 `dist.all_reduce`、看它是否在 `backward()` 返回之前以 `async_op=True` 被调用来判断的；重计算的效果是用 `torch.autograd.graph.saved_tensors_hooks` 统计前向保存了多少字节来判断的。

```bash
cd assignments/a2-trainsys
python run_checks.py                 # 4 项检查全部通过
```

## 报告

用这三个组件在 2～4 个进程上训练[大模型手册大作业](llm://training/assignment/)的模型，写一页报告：

1. **显存账本**：参数、梯度、优化器状态、激活各占多少，和[总论](../basics/overview.md)的公式对比；
2. **桶大小**：从 16 KB 到 16 MB，每步时间怎样变化，为什么（参考 [DDP](../data/ddp.md) 一章的时间线模型）；
3. **重计算的代价**：多花的时间和省下的显存，与"多一次前向"的估算是否一致。

## 延伸

- 改成 ZeRO-2：梯度也切分，用 reduce-scatter 代替 all-reduce，验证通信量不变；
- 让桶直接分配在连续的缓冲区里，省掉拼接的拷贝；
- 到 GPU 上换成 NCCL，用 profiler 确认通信真的和反向重叠了。
