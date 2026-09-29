# 大作业二：训练系统——DDP、ZeRO-1 与激活重计算

只给接口和检查脚本。你要自己实现三个组件，让它们组合在一起训练时与单进程的结果逐元素一致，再用它们训练一个模型、把显存和吞吐与估算对比。

## 要做什么

| 文件 | 内容 | 检查 |
| --- | --- | --- |
| `trainsys/ddp.py` | 分桶、与反向重叠的数据并行（`BucketedDDP`） | 与单进程一致；反向过程中发起了异步 all-reduce |
| `trainsys/zero1.py` | 切分优化器状态的 AdamW（`ZeRO1`） | 与单进程 AdamW 一致；每个 rank 的状态约为 1 / world |
| `trainsys/recompute.py` | 激活重计算（`checkpoint`，不能用 `torch.utils.checkpoint`） | 梯度一致；为反向保存的字节数至少减半 |
| 组合 | 三者一起训练一个带残差块的网络 | 与单进程一致 |

## 步骤

```bash
pip install torch                            # CPU 版即可，多进程用 gloo 后端
python run_checks.py                         # 4 项检查全部通过即完成主体部分
```

每个文件开头写明了检查的内容。检查脚本在 `checks/` 下，可以单独运行，例如 `torchrun --standalone --nproc-per-node 2 checks/check_ddp.py`。

## 报告（交付物的一部分）

用这三个组件在 2～4 个进程上训练[大作业一](../a1-lm/README.md)的模型（或者一个更大的版本），写一页报告：

1. **显存账本**：参数、梯度、优化器状态、激活各占多少，和[分布式训练手册](https://anrans.github.io/ai-infra-handbooks/train/basics/overview/)的公式对比（CPU 上可以统计张量的字节数，GPU 上用 `torch.cuda.max_memory_allocated`）；
2. **吞吐**：DDP 的桶大小从 16 KB 到 16 MB，每步时间怎样变化，为什么；
3. **重计算的代价**：打开重计算后，每步多花了多少时间、省下了多少显存，和"多一次前向约 33%"的估算是否一致。

## 延伸

- 把 ZeRO-1 改成 ZeRO-2（梯度也切分，用 reduce-scatter 代替 all-reduce），检查通信量；
- 让 DDP 的桶直接分配在一块连续的缓冲区里，省掉 `torch.cat` 的拷贝；
- 在 GPU 上用 NCCL 跑同样的检查，并用 profiler 看通信与反向是否真的重叠。
