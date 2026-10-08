# PyTorch 速成

<p class="lead">八章把 PyTorch 用熟：张量与形状、索引与掩码、autograd、nn.Module、数据与训练循环、保存与复现、调试与提速。这本书只讲<strong>怎么用</strong>——每一节都是一段几秒钟就能跑完的代码加上它真实的输出，读完你能看懂也能写出主流框架里的训练代码。想知道这些东西<strong>怎么实现</strong>，去 CUDA 手册的「框架与编译器」几章；想把它们用起来完整训练一个模型，去《从零训练一个小模型》。</p>

## 这本书适合谁

- 会 Python，看得懂神经网络在算什么，但 PyTorch 只停留在"抄得动别人的训练脚本"；
- 读源码时经常卡在形状、`gather`、`detach`、`state_dict` 这类地方；
- 想把"形状对不上""梯度是 None""显存莫名其妙涨"这类问题自己定位掉。

读完并跟着敲完，你应该能：

- 一眼看出一段张量操作的形状怎么变，知道什么时候是视图、什么时候复制了数据；
- 用 `gather` / `scatter_` / `masked_fill` / `einsum` 写出注意力和损失函数里的那些操作；
- 讲清 `no_grad`、`detach`、`eval()` 各自管什么，为什么它们不能互相替代；
- 写出一个带评估、调度和裁剪的训练循环，并且中断之后能原样续上；
- 读懂报错，用 profiler 找到最慢的算子。

## 八章的路线

<div class="roadmap" markdown>

| 章节 | 做什么 | 学完能做什么 | 建议用时 |
| --- | --- | --- | --- |
| [（一）张量](tensor.md) | 形状、dtype、设备，以及哪些操作共享内存 | 不再被 dtype 和设备的报错卡住 | 1 小时 |
| [（二）形状的功夫](shape.md) | view / reshape / permute、广播、einsum | 看一眼就知道形状怎么变，能写多头注意力的下标 | 2 小时 |
| [（三）索引、归约与掩码](indexing.md) | 视图与拷贝、gather / scatter_、因果掩码、dim 与 keepdim | 写得出交叉熵和注意力里的取数操作 | 2 小时 |
| [（四）autograd 怎么用](autograd.md) | backward、梯度累加、no_grad 与 detach | 讲清梯度为什么是 None，知道哪些原地操作会炸 | 2 小时 |
| [（五）nn.Module](module.md) | 参数与 buffer、state_dict、train / eval | 写出结构清晰、存得下也加载得回来的模型 | 2 小时 |
| [（六）数据与训练循环](training.md) | Dataset、DataLoader、collate_fn，一个完整的训练 | 独立写出一个能跑的训练脚本 | 3 小时 |
| [（七）保存、加载与复现](checkpoint.md) | checkpoint 存什么、种子与确定性 | 中断之后续训的结果和没中断一样 | 1 小时 |
| [（八）调试与提速](debug.md) | 读懂报错、经典的坑、inference_mode、autocast、profiler | 自己定位形状、梯度和性能问题 | 2 小时 |

</div>

八章彼此独立，哪一章卡住就单独看哪一章；代码都在 CPU 上几秒钟跑完。

## 接下来往哪走

| 想继续的方向 | 去哪本 |
| --- | --- |
| 把这些用起来，完整训练一个会写《三国演义》的小模型 | [从零训练一个小模型](scratch://) |
| 张量在内存里到底长什么样、autograd 怎么建图、torch.compile 做了什么 | [CUDA 进阶手册的「框架与编译器」](cuda://framework/tensor/) |
| Transformer 的每个部件为什么这样设计 | [大模型原理](llm://) |
| 一张卡放不下的时候：DDP、ZeRO、张量并行 | [分布式训练手册](train://) |
| 写得地道一点的 Python | [Python 进阶](python://) |

各本手册的逐章路线见[学习路线图](root://roadmap/)，求职冲刺的安排见[冲刺计划](root://plan/)。

## 怎么验证的

- 每一页的代码都在 **CPU 版 PyTorch** 上实际运行，页面上的输出与运行结果逐行一致；
- 报错信息也是真跑出来的——这本书里出现的每一条 `RuntimeError` 都可以自己复现。

```bash
# 需要 CPU 版 PyTorch（2.4 以上）与 numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```
