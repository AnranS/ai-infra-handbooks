# 大作业：从零训练一个小语言模型

<p class="lead">读完这本书，最好的检验是亲手从零训练一个语言模型：自己写分词器、模型、优化器和训练循环，在限定时间内把验证集的损失降到达标线以下。这个大作业参考 CS336 的做法，只给接口、测试和评分脚本，不给骨架——每一行代码都要你自己写，写不出来的地方就是需要回头重读的章节。</p>

代码与说明在仓库的 [`assignments/a1-lm/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a1-lm)。

## 要做什么

| 文件 | 内容 | 对应章节 | 检查 |
| --- | --- | --- | --- |
| `lm/bpe.py` | 字节级 BPE：训练、编码、解码，合并规则与平局的处理都写在文件开头 | [分词](../basics/tokenization.md) | 5 个测试 |
| `lm/model.py` | pre-norm Transformer：RMSNorm、带 RoPE 的因果多头注意力、SwiGLU | [Transformer 解剖](../transformer/build-llm.md) | 4 个测试（形状、参数量、因果性） |
| `lm/optim.py` | 自己实现的 AdamW（与 PyTorch 逐元素一致）、线性预热 + 余弦调度、梯度裁剪 | [预训练](pretraining.md) | 3 个测试 |
| `train.py`（自己新建） | 训练循环：分词、随机取窗口、前向反向、保存检查点 | — | 评分脚本 |

规则：可以用 PyTorch 的张量运算、autograd、`nn.Linear`、`nn.Embedding`；不能用 `torch.optim.AdamW`、`nn.MultiheadAttention`、`nn.Transformer*`、`clip_grad_norm_` 和现成的分词器库。

## 步骤与达标线

```bash
cd assignments/a1-lm
python data.py                       # 生成约 4 MB 的训练语料（确定性的"小故事"，有需要记住人名的长程依赖）
python -m pytest -q tests            # 12 个测试全部通过
python train.py                      # 自己写，保存 out/model.pt
python evaluate.py out/model.pt      # 验证集每字节交叉熵 ≤ 0.10 nats/byte
```

- **评分按"每字节"计算**：总损失除以验证集的字节数，与词表大小无关，不同的分词方案可以公平比较；
- **防作弊**：评分脚本先检查分词器能无损还原验证集、模型是因果的（改动后面的 token 不影响前面的输出）；
- **时间限制**：CPU 上 10 分钟以内。参考实现（词表 512、4 层、d_model 128、上下文 256、1500 步）用 16 个线程约 2.5 分钟，达到 0.089 nats/byte。

## 做完之后

- 测训练吞吐和模型 FLOPs 利用率，和[估算](../inference/estimation.md)一章的公式对比；
- 消融：去掉 RoPE、改成 post-norm、换词表大小，各自对损失和速度的影响；
- 把模型交给[分布式训练手册的大作业](train://practice/assignment/)做多进程训练，或者接到推理引擎上做推理。
