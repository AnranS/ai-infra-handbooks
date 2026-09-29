# 大作业一：从零训练一个小语言模型

只给接口、测试和评分脚本，不给骨架。你要自己写出分词器、模型、优化器和训练循环，在限定时间内把验证集的损失降到达标线以下。

## 要做什么

| 文件 | 内容 | 检查方式 |
| --- | --- | --- |
| `lm/bpe.py` | 字节级 BPE：训练、编码、解码（规则见文件开头的说明） | `tests/test_bpe.py` |
| `lm/model.py` | pre-norm Transformer：RMSNorm、带 RoPE 的因果多头注意力、SwiGLU | `tests/test_model.py` |
| `lm/optim.py` | 自己实现的 AdamW、学习率调度（线性预热 + 余弦）、梯度裁剪 | `tests/test_optim.py` |
| `train.py`（自己新建） | 训练循环：读数据、分词、随机取窗口、前向反向、保存检查点 | `evaluate.py` |

规则：可以用 PyTorch 的张量运算、autograd、`nn.Parameter`、`nn.Linear`、`nn.Embedding`；不能用 `torch.optim.AdamW`、`nn.MultiheadAttention`、`nn.Transformer*`、`torch.nn.utils.clip_grad_norm_`，也不能用现成的分词器库。注意力请先自己写（softmax + 因果掩码），跑通之后可以换成 `F.scaled_dot_product_attention` 对比速度。

## 步骤

```bash
pip install torch pytest                     # CPU 版即可
python data.py                               # 生成 data/train.txt（约 4 MB）和 data/val.txt
python -m pytest -q tests                    # 实现 lm/ 下的三个文件，直到 12 个测试全部通过
python train.py                              # 自己写：训练并保存 out/model.pt
python evaluate.py out/model.pt              # 验证集每字节交叉熵 ≤ 0.10 nats/byte 即达标
```

检查点的格式：`torch.save({"config": vars(cfg), "model": model.state_dict(), "merges": merges}, "out/model.pt")`。评分脚本会先检查分词器能无损还原验证集、模型是因果的（改动后面的 token 不影响前面的输出），再计算损失。

**时间限制**：在 CPU 上 10 分钟以内（参考实现用 16 个线程约 2.5 分钟，达到 0.089 nats/byte），或一张 GPU 上 2 分钟以内。

<details><summary>卡住了？参考的超参数</summary>

词表 512（在前 40 万个字符上训练 BPE 就够了）；4 层、d_model 128、4 个头、d_ff 384、上下文 256；batch 32；1500 步；AdamW 学习率 3e-3、betas (0.9, 0.95)、weight decay 0.1，预热 100 步后余弦降到 3e-4；梯度裁剪 1.0。

</details>

## 做完之后

- 算一下训练的吞吐（token/s）和模型 FLOPs 利用率（$6N$ 每 token 加上注意力），和[大模型手册](https://anrans.github.io/ai-infra-handbooks/llm/)里的估算对比；
- 消融：去掉 RoPE、改成 post-norm、词表 256 / 1024 / 4096，各自对损失和速度的影响；
- 用 `torch.compile` 或者换成 `F.scaled_dot_product_attention`，速度提升多少；
- 把训练好的模型接到[大作业三](../a3-engine-gpu/README.md)或手写 mini-sglang 的引擎上做推理。
