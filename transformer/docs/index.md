# 手写 Transformer

<p class="lead">从一个 <code>main.py</code> 开始，一章一个文件，把 Qwen 最新一代的架构<strong>手写出来</strong>：零中心 RMSNorm、SwiGLU、只转 1/4 维的 RoPE、带输出门的 GQA 注意力、Gated DeltaNet 线性注意力（递推版和分块版）、KV cache 与记忆表、采样、聊天模板、批处理、MTP 投机解码、MoE。每一步都能跑，前九章每换掉一块零件就和 Hugging Face 的实现对一次 logits，误差 1e-5 量级；之后的每一章都用"带状态 == 不带状态""批 == 单条""手写模板 == 官方模板"这样的方式验证。以 <strong>Qwen3.5-0.8B</strong> 为例——它是 3.5 / 3.6 / 3.8 这一代里唯一能在 CPU 和 Mac 上舒服跑起来的，而写出来的代码对 27B 和 MoE 版同样成立。</p>

## 这本书适合谁

- 读过《大模型原理》知道注意力是什么，但没有从权重文件开始把一个真模型写出来过；
- 想弄清这一代开源模型和 Llama 式架构的区别：线性注意力、门控、QK-Norm、M-RoPE、MTP 到底是什么；
- 准备做推理系统，想亲手量一量 KV cache、记忆表、批处理、投机解码各自省的是什么。

读完并跟着敲完，你应该能：

- 对着 config.json 和 safetensors 的目录，说出每个张量是什么、形状为什么是那样；
- 不看参考实现写出 Qwen3.5 的前向，并且知道怎么验证它对不对；
- 讲清楚 Gated DeltaNet 的 delta 规则、为什么能分块并行、状态为什么不随长度增长；
- 把一个"只会算 logits"的模型变成一个能流式对话、能拼批、能投机解码的小引擎。

## 怎么展开

<div class="roadmap" markdown>

| 阶段 | 章 | 做什么 | 怎么验证 |
| --- | --- | --- | --- |
| [第 0 步](main.md) | main.py | HF 当黑盒，自己写解码循环；打开权重文件画全书地图 | 和 `model.generate` 逐 token 一致 |
| 阶段一 · 拆黑盒 | [1](weights.md) [2](embedding.md) [3](norm.md) [4](mlp.md) [5](rope.md) [6](attention.md) [7](deltanet.md) [8](chunk.md) [9](model.md) | 一章一个文件，把 HF 模型里的零件逐个换成自己写的 | 每换一块对一次 logits；第 9 章起不再 import transformers |
| 阶段二 · 从模型到引擎 | [10](cache.md) [11](sample.md) [12](chat.md) [13](batch.md) [14](mtp.md) | KV cache 与记忆表、采样、聊天模板与思考模式、批处理、MTP 投机解码 | 带状态 == 不带状态；手写模板 == 官方模板；批 == 单条；投机 == 贪心 |
| 阶段三 · 更快、更大 | [15](speed.md) [16](scale.md) | 线程、分块、bf16、设备；全系配置的参数账，手写 MoE | 真实计时；MoE 和 transformers 用随机权重对 |

</div>

每一章的代码都是接力的：第 N 章的脚本用到前面所有章写好的文件。整个包最后 17 个文件、约 800 行（连注释），文件树见[第 0 步](main.md#后面怎么走)。

## 怎么跑

目录结构、环境配置和每一章改哪些文件，见[项目结构与环境搭建](setup.md)；最短的版本是三条命令——Python 3.10 以上、PyTorch（CPU 版即可）、`safetensors`、`tokenizers`，前九章和第 12 章对答案时还要 `transformers`（5.x，含 Qwen3.5 的实现）。权重 1.7 GB：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install safetensors tokenizers "transformers>=5.0"
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B      # 国内可先 export HF_ENDPOINT=https://hf-mirror.com
```

代码都在仓库的 `transformer/examples/` 下（`qwen/` 包 + 每章一个 `chNN_*.py`），在 `transformer/` 目录下运行：`python examples/main.py`、`python examples/ch01_weights.py` ……。脚本会从当前目录往上找 `models/Qwen3.5-0.8B`，也可以用环境变量 `QWEN_MODEL` 指定。

- **内存**：fp32 加载一份模型 3 GB；前九章同时加载 HF 的参考实现，约 6 GB。
- **CPU**：脚本把线程数设为逻辑核的一半（第 15 章解释为什么）；一台 32 核的机器上 decode 约 18 tok/s，Apple Silicon 的 Mac 用 CPU 也能跑。
- **GPU**：所有代码都能 `.to("cuda")` / `.to("mps")`，第 15 章的脚本会自动用。

## 接下来往哪走

| 想继续的方向 | 去哪本 |
| --- | --- |
| 把这个小引擎变成真正的推理服务：调度、Radix cache、CUDA Graph、张量并行 | [手写 mini-sglang](minisgl://) |
| 这些算子在 GPU 上怎么写快 | [CUDA 进阶手册](cuda://) |
| 注意力、RoPE、MoE 背后的原理与变体 | [大模型原理](llm://) |
| 从零训练一个这样的模型 | [从零训练一个小模型](scratch://) |
| 推理服务的工程全貌：vLLM / SGLang 的架构与调度 | [大模型推理服务](serving://) |

各本手册的逐章路线见[学习路线图](root://roadmap/)，求职冲刺的安排见[冲刺计划](root://plan/)。

## 怎么验证的

- 每一章的脚本都在 CPU 上真实运行，页面上的输出与运行结果逐行一致（`transformer/tools/check_code.py` 按导航顺序接力重跑全书）；
- 前九章每换掉一块零件都和 Hugging Face transformers 5.17 的 `Qwen3_5ForCausalLM` 对 logits，最大差在 1e-5 量级；分块版线性注意力和递推版逐位对；手写聊天模板和官方 Jinja 模板逐 token 对；MoE 层和 transformers 用随机权重对到 0。
- 带时间的数字（tok/s、ms）是一台 32 核 Xeon 上的真实计时，换机器会不同。
