# 大作业四：给 mini-sglang 接入 Qwen3.5（混合架构）的文本部分

在[手写 mini-sglang](https://anrans.github.io/ai-infra-handbooks/minisgl/) 的基础上，让它能服务 Qwen3.5-0.8B：24 层里 18 层是 Gated DeltaNet（线性注意力），6 层是门控全注意力。这是一次完整的"新模型接入"：读 config 和参考实现、改配置解析和权重加载、写新的层，再把**每个请求的状态**接进调度器和内存管理——推理引擎为 KV 设计的机制在这里都要重新想一遍。只给接口和检查脚本，不给骨架；参考实现不公开。

先读推理系统手册的[新模型接入与精度对齐](https://anrans.github.io/ai-infra-handbooks/serving/ops/new-model/)和[线性注意力与混合架构](https://anrans.github.io/ai-infra-handbooks/serving/frontier/linear-attn/)两章。只需要 CPU。

## 要做什么

| 部分 | 要求 | 检查 |
| --- | --- | --- |
| 配置 | `ModelConfig.from_hf` 读 `text_config`；新增 `layer_types`、`linear_num_key_heads`、`linear_num_value_heads`、`linear_key_head_dim`、`linear_value_head_dim`、`linear_conv_kernel_dim` 字段，`full_attention_layers`、`linear_attention_layers`、`is_hybrid` 三个属性；`rotary_config.rotary_dim` 按 `partial_rotary_factor` 计算。纯注意力模型的行为不变 | `check_config.py` |
| 权重 | 架构名 `Qwen3_5ForConditionalGeneration` 注册到模型表；加载器去掉 `model.language_model.` 前缀，跳过视觉编码器（`model.visual.*`）和 MTP 层（`mtp.*`）；加载出的键与模型的 `state_dict` 完全一致 | `check_weights.py` |
| 计算 | 门控全注意力层、Gated DeltaNet 层；prefill 和 decode 都与 transformers 的实现数值一致 | `check_generate.py` |
| 状态 | 每个请求一份卷积缓存和递推状态；新请求从零状态开始，分块 prefill 的段与段之间传递状态，请求槽被复用时不能读到上一个请求的状态 | `check_generate.py` |
| 内存 | KV 池只为 6 个全注意力层分配；状态池的大小算进显存规划 | 报告 |
| 前缀缓存 | radix 缓存要么在创建时明确拒绝（`ValueError` / `NotImplementedError`），要么正确处理状态（见"延伸"） | `check_prefix_cache.py` |

`check_generate.py` 用离线接口 `LLM(model, dtype=torch.float32, cache_type="naive", device="cpu", ...)` 在 CPU 上贪心解码 24 个 token，与 Hugging Face transformers 的 `generate` 逐 token 比较，覆盖四种情况：单个请求、5 个长短不一的请求一起跑、分块 prefill（每步最多 16 个 token）、最多 2 个并发请求（请求槽被反复复用）。

## 步骤

```bash
# 模型：仓库根目录的 models/Qwen3.5-0.8B（env/setup-macos.sh 会下载），或者设置 A4_MODEL
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir models/Qwen3.5-0.8B
pip install torch "transformers>=5.2" safetensors        # transformers 用来算标准答案
MINISGL=/path/to/your/mini-sglang/python python run_checks.py
```

没有设置 `MINISGL` 时检查的是本仓库的 `minisgl/python`（没有接入 Qwen3.5），四项检查应当都失败。

## 提示：容易漏掉的细节

参考实现是 transformers 的 `models/qwen3_5/modeling_qwen3_5.py`（没有安装 fla 库时，它的 prefill 走分块的纯 PyTorch 实现、decode 走逐 token 的递推）。按[新模型接入](https://anrans.github.io/ai-infra-handbooks/serving/ops/new-model/)一章的方法逐层对齐，下面这些地方最容易出错：

<details><summary>展开</summary>

- **两种 RMSNorm**：层前后的归一化、最后的归一化、`q_norm`、`k_norm` 都是 `x * (1 + w)`（权重初始化为 0），而 Gated DeltaNet 输出处的门控归一化是普通的 `w * x`，再乘 `silu(z)`。可以在加载时把 1 折进权重，这样就能复用现有的 RMSNorm；
- **q_proj 的输出里夹着门**：每个头输出 `[query | gate]` 两段（先按头 reshape 成 `2 × head_dim`，再对半切），注意力的输出乘 `sigmoid(gate)` 之后才进 `o_proj`；
- **只转一部分维度**：`partial_rotary_factor = 0.25`，每个头 256 维里只有前 64 维做 RoPE（NeoX 风格，前后两半配对），base 是 1e7。多模态的交错 M-RoPE 在纯文本时三个位置分量相同，退化成普通的 RoPE；
- **卷积**：`in_proj_qkv` 的输出先过一个窗口为 4 的逐通道因果卷积和 SiLU，再切成 q、k、v。decode 时需要前 3 个 token 的输入，这就是"卷积缓存"；
- **Gated DeltaNet 的递推**（每个头，状态 $S$ 是 $d_k \times d_v$，用 fp32）：q、k 先做 L2 归一化，q 再乘 $d_k^{-1/2}$；$\beta = \sigma(b)$，$g = -e^{A_{\log}} \cdot \mathrm{softplus}(a + \text{dt\_bias})$；每个 token：$S \leftarrow e^{g} S$，$\delta = \beta\,(v - S^\top k)$，$S \leftarrow S + k\,\delta^\top$，输出 $o = S^\top q$；
- **状态放在哪里**：按请求的 `table_idx` 索引一个状态池（`[线性层数, 请求槽数, …]`），请求的 `cached_len == 0` 时清零；dummy 请求也占一个槽；
- **权重的 dtype**：`A_log` 和门控归一化的权重在 checkpoint 里是 fp32，其余是 bf16。

</details>

## 报告（交付物的一部分）

1. **内存账本**：每个请求的状态（递推状态 + 卷积缓存）多少字节、每个 token 的 KV 多少字节；在一张 24 GB 的卡上用 bf16 部署时，平均上下文 1K、8K、32K 各能同时服务多少个请求，和"24 层都是全注意力"的假想模型比较；
2. **decode 的开销**：CPU 上测 decode 一步的时间随上下文长度的变化（512～8K），和 Qwen3-0.6B 比较，解释差别；
3. **分块算法**：参考实现的 prefill 可以逐 token 递推；把它换成[分块写法](https://anrans.github.io/ai-infra-handbooks/serving/frontier/linear-attn/#同一个计算的两种写法)（块内矩阵乘、块间传状态），比较 1K 提示词的 prefill 时间。

## 延伸

- **带状态的前缀缓存**：请求结束时，在提示词结尾存一份状态检查点，挂在 radix 树的节点上；新请求只能在"存过检查点"的位置命中（`check_prefix_cache.py` 的第二种通过方式）；
- **MTP 投机解码**：checkpoint 里自带 1 层 MTP（`mtp.*`），用它做草稿；被拒绝的草稿 token 要回滚线性层的状态；
- **GPU**：用 flash-linear-attention 的 kernel 替换递推循环，给状态池加上 CUDA Graph 需要的固定地址。
