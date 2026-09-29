# 大作业：接入混合架构模型 Qwen3.5

<p class="lead">这本书的 mini-sglang 只认识"每一层都是注意力"的模型：每个请求的全部历史都在 KV 池里，前缀缓存、分块 prefill、请求槽的复用都建立在这个前提上。Qwen3.5 打破了它——24 层里 18 层是 Gated DeltaNet，每个请求带着一份固定大小、会被原地更新的状态。这个大作业要求你把它接进来：输出与 Hugging Face 逐 token 一致，并且弄清楚引擎里哪些机制因此要重做。</p>

说明和检查脚本在仓库的 [`assignments/a4-hybrid-model/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/assignments/a4-hybrid-model)，只需要 CPU。动手之前先读推理系统手册的[新模型接入与精度对齐](serving://ops/new-model/)和[线性注意力与混合架构](serving://frontier/linear-attn/)两章。

## 要做什么

| 部分 | 要求 |
| --- | --- |
| 配置 | `ModelConfig` 读出每层的类型和线性注意力层的形状；RoPE 只转每个头的前 1/4 维 |
| 权重 | 从多模态 checkpoint 里只取语言部分，跳过视觉编码器和 MTP 层，键与模型完全对上 |
| 计算 | 门控全注意力层、Gated DeltaNet 层，prefill 和 decode 都与参考实现一致 |
| 状态 | 每个请求一份卷积缓存和递推状态，按请求槽存放；新请求从零开始，分块 prefill 的段与段之间传递状态 |
| 内存 | KV 池只为 6 个全注意力层分配，状态池的大小算进显存规划 |
| 前缀缓存 | radix 缓存要么明确拒绝，要么在存过状态检查点的位置才命中 |

四项检查：配置解析、权重加载、贪心解码与 Hugging Face 逐 token 一致（单个请求、变长批处理、分块 prefill、请求槽复用）、前缀缓存的处理。

## 为什么值得做

- **这是接新模型的真实工作量**：读 config 和参考实现、找出"和已有模型不一样"的每一处（两种 RMSNorm、夹在 q_proj 里的门、只转一部分维度的 RoPE），再逐层对齐；
- **它逼你想清楚"状态"**：KV 可以按块共享、按位置截断，状态只能整份保存、整份拷贝。请求槽被复用时状态必须清零，分块 prefill 时状态必须接力，前缀缓存和投机解码都要为它重新设计——这些正是 vLLM、SGLang 支持 Qwen3-Next、Qwen3.5、Kimi 等混合模型时做过的事；
- **报告里的数字能直接用在面试里**：每个请求的状态多大、短请求时混合模型为什么反而更占显存、decode 的开销为什么几乎不随上下文增长。

## 交付物

通过全部检查的代码、一页报告（内存账本、decode 开销随上下文的变化、分块算法与逐 token 递推的 prefill 时间对比），以及一段说明：你的引擎如果要在 GPU 上服务这个模型，还缺哪些东西。
