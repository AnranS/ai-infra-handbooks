# 算子层：BaseOP 与各种 Layer

<p class="lead">mini-sglang 没有用 <code>torch.nn.Module</code>，而是自己写了一个 99 行的 <code>BaseOP</code>。这一章先弄清为什么，再实现组成 decoder 的所有层：线性层、词嵌入与输出层、RMSNorm、RoPE、激活函数和注意力层。层本身都很薄，真正的计算交给 <code>minisgl.kernel</code> 里的算子，它们有 PyTorch 参考实现，在 GPU 上换成 FlashInfer 或自定义 kernel。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 推理引擎需要 `nn.Module` 的哪些功能？不需要哪些？
    2. 模型为什么要先在 `meta` 设备上构建？这时候 RoPE 的 cos/sin 表会遇到什么问题？
    3. "融合的残差加 RMSNorm"返回的两个张量分别是什么？
    4. prefill 时 LM head 为什么只算每个请求最后一个位置？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 需要：按层次组织参数、按名字收集和加载权重；不需要：autograd、训练 / 评估模式、hook、`parameters()` 的注册机制、`to()` 的设备搬运等——推理只要前向，权重加载一次就不变。
    2. 先在 meta 设备上建模型只分配元数据，不占内存、不做初始化，之后直接用真实的权重替换，秒建模型。但 RoPE 的 cos / sin 表不是权重，在 meta 设备上算出来也没有数据，所以要放在真实的设备上单独计算（所有层共用一张表）。
    3. 一个是归一化后的输出（送进下一个子层），另一个是加上残差之后的新残差（留给下一次残差相加）。
    4. 只有每个请求最后一个位置的 logits 用来采样下一个 token，其余位置的 logits 没用；LM head 是最大的矩阵乘之一（词表很大），只算最后位置省下大部分计算和显存。

**本章要写的文件**：`layers/base.py`、`layers/linear.py`、`layers/embedding.py`、`layers/norm.py`、`layers/rotary.py`、`layers/activation.py`、`layers/attention.py`，以及算子 `kernel/torch_ops.py`、`kernel/__init__.py`。

## 为什么不用 nn.Module

推理引擎用到 `nn.Module` 的地方只有一个：**按名字收集和加载权重**，名字要与 checkpoint 里的键名一致（`model.layers.0.self_attn.o_proj.weight`）。而 `nn.Module` 的其余机制——`Parameter`、自动求导、hook、`train()`/`eval()`、`__call__` 的额外开销——都用不到。`BaseOP` 用 Python 对象的 `__dict__` 实现了按名字收集和加载，其余一概没有：

@@code python/minisgl/layers/base.py:BaseOP@@

规则只有三条：

1. 不以下划线开头的 `Tensor` 属性是权重，名字是"属性路径"；
2. 不以下划线开头的 `BaseOP` 属性是子模块，递归处理；
3. 以下划线开头的属性（比如 RoPE 的 `_cos_sin_cache`、线性层的 `_comm`）不参与。

`load_state_dict` 不是把数据拷进已有的张量，而是**用 checkpoint 里的张量替换属性**（`setattr`）。这配合"先在 meta 设备上建模型"：`torch.device("meta")` 上创建的张量只有形状、没有内存，建一个 70B 的模型也是瞬间完成、不占内存；加载时用真实张量换掉它们。加载结束后如果 `state_dict` 里还有没被取走的键，就报错——多余的键通常意味着模型结构写错了。

`OPList` 让子模块的名字是 `0`、`1`、`2`……，对应 checkpoint 里的 `layers.0`、`layers.1`；`StateLessOP` 用于没有权重的层（RoPE、注意力层本身），它们的 `state_dict` 永远为空。

@@code python/minisgl/layers/base.py:OPList@@

用 Qwen3-0.6B 的配置在 meta 设备上建一层 decoder，看看收集到的权重名：

@@code examples/ch02_layers.py@@

@@output ch02_layers@@

名字与 Hugging Face 的 checkpoint 一致，只有两处不同：`q_proj`、`k_proj`、`v_proj` 合并成了 `qkv_proj`，`gate_proj`、`up_proj` 合并成了 `gate_up_proj`。合并后三个矩阵乘变成一个，输入只读一次。合并由下一章的权重加载器在读取时完成。

注意 `q_norm` 同时是 `RopeAttn` 和它内部 `AttentionLayer` 的属性，但只出现了一次：`AttentionLayer` 是 `StateLessOP`，不收集任何权重，它只是引用 `RopeAttn` 持有的那一份。

## 算子：参考实现与分派

层只负责组织参数，计算交给 `minisgl.kernel`。每个算子都有一个 PyTorch 参考实现，计算顺序刻意与 Hugging Face 保持一致，这样在 float32 下可以逐位对齐：

@@code python/minisgl/kernel/torch_ops.py:rmsnorm@@

@@code python/minisgl/kernel/torch_ops.py:fused_add_rmsnorm@@

`kernel/__init__.py` 负责分派：CUDA 张量且装了 FlashInfer 时用 FlashInfer，否则用参考实现。

@@code python/minisgl/kernel/__init__.py:rmsnorm@@

!!! diff "与官方的差异：算子分派"
    官方的各层直接 `from flashinfer import rmsnorm` 等，只能在 GPU 上运行。我们让所有层都通过 `minisgl.kernel` 调用算子，CPU 上用参考实现。上面例子的最后两行显示，参考实现的 RMSNorm、RoPE 与 Hugging Face 的误差是 0。

## 线性层

@@code python/minisgl/layers/linear.py:_LinearTPImpl@@

`_LinearTPImpl` 记录了完整的和本 rank 的输入、输出维度，`forward` 就是一个 `F.linear`。四个子类对应张量并行下的四种切法：

| 类 | 用在哪 | TP 下怎么切 |
| --- | --- | --- |
| `LinearReplicated` | MoE 的路由层 | 不切，每个 rank 一份完整的 |
| `LinearColParallelMerged` | `gate_up_proj` | 按输出维切，多个投影合并 |
| `LinearQKVMerged` | `qkv_proj` | 按注意力头切，KV 头不够分时复制 |
| `LinearRowParallel` / `LinearOProj` | `down_proj`、`o_proj` | 按输入维切，结果 all-reduce |

TP=1 时它们都退化成普通的线性层。切分的细节和通信放到[张量并行](../perf/tensor-parallel.md)一章；现在只需要记住：构造函数里出现的 `get_tp_info().size` 在单卡时是 1。

## 词嵌入与输出层

@@code python/minisgl/layers/embedding.py:VocabParallelEmbedding@@

查表交给 `indexing` 算子。张量并行时每个 rank 只保存词表的一段 `vocab_range = (起点, 长度)`，查不到的 token 填 0，最后 all-reduce 相加——每个 token 只在一个 rank 上查到非零向量。

@@code python/minisgl/layers/embedding.py:ParallelLMHead.forward@@

输出层有一个重要的优化：**prefill 时只取每个请求最后一个位置的隐状态**去算 logits。一个 1000 个 token 的提示词，prefill 需要的只是第 1000 个位置的下一个 token 分布；如果对所有位置都算，LM head（`[1000, 1024] × [1024, 151936]`）的计算量和显存都会浪费上千倍。最后位置的下标由注意力元数据的 `get_last_indices` 给出（第 5 章）。decode 时每个请求本来就只有一个位置，不需要挑。

Qwen3-0.6B 共享词嵌入（`tie_word_embeddings`），`ParallelLMHead` 直接用 `embed_tokens` 的权重，自己的 `state_dict` 为空；加载时如果 checkpoint 里仍然带着 `lm_head.weight`（Qwen3-0.6B 就带着），就丢掉。

## RMSNorm 与残差

@@code python/minisgl/layers/norm.py:RMSNormFused@@

decoder 层里残差相加总是紧跟着归一化：`residual = x + residual; x = norm(residual)`。`RMSNormFused` 把两步合成一个算子，同时返回归一化结果和新的残差；在 GPU 上它是 FlashInfer 的一个 kernel，少读写一遍隐状态。第一层还没有残差时，`x` 本身就是残差。于是整个 decoder 层写成：

@@code python/minisgl/models/decoder.py:DecoderLayer.forward@@

## RoPE

@@code python/minisgl/layers/rotary.py:RotaryEmbedding@@

`cos_sin_cache` 在构造时对所有位置一次性算好，前向时按 `positions` 查表。算子按 NeoX 风格（前后两半配对）旋转，与 Llama、Qwen 系列的 Hugging Face 实现相同：

@@code python/minisgl/kernel/torch_ops.py:apply_rope_inplace@@

`get_rope` 用 `functools.cache` 缓存：所有层的 RoPE 参数相同，共用一张表。还有一个细节——模型是在 `torch.device("meta")` 上构建的，而 cos/sin 表是真实数据，不能放在 meta 上。所以 `get_rope` 检测到当前默认设备是 meta 时，改用 `set_rope_device` 事先指定的真实设备：

@@code python/minisgl/layers/rotary.py:get_rope@@

忘了调用 `set_rope_device` 就在 meta 上建模型，会得到 `RuntimeError: Call set_rope_device() before building a model on meta device`。

Llama 3.1 的长上下文扩展（`rope_type: llama3`）对频率做后处理：高频分量不变，低频分量除以 `factor`，中间平滑过渡。它的正确性由第 20 章里一个用随机权重构造的小 Llama 3 模型与 Hugging Face 对比验证。

!!! diff "与官方的差异：YaRN"
    官方的 `_get_rope` 还支持 `yarn`，但只修改了频率，没有乘 Hugging Face 实现里的注意力缩放系数（`0.1 * ln(factor) + 1`）。我们只实现了 `default` 和 `llama3` 两种，YaRN 留作练习。

## 注意力层

@@code python/minisgl/layers/attention.py:AttentionLayer.forward@@

注意力层也不持有权重：它把合并的 `qkv` 拆成三份，Qwen3 还要对每个头的 q、k 做一次 RMSNorm（QK-Norm），然后加上 RoPE，最后把 q、k、v 交给全局上下文里的**注意力后端**。写 KV 缓存和计算注意力都是后端的事，第 5 章实现。

!!! upstream "官方实现"
    - `BaseOP`：@@upstream layers/base.py:BaseOP@@
    - 输出层只取最后位置：@@upstream layers/embedding.py:ParallelLMHead.forward@@
    - RoPE 的 meta 设备处理：@@upstream layers/rotary.py:get_rope@@
    - 注意力层：@@upstream layers/attention.py:AttentionLayer.forward@@

    官方在注意力层里直接对 `qkv.split(...)` 得到的视图原地做 QK-Norm 和 RoPE（FlashInfer 的算子支持带步长的输入）。我们的参考算子先 `.contiguous()` 拷一份，写法更直白，代价是多一次拷贝。

## 测试

`tests/test_ch02_layers.py` 检查 `BaseOP` 的命名规则、多余键报错、RMSNorm 与 RoPE 与 Hugging Face 一致，以及词表并行查表：

@@code tests/test_ch02_layers.py:test_vocab_parallel_indexing_masks_other_shards@@

!!! interview "怎么讲清楚"
    讲算子层：推理引擎只需要 `nn.Module` 的一小部分——按名字收集和加载权重、组织层级；不需要自动求导、钩子、训练状态，所以 mini-sglang 用轻量的 `BaseOP`，层只组织参数、计算交给 kernel（CPU 上是与 HF 数值一致的参考实现，GPU 上是 FlashInfer）。模型先在 `meta` 设备上建空壳（不分配内存，秒建），加载时直接替换成真实权重；但 RoPE 的 cos/sin 表是计算出来的，必须放在真实设备上。融合的残差加 RMSNorm 返回归一化后的输出和新的残差；prefill 时 LM head 只算每个请求最后一个位置，省掉大量无用的词表投影。

## 练习

1. 给 `BaseOP` 加一个 `num_params()` 方法，返回权重元素总数。用它算出 Qwen3-0.6B 的参数量（注意共享词嵌入不要重复计算）。
2. 把 `ParallelLMHead.forward` 里"prefill 只取最后位置"的逻辑去掉，prefill 一个 1000 token 的请求，LM head 的输出张量有多大（float32）？
3. 实现 YaRN：在 `_get_rope` 里加一个分支，并让 cos/sin 乘上注意力缩放系数。用 `transformers` 的 `Qwen3Config(rope_scaling={"rope_type": "yarn", ...})` 构造一个小模型验证。

??? success "参考答案"
    1. `sum(t.numel() for t in self.state_dict().values())`。因为共享词嵌入时 `ParallelLMHead.state_dict()` 为空，结果自然不会重复：Qwen3-0.6B 约 5.96 亿。
    2. `[1000, 151936]` 个 float32，约 580 MiB；只取最后位置时是 0.6 MB。
    3. 参考 Hugging Face `_compute_yarn_parameters`：频率按 `ramp` 在 `inv_freq / factor` 与 `inv_freq` 之间插值，`attention_factor = 0.1 * ln(factor) + 1`，构造 cos/sin 表时乘上它。

## 小结

- [x] `BaseOP` 用 `__dict__` 按属性路径收集和加载权重；下划线开头的属性不是权重；加载是替换而不是拷贝，配合 meta 设备秒建模型。
- [x] 层只组织参数，计算交给 `minisgl.kernel`：CPU 上是与 HF 数值一致的参考实现，GPU 上是 FlashInfer。
- [x] 输出层在 prefill 时只算每个请求最后一个位置的 logits。
- [x] RoPE 表必须放在真实设备上；所有层共用一张表。
- [x] 注意力层只做拆分、QK-Norm 和 RoPE，写缓存和算注意力交给注意力后端。
