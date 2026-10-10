# 模型与权重加载

<p class="lead">有了各种层，这一章把它们拼成完整的模型：从 Hugging Face 的 <code>config.json</code> 提取引擎关心的字段，按结构名创建模型，再用一个流式加载器把 safetensors 里的权重一边读、一边合并、一边切分地装进去。支持 Llama、Qwen2、Qwen3 和 Qwen3-MoE 四种结构。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Llama、Qwen2、Qwen3 的 decoder 层结构有哪些差别？
    2. 权重加载器为什么要"流式"？一次性 `torch.load` 整个 checkpoint 有什么问题？
    3. checkpoint 里分开存的 `q_proj`、`k_proj`、`v_proj`，加载时怎样合并成 `qkv_proj`？合并的顺序重要吗？
    4. transformers 4.x 和 5.x 的 config 里，RoPE 的 `rope_theta` 分别放在哪？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 结构相同，差别只在三个开关：Qwen2 的 q、k、v 投影有偏置；Qwen3 在 RoPE 之前对 q、k 做 QK-Norm（且没有 qkv 偏置）；Qwen3-MoE 把 MLP 换成 MoE。
    2. 一次性 `torch.load` 要把整个 checkpoint 读进内存，峰值内存可能是模型的好几倍（原始权重 + 切分、合并后的副本）；流式加载边读边切分边合并，峰值内存小。
    3. 读到 q、k、v 各自的权重后，（张量并行时先各自切分）沿输出维拼接成一个 `qkv_proj` 的权重。顺序很重要：必须和模型里把 `qkv_proj` 的输出切回 q、k、v 的顺序一致，否则 q、k、v 就对调了。
    4. transformers 4.x 在 config 的顶层（`rope_theta`）；5.x 放进了 `rope_parameters` 字典里。`ModelConfig` 要两种位置都认。

**本章要写的文件**：`models/config.py`、`models/base.py`、`models/utils.py`、`models/decoder.py`、`models/register.py`、`models/weight.py`。

@@tree@@

这一步的 main：`examples/ch03_models.py`——它只用到上面这些文件；`python tools/steps.py check` 会逐章搭出这棵树、跑这个 main。

## ModelConfig：只取需要的字段

Hugging Face 的配置类有几十个字段，而且不同模型的字段名、默认值各不相同。`ModelConfig` 把引擎关心的字段统一成一个冻结的 dataclass：

@@code python/minisgl/models/config.py:ModelConfig@@

几个容易踩坑的地方：

- **`head_dim`**：Qwen3 的 `head_dim`（128）不等于 `hidden_size / num_attention_heads`（1024 / 16 = 64），必须优先读显式字段；
- **RoPE 参数**：transformers 5 把 `rope_theta` 和 `rope_scaling` 合并进了 `rope_parameters` 字典（`{"rope_theta": 1000000, "rope_type": "default"}`），4.x 版本则是两个独立字段。`from_hf` 两种都支持；
- **Qwen2 的偏置**：Qwen2 的 q/k/v 投影带偏置，但它的 config 里没有 `attention_bias` 这个字段，要按 `model_type` 判断。

@@code examples/ch03_models.py@@

@@output ch03_models@@

## 模型结构

四种结构的 decoder 层几乎相同，差别只有三个开关：

| 结构 | q/k/v 偏置 | QK-Norm | MLP |
| --- | --- | --- | --- |
| Llama | 看 `attention_bias`（通常没有） | 无 | SwiGLU |
| Qwen2 | 有 | 无 | SwiGLU |
| Qwen3 | 无 | 有 | SwiGLU |
| Qwen3-MoE | 无 | 有 | MoE（第 20 章） |

所以我们用一个通用的 `decoder.py`：

@@code python/minisgl/models/decoder.py:CausalLM@@

@@code python/minisgl/models/decoder.py:Qwen3ForCausalLM@@

子层在 `models/utils.py` 里：`RopeAttn`（qkv_proj → 注意力层 → o_proj）、`GatedMLP`（gate_up_proj → silu × up → down_proj）和 `MoEMLP`。

@@code python/minisgl/models/utils.py:GatedMLP@@

`CausalLM.forward()` 没有参数，从全局上下文读取 `input_ids`，返回 `[batch_size, vocab]` 的 logits——每个请求一行，这是 LM head 在 prefill 时只取最后位置的结果。

!!! diff "与官方的差异：模型文件"
    官方为每种结构各写一个文件（`llama.py`、`qwen2.py`、`qwen3.py`、`qwen3_moe.py`、`mistral.py`，每个 80 多行，内容大同小异），并通过注册表按 `architectures[0]` 懒加载。我们合并成一个 `decoder.py` 加三个开关，没有实现 Mistral。

## 流式权重加载

@@code python/minisgl/models/weight.py:load_weight@@

加载器是一个生成器，每次 `yield` 一个"已经处理好"的 `(名字, 张量)`：

1. **逐个读取**：`safetensors.safe_open` 按需从文件中读出一个张量，不必把整个 checkpoint 读进内存。一个 70B 的 bf16 模型有 140 GB，一次性读入会让 CPU 内存爆掉；流式读取时峰值只多出一个张量加一小块合并缓冲区。
2. **按 TP 切分**：`shard_tensor` 取出本 rank 的那一份（第 16 章详细讲）。先切分再合并，合并后的张量天然就是本 rank 的 `[q_本地; k_本地; v_本地]`。
3. **合并投影**：读到 `q_proj` 时先放进缓冲区，等同一层的 `k_proj`、`v_proj` 都读到了，再按 **q、k、v 的固定顺序**拼接后 yield。顺序必须与注意力层拆分时一致（`qkv.split([q_dim, kv_dim, kv_dim])`）；文件里的读取顺序是字母序（上面的运行结果里 `down_proj` 先于合并后的 `gate_up_proj` 出现），与拼接顺序无关。
4. **打包专家**：MoE 模型的每个专家分开存（`experts.0.gate_proj`、`experts.1.gate_proj`……），读齐所有专家后 `torch.stack` 成 `[E, ...]` 的三维张量。

最后两个断言保证没有"只读到一半"的合并组。

## 装进模型

引擎（第 6 章）用三行完成加载：

```python
with torch.device("meta"), torch_dtype(config.dtype):   # 空壳：只有形状，没有内存
    self.model = create_model(config.model_config)
self.model.load_state_dict({k: v.to(self.dtype) for k, v in load_weight(path, device)})
```

`torch_dtype` 临时修改默认 dtype，让构建时的 `torch.empty(...)` 就是目标精度，这样 `load_state_dict` 里的 dtype 断言才能通过。`load_state_dict` 用 checkpoint 里的真实张量替换 meta 张量；Qwen3-0.6B 的 checkpoint 多带了一个 `lm_head.weight`（共享词嵌入时它等于 `embed_tokens.weight`），被 `ParallelLMHead.load_state_dict` 丢掉，其余键必须一一对应，否则报错。

!!! upstream "官方实现"
    - 配置：@@upstream models/config.py:ModelConfig.from_hf@@（官方依赖锁定 `transformers<=4.57.3`，本书在 transformers 5.17 上验证）
    - Qwen3 结构：@@upstream models/qwen3.py:Qwen3DecoderLayer@@
    - 流式加载：@@upstream models/weight.py:load_weight@@
    - 切分：@@upstream models/weight.py:_shard_tensor@@

## 测试

@@code tests/test_ch03_models.py:test_weight_loader_produces_exactly_the_model_keys@@

这个测试保证加载器产出的键集合与模型的 `state_dict` 完全相同（除了被丢掉的 `lm_head.weight`），形状也一一对应。`test_model_config_from_hf` 检查 Qwen3 与 Qwen2.5 两种配置的关键字段（Qwen2.5 只用到 `config.json`，放在 `tests/configs/` 里，不需要下载模型）。

!!! interview "怎么讲清楚"
    讲模型加载：Llama、Qwen2、Qwen3、Qwen3-MoE 共用一种 decoder 结构，差别只是三个开关——qkv 是否带偏置（Qwen2）、是否有 QK-Norm（Qwen3）、MLP 是否是 MoE。权重要流式加载：一次性读进整个 checkpoint 峰值内存会翻倍，流式地边读、边按张量并行切分、边把 `q_proj` / `k_proj` / `v_proj` 按顺序拼成一个 `qkv_proj`（顺序必须和前向里拆分的顺序一致）、把 gate 和 up 合并、把 MoE 专家打包成三维张量。配置要兼容新旧两种写法（比如 transformers 5.x 把 `rope_theta` 挪进了 RoPE 参数里），`head_dim` 不一定等于 `hidden / heads`。

## 练习

1. 用 `ModelConfig` 写一个函数，算出每个 token 的 KV 缓存占多少字节（下一章会用到）。Qwen3-0.6B 在 bf16 下是多少？
2. 给加载器加上 `tqdm` 进度条，但只在 TP rank 0 上显示。
3. 如果某个 checkpoint 把 `q_proj` 和 `k_proj` 放在两个不同的 safetensors 文件里，我们的加载器还能正确合并吗？为什么？

??? success "参考答案"
    1. `2 * num_layers * num_kv_heads * head_dim * 2 字节` = 2 × 28 × 8 × 128 × 2 = 114688 字节，约 112 KB。
    2. `for file in tqdm(files, disable=not get_tp_info().is_primary())`，官方就是这样写的。
    3. 能。合并缓冲区 `merge_buf` 跨文件保留，只要同一组的所有投影最终都读到了就会合并；读完所有文件还有未完成的组才报错。

## 小结

- [x] `ModelConfig` 把各种 HF 配置统一成引擎需要的字段，注意 `head_dim`、RoPE 参数的新旧两种位置、Qwen2 的偏置。
- [x] Llama、Qwen2、Qwen3、Qwen3-MoE 共用一个 decoder 结构，差别是 qkv 偏置、QK-Norm、MoE 三个开关。
- [x] 流式加载器边读边切分边合并：峰值内存小，q/k/v 与 gate/up 合并为一次矩阵乘，MoE 专家打包成三维张量。
- [x] 模型先在 meta 设备上用目标 dtype 建空壳，再用 `load_state_dict` 替换成真实权重。
