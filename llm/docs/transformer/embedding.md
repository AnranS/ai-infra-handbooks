# 嵌入层与输出层

<p class="lead">模型的入口把 token 编号变成向量，出口把向量变回"下一个 token 的分数"。这两头看似简单，却占了小模型相当一部分参数；它们之间流动的那个向量（残差流）是理解整个 Transformer 的钥匙。这一章还会用"logit lens"偷看模型中间层在"想"什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 嵌入层做的是什么运算？它和一个线性层有什么关系？
    2. 什么是权重共享（tied embeddings）？哪些模型使用它？
    3. logits 是怎么算出来的？为什么推理时只需要对最后一个位置计算？
    4. 在 Qwen3-0.6B 中，嵌入矩阵占总参数量的多少？在 70B 的模型中呢？
    5. 什么是 logit lens？它说明了什么？

## 嵌入层：查表

嵌入层就是一个形状为 `[V, d]` 的矩阵 $E$，第 i 行是 token i 的向量。输入 token 编号，输出对应的行：

```pycon
>>> import torch
>>> import torch.nn as nn
>>> torch.manual_seed(0)  # doctest: +ELLIPSIS
<torch._C.Generator object at 0x...>
>>> emb = nn.Embedding(10, 4)
>>> ids = torch.tensor([3, 7, 3])
>>> out = emb(ids)
>>> out.shape
torch.Size([3, 4])
>>> torch.equal(out[0], emb.weight[3]) and torch.equal(out[0], out[2])
True
>>> onehot = nn.functional.one_hot(ids, 10).float()       # 等价写法：one-hot 向量乘以嵌入矩阵
>>> torch.allclose(onehot @ emb.weight, out)
True
```

数学上它等价于"one-hot 向量乘以矩阵"，但实现上只是按下标读取几行，几乎没有计算量。

## 输出层：LM Head

最后一层的隐藏状态 $h$（经过最终的 RMSNorm）乘以输出矩阵 $W_{out}$（形状 `[V, d]`），得到每个 token 的分数：

$$
\text{logits} = h\, W_{out}^\top \in \mathbb{R}^{V}
$$

可以把它理解为：**用 $h$ 和词表里每个 token 的"输出向量"做点积**，越像的 token 分数越高。

### 权重共享

很多模型让输出矩阵直接复用嵌入矩阵，$W_{out} = E$，叫做**权重共享（tied embeddings）**。输入时"token → 向量"，输出时"向量 → 和哪个 token 最像"，用同一套向量很自然，而且省参数。Qwen3-0.6B 就是这样：

```pycon
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> path = "models/Qwen3-0.6B"
>>> tok = AutoTokenizer.from_pretrained(path)
>>> model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
>>> E = model.model.embed_tokens.weight
>>> E.shape
torch.Size([151936, 1024])
>>> model.lm_head.weight.data_ptr() == E.data_ptr()      # 同一块内存
True
>>> total = sum(p.numel() for p in model.parameters())   # 共享的参数只算一次
>>> total, E.numel(), round(E.numel() / total, 3)
(596049920, 155582464, 0.261)
```

嵌入矩阵占了 0.6B 模型 26.1% 的参数。模型越大，这个比例越小：LLaMA-3-70B 的词表是 128256、隐藏维度 8192，嵌入和输出层各约 10.5 亿参数，加起来只占 3% 左右。所以**小模型倾向于共享权重，大模型通常不共享**（Qwen 系列 7B、8B 以上的型号和 LLaMA-3 都不共享）。

## 嵌入向量里有什么

训练之后，意思相近的 token，嵌入向量也相近。用余弦相似度找几个 token 的"近邻"：

```pycon
>>> En = nn.functional.normalize(E[:151643], dim=-1)      # 只看真实存在的 token
>>> for w in ["北京", "猫", " king", "三"]:
...     i = tok.encode(w)[0]
...     top = (En @ En[i]).topk(6).indices[1:]          # 去掉自己
...     print(repr(w), [tok.decode([t]) for t in top])
'北京' [' Beijing', '北京市', '在北京', '上海', '广州']
'猫' ['貓', ' cat', ' cats', '猫咪', ' Cat']
' king' [' King', 'King', ' kings', ' KING', '国王']
'三' [' three', 'three', ' Three', 'Three', '四']
```

跨语言的对应（"北京"和" Beijing"、"猫"和" cat"、" king"和"国王"）、繁简体、大小写变体、同类概念（"上海"、"广州"，"四"）都靠得很近。模型从未被告知这些关系，完全是从"预测下一个 token"的训练中学出来的。

## 残差流

嵌入层输出的向量 `[B, T, d]` 进入第一层，之后每一层都把自己的计算结果**加**到这个向量上（残差连接，见[归一化与残差流](norm-residual.md)），最后经过 RMSNorm 和 LM Head 变成 logits。这条从嵌入到输出、贯穿所有层的 d 维向量，叫做**残差流（residual stream）**。可以把每一层理解为：从残差流里读取信息，计算，再把结果写回去。

### logit lens：偷看中间层

既然最后一层的残差流经过 "RMSNorm + LM Head" 就能变成预测，那么**中间层的残差流**直接套上同样的变换，会预测出什么？这个技巧叫 **logit lens**：

```pycon
>>> msgs = [{"role": "user", "content": "中国的首都是哪里？"}]
>>> prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + "中国的首都是"
>>> ids = tok(prompt, return_tensors="pt").input_ids
>>> with torch.no_grad():
...     out = model(ids, output_hidden_states=True)
>>> len(out.hidden_states)            # 嵌入输出 + 28 层每层的输出
29
>>> with torch.no_grad():
...     for layer in [0, 4, 8, 14, 20, 24, 27]:
...         h = model.model.norm(out.hidden_states[layer][0, -1])
...         p = model.lm_head(h).softmax(-1)
...         print(layer, repr(tok.decode(p.argmax())), round(p.max().item(), 3))
0 '都是' 1.0
4 'omorphic' 0.181
8 '**' 0.187
14 '1' 0.463
20 ' **' 0.601
24 '北京' 0.79
27 '北京' 0.823
>>> p = out.logits[0, -1].softmax(-1)  # 最后一层（transformers 返回的最后一个 hidden state 已经过最终的 RMSNorm）
>>> repr(tok.decode(p.argmax())), round(p.max().item(), 3)
("'北京'", 0.433)
```

读这个结果：

- 第 0 层（刚嵌入）预测的是输入 token 本身（因为权重共享，嵌入向量和自己最像）；
- 中间层的"预测"多是 `**`、`1` 这样的格式符号：此时残差流里的信息还不能直接解读为下一个 token；
- 到第 24 层，"北京"已经浮现出来（0.79），在最后一层之前达到 0.82；
- 最后一层的输出仍然是"北京"，但概率反而降到 0.43——最后一层把一部分概率分给了其他说法（比如"北京市"），起到"校准"的作用。

这说明**答案是在最后几层才逐渐形成的**，前面的层在做更抽象的处理。这类"可解释性"分析不是推理优化的必备知识，但它能帮你建立对残差流的直觉。

!!! inference "推理视角"
    - **嵌入层是一次"gather"（按下标读取），几乎不花时间**；输出层是一个 `[T, d] × [d, V]` 的大矩阵乘法，而且 V 很大。推理时只有**最后一个位置**的 logits 有用（用来选下一个 token），所以 prefill 阶段推理引擎只对最后一个位置计算 LM Head，省掉 T−1 倍的计算；
    - **张量并行时**，嵌入层和输出层常按词表维度切到多张卡上（vocab parallel），输出层算完后需要把各卡的 logits 拼起来，或者在各卡上分别求局部最大值再合并；
    - **大词表的采样**：15 万维的 logits 做 softmax、排序、top-p，本身就不便宜，推理引擎会专门优化采样 kernel，见[解码与采样](../inference/decoding.md)。

!!! interview "面试怎么答"
    输入输出层的考点：嵌入是 `[V, d]` 的查表（gather），输出层是隐藏状态和每个 token 向量的点积，是一个很大的 GEMM；小模型常共享两者——Qwen3-0.6B 的嵌入就占了 26% 的参数，而 LLaMA-3-70B 的嵌入和输出层加起来只占 3% 左右。推理时 LM Head 只算最后一个位置（对 0.6B 模型能省下每 token 约四分之一的计算），训练时每个位置都要算。logit lens 说明答案在靠后的几层才成形。

## 练习

**1. 参数计算。** Qwen2.5-7B 的词表大小为 152064、隐藏维度为 3584，不共享嵌入权重。嵌入层和输出层一共有多少参数？占 76.2 亿总参数的多少？

??? success "参考答案"
    ```python
    V, d, total = 152064, 3584, 7_615_616_512
    emb = 2 * V * d                       # 嵌入层 + 输出层，不共享
    assert emb == 1_089_994_752
    print(f"{emb / total:.1%}")           # 约 14.3%
    ```

    约 10.9 亿参数，占 14.3%。可见词表很大时，即使是 7B 模型，嵌入和输出层的占比也不小。

**2. 思考题。** 如果输入的是 prompt 的 1000 个 token，prefill 时只对最后一个位置计算 LM Head，能省下多少计算量（以 Qwen3-0.6B 为例，只算 LM Head）？

??? success "参考答案"
    LM Head 对每个位置的计算量是 2 × 1024 × 151936 ≈ 3.11 亿次运算。1000 个位置全算需要 3110 亿次，只算最后一个位置只需 3.11 亿次，省下 99.9%。对 0.6B 模型来说，LM Head 约占每个 token 计算量的四分之一，所以这是非常显著的节省。训练时则不能省，因为每个位置都要计算损失。

## 小结

- [x] 嵌入层是 `[V, d]` 的查找表；输出层把最终隐藏状态和每个 token 的输出向量做点积得到 logits。
- [x] 小模型常共享嵌入和输出权重；词表大时这两层占比可观。
- [x] 嵌入向量编码了语义关系；贯穿所有层的隐藏向量叫残差流。
- [x] logit lens 显示答案在靠后的几层才形成。
- [x] 推理时 LM Head 只需算最后一个位置；嵌入是 gather，输出层是大 GEMM。
