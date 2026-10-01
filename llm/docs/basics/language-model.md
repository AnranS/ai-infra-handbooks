# 语言模型：从概率到下一个词

<p class="lead">不管模型有多大、结构多复杂，大语言模型做的事情只有一件：给定前面的文字，算出下一个 token 的概率分布。训练是让这个分布越来越准，推理是反复从这个分布里取词。理解了这一点，后面所有的内容（包括推理为什么难优化）都有了落脚点。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. "自回归"是什么意思？一句话的概率怎么分解？
    2. 模型的一次前向传播输出的是什么？形状是多少？
    3. 交叉熵损失和困惑度（perplexity）是什么关系？
    4. 训练时为什么一次前向就能算出一整句话每个位置的损失？
    5. 为什么说推理的慢，根源在于"自回归"？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 每个 token 的概率只依赖它前面的 token：$P(x_1 \dots x_T) = \prod_t P(x_t \mid x_{<t})$，生成时一个接一个地产生。
    2. 每个位置一个词表大小的 logits 向量，形状 `[B, T, V]`；softmax 之后是"下一个 token"的概率分布。
    3. 困惑度 = exp(平均交叉熵)，交叉熵就是平均每个 token 的负对数似然；困惑度可以理解为模型每一步在多少个"等可能的"选项里犹豫。
    4. 训练时整句都已知，因果掩码保证每个位置只看到自己之前的 token，所以一次前向就能同时算出所有位置的预测和损失（teacher forcing）。
    5. 每个新 token 依赖前面所有 token，只能一步一步地生成；每一步都要读一遍全部权重、只算一个 token，GPU 算力大量闲置——decode 因此串行而且访存受限。

## 语言模型就是一个条件概率函数

把一段文本切成一串 token $x_1, x_2, \ldots, x_T$（怎么切见[分词](tokenization.md)）。根据概率的链式法则，整段文本的概率可以分解为：

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})
$$

**语言模型**就是一个能计算右边每一项的函数：输入前面的 token，输出下一个 token 在整个词表上的概率分布。这种"一次预测一个、依赖之前所有"的方式叫**自回归（autoregressive）**。GPT、LLaMA、Qwen、DeepSeek 都是这一类模型。

生成文本就是反复做同一件事：

1. 把已有的 token 输入模型，得到下一个 token 的分布；
2. 按某种策略从分布中选一个 token（取概率最大的，或者随机采样，见[解码与采样](../inference/decoding.md)）；
3. 把它接到输入后面，回到第 1 步，直到生成结束符或达到长度上限。

## 看一眼真实模型的输出

用 Qwen3-0.6B 看看"下一个 token 的分布"长什么样（模型下载方式见[首页](../index.md#准备环境)）：

```pycon
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> path = "models/Qwen3-0.6B"
>>> tok = AutoTokenizer.from_pretrained(path)
>>> model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
>>> ids = tok("中国的首都是", return_tensors="pt").input_ids
>>> ids
tensor([[105538,  59975, 100132]])
>>> with torch.no_grad():
...     logits = model(ids).logits
...
>>> logits.shape
torch.Size([1, 3, 151936])
```

输入是 3 个 token（"中国"、"的"、"首都是"），输出的形状是 `[batch, 序列长度, 词表大小]` = `[1, 3, 151936]`。注意 3 是序列长度（三个输入位置），不是 logits 的长度：**每个位置**都输出一个长度为 151936 的向量，叫 **logits**（未归一化的分数），词表里每一个词都有一个分数，对应"在这个位置之后，下一个 token 是这个词"的打分。用 softmax 把最后一个位置（"首都是"之后）的 151936 个分数变成概率，`topk(5)` 只是从这 151936 个候选里挑概率最大的 5 个打印出来看看：

```pycon
>>> probs = logits[0, -1].softmax(dim=-1)
>>> top = probs.topk(5)
>>> for p, i in zip(top.values, top.indices):
...     print(repr(tok.decode(i)), round(p.item(), 3))
'____' 0.122
'北京' 0.094
'城市' 0.048
'位于' 0.043
'建' 0.037
```

有意思的是，概率最高的不是"北京"，而是下划线"____"。因为我们没有使用对话模板，模型把"中国的首都是"当成了一道填空题的题干，这在它见过的训练文本里很常见。模型只是在模仿训练数据的统计规律，这件事值得牢记。加上对话模板后的行为见[分词](tokenization.md#特殊-token-与对话模板)一章。

!!! inference "推理视角"
    模型每一步都要对**整个词表**打分，最后一层（LM Head）是一个 `hidden_size × vocab_size` 的矩阵乘法，Qwen3-0.6B 的这个矩阵有 1024 × 151936 ≈ 1.56 亿个参数（和词嵌入共享），占整个模型的四分之一以上。词表越大，这一层的开销越大，这也是推理引擎经常单独优化 logits 计算和采样的原因。

## 训练目标：交叉熵

模型怎么学会给出好的分布？让它在海量文本上，**对每个位置真实出现的下一个 token 给出尽量高的概率**。形式化地，最小化负对数似然，也就是**交叉熵损失**：

$$
\mathcal{L} = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log P_\theta(x_{t+1} \mid x_1, \ldots, x_t)
$$

它的指数叫**困惑度（perplexity）**：$\text{PPL} = e^{\mathcal{L}}$。直观理解：困惑度为 10，相当于模型在每个位置平均在 10 个候选之间"犹豫"。

在真实模型上算一下。transformers 的模型传入 `labels` 时会自动计算损失（内部会把标签错开一位）：

```pycon
>>> text = "北京是中国的首都，也是全国的政治和文化中心。"
>>> ids = tok(text, return_tensors="pt").input_ids
>>> with torch.no_grad():
...     out = model(ids, labels=ids)
>>> ids.shape, round(out.loss.item(), 3), round(torch.exp(out.loss).item(), 2)
(torch.Size([1, 12]), 3.703, 40.58)
```

手动计算一遍，看看每个 token 的损失：

```pycon
>>> with torch.no_grad():
...     logp = model(ids).logits[0, :-1].log_softmax(dim=-1)   # 位置 t 的输出预测第 t+1 个 token
>>> nll = -logp.gather(1, ids[0, 1:, None]).squeeze(1)          # 取出真实 token 的负对数概率
>>> round(nll.mean().item(), 3)
3.703
>>> for t, v in zip(ids[0, 1:], nll):
...     print(tok.decode(t), round(v.item(), 2))
是中国 17.86
的 1.64
首都 1.93
， 0.29
也是 2.35
全国 4.6
的政治 3.22
和 5.15
文化 3.16
中心 0.0
。 0.53
```

能看出模型"知道"的东西："政治和文化"之后，"中心"几乎没有悬念（损失 0.0）；看到"北京是中国的"之后，"首都"也很有把握（1.93）；而"和"之后接什么本来就不确定（5.15）。第一个位置的损失特别大（17.86）：模型只看到"北京"两个字，完全没料到下一个 token 是"是中国"，这一项就把平均损失从约 2.3 拉到了 3.7——**困惑度对个别意外的 token 很敏感**，比较模型时要用足够长、足够多样的文本。

把每个位置的分布摆出来看，"条件概率函数"和"交叉熵"就不抽象了：拖动位置，看模型在每一步认为最可能的候选是什么、真实的下一个 token 排在哪里（数据就是上面这次计算的结果）：

<div class="aig-widget" data-widget="nextword"></div>

## 训练可以并行，推理只能串行

注意上面的计算：**一次前向传播**就得到了所有 11 个位置的预测。这是因为训练时整句话是已知的，模型通过**因果掩码**（每个位置只能看到它之前的 token，见[注意力](../transformer/attention.md#因果掩码)）保证第 t 个位置的输出只依赖前 t 个 token。于是所有位置可以同时计算，这叫 **teacher forcing**。

推理时就不一样了：第 t+1 个 token 要等第 t 个 token 生成出来才知道，**只能一个一个地生成**。生成 500 个 token，就要按顺序执行 500 次前向传播。

![图：训练一次前向算出所有位置的预测；推理只能一个一个地生成](../assets/figures/train-vs-infer.svg){.aig-svg}

!!! inference "推理视角"
    这个"训练并行、推理串行"的不对称，是推理优化几乎所有问题的根源：

    - 每生成一个 token 都要把**全部权重读一遍**，但只算一个 token，计算量很小，GPU 大部分时间在等显存，这就是 decode 阶段的**访存瓶颈**；
    - 于是有了 **KV Cache**（不重复计算历史 token，见 [KV Cache](../inference/kv-cache.md)）、**批处理**（多个请求一起读权重）、**投机解码**（一次验证多个 token，见[推理服务](../inference/serving.md#投机解码)）、**量化**（少读一些字节）。

## 一个迷你语言模型

为了把"模型 → logits → 损失 → 训练 → 采样"完整走一遍，下面用 PyTorch 训练一个最简单的语言模型：**二元（bigram）模型**，下一个字只依赖当前这一个字。它的"参数"就是一张 `词表 × 词表` 的 logits 表：

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
corpus = "春眠不觉晓处处闻啼鸟夜来风雨声花落知多少床前明月光疑是地上霜举头望明月低头思故乡"
chars = sorted(set(corpus))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in corpus])
V = len(chars)

logits_table = torch.zeros(V, V, requires_grad=True)     # 第 i 行：当前字为 i 时，下一个字的 logits
opt = torch.optim.Adam([logits_table], lr=0.1)
for step in range(300):
    logits = logits_table[data[:-1]]                     # [T-1, V]：每个位置对下一个字的打分
    loss = F.cross_entropy(logits, data[1:])             # 和真实的下一个字比较
    opt.zero_grad()
    loss.backward()
    opt.step()

print(f"vocab={V}, final loss={loss.item():.3f}, perplexity={loss.exp().item():.2f}")
assert loss.item() < 0.5

# 生成：从"床"开始，每次按分布采样下一个字
idx = stoi["床"]
out = ["床"]
for _ in range(9):
    probs = logits_table[idx].softmax(dim=-1)
    idx = torch.multinomial(probs, 1).item()
    out.append(chars[idx])
print("".join(out))
```

这个模型能力极其有限：它只看前一个字，而"明"后面可能是"月"也可能是其他字，它无法区分语境。**Transformer 做的事情，本质上就是把"只看前一个字"扩展成"看前面所有的字，并且能学会该重点看哪些字"**。结构越来越复杂，但输入 token、输出下一个 token 的 logits、用交叉熵训练、按分布采样生成，这条主线始终不变。

!!! interview "面试怎么答"
    被问"大模型是怎么生成文字的"，三句话讲清：模型对每个位置输出一个词表大小的 logits，softmax 之后是下一个 token 的条件概率；训练时整句已知，靠因果掩码一次前向算出所有位置的交叉熵（困惑度是它的指数）；推理时只能一个一个地生成，每一步都要把全部权重读一遍。最后点题："训练并行、推理串行"让 decode 慢而且访存受限，KV Cache、批处理、投机解码都是从这里出发的。

## 练习

**1. 概率与损失。** 如果模型对正确的下一个 token 给出的概率分别是 0.5、0.25、0.125，这三个位置的平均交叉熵损失和困惑度是多少？

??? success "参考答案"
    损失 = −(ln 0.5 + ln 0.25 + ln 0.125) / 3 = (0.693 + 1.386 + 2.079) / 3 ≈ 1.386，即 ln 4。困惑度 = e^1.386 = 4。直观上，三个位置分别相当于在 2、4、8 个候选中"平均"猜中，几何平均恰好是 4。

    ```python
    import math
    probs = [0.5, 0.25, 0.125]
    loss = -sum(math.log(p) for p in probs) / len(probs)
    assert abs(loss - math.log(4)) < 1e-12 and abs(math.exp(loss) - 4) < 1e-9
    ```

**2. 理解形状。** 一个模型的词表大小为 V，输入形状为 `[B, T]` 的 token id，输出 logits 的形状是多少？训练时计算损失，需要从中取出哪些值？

??? success "参考答案"
    logits 形状为 `[B, T, V]`。位置 t 的 logits 预测第 t+1 个 token，所以取前 T−1 个位置的 logits（`[B, T-1, V]`），和第 2 到第 T 个 token（`[B, T-1]`）计算交叉熵。推理时只关心最后一个位置的 logits（`[B, V]`），前面位置的 logits 都是浪费，推理引擎在 prefill 阶段通常只对最后一个位置计算 LM Head。

**3. 思考题。** 为什么说大模型"只是在模仿训练数据的统计规律"？这和它表现出的推理能力矛盾吗？

??? success "参考思路"
    训练目标确实只是预测下一个 token。但要在海量、多样的文本上把这件事做好，模型必须在内部形成对语法、事实、逻辑结构的表示：比如要准确预测一道数学题的下一步，最有效的办法就是"学会"解这类题。所以"预测下一个 token"是一个极其通用的学习信号。同时也要记住它的局限：模型会复现训练数据中常见的模式（比如上面的填空题格式），也会在没有依据时给出看似合理的错误答案。后训练（SFT、RLHF）就是在预训练的基础上，把模型的行为引导到我们想要的方向，见[后训练](../training/post-training.md)。

## 小结

- [x] 语言模型计算 $P(x_t \mid x_{<t})$，一次前向对每个位置输出一个词表大小的 logits 向量。
- [x] 训练目标是交叉熵（负对数似然），困惑度是它的指数。
- [x] 训练时整句已知，靠因果掩码并行计算所有位置；推理时只能逐个生成。
- [x] 推理的自回归特性导致 decode 串行、访存受限，这是 KV Cache、批处理、投机解码等优化的出发点。
