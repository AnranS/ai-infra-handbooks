# 分词：文本如何变成数字

<p class="lead">模型只认识整数。分词器（tokenizer）负责把文本切成 token 并映射成编号，生成结束后再把编号还原成文本。它看起来只是预处理，却决定了序列有多长（也就是计算量和 KV Cache 有多大）、嵌入层和输出层有多大、多语言的效率如何，还带来了流式输出、对话模板等一系列工程细节。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么大模型既不按字符切分，也不按单词切分？
    2. BPE 的训练过程是怎样的？编码一段新文本时怎么应用学到的合并规则？
    3. 什么是 byte-level BPE？它为什么不会出现"未知词"？
    4. `<|im_start|>`、`<|im_end|>` 是什么？对话模板有什么用？
    5. 流式输出时，为什么不能把每个 token 单独解码后直接拼接？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 按字符切序列太长（计算和 KV 成本高）、单个字符信息量低；按单词切词表爆炸，而且总有没见过的词。子词在两者之间折中：常见词是一个 token，生僻词拆成几段。
    2. 训练：从字节（或字符）出发，统计相邻片段的频次，反复把最常见的一对合并成新 token，直到词表达到目标大小，记下合并的顺序。编码：先预切分，再按学到的合并顺序依次应用合并规则。
    3. 以字节而不是 Unicode 字符为基本单位，256 个字节覆盖了任何文本，所以任何输入都能编码、不会有未知词，解码也是无损的。
    4. 对话模板里的特殊 token，标记一条消息的开始和结束（以及角色）。模型是按这种格式训练的，推理时必须用同样的模板（`apply_chat_template`）拼接对话，否则效果明显变差。
    5. 一个 token 可能只是一个汉字 UTF-8 编码的一部分字节，单独解码会得到乱码；要增量反分词：把还不完整的字节缓存起来，等凑成完整的字符再输出。

## 字符、单词，还是子词

- **按字符**：词表小，但序列很长（一个英文单词要好几个 token），计算量和上下文占用都大；
- **按单词**：序列短，但词表巨大，而且新词、拼写错误、罕见词会变成"未知"；
- **按子词（subword）**：常见的词是一个 token，罕见的词拆成几个片段。这是折中方案，现代大模型几乎都用子词分词，最主流的算法是 **BPE（Byte Pair Encoding）**。

## BPE：从字节开始不断合并

BPE 的训练过程非常简单：

1. 初始词表是 256 个字节（**byte-level BPE**：任何文本都能表示成 UTF-8 字节序列，所以永远不会出现未知词）；
2. 统计语料中所有相邻 token 对的出现次数；
3. 把出现次数最多的一对合并成一个新 token，加入词表，记下这条**合并规则**；
4. 重复第 2、3 步，直到词表达到目标大小。

**编码**新文本时，先把文本切成字节，然后按合并规则**学习时的先后顺序**反复合并，直到没有可用的规则。

先在一个几十个词的小语料上把这个过程一步一步走一遍（为了看得清按字符合并，真实实现按字节）：

<div class="aig-widget" data-widget="bpe"></div>

下面从零实现一个 byte-level BPE：

```python title="bpe.py"
"""bpe.py —— 从零实现的 byte-level BPE 分词器（教学用）。"""

import re
from collections import Counter

# 预切分：按"可选的前导空格 + 连续的字母/数字"、"标点"、"空白"切开，合并不会跨越这些边界
PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+")


def merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class ByteBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}          # (左, 右) -> 新 token 的编号；编号越小越先合并
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        words = Counter(PATTERN.findall(text))                 # 相同的片段只统计一次，乘以频次
        seqs = {w: list(w.encode("utf-8")) for w in words}
        for new_id in range(256, vocab_size):
            counts = Counter()
            for w, freq in words.items():
                s = seqs[w]
                for pair in zip(s, s[1:]):
                    counts[pair] += freq
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))   # 次数最多，平局取编号小的，保证确定性
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w in words:
                seqs[w] = merge(seqs[w], best, new_id)

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in PATTERN.findall(text):
            seq = list(piece.encode("utf-8"))
            while len(seq) >= 2:
                # 在当前所有相邻对中，找最早学到的那条合并规则
                pair = min(zip(seq, seq[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                seq = merge(seq, pair, self.merges[pair])
            ids.extend(seq)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

在一小段中英文混合的语料上训练，看看它学到了什么：

```python
from bpe import ByteBPE

corpus = """
大语言模型的推理分为预填充和解码两个阶段。预填充阶段一次处理整个提示词，解码阶段每次生成一个词。
解码阶段需要读取全部模型权重和键值缓存，因此是访存瓶颈。推理优化的核心是减少访存、提高并行度。
The inference of large language models has two phases: prefill and decode. The prefill phase processes
the whole prompt at once, while the decode phase generates one token at a time. Decoding is memory bound,
so inference optimization focuses on reducing memory traffic and increasing parallelism.
""" * 20

bpe = ByteBPE()
bpe.train(corpus, vocab_size=256 + 150)
learned = [bpe.vocab[i].decode("utf-8", errors="replace") for i in range(256, 256 + 150)]
print("最早学到的合并:", learned[:12])
print("较晚学到的合并:", learned[-8:])

text = "推理优化的核心是减少访存。Decoding is memory bound."
ids = bpe.encode(text)
print(len(text.encode("utf-8")), "字节 ->", len(ids), "个 token")
assert bpe.decode(ids) == text                                   # 编码-解码必须无损
for unseen in ["从未见过的句子🙂", "tokenizer handles anything!"]:
    assert bpe.decode(bpe.encode(unseen)) == unseen               # 没见过的文本也能处理：退化成字节
```

最早学到的合并是英文里最常见的字母组合（如 `' p'`、`'in'`、`'re'`），以及常见汉字的 UTF-8 字节片段（一个汉字占 3 个字节，片段单独解码不是完整字符，所以打印成 �）；越往后，学到的片段越长，开始出现完整的英文单词和由多个字节拼成的汉字。没见过的内容（比如 emoji）会退化成单个字节，但依然能无损还原，这就是 byte-level 的好处。

真实的分词器（GPT-4 的 tiktoken、Qwen 的分词器）原理相同，只是在几 TB 的语料上训练出了十几万个 token，预切分的正则也更讲究（比如数字最多 3 位一组）。

## 真实的分词器

看看 Qwen3 的分词器怎么处理各种文本（它和 Qwen2.5 用的是同一套词表）：

```pycon
>>> from transformers import AutoTokenizer
>>> tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
>>> len(tok), tok.vocab_size
(151669, 151643)
>>> for s in ["Hello world!", "大语言模型推理优化", "def add(a, b): return a + b", "12345678"]:
...     ids = tok.encode(s)
...     print(len(ids), [tok.decode([i]) for i in ids])
3 ['Hello', ' world', '!']
5 ['大', '语言', '模型', '推理', '优化']
10 ['def', ' add', '(a', ',', ' b', '):', ' return', ' a', ' +', ' b']
8 ['1', '2', '3', '4', '5', '6', '7', '8']
```

几个值得注意的地方：

- 普通 token 有 151643 个，加上 26 个特殊 token 共 151669 个（比 Qwen2.5 多了 `<think>`、`</think>` 等几个）；而模型配置里的 `vocab_size` 是 151936，嵌入矩阵比实际词表大一些。补齐到 128、256 的倍数有利于矩阵乘法的效率，多出来的行不会被用到；
- 英文单词通常带着前面的空格成为一个 token（`' world'`）；
- 中文常用词被合并成一个 token，"大语言模型推理优化"只用了 5 个 token；
- Qwen 把数字拆成单个数字，这有利于模型学习算术。

`tok.convert_ids_to_tokens` 能看到 token 的原始形式。byte-level BPE 为了能把任意字节显示出来，把每个字节映射成了一个可打印字符，所以中文 token 的原始形式看起来是乱码，这是正常的：

```pycon
>>> tok.convert_ids_to_tokens(tok.encode("推理"))
['æİ¨çĲĨ']
```

!!! inference "推理视角"
    - **token 数就是成本**：计算量、KV Cache 大小、延迟都和 token 数成正比。同样的内容，分词效率越高（每个 token 覆盖的字符越多），推理越便宜。Qwen 的词表对中文做了大量优化，常用词基本都是一个 token；
    - **词表大小影响模型大小**：嵌入层和输出层都是 `vocab × d`，15 万的词表在 0.6B 的模型里占了四分之一的参数；
    - **分词器本身也要算**：在高并发的推理服务里，分词和反分词在 CPU 上执行，可能成为瓶颈。vLLM、SGLang 都把它们放在独立的进程里异步执行。

## 特殊 token 与对话模板

对话模型需要知道"哪里是系统提示、哪里是用户说的、哪里该轮到自己回答"。这些结构用**特殊 token** 标记。Qwen 使用的是 ChatML 格式：

```pycon
>>> msgs = [{"role": "system", "content": "你是一个助手。"}, {"role": "user", "content": "你好"}]
>>> text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
>>> print(text)
<|im_start|>system
你是一个助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>
<BLANKLINE>
</think>
>>> tok.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>", "<|endoftext|>"])
[151644, 151645, 151643]
>>> tok.eos_token
'<|im_end|>'
```

- `add_generation_prompt=True` 在末尾加上 `<|im_start|>assistant\n`，告诉模型"现在轮到你了"；
- Qwen3 是"混合思考"模型：默认先在 `<think>` 和 `</think>` 之间写推理过程，再给出回答。`enable_thinking=False` 会直接放进一对空的 `<think></think>`，让它跳过思考、直接回答——本书的例子为了输出短、可复现，都这样做。推理引擎要为此解析出思考内容（例如 OpenAI 接口里的 `reasoning_content`），见[推理系统手册的采样与 API](serving://engine/sampler-api/)；
- 模型生成 `<|im_end|>` 时表示回答结束，所以它就是对话模型的结束符（eos）；
- 模型在后训练阶段就是按这个格式学习的（见[后训练](../training/post-training.md)），**推理时的格式必须和训练时一模一样**，多一个空格、少一个换行都可能让效果变差。所以一定要用 `apply_chat_template`，不要自己拼字符串。

回到[上一章](language-model.md#看一眼真实模型的输出)的例子：加上对话模板后，模型就知道这是一个问题，而不是一道填空题：

```pycon
>>> import torch
>>> from transformers import AutoModelForCausalLM
>>> model = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B", dtype=torch.float32).eval()
>>> msgs = [{"role": "user", "content": "中国的首都是哪里？"}]
>>> ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), return_tensors="pt").input_ids
>>> with torch.no_grad():
...     probs = model(ids).logits[0, -1].softmax(-1)
>>> tok.decode(probs.argmax()), round(probs.max().item(), 3)
('中国的', 0.954)
```

模型准备以"中国的首都是北京"这样的句式作答，而且相当确定。

!!! inference "推理视角"
    对话模板意味着每个请求都以相同的系统提示和格式 token 开头。推理引擎的**前缀缓存**（prefix caching，见[推理服务](../inference/serving.md#kv-cache-管理与前缀缓存)）正是利用这一点：多个请求共享的前缀，只需要计算一次 KV Cache。

## 流式输出与增量反分词

流式输出（一个字一个字地显示）时，需要把每一步生成的 token 转换成文本。问题是：**一个 token 不一定对应完整的字符**。byte-level BPE 的 token 可能只是一个汉字的一部分字节：

```pycon
>>> ids = tok.encode("龘")
>>> ids, [tok.decode([i]) for i in ids]
([82912, 246], ['�', '�'])
>>> tok.decode(ids)
'龘'
```

"龘"不在常用词表里，被拆成了两个字节级 token，单独解码都是无效的 UTF-8（显示为替换字符 �），合在一起才是完整的字。所以流式输出不能"每个 token 单独解码再拼接"，正确的做法是**增量反分词**：维护已生成的全部 token，每次解码最近的一段，只有当新增的文本不以不完整的字节结尾时才输出。vLLM 的 detokenizer 就是这样实现的。

!!! interview "面试怎么答"
    被问"分词对推理有什么影响"：先讲 byte-level BPE——从字节开始按频次合并，任何文本都能无损编码、没有未知词；再讲成本——token 数决定计算量和 KV Cache，同样的内容换一种语言或写成代码，token 数可以差出几倍，词表大小决定嵌入层和输出层有多大；最后讲工程上的坑：对话模板必须用 `apply_chat_template` 与训练保持一致，流式输出时一个 token 可能只是半个汉字，要做增量反分词。

## 练习

**1. 增量反分词。** 写一个函数 `stream_decode(tok, ids)`，模拟流式输出：逐个加入 token，每次只输出新增的、已经完整的文本片段。用"龘龘你好"验证拼接结果等于整体解码的结果，并且没有输出过替换字符。

??? success "参考答案"
    ```python
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    def stream_decode(tok, ids):
        pieces, emitted = [], ""
        for n in range(1, len(ids) + 1):
            text = tok.decode(ids[:n])
            if text.endswith("�"):        # 结尾是不完整的字节，先不输出
                continue
            pieces.append(text[len(emitted):])
            emitted = text
        return pieces

    ids = tok.encode("龘龘你好")
    pieces = stream_decode(tok, ids)
    assert "".join(pieces) == tok.decode(ids) == "龘龘你好"
    assert all("�" not in p for p in pieces)
    print(pieces)
    ```

    这个版本每次都从头解码，复杂度是 O(n²)。真实的实现只解码最近的几个 token（维护一个"前缀偏移"和"读偏移"），开销是常数。

**2. 估算题。** 一个服务每天处理 1 亿个请求，每个请求的系统提示和对话模板共 300 个 token。如果这些前缀的 KV Cache 能被完全复用，每天可以省下多少 token 的 prefill 计算？

??? success "参考答案"
    1 亿 × 300 = 300 亿个 token 的 prefill 计算。按 7B 模型每个 token 约 2 × 7e9 = 1.4e10 次运算计算，约 4.2e20 次运算，相当于一张 H100（约 1e15 次/秒的 BF16 算力，按 50% 利用率计算）连续算大约 10 天。这就是前缀缓存在实际服务中价值巨大的原因。

## 小结

- [x] 现代大模型用子词分词，主流算法是 byte-level BPE：从字节开始，按频次不断合并。
- [x] 编码时按合并规则的学习顺序应用；byte-level 保证任何文本都能无损编码。
- [x] token 数决定计算量和 KV Cache 大小；词表大小决定嵌入层和输出层的大小。
- [x] 对话模型依赖特殊 token 组成的对话模板，推理时必须用 `apply_chat_template` 保持与训练一致。
- [x] 一个 token 可能只是半个字符，流式输出需要增量反分词。
