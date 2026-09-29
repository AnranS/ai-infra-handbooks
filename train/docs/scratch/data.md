# 从零训练（一）：语料与分词器

<p class="lead">接下来三章从一个空目录开始，在自己的电脑上训练一个会写《三国演义》的小语言模型：准备语料、训练分词器、写模型和训练循环、评估与生成，最后把它训得更快、更大。每一步都是大模型预训练的缩影——真实的预训练多了几个数量级的数据和算力，流程却是同一个。全部代码在 CPU 上就能跑完，训练一次约 3 分钟。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么不直接按字（或按字节）切分，而要训练一个 BPE 分词器？词表大小怎么选？
    2. 训练分词器时为什么要做"预切分"？
    3. 验证集为什么按章节切，而不是随机抽取片段？
    4. 真实的预训练语料要经过哪些处理？

## 全景

| 步骤 | 本教程 | 真实的预训练 |
| --- | --- | --- |
| 语料 | 《三国演义》，约 60 万字、1.8 MB | 网页、书籍、代码、论文等，十几万亿 token，经过去重、质量过滤、配比 |
| 分词器 | BPE，词表 8192 | BPE，词表 10 万～25 万，覆盖多语言和代码 |
| 模型 | 4 层、1.8M 参数的 GPT | 几十到上百层、数十亿到上万亿参数 |
| 训练 | 一台笔记本的 CPU，3 分钟 | 数千张 GPU，数周到数月 |
| 评估 | 验证集 loss、看生成的文字 | 验证集 loss、几十个下游评测集 |

这一章做前两步：把语料变成 token 序列。

## 语料

语料取自 Project Gutenberg 的《三国志演义》（#23950，公有领域）：去掉了版权页、合并了排版造成的断行、用 OpenCC 转成简体，并修正了原文排版里的两处错字和一处和上一段粘在一起的回目。整理好的文本放在手册的站点上，下面的脚本会自动下载（在仓库里运行时直接读本地副本）。

真实的预训练语料处理要复杂得多，决定模型质量的往往是这一步（见大模型手册的[预训练](llm://training/pretraining/)）：

- **去重**：完全相同的文档用哈希去掉，近似重复的用 MinHash；重复的数据会被模型"背下来"，浪费算力、增加泄露隐私的风险；
- **质量过滤**：规则（长度、符号比例、重复行）加上分类器（用高质量语料训练的打分模型），去掉广告、乱码、模板页；
- **配比**：网页、代码、数学、书籍等各占多少，训练的不同阶段配比不同（后期提高高质量数据的比例）；
- **去污染**：从训练集里去掉评测集的题目，否则评测分数没有意义。

## 分词器：BPE

模型只认识整数。最简单的做法是按字切分：每个汉字一个 token，词表只要四千多个。问题是序列太长——同样的上下文长度装下的文字少，注意力的计算量随长度平方增长。按字节切分更糟，一个汉字是 3 个字节。

**BPE**（字节对编码）从单个字符出发，反复把语料里最常相邻出现的一对片段合并成一个新 token，直到词表达到指定大小（原理和手写实现见大模型手册的[分词](llm://basics/tokenization/)）。这里用 Hugging Face 的 `tokenizers` 库训练，它是用 Rust 写的，几秒钟就能训完：

```python title="prepare.py"
import re
import shutil
import urllib.request
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

URL = "https://anrans.github.io/ai-infra-handbooks/train/assets/data/sanguo.txt"
corpus = Path("sanguo.txt")
if not corpus.exists():                                       # 在仓库里运行：复制本地副本；否则从手册站点下载（约 1.8 MB）
    local = next((p / "train/docs/assets/data/sanguo.txt" for p in [Path.cwd(), *Path.cwd().parents]
                  if (p / "train/docs/assets/data/sanguo.txt").exists()), None)
    shutil.copy(local, corpus) if local else urllib.request.urlretrieve(URL, corpus)
text = corpus.read_text(encoding="utf-8")
chapters = re.split(r"(?=^第.{1,4}回：)", text, flags=re.M)[1:]
print(f"语料：{len(text):,} 个字符，{len(set(text)):,} 种不同的字符，{len(chapters)} 回")

# ① BPE 分词器：从单个字符出发，反复合并最常一起出现的相邻片段
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Sequence([                 # 合并不跨过换行和标点：词不会粘上标点
    pre_tokenizers.Split("\n", behavior="isolated"), pre_tokenizers.Punctuation(behavior="isolated")])
tok.decoder = decoders.Fuse()                                 # 解码时把片段直接拼起来（中文没有空格）
trainer = trainers.BpeTrainer(vocab_size=8192, special_tokens=["<unk>", "<eos>"], show_progress=False)
tok.train_from_iterator(chapters, trainer)
tok.save("tokenizer.json")
print(f"词表：{tok.get_vocab_size()} 个 token")

sample = "玄德曰：「吾乃中山靖王之后，孝景皇帝阁下玄孙。」"
enc = tok.encode(sample)
print("分词示例：", " / ".join(enc.tokens))
print("解码回来一致：", tok.decode(enc.ids) == sample)
vocab = tok.get_vocab()
longest = sorted((t for t in vocab if not t.startswith("<")), key=lambda t: (-len(t), vocab[t]))[:12]
print("最长的几个 token：", "、".join(longest))

# ② 整本书编码成 token 序列；最后 12 回做验证集（按回切，避免验证集和训练集是相邻的句子）
eos = tok.token_to_id("<eos>")
def encode(chs):
    ids = []
    for ch in chs:
        ids += tok.encode(ch).ids + [eos]                     # 每一回结尾加一个 <eos>
    return torch.tensor(ids, dtype=torch.int16)               # 词表 8192 < 32768，int16 就够
train, val = encode(chapters[:-12]), encode(chapters[-12:])
torch.save({"train": train, "val": val}, "tokens.pt")
n_chars = sum(len(c) for c in chapters)
print(f"训练集 {len(train):,} 个 token，验证集 {len(val):,} 个；平均每个 token {n_chars / (len(train) + len(val)):.2f} 个字")
```

```text title="输出"
语料：597,228 个字符，4,035 种不同的字符，120 回
词表：8192 个 token
分词示例： 玄德曰 / ： / 「 / 吾乃 / 中山 / 靖 / 王 / 之后 / ， / 孝 / 景 / 皇帝 / 阁 / 下 / 玄 / 孙 / 。 / 」
解码回来一致： True
最长的几个 token： 且看下文分解、后人有诗叹曰、后人有诗赞曰、未知胜负如何、且听下文分解、分付如此如此、后人有诗赞、后人有诗曰、却说司马懿、常山赵子龙、早有人报知、为前部先锋
训练集 410,934 个 token，验证集 40,717 个；平均每个 token 1.32 个字
```

几个值得看的地方：

- **最长的 token 是这本书的套话**："且看下文分解""后人有诗叹曰""常山赵子龙"——BPE 只看频率，出现得足够多的片段都会被合并成一个 token。真实的分词器用海量、多样的语料训练，学到的是通用的词和词缀；
- **预切分**：合并不跨过换行和标点，否则会学出"曰：「"这种把标点粘进去的 token，浪费词表。GPT 系列的分词器还会在空格、数字、字母的边界上预切分；
- **压缩率**：平均每个 token 1.32 个字，同样 128 个 token 的上下文能装下约 170 个字；
- **词表大小是个权衡**：词表越大，序列越短，但嵌入层和输出层越大。我们的模型只有 128 维，8192 × 128 的词嵌入就有 1.05M 参数，占了整个模型（1.8M）的一半以上——小模型常常就是这样，所以要让输入和输出共享同一个词嵌入矩阵（下一章）；
- **`<eos>`**：每一回的结尾加一个特殊 token，模型能学到"一回在这里结束"，生成时也可以用它判断停止。

## 训练集与验证集

验证集是用来判断模型"学会了"还是"背下来了"的：训练集 loss 一直下降，验证集 loss 却停下来或上升，就是过拟合。验证集必须和训练集**真正分开**：如果随机抽取片段，相邻的两段文字（甚至重叠的片段）会分别落进训练集和验证集，验证集 loss 就偏乐观。这里按章节切：前 108 回训练，最后 12 回验证——后面的情节模型从没见过，但人物和文风是相通的。

token 序列用 `int16` 存（词表 8192 小于 32768），40 多万个 token 只占不到 1 MB。真实的训练把几万亿个 token 存成很多个二进制分片，用内存映射按需读取，数据加载器要能精确地记录"读到了哪里"，这样中断之后才能接着训练（下一章的续训会用到同样的思路）。

!!! interview "面试怎么答"
    被问"从零训练一个模型，数据怎么准备"：流程是语料 → 分词器 → token 序列 → 模型 → 训练循环 → 评估，而决定质量的常常是语料处理（去重、过滤、配比、去污染）。分词器用 BPE：从字符（或字节）出发反复合并最常见的相邻片段，预切分防止合并跨过标点和换行；词表大小在序列长度和嵌入层大小之间权衡（小模型里嵌入层可能占一半参数）。验证集要和训练集真正分开（按章节、按文档），随机抽片段会让验证 loss 偏乐观；token 序列存成紧凑的二进制，便于按需读取和续训。

## 练习

**1. 词表大小的影响。** 把 `vocab_size` 改成 4096 和 16384，平均每个 token 多少个字？模型的词嵌入参数各是多少？

??? success "参考思路"
    词表越大，能合并的片段越多，每个 token 的字数越多（序列越短），但增长越来越慢：高频的词先被合并，后面的合并只覆盖越来越少的出现次数。词嵌入的参数量是 `vocab_size × d_model`，在 `d_model = 128` 时分别是 0.5M、1.0M、2.1M。对一个总共不到 2M 参数的模型，16384 的词表会让大部分参数都花在词嵌入上，而大部分 token 在训练集里只出现过几次，学不好。真实的大模型里，词嵌入只占总参数的很小一部分，所以词表可以做到十几万。

**2. 为什么不用随机切分？** 如果把整本书切成 128 个 token 的片段，随机抽 10% 做验证集，验证集 loss 会比按章节切更低还是更高？为什么？

??? success "参考答案"
    会更低（更乐观）。随机抽取的验证片段，前后相邻的文字都在训练集里，模型见过同样的人物、同样的情节甚至同样的句子，预测起来容易得多，这个 loss 并不能反映模型对"没见过的文字"的能力。按章节切更接近真实的使用场景：模型要续写的是新的情节。真实的预训练也要注意这个问题（去重、去污染），否则评测结果会虚高。

## 小结

- [x] 预训练的流程：语料 → 分词器 → token 序列 → 模型 → 训练循环 → 评估；决定质量的常常是语料的处理（去重、过滤、配比、去污染）。
- [x] BPE 从字符出发反复合并最常见的相邻片段；预切分防止合并跨过标点和换行；词表大小在序列长度与嵌入层大小之间权衡。
- [x] 验证集要和训练集真正分开（按章节、按文档），否则 loss 偏乐观；token 序列存成紧凑的二进制，便于按需读取和续训。
