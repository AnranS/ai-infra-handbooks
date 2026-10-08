import re
import shutil
import urllib.request
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

URL = "https://anrans.github.io/ai-infra-handbooks/scratch/assets/data/sanguo.txt"
corpus = Path("sanguo.txt")
if not corpus.exists():                                       # 在仓库里运行：复制本地副本；否则从手册站点下载（约 1.8 MB）
    local = next((p / "scratch/docs/assets/data/sanguo.txt" for p in [Path.cwd(), *Path.cwd().parents]
                  if (p / "scratch/docs/assets/data/sanguo.txt").exists()), None)
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
