# From scratch (1): the corpus and the tokenizer

<p class="lead">The next three chapters start from an empty directory and train a small language model on your own computer that writes in the style of <em>Romance of the Three Kingdoms</em>: preparing the corpus, training a tokenizer, writing the model and the training loop, evaluating and generating, and finally training it faster and larger. Every step is a miniature of large-model pretraining; real pretraining has several orders of magnitude more data and compute, but the procedure is the same. All of the code runs to completion on a CPU, and one training run takes about 3 minutes.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why not split by character (or by byte) directly, rather than training a BPE tokenizer? How is the vocabulary size chosen?
    2. Why does training a tokenizer need pre-tokenization?
    3. Why is the validation set split by chapter rather than sampled at random?
    4. What processing does a real pretraining corpus go through?

??? success "Answers for the self-test (answer first, then open this)"
    1. Splitting by character or byte makes the sequence too long and each token carries little information, so both training and inference cost more; BPE merges common fragments into one token and shortens the sequence. The vocabulary size trades the sequence length against the size of the embedding and output layers: in a small model the embedding easily takes half the parameters.
    2. To stop merges crossing newlines and punctuation, which would otherwise produce tokens with punctuation glued on and waste the vocabulary.
    3. The text on either side of a randomly sampled fragment is in the training set (and may even overlap it), so the model has seen the same characters and sentences and the validation loss is optimistic; split by chapter, the validation set holds events the model has genuinely never seen, which is closer to real use.
    4. Deduplication (exact and near), quality filtering (rules, classifiers), removing harmful content and personal information, language identification, proportioning the sources, and decontamination (removing anything overlapping the evaluation sets).

## The whole picture {#全景}

| Step | This tutorial | Real pretraining |
| --- | --- | --- |
| The corpus | *Romance of the Three Kingdoms*, about 600,000 characters, 1.8 MB | web pages, books, code, papers and more, tens of trillions of tokens, deduplicated, quality-filtered and proportioned |
| The tokenizer | BPE with a vocabulary of 8192 | BPE with a vocabulary of 100,000 to 250,000, covering many languages and code |
| The model | a 4-layer GPT with 1.8M parameters | tens to hundreds of layers, billions to trillions of parameters |
| Training | a laptop's CPU, 3 minutes | thousands of GPUs, weeks to months |
| Evaluation | the validation loss, and reading what it writes | the validation loss and dozens of downstream evaluation sets |

This chapter does the first two steps: turning the corpus into a token sequence.

## The corpus {#语料}

The corpus is *Romance of the Three Kingdoms* from Project Gutenberg (#23950, public domain): the copyright page removed, lines rejoined where the typesetting had broken them, converted to simplified characters with OpenCC, with two typographical errors and one chapter heading that had run into the previous paragraph corrected. The tidied text lives on this handbook's site and the script below downloads it automatically (it reads the local copy when run inside the repository).

Processing a real pretraining corpus is far more involved, and this step is often what decides the model's quality (see [Pretraining](llm://training/pretraining/) in the large-model handbook):

- **Deduplication**: identical documents removed by hash, near-duplicates by MinHash; repeated data gets memorised by the model, wasting compute and increasing the risk of leaking private information.
- **Quality filtering**: rules (length, the proportion of symbols, repeated lines) plus a classifier (a scoring model trained on high-quality text), removing advertisements, mojibake and template pages.
- **Proportioning**: how much of the mix is web pages, code, mathematics and books, with different proportions at different stages of training (raising the share of high-quality data later on).
- **Decontamination**: removing the evaluation sets' questions from the training data, without which the evaluation scores mean nothing.

## The tokenizer: BPE {#分词器bpe}

Walk BPE's merges through step by step on a small corpus first, with the tool from the large-model handbook:

<div class="aig-widget" data-widget="bpe"></div>

A model only understands integers. The simplest approach is to split by character: one token per character, with a vocabulary of a little over four thousand. The problem is that the sequence gets too long, so the same context length holds less text, and attention's computation grows with the square of the length. Splitting by byte is worse, since a Chinese character is 3 bytes.

**BPE** (byte-pair encoding) starts from individual characters and repeatedly merges the pair of fragments that occur adjacently most often into a new token, until the vocabulary reaches the given size (the principle and a hand-written implementation are in [Tokenization](llm://basics/tokenization/) in the large-model handbook). Here we train it with Hugging Face's `tokenizers` library, written in Rust, which finishes in a few seconds:

```python title="prepare.py"
import re
import shutil
import urllib.request
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

URL = "https://anrans.github.io/ai-infra-handbooks/scratch/assets/data/sanguo.txt"
corpus = Path("sanguo.txt")
if not corpus.exists():                                       # run inside the repository: copy the local file; otherwise download from the handbook's site (about 1.8 MB)
    local = next((p / "scratch/docs/assets/data/sanguo.txt" for p in [Path.cwd(), *Path.cwd().parents]
                  if (p / "scratch/docs/assets/data/sanguo.txt").exists()), None)
    shutil.copy(local, corpus) if local else urllib.request.urlretrieve(URL, corpus)
text = corpus.read_text(encoding="utf-8")
chapters = re.split(r"(?=^第.{1,4}回：)", text, flags=re.M)[1:]
print(f"语料：{len(text):,} 个字符，{len(set(text)):,} 种不同的字符，{len(chapters)} 回")

# 1. the BPE tokenizer: start from individual characters and repeatedly merge the adjacent pair that occurs together most often
tok = Tokenizer(models.BPE(unk_token="<unk>"))
tok.pre_tokenizer = pre_tokenizers.Sequence([                 # merges do not cross newlines or punctuation, so a word never has punctuation glued on
    pre_tokenizers.Split("\n", behavior="isolated"), pre_tokenizers.Punctuation(behavior="isolated")])
tok.decoder = decoders.Fuse()                                 # the decoder joins the fragments directly (Chinese has no spaces)
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

# 2. encode the whole book as a token sequence; the last 12 chapters are the validation set (split by chapter, so the validation and training sets are not adjacent sentences)
eos = tok.token_to_id("<eos>")
def encode(chs):
    ids = []
    for ch in chs:
        ids += tok.encode(ch).ids + [eos]                     # add an <eos> at the end of each chapter
    return torch.tensor(ids, dtype=torch.int16)               # a vocabulary of 8192 is below 32768, so int16 is enough
train, val = encode(chapters[:-12]), encode(chapters[-12:])
torch.save({"train": train, "val": val}, "tokens.pt")
n_chars = sum(len(c) for c in chapters)
print(f"训练集 {len(train):,} 个 token，验证集 {len(val):,} 个；平均每个 token {n_chars / (len(train) + len(val)):.2f} 个字")
```

```text title="output"
语料：597,228 个字符，4,035 种不同的字符，120 回
词表：8192 个 token
分词示例： 玄德曰 / ： / 「 / 吾乃 / 中山 / 靖 / 王 / 之后 / ， / 孝 / 景 / 皇帝 / 阁 / 下 / 玄 / 孙 / 。 / 」
解码回来一致： True
最长的几个 token： 且看下文分解、后人有诗叹曰、后人有诗赞曰、未知胜负如何、且听下文分解、分付如此如此、后人有诗赞、后人有诗曰、却说司马懿、常山赵子龙、早有人报知、为前部先锋
训练集 410,934 个 token，验证集 40,717 个；平均每个 token 1.32 个字
```

A few things worth looking at:

- **The longest tokens are the book's stock phrases**: "and what happened next will be told in the following chapter", "a later poet wrote in praise", "Zhao Zilong of Changshan". BPE only looks at frequency, and any fragment that occurs often enough gets merged into one token. A real tokenizer is trained on a vast and varied corpus and learns general words and affixes.
- **Pre-tokenization**: merges do not cross newlines or punctuation, which would otherwise produce tokens with punctuation glued on and waste the vocabulary. The GPT family's tokenizers also pre-tokenize at the boundaries between spaces, digits and letters.
- **The compression ratio**: 1.32 characters per token on average, so a context of 128 tokens holds about 170 characters.
- **The vocabulary size is a trade-off**: a larger vocabulary means shorter sequences and larger embedding and output layers. Our model has only 128 dimensions, so an 8192 x 128 embedding is 1.05M parameters, over half of the whole model's 1.8M. Small models are often like this, which is why the input and output share one embedding matrix (the next chapter).
- **`<eos>`**: a special token at the end of each chapter lets the model learn that a chapter ends here, and generation can use it to decide when to stop.

## The training and validation sets {#训练集与验证集}

The validation set is how you tell whether the model has learned or memorised: the training loss keeps falling while the validation loss stalls or rises, which is overfitting. The validation set has to be **genuinely separate** from the training set: sampling fragments at random puts two adjacent (or even overlapping) pieces of text into the training and validation sets respectively, and the validation loss comes out optimistic. Here it is split by chapter: the first 108 chapters train and the last 12 validate, so the model has never seen the later events while the characters and the style carry over.

The token sequence is stored as `int16` (a vocabulary of 8192 is below 32768), so over 400,000 tokens take under 1 MB. Real training stores trillions of tokens in many binary shards read on demand through memory mapping, and the data loader has to record exactly where it has read to, so that training can resume after an interruption (the next chapter's resumption uses the same idea).

!!! interview "How to explain it"
    To explain how the data is prepared to train a model from scratch: the procedure is corpus, tokenizer, token sequence, model, training loop, evaluation, and what decides the quality is usually the corpus processing (deduplication, filtering, proportioning, decontamination). The tokenizer is BPE: start from characters (or bytes) and repeatedly merge the commonest adjacent pair, with pre-tokenization stopping merges from crossing punctuation and newlines; the vocabulary size trades the sequence length against the embedding's size (in a small model the embedding may be half the parameters). The validation set has to be genuinely separate from the training set (by chapter, by document), since sampling fragments at random makes the validation loss optimistic; the token sequence is stored as compact binary for reading on demand and for resuming.

## Exercises {#练习}

**1. The vocabulary size's effect.** Change `vocab_size` to 4096 and 16384. How many characters per token on average? How many parameters does the model's embedding have in each case?

??? success "The approach"
    The larger the vocabulary, the more fragments can be merged and the more characters per token (the shorter the sequence), but the growth slows: the high-frequency words are merged first and later merges cover fewer and fewer occurrences. The embedding's parameter count is `vocab_size x d_model`, which at `d_model = 128` is 0.5M, 1.0M and 2.1M respectively. For a model of under 2M parameters in total, a vocabulary of 16384 puts most of the parameters in the embedding, while most of those tokens occur only a few times in the training set and are not learned well. In a real large model the embedding is a very small share of the total, which is why the vocabulary can run to a hundred thousand or more.

**2. Why not split at random?** If you cut the whole book into 128-token fragments and sampled 10% at random as the validation set, would the validation loss be lower or higher than splitting by chapter? Why?

??? success "Answer"
    Lower (more optimistic). The text on either side of a randomly sampled validation fragment is in the training set, so the model has seen the same characters, the same events and even the same sentences, which makes the prediction far easier; that loss does not reflect the model's ability on text it has not seen. Splitting by chapter is closer to real use: the model has to continue a new part of the story. Real pretraining has to watch for the same problem (deduplication, decontamination), or the evaluation scores come out inflated.

## Summary {#小结}

- [x] The pretraining procedure: corpus, tokenizer, token sequence, model, training loop, evaluation; what decides the quality is usually the corpus processing (deduplication, filtering, proportioning, decontamination).
- [x] BPE starts from characters and repeatedly merges the commonest adjacent pair; pre-tokenization stops merges from crossing punctuation and newlines; the vocabulary size trades the sequence length against the embedding's size.
- [x] The validation set has to be genuinely separate from the training set (by chapter, by document) or the loss is optimistic; the token sequence is stored as compact binary for reading on demand and for resuming.
