# Tokenization: how text becomes numbers

<p class="lead">A model only understands integers. The tokenizer splits text into tokens and maps them to ids, and after generation turns the ids back into text. It looks like mere preprocessing, yet it decides how long the sequence is (and so how much compute and KV cache it needs), how big the embedding and output layers are, and how efficient the model is across languages, and it brings a whole series of engineering details such as streaming output and chat templates.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. Why do large models split text neither into characters nor into words?
    2. How is BPE trained? How are the learned merge rules applied when encoding new text?
    3. What is byte-level BPE? Why does it never produce "unknown words"?
    4. What are `<|im_start|>` and `<|im_end|>`? What is a chat template for?
    5. In streaming output, why can't you decode each token on its own and concatenate the results?
    6. Which tables does a real tokenizer have inside? What are the keys and values of the merge-rule table?

??? success "Answers (try first, then expand to compare)"
    1. Characters make sequences too long (high compute and KV cost) and each character carries little information; words make the vocabulary explode, and there are always words never seen before. Subwords are the compromise: a common word is one token, and a rare word is split into a few pieces.
    2. Training: start from bytes (or characters), count how often adjacent pieces occur, and repeatedly merge the most common pair into a new token until the vocabulary reaches its target size, recording the order of the merges. Encoding: pre-tokenize first, then apply the merge rules in the order they were learned.
    3. It uses bytes rather than Unicode characters as the basic unit. The 256 bytes cover any text, so any input can be encoded, there are no unknown words, and decoding is lossless.
    4. They are special tokens of the chat template that mark where a message starts and ends (and its role). The model was trained on this format, so at inference time conversations must be put together with the same template (`apply_chat_template`), or quality drops noticeably.
    5. A token may be only some of the bytes of a Chinese character's UTF-8 encoding, and decoding it alone gives garbage; use incremental detokenization: hold back incomplete bytes and output them once they form a complete character.
    6. Three hash tables: the vocabulary (piece → id), the reverse vocabulary (id → piece, for decoding) and the merge-rule table (the key is the ids of two adjacent pieces, the value is the rule's rank and the merged id). Merging uses a min-heap to take the next pair by rank and a doubly linked list to join two pieces, and results for common pieces come straight from a cache.

## Characters, words, or subwords {#字符单词还是子词}

- **By character**: a small vocabulary, but very long sequences (an English word takes several tokens), so both compute and context use are large;
- **By word**: short sequences, but a huge vocabulary, and new words, typos and rare words become "unknown";
- **By subword**: a common word is one token, and a rare word is split into a few pieces. This is the compromise, and nearly all modern large models use subword tokenization, most commonly the **BPE (Byte Pair Encoding)** algorithm.

## BPE: start from bytes and keep merging {#bpe从字节开始不断合并}

Training BPE is very simple:

1. The initial vocabulary is the 256 bytes (**byte-level BPE**: any text can be written as a sequence of UTF-8 bytes, so there are never unknown words);
2. Count how often every pair of adjacent tokens occurs in the corpus;
3. Merge the most frequent pair into a new token, add it to the vocabulary, and record this **merge rule**;
4. Repeat steps 2 and 3 until the vocabulary reaches its target size.

To **encode** new text, split it into bytes and keep merging following the merge rules **in the order they were learned**, until no rule applies.

First walk through the process step by step on a small corpus of a few dozen words (merging characters to keep it readable; the real implementation works on bytes):

<div class="aig-widget" data-widget="bpe"></div>

Below is a byte-level BPE implemented from scratch:

```python title="bpe.py"
"""bpe.py —— 从零实现的 byte-level BPE 分词器（教学用）。"""

import re
from collections import Counter

# pre-tokenize: split into "optional leading space + run of letters/digits", "punctuation" and "whitespace"; merges never cross these boundaries
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
        self.merges: dict[tuple[int, int], int] = {}          # (left, right) -> id of the new token; smaller ids merge first
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        words = Counter(PATTERN.findall(text))                 # count each distinct piece once, weighted by its frequency
        seqs = {w: list(w.encode("utf-8")) for w in words}
        for new_id in range(256, vocab_size):
            counts = Counter()
            for w, freq in words.items():
                s = seqs[w]
                for pair in zip(s, s[1:]):
                    counts[pair] += freq
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))   # most frequent; ties go to the smaller ids, so the result is deterministic
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w in words:
                seqs[w] = merge(seqs[w], best, new_id)

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in PATTERN.findall(text):
            seq = list(piece.encode("utf-8"))
            while len(seq) >= 2:
                # among all current adjacent pairs, find the merge rule learned earliest
                pair = min(zip(seq, seq[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                seq = merge(seq, pair, self.merges[pair])
            ids.extend(seq)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

Train it on a small corpus mixing Chinese and English, and see what it learns:

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
assert bpe.decode(ids) == text                                   # encode-decode must be lossless
for unseen in ["从未见过的句子🙂", "tokenizer handles anything!"]:
    assert bpe.decode(bpe.encode(unseen)) == unseen               # unseen text still works: it falls back to bytes
```

The earliest merges are the most common letter combinations in English (such as `' p'`, `'in'`, `'re'`) and UTF-8 byte fragments of common Chinese characters (a Chinese character takes 3 bytes, and a fragment on its own is not a complete character, so it prints as �); the later the merge, the longer the piece, and complete English words and Chinese characters assembled from several bytes start to appear. Content never seen (emoji, say) falls back to single bytes, yet can still be restored losslessly: that is the benefit of working at the byte level.

Real tokenizers (GPT-4's tiktoken, Qwen's tokenizer) work the same way, only trained on terabytes of text into well over a hundred thousand tokens, with more careful pre-tokenization regexes (digits in groups of at most 3, for example).

## A real tokenizer {#真实的分词器}

See how Qwen3's tokenizer handles various kinds of text (it uses the same vocabulary as Qwen2.5):

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

A few things worth noting:

- There are 151643 regular tokens, plus 26 special tokens for a total of 151669 (a few more than Qwen2.5, such as `<think>` and `</think>`); the `vocab_size` in the model config is 151936, so the embedding matrix is a bit larger than the actual vocabulary. Padding to a multiple of 128 or 256 helps matrix multiplication efficiency, and the extra rows are never used;
- An English word usually becomes one token together with the space before it (`' world'`);
- Common Chinese words are merged into single tokens: "大语言模型推理优化" ("large language model inference optimization") takes only 5 tokens;
- Qwen splits numbers into single digits, which helps the model learn arithmetic.

`tok.convert_ids_to_tokens` shows a token's raw form. To be able to display arbitrary bytes, byte-level BPE maps each byte to a printable character, so the raw form of a Chinese token looks like garbage. That is normal:

```pycon
>>> tok.convert_ids_to_tokens(tok.encode("推理"))
['æİ¨çĲĨ']
```

!!! inference "Inference view"
    - **Token count is cost**: compute, KV cache size and latency are all proportional to the number of tokens. For the same content, the more efficient the tokenization (the more characters each token covers), the cheaper the inference. Qwen's vocabulary is heavily optimized for Chinese, and common words are almost all single tokens;
    - **Vocabulary size affects model size**: the embedding and output layers are both `vocab × d`, and a 150k vocabulary takes a quarter of the parameters of the 0.6B model;
    - **The tokenizer itself costs time**: when this book was written it measured about 1 microsecond per token single-threaded on a development machine (4797 characters into 2754 tokens in 2.9 milliseconds), so a long prompt of 100k tokens takes about 0.1 seconds. In a high-concurrency inference service, tokenization and detokenization run on the CPU and can become a bottleneck. vLLM and SGLang both run them asynchronously in separate processes.

## Inside the tokenizer: which tables it looks up {#分词器内部查的是哪几张表}

Turning text into token ids takes four steps. Walk through them with a sentence mixing Chinese and English:

```pycon
>>> s = "我想学推理 hello"
>>> b = s.encode("utf-8")
>>> len(b), list(b[:3]), list(b[-6:])     # a Chinese character is 3 bytes; space is 32, hello is 104 101 108 108 111
(21, [230, 136, 145], [32, 104, 101, 108, 108, 111])
>>> [p for p, _ in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(s)]   # pre-tokenized, bytes already mapped to stand-in characters
['æĪĳæĥ³åŃ¦æİ¨çĲĨ', 'Ġhello']
>>> tok.convert_ids_to_tokens(tok.encode(s))
['æĪĳæĥ³', 'åŃ¦', 'æİ¨çĲĨ', 'Ġhello']
>>> tok.encode(s)
[104100, 47764, 113272, 23811]
```

1. **Convert to bytes**: in UTF-8, this sentence is 21 bytes. Byte-level BPE gives each of the 256 byte values a printable "stand-in character", so that every byte can be handled as one character of a string: the stand-in for a space is `Ġ`, and the stand-ins of the Chinese bytes strung together are the garbage above.
2. **Pre-tokenize**: a regex splits the text into segments (words, "space + word", numbers and punctuation each form a segment), and merges never cross a segment boundary. This sentence splits into two segments.
3. **Merge by the merge rules**: each segment starts as single bytes, and repeatedly the rule with the best rank among the adjacent pairs is applied, until no rule applies.
4. **Get the ids**: when merging ends, every piece is a token in the vocabulary.

The merge rules and the vocabulary are stored in `tokenizer.json` in the model directory. Read them out and replay step 3 by hand:

```python
import json

model = json.load(open("models/Qwen3-0.6B/tokenizer.json", encoding="utf-8"))["model"]
vocab = model["vocab"]                                         # piece -> id
rank = {tuple(p): r for r, p in enumerate(model["merges"])}    # (left, right) -> rank; smaller merges first
print(len(vocab), "个片段，", len(rank), "条合并规则")

parts = list("Ġhello")                                         # start from single bytes (their stand-in characters)
print("开始：", [(p, vocab[p]) for p in parts])
while True:
    pairs = [(rank[(a, c)], k) for k, (a, c) in enumerate(zip(parts, parts[1:])) if (a, c) in rank]
    if not pairs:
        break
    r, k = min(pairs)                                          # the pair with the best rank
    parts[k:k + 2] = [parts[k] + parts[k + 1]]
    print(f"排名 {r:5d}：", [(p, vocab[p]) for p in parts])
```

```text title="输出"
151643 个片段， 151387 条合并规则
开始： [('Ġ', 220), ('h', 71), ('e', 68), ('l', 75), ('l', 75), ('o', 78)]
排名    45： [('Ġ', 220), ('h', 71), ('el', 301), ('l', 75), ('o', 78)]
排名    49： [('Ġh', 305), ('el', 301), ('l', 75), ('o', 78)]
排名   129： [('Ġh', 305), ('el', 301), ('lo', 385)]
排名  4535： [('Ġh', 305), ('ello', 4791)]
排名 23555： [('Ġhello', 23811)]
```

The 6 bytes are merged 5 times into one token, matching the 23811 the tokenizer gives. The rank is the order in which the rule was learned in training: e + l is rule 45, while the whole `Ġhello` only appears at rule 23555.

![Figure: the whole path from text through tokenization to embedding, and how ␣hello is merged](../assets/figures/bpe-lookup.svg){.aig-svg}

### How the real library does it {#真实的库怎么实现}

Behind the "fast tokenizer" of transformers is Hugging Face's tokenizers library (written in Rust). The core of its BPE model is three hash tables (from `tokenizers/src/models/bpe/model.rs` in version 0.23.2):

```rust
pub type Vocab = AHashMap<String, u32>;          // piece -> id
type VocabR = AHashMap<u32, String>;              // id -> piece, for decoding
pub type MergeMap = AHashMap<Pair, (u32, u32)>;   // (left id, right id) -> (rank, merged id)
```

One hash-table lookup: hash the key into a number, take it modulo the table size to get a slot, and compare the key stored in the slot with the one you are looking for (on a collision, look at the next slot by the probing rule). On average that is one hash and one or two comparisons, regardless of the table size: looking up among 150k entries is as fast as among 10.

Compared with the teaching implementation `ByteBPE.encode` above, the library differs in three ways:

- **The merge-rule table gives the new id directly**: the key is a pair of ids and the value is (rank, merged id), so merging needs no string concatenation and no second vocabulary lookup. The vocabulary is consulted only once, at the start, to turn single bytes into ids.
- **A min-heap plus a doubly linked list**: `ByteBPE.encode` rescans every adjacent pair after each merge, which is O(n²) for a text of n bytes. The library links the pieces into a doubly linked list and puts every mergeable adjacent pair into a min-heap ordered by rank: each time it pops the best-ranked pair, joins the two pieces by updating two pointers, then looks up the new pairs the merged piece forms with its left and right neighbors and pushes them onto the heap, for O(n log n) overall.
- **A cache**: each pre-tokenized segment and its merge result go into a cache table. Common pieces such as " the" or "的" hit the cache the second time they appear, skipping the whole merge.

Special tokens (such as `<|im_start|>`) do not go through BPE; a separate "added vocabulary" matches them as whole strings before pre-tokenization. That is why `model.vocab` in `tokenizer.json` has only the 151643 regular tokens, and adding the 26 special tokens gives the 151669 of `len(tok)`. Decoding goes the other way: the reverse vocabulary maps ids back to pieces, which are joined, mapped back to raw bytes and decoded as UTF-8.

## Special tokens and chat templates {#特殊-token-与对话模板}

A chat model needs to know "which part is the system prompt, which is what the user said, and where it is its turn to answer". This structure is marked with **special tokens**. Qwen uses the ChatML format:

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

- `add_generation_prompt=True` appends `<|im_start|>assistant\n` at the end, telling the model "now it is your turn";
- Qwen3 is a "hybrid thinking" model: by default it first writes its reasoning between `<think>` and `</think>` and then gives the answer. `enable_thinking=False` inserts an empty `<think></think>` pair so that it skips thinking and answers directly; this book's examples all do this to keep the output short and reproducible. Inference engines have to parse out the thinking content for this (for example `reasoning_content` in the OpenAI API); see [sampling and the API](serving://engine/sampler-api/) in the inference systems book;
- When the model generates `<|im_end|>`, the answer is over, so it is the chat model's end-of-sequence token (eos);
- The model learned this exact format in post-training (see [post-training](../training/post-training.md)), so **the format at inference time must be exactly the same as in training**: one extra space or one missing newline can hurt quality. Always use `apply_chat_template` instead of building the string yourself.

Back to the example from the [previous chapter](language-model.md#看一眼真实模型的输出): with the chat template, the model knows this is a question rather than a fill-in-the-blank exercise:

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

The model is about to answer with a sentence like "中国的首都是北京" ("The capital of China is Beijing"), and it is quite sure.

!!! inference "Inference view"
    A chat template means every request starts with the same system prompt and format tokens. **Prefix caching** in inference engines (see [serving](../inference/serving.md#kv-cache-管理与前缀缓存)) exploits exactly this: a prefix shared by many requests needs its KV cache computed only once.

## Streaming output and incremental detokenization {#流式输出与增量反分词}

In streaming output (showing text word by word), the tokens generated at each step must be turned into text. The problem: **a token does not necessarily correspond to a complete character**. A byte-level BPE token may be only some of the bytes of a Chinese character:

```pycon
>>> ids = tok.encode("龘")
>>> ids, [tok.decode([i]) for i in ids]
([82912, 246], ['�', '�'])
>>> tok.decode(ids)
'龘'
```

"龘" is not in the common vocabulary and is split into two byte-level tokens. Each decodes on its own to invalid UTF-8 (shown as the replacement character �), and only together do they form the complete character. So streaming output cannot "decode each token alone and concatenate"; the right way is **incremental detokenization**: keep all the generated tokens, decode the most recent stretch each time, and output new text only when it does not end with incomplete bytes. vLLM's detokenizer is implemented this way.

!!! interview "In an interview"
    Asked "how does tokenization affect inference": start with byte-level BPE, which merges by frequency starting from bytes, so any text can be encoded losslessly with no unknown words; then cost, where the token count decides compute and KV cache, the same content in another language or written as code can differ several-fold in tokens, and the vocabulary size decides how big the embedding and output layers are; and finish with the engineering pitfalls: chat templates must go through `apply_chat_template` to match training, and in streaming output one token may be half a Chinese character, so you need incremental detokenization.

## Exercises {#练习}

**1. Incremental detokenization.** Write a function `stream_decode(tok, ids)` that simulates streaming output: add tokens one at a time and each time output only the new text that is already complete. Check with "龘龘你好" that the concatenated pieces equal the result of decoding everything at once, and that no replacement character was ever output.

??? success "Answer"
    ```python
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

    def stream_decode(tok, ids):
        pieces, emitted = [], ""
        for n in range(1, len(ids) + 1):
            text = tok.decode(ids[:n])
            if text.endswith("�"):        # ends with incomplete bytes: hold it back for now
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

    This version decodes from the start every time, which is O(n²). A real implementation decodes only the last few tokens (keeping a "prefix offset" and a "read offset"), at constant cost.

**2. Estimation.** A service handles 100 million requests a day, and each request's system prompt and chat template add up to 300 tokens. If the KV cache of these prefixes could be fully reused, how many tokens of prefill computation would be saved per day?

??? success "Answer"
    100 million × 300 = 30 billion tokens of prefill. At about 2 × 7e9 = 1.4e10 operations per token for a 7B model, that is about 4.2e20 operations, roughly 10 days of continuous work for one H100 (about 1e15 BF16 operations per second, at 50% utilization). That is why prefix caching is so valuable in real services.

## Summary {#小结}

- [x] Modern large models use subword tokenization, mostly byte-level BPE: start from bytes and keep merging by frequency.
- [x] Encoding applies the merge rules in the order they were learned; working at the byte level guarantees any text can be encoded losslessly.
- [x] A real tokenizer is three hash tables inside (vocabulary, reverse vocabulary, merge rules); it merges by rank with a min-heap and a doubly linked list, and common pieces come from a cache.
- [x] The token count decides compute and KV cache size; the vocabulary size decides the size of the embedding and output layers.
- [x] Chat models rely on chat templates made of special tokens, and inference must use `apply_chat_template` to match training.
- [x] One token may be half a character, so streaming output needs incremental detokenization.
