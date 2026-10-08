# From scratch (4): from continuation to conversation

<p class="lead">The model from the first three chapters does exactly one thing: it keeps writing from whatever text you give it. To make it "follow an instruction and answer a question" takes one more round of training, and that round is called supervised fine-tuning (SFT). This chapter builds an instruction set out of the same corpus, writes a chat template, puts the loss on the assistant's replies only, and fine-tunes the same 1.8M-parameter model: it starts answering in the conversation format, and goes from 0% to over 20% on "who said this". The pipeline is exactly the one real SFT uses, with several orders of magnitude less data.</p>

!!! note "What this chapter runs"
    **Main line**: `chat_data.py` (about under a second) → three jsonl files of instruction data; `chat.py` (the chat template, imported later);
    `sft.py` (about 3.2 minutes) → produces `sft.pt` and `tokenizer_chat.json`. Afterwards, use `talk.py` to say a couple of things to it.

!!! question "Self-test: if you can answer these, skip the chapter"
    1. How do the training objectives of pretraining and instruction tuning differ? Does the model's architecture change?
    2. What is a chat template, and why use special tokens instead of just writing "User:"?
    3. Why does SFT compute the loss only over the assistant's reply? What happens if the user's question is included too?
    4. After adding new special tokens to a base model, what has to change in the model?
    5. When a sample is padded to a fixed length, how should the padding positions be handled?

??? success "Answers (try first, then expand to compare)"
    1. The objective is the same one — predict the next token. What differs is the data and the span the loss covers: pretraining computes it at every position of a huge body of text, SFT computes it only over the reply in "instruction + reply". The architecture does not change at all, which is why SFT can continue straight from the pretrained weights.
    2. A chat template fixes how a multi-turn conversation is laid out as one token sequence: who is speaking, and where a turn starts and ends. Special tokens (`<|im_start|>` / `<|im_end|>`) are used so that those markers **cannot be forged by ordinary text** — if a turn were just "User:", anyone could type "User:" inside their message and fake a new turn, which is one form of prompt injection.
    3. Because what we are teaching is "given this question, say that". Putting the question into the loss also makes the model learn "how to make up questions", which dilutes the gradient; it shows up more with multiple turns. The way to do it is to mark the positions you do not want learned as `-100`, which cross-entropy ignores.
    4. The embedding (and the output layer that shares its weights) gains a few rows, randomly initialized; every other weight is copied over. The representation of the new tokens is learned entirely during SFT.
    5. Mark the padding positions `-100` so they contribute no loss. When padding on the right, causal attention guarantees they cannot affect the real tokens before them, so no extra attention mask is needed; padding on the left requires one.

## First, build an instruction set {#先造一份指令数据}

Real SFT data is written by people or synthesized by a large model: hundreds of thousands to millions of "instruction + reply" pairs covering question answering, rewriting, code, mathematics and tool calls. We have none of that, but we can mechanically build three kinds of task out of the same book, which is enough to walk the whole pipeline through:

```python title="chat_data.py"
"""从同一份语料机械地造出指令数据：续写、下一句、谁说的。真实的 SFT 数据是人写的或大模型合成的，格式是一样的"""
import collections
import json
import random
import re
from pathlib import Path

random.seed(0)
chapters = re.split(r"(?=^第.{1,4}回：)", Path("sanguo.txt").read_text(encoding="utf-8"), flags=re.M)[1:]
SAY = re.compile(r"([一-龥]{1,3})曰：「([^」]{6,28})」")       # 「Someone 曰:「…」」, i.e. "X said: ..."
STOP = {"问", "公", "众", "答", "又", "大", "曰", "言", "报", "或", "左右", "众人", "老人", "童子", "军士",
        "一人", "二人", "后人", "来人", "细作", "门吏", "近臣", "侍臣"}                # 「问曰」(asked) and 「众人曰」(everyone said) are not names


def clean(s):
    """整句、引号配对的句子才用（半句话当答案会教坏模型）"""
    s = s.strip()
    return 10 <= len(s) <= 30 and s.count("「") == s.count("」") and s.count("『") == s.count("』")


def samples(chs, names):
    """每一回出三类样本；前两类的答案都是"下一句"，只是问法不同——模型必须看指令才知道要做什么"""
    out = []
    for ch in chs:
        body = ch.split("\n", 1)[1] if "\n" in ch else ch
        sents = [s.strip() for s in re.split(r"(?<=[。！？」])", body) if clean(s)]
        out += [[("接下来写：" + sents[i], sents[i + 1])] for i in range(0, len(sents) - 1, 7)]
        out += [[(f"「{sents[i]}」的下一句是什么？", sents[i + 1])] for i in range(1, len(sents) - 1, 9)]
        out += [[(f"这句话是谁说的：「{what}」", who)] for who, what in SAY.findall(body) if who in names]
    return out


count = collections.Counter(w for c in chapters for w, _ in SAY.findall(c) if w not in STOP)
NAMES = {w for w, _ in count.most_common(12)}
train = samples(chapters[:-12], NAMES)
val = samples(chapters[-12:], NAMES)                                    # the last 12 chapters: a plot the model has never seen
random.shuffle(train)
who = [t for t in train if "这句话是谁说" in t[0][0]][:600]              # hold out a separate "who said it" set: the cast changes almost completely in the last 12 chapters
train = [t for t in train if t not in who]
for pool, n in ((train, 400), (val, 20)):                               # chain some of them end to end into two-turn conversations
    for i in random.sample(range(len(pool) - 1), n):
        pool[i] = pool[i] + pool[i + 1]

SYSTEM = ["你是一个读过《三国演义》的小助手。", "你是三国问答助手，请简短作答。", "你熟悉《三国演义》，请帮用户补全原文。"]


def dump(path, data):
    """两成样本带 system：模型要学会"有没有系统提示都能答"（真实的 SFT 数据集也这么掺）"""
    with open(path, "w", encoding="utf-8") as f:
        for turns in data:
            conv = [m for u, a in turns for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]
            if random.random() < 0.2:
                conv = [{"role": "system", "content": random.choice(SYSTEM)}] + conv
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")


for path, data in (("sft_train.jsonl", train), ("sft_val.jsonl", val), ("sft_who.jsonl", who)):
    dump(path, data)
kinds = {"续写": "接下来写", "下一句": "的下一句", "谁说的": "这句话是谁说"}
print(f"说话最多的 12 个人：{'、'.join(w for w, _ in count.most_common(12))}")
print(f"训练 {len(train)} 条（两轮的 {sum(len(t) > 1 for t in train)} 条）、验证 {len(val)} 条、"
      f"「谁说的」留出 {len(who)} 条（前 400 条给下一章的 LoRA，后 200 条当测试）")
print("训练集三类各：", "、".join(f"{k} {sum(m in t[0][0] for t in train)}" for k, m in kinds.items()))
for line in open("sft_train.jsonl", encoding="utf-8").readlines()[:2]:
    conv = json.loads(line)["conversations"]
    print("样本：" + " → ".join(f"[{m['role'][0]}] {m['content']}" for m in conv))
```

```text title="output"
说话最多的 12 个人：玄德、操、孔明、肃、权、瑜、布、关公、飞、懿、云长、绍
训练 3861 条（两轮的 400 条）、验证 324 条、「谁说的」留出 600 条（前 400 条给下一章的 LoRA，后 200 条当测试）
训练集三类各： 续写 1917、下一句 1486、谁说的 458
样本：[u] 接下来写：荆、襄之民，闻曹兵至，未战而胆先寒，安能与之敌哉？ → [a] 众视之，乃山阳高平人，姓王，名粲，字仲宣。
样本：[u] 「前队才行，两下火起，乃是马超伏兵追赶。」的下一句是什么？ → [a] 操令军士急行，晓夜奔走无停；直至京兆，方始安心。
```

- **The three tasks share one pool of answers**: the correct answer to both "Write what comes next: X" and "What is the sentence after 「X」?" is the following sentence; only the way of asking differs. That is deliberate — the model has to read the instruction to know what to produce, rather than just look at the last sentence;
- **600 "who said it" samples are held out**: the cast changes almost completely in the last 12 chapters (Xuande, Cao Cao and Kongming are all gone by then), so a chapter-based split would make the validation set useless for this task. We hold out a batch from the training chapters instead, never train on it, and use it purely to measure accuracy;
- **A fifth of the samples carry a system prompt**: real SFT datasets mix system prompts in at some ratio, so that the model learns to answer with or without one;
- **Every character in the instruction has to be in the vocabulary.** Our tokenizer has only ever seen *Romance of the Three Kingdoms*, which writes 著 and never once 着, so 「接着往下写」 comes out as `接 <unk> 往 下 写`. A real project either uses a general-purpose tokenizer or extends the vocabulary; here we reworded it to 「接下来写」.

## The chat template: laying a conversation out as tokens {#聊天模板把对话拼成一串-token}

The model sees nothing but a sequence of tokens, so a multi-turn conversation has to be laid out as one. The common approach is a format like ChatML: wrap each turn in special tokens that name the speaker.

```python title="chat.py"
"""聊天模板：把多轮对话拼成一串 token，标出"哪些位置要算 loss"（只有助手说的话算），再按模板生成"""
import json

import torch
from tokenizers import Tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ROLE = {"system": "系统", "user": "用户", "assistant": "助手"}    # the role names must be in the vocabulary too, and ours has only ever seen Chinese
IGNORE = -100                                                 # the label value F.cross_entropy ignores by default


def load_tokenizer(path="tokenizer.json"):
    """给预训练好的分词器补两个聊天用的特殊 token，id 接在原词表后面"""
    tok = Tokenizer.from_file(path)
    tok.add_special_tokens([IM_START, IM_END])
    return tok


def chat_ids(tok, conversations, add_generation_prompt=False):
    """拼成 (token, 标签)：标签在系统段、用户段上是 -100，只有助手回答（含收尾的 <|im_end|>）要预测"""
    ids, labels = [], []

    def put(text, learn=False):
        piece = tok.encode(text, add_special_tokens=False).ids
        ids.extend(piece)
        labels.extend(piece if learn else [IGNORE] * len(piece))

    for msg in conversations:
        if msg["role"] == "assistant":
            put(f"{msg['content']}{IM_END}\n", learn=True)     # learn the <|im_end|> too, or the model never knows where to stop
        else:
            put(f"{IM_START}{ROLE[msg['role']]}\n{msg['content']}{IM_END}\n")
            if msg["role"] == "user":
                put(f"{IM_START}{ROLE['assistant']}\n")        # the opening of the assistant turn is a prompt, not something to learn
    if add_generation_prompt:                                  # used when you give only the question and let the model continue
        put(f"{IM_START}{ROLE['assistant']}\n")
    return ids, labels


def encode(tok, conversations, max_len=128):
    """补齐到定长，并把标签错开一位，可以直接喂给 model(x, y)"""
    ids, labels = chat_ids(tok, conversations)
    ids, labels = ids[:max_len], labels[:max_len]
    n = max_len - len(ids)
    x = torch.tensor(ids + [tok.token_to_id(IM_END)] * n)      # pad on the right; with causal attention the padding cannot affect anything before it
    y = torch.tensor(labels[1:] + [IGNORE] * (n + 1))          # position i has to predict token i+1
    return x, y


def read(path):
    return [json.loads(line)["conversations"] for line in open(path, encoding="utf-8")]


def load(tok, path, max_len=128):
    xs, ys = zip(*(encode(tok, c, max_len) for c in read(path)))
    return torch.stack(xs), torch.stack(ys)


@torch.no_grad()
def answer(model, tok, prompt, max_new=40, templated=True):
    """按模板补上"该助手说话了"，贪心生成到 <|im_end|> 为止；templated=False 时直接把问题当正文续写"""
    training, seq_len = model.training, model.cfg.seq_len
    model.eval()
    ids = (chat_ids(tok, [{"role": "user", "content": prompt}], add_generation_prompt=True)[0] if templated
           else tok.encode(prompt, add_special_tokens=False).ids)
    idx, out = torch.tensor([ids]), []
    for _ in range(max_new):
        nxt = model(idx[:, -seq_len:])[:, -1].argmax(-1, keepdim=True)
        if nxt.item() == tok.token_to_id(IM_END):
            break
        out.append(nxt.item())
        idx = torch.cat([idx, nxt], dim=1)
    model.train(training)
    return tok.decode(out, skip_special_tokens=False).replace("\n", "⏎")


def accuracy(model, tok, pairs, max_new=6):
    """「谁说的」这类短答案的准确率：贪心生成，要求和参考答案完全一致"""
    return sum(answer(model, tok, q, max_new).strip() == a for q, a in pairs) / len(pairs)


if __name__ == "__main__":
    tok = load_tokenizer()
    print(f"词表 {tok.get_vocab_size()} 个：原来 8192，加上 {IM_START}（{tok.token_to_id(IM_START)}）"
          f"和 {IM_END}（{tok.token_to_id(IM_END)}）")
    conv = next(c for c in read("sft_train.jsonl") if len(c) == 4)
    ids, _ = chat_ids(tok, conv)
    print("—— 拼好的样本 ——")
    print(tok.decode(ids, skip_special_tokens=False).replace(IM_START, "⟨s⟩").replace(IM_END, "⟨e⟩"))
    x, y = encode(tok, conv)
    print(f"补齐到 {len(x)} 个 token，其中要算 loss 的 {(y != IGNORE).sum().item()} 个")
    first = (y != IGNORE).nonzero()[0].item()
    print(" 位置   输入 token        要预测的")
    for i in range(first - 4, first + 8):
        nxt = repr(tok.decode([y[i].item()], skip_special_tokens=False)) if y[i] != IGNORE else "—（不算 loss）"
        print(f"{i:5d}   {tok.decode([x[i].item()], skip_special_tokens=False)!r:18s} {nxt}")
```

```text title="output"
词表 8194 个：原来 8192，加上 <|im_start|>（8192）和 <|im_end|>（8193）
—— 拼好的样本 ——
⟨s⟩用户
接下来写：魏兵回见司马懿，细告前事。⟨e⟩
⟨s⟩助手
魏主闻张郃死，挥泪叹息，令人收其尸，厚葬之。⟨e⟩
⟨s⟩用户
「谭引败军奔平原，尚收兵还。」的下一句是什么？⟨e⟩
⟨s⟩助手
袁谭与郭图再议进兵，令岑璧为将，领兵前来。⟨e⟩

补齐到 128 个 token，其中要算 loss 的 36 个
 位置   输入 token        要预测的
   17   '\n'               —（不算 loss）
   18   '<|im_start|>'     —（不算 loss）
   19   '助'                —（不算 loss）
   20   '手'                —（不算 loss）
   21   '\n'               '魏主'
   22   '魏主'               '闻'
   23   '闻'                '张郃'
   24   '张郃'               '死'
   25   '死'                '，'
   26   '，'                '挥'
   27   '挥'                '泪'
   28   '泪'                '叹息'
```

Reading that output:

- **The special tokens come after the original vocabulary**: `<|im_start|>` is 8192 and `<|im_end|>` is 8193, so the vocabulary goes from 8192 to 8194. They have to be **single, indivisible tokens** that ordinary text cannot be split into, which is what stops a user from forging a new turn inside their message;
- **Role names have to be tokenizable too**: our vocabulary has only seen Chinese, so the English `system` / `user` / `assistant` would turn into `<unk>`; we write the roles as 系统 / 用户 / 助手 instead. Real models have those English words in their vocabulary, which is why they can use ChatML as it is;
- **Labels are shifted by one**: position $i$ has to predict token $i+1$, so the labels move one step to the left. The row "input `\n`, predict `孟获`" in the table is the first token of the assistant's reply;
- **Only the assistant's reply carries labels**: the system turn, the user turn, and the `<|im_start|>助手\n` prompt itself are all `-100`. Of the 128 positions, only 25 contribute loss;
- **`<|im_end|>` has to be learned too**: it is the "I am done" signal. Without it, the model never knows where to stop when generating.

## Loss on the reply only {#只在回答上算-loss}

`-100` is the default `ignore_index` of PyTorch's cross-entropy, so the model from chapter 2 needs no changes at all — `F.cross_entropy(logits, targets)` skips those positions automatically. Two details are worth noting:

- **Pad on the right**: the padding tokens are labelled `-100` as well, so they produce no gradient; and because causal attention lets each position see only itself and what is to its left, padding on the right cannot affect the real tokens before it, so no extra attention mask is needed. Pad on the left and you must pass a mask, or the real tokens will "see" the padding;
- **The positions that carry loss are sparse**: only a little over a tenth of the positions in this batch contribute, and the rest of the compute goes into "reading the question". Real SFT **packs** several short samples into one sequence, with a segmented attention mask, to raise that fraction.

## Fine-tuning {#微调}

```python title="sft.py"
"""指令微调：从预训练的 checkpoint 出发，只在"助手说的话"上算 loss"""
import math

import torch

from chat import IGNORE, accuracy, answer, load, load_tokenizer, read
from model import GPT, GPTConfig

torch.manual_seed(0)
tok = load_tokenizer()
ckpt = torch.load("ckpt.pt", weights_only=False)
cfg = GPTConfig(**{**ckpt["model_config"], "vocab_size": tok.get_vocab_size()})
model = GPT(cfg)
state = ckpt["model"]
with torch.no_grad():                                         # two more tokens in the vocabulary: two more rows in the embedding, every other weight copied over
    state["embed.weight"] = torch.cat([state["embed.weight"], model.embed.weight[8192:].clone()])
    state["head.weight"] = state["embed.weight"]              # the input and the output share one matrix
model.load_state_dict(state)
print(f"预训练模型 {ckpt['step']} 步、{model.num_params() / 1e6:.2f}M 参数；词表 8192 → {cfg.vocab_size}，词嵌入补了两行")

x_tr, y_tr = load(tok, "sft_train.jsonl", cfg.seq_len)
x_va, y_va = load(tok, "sft_val.jsonl", cfg.seq_len)
who = [(c[-2]["content"], c[-1]["content"]) for c in read("sft_who.jsonl")][-200:]
print(f"训练 {len(x_tr)} 条、验证 {len(x_va)} 条；每条 {cfg.seq_len} 个位置，"
      f"只有 {(y_tr != IGNORE).float().mean():.1%} 要算 loss")


@torch.no_grad()
def val_loss():
    model.eval()
    loss = sum(model(x_va[i:i + 64], y_va[i:i + 64])[1].item() for i in range(0, len(x_va), 64))
    model.train()
    return loss / math.ceil(len(x_va) / 64)


ASK = ["这句话是谁说的：「吾与汝同生共死，汝可速去。」", "接下来写：孔明曰：「吾自有计。」",
       "「操大喜，遂引兵望寿春而来。」的下一句是什么？"]
cont = [c[-2]["content"] for c in read("sft_val.jsonl") if "接下来写" in c[-2]["content"]][:100]


def task_style():
    """答案的"长相"对不对：「谁说的」该是两三个字的人名，续写该是一整句话"""
    short = sum(len(answer(model, tok, q, 8)) <= 4 for q, _ in who[:100])
    long_ = sum(len(answer(model, tok, q, 40)) >= 8 for q in cont)
    return short / 100, long_ / len(cont)


print(f"\n微调前：验证 loss {val_loss():.3f}，「谁说的」准确率 {accuracy(model, tok, who):.1%}"
      f"（12 个人里挑，随机是 8.3%）")
print("把问题直接喂给它（它没见过聊天模板），只会顺着往下编：")
for q in ASK[:2]:
    print(f"  问：{q}\n  答：{answer(model, tok, q, 20, templated=False)}")

steps, batch, peak, warmup = 400, 32, 1e-3, 40
opt = torch.optim.AdamW(model.parameters(), lr=peak, betas=(0.9, 0.95), weight_decay=0.1)
gen = torch.Generator().manual_seed(1)
print(f"\n微调 {steps} 步 × {batch} 条 = {steps * batch / len(x_tr):.1f} 遍，峰值学习率 {peak:g}（预训练用的是 3e-3）")
print("  步   训练 loss   验证 loss")
running = 0.0
for step in range(steps):
    lr = peak * (step + 1) / warmup if step < warmup else \
        0.1 * peak + 0.45 * peak * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
    for g in opt.param_groups:
        g["lr"] = lr
    i = torch.randint(0, len(x_tr), (batch,), generator=gen)
    _, loss = model(x_tr[i], y_tr[i])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    running += loss.item()
    if (step + 1) % 100 == 0:
        print(f"{step + 1:4d}   {running / 100:8.3f}   {val_loss():8.3f}")
        running = 0.0

short, long_ = task_style()
print(f"\n微调后：「谁说的」准确率 {accuracy(model, tok, who):.1%}；"
      f"答人名的题里 {short:.0%} 答了 4 个字以内，续写的题里 {long_:.0%} 答出了完整的一句")
for q in ASK:
    print(f"  问：{q}\n  答：{answer(model, tok, q)}")
torch.save({"model": model.state_dict(), "model_config": vars(cfg)}, "sft.pt")
tok.save("tokenizer_chat.json")
```

```text title="output"
预训练模型 600 步、1.80M 参数；词表 8192 → 8194，词嵌入补了两行
训练 3861 条、验证 324 条；每条 128 个位置，只有 11.6% 要算 loss

微调前：验证 loss 7.049，「谁说的」准确率 0.0%（12 个人里挑，随机是 8.3%）
把问题直接喂给它（它没见过聊天模板），只会顺着往下编：
  问：这句话是谁说的：「吾与汝同生共死，汝可速去。」
  答：⏎玄德从之，遂令人往成都。玄德自遣大军，与赵云、张嶷、张嶷
  问：接下来写：孔明曰：「吾自有计。」
  答：⏎玄德从之，遂令人往成都。玄德自遣大军，与赵云、张嶷、张嶷

微调 400 步 × 32 条 = 3.3 遍，峰值学习率 0.001（预训练用的是 3e-3）
  步   训练 loss   验证 loss
 100      5.128      5.775
 200      4.559      5.842
 300      4.308      5.883
 400      4.162      5.896

微调后：「谁说的」准确率 23.5%；答人名的题里 100% 答了 4 个字以内，续写的题里 94% 答出了完整的一句
  问：这句话是谁说的：「吾与汝同生共死，汝可速去。」
  答：玄德
  问：接下来写：孔明曰：「吾自有计。」
  答：操
  问：「操大喜，遂引兵望寿春而来。」的下一句是什么？
  答：操见玄德，言：「我已来」
```

Line by line:

- **The learning rate is an order of magnitude below pretraining** (1e-3 against 3e-3): the knowledge in the base model took hundreds of times more compute to acquire, and fine-tuning only adjusts the posture; too large a learning rate washes it away (catastrophic forgetting). In real projects the SFT learning rate is typically 1/10 to 1/30 of the pretraining peak;
- **Ask the model a question before fine-tuning and it just keeps making things up**: and two different questions get the same answer — that is not a bug. Under greedy decoding a 1.8M model collapses onto one "most likely" path (「张嶷、张嶷、张嶷」 is textbook greedy degeneration); it is not treating the text in front of it as a question at all;
- **The validation loss bottoms out at step 100 and rises after that**: 4000 instruction samples are too few for 400 steps of 32, and the model starts memorizing answers. Real SFT also runs only 2 to 3 epochs and picks the checkpoint by validation loss; we deliberately train to the end here so the curve is visible;
- **"Who said it" goes from 0% to over 20%**: a random guess among 12 people is 8.3%. It has picked up a little of "who tends to say what", but it is nowhere near reliable — with 1.8M parameters and a few thousand samples, getting the format right is the main gain;
- **Format sticks, content does not**: 100% of the name questions get an answer of four characters or fewer, and 94% of the continuation questions get a complete sentence — it really did read "does this question want a short answer or a long one". But the second example above (asked to continue, it answered 「操」) is one of the remaining 6%, and the third has the right shape with made-up content. **Format is cheap, content is expensive**: the first is learned in a few hundred steps, the second takes the scale of pretraining;
- **The three tasks sharing answers is deliberate**: if each kind of task had an answer of a distinctive shape, the model could guess right just by looking at the final punctuation. Making "write what comes next" and "what is the next sentence" have exactly the same answers is what proves it is really reading the instruction.

## Say something to it {#跟它说句话}

The last few lines of the training script only asked it two questions for you. To talk to it yourself, write a minimal command line:

```python title="talk.py" run="no"
"""跟训好的模型聊天：python talk.py（Ctrl-C 退出）"""
import torch

from chat import IM_END, chat_ids, load_tokenizer
from model import GPT, GPTConfig

tok = load_tokenizer("tokenizer_chat.json")
sft = torch.load("sft.pt", weights_only=False)
cfg = GPTConfig(**sft["model_config"])
model = GPT(cfg)
model.load_state_dict(sft["model"])
model.eval()
print("它只读过《三国演义》，会做三件事：接下来写 X、「X」的下一句是什么、这句话是谁说的。Ctrl-C 退出。")
history = []
while True:
    try:
        question = input("\n你：").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not question:
        continue
    history.append({"role": "user", "content": question})
    idx, out = torch.tensor([chat_ids(tok, history, add_generation_prompt=True)[0]]), []
    with torch.no_grad():
        for _ in range(60):
            logits = model(idx[:, -cfg.seq_len:])[:, -1] / 0.8          # temperature 0.8, a bit livelier than greedy
            nxt = torch.multinomial(logits.softmax(-1), 1)
            if nxt.item() == tok.token_to_id(IM_END):
                break
            out.append(nxt.item())
            idx = torch.cat([idx, nxt], dim=1)
    reply = tok.decode(out, skip_special_tokens=False)
    print("模型：" + reply)
    history = (history + [{"role": "assistant", "content": reply}])[-4:]  # the context is only 128 tokens, so keep the last two turns
```

It is interactive, so this page does not run it automatically. Here is a real session:

```text
它只读过《三国演义》，会做三件事：接下来写 X、「X」的下一句是什么、这句话是谁说的。Ctrl-C 退出。

你：这句话是谁说的：「吾自有计。」
模型：操

你：接下来写：玄德大喜，遂引兵而进。
模型：次日，人报魏兵溃散。

你：「操大怒，拔剑欲斩之。」的下一句是什么？
模型：孔明
```

Whether the answers are right is another matter, but it really is answering in the conversation format and knows where to stop — which is the whole point of this chapter.

## What real SFT data looks like {#真实的-sft-数据长什么样}

The format is the same; the content is much richer. Take the open-source [MiniMind](https://github.com/jingyaogong/minimind) (a 64M-parameter project that trains from scratch to a chat model in about two hours on one GPU): its SFT data is one JSON object per line:

```json
{"conversations": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"},
  {"role": "user", "content": "再见"},
  {"role": "assistant", "content": "再见！"}
]}
{"conversations": [
  {"role": "system", "content": "# Tools ...", "tools": "[{\"name\": \"translate_text\", ...}]"},
  {"role": "user", "content": "把'你好世界'翻译成 english"},
  {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"translate_text\",\"arguments\":{...}}]"},
  {"role": "tool", "content": "{\"translated_text\":\"Hello World\"}"},
  {"role": "assistant", "content": "Hello World"}
]}
```

The extra pieces work exactly like this chapter's, with a few more roles and tags in the template:

- **Tool calls**: the `system` turn lists the available tools, the assistant emits a `tool_calls` block instead of natural language, the `tool` role returns the result, and the assistant answers from it. What is learned is still only "what the assistant says", including that JSON;
- **Thinking**: `<think> … </think>` wraps the reasoning; during training it carries loss just like the answer, and at inference you can choose not to show it. Datasets usually mix in a fraction of "empty thinking" (`<think>\n\n</think>`) samples so the model learns that a simple question needs no deliberation;
- **The share of system prompts**, **the share of multi-turn samples** and **the length distribution** are all part of the recipe;
- **How the answer spans are located**: rendering with a ready-made `apply_chat_template` gives you no turn boundaries, so the common approach is to render to text and then **search** for `<|im_start|>助手\n` and `<|im_end|>` to unmask what lies between. We label as we build, which is more direct and harder to get wrong.

!!! interview "How to explain it"
    To explain "the difference between pretraining and SFT": the objective has not changed; the data and the span of the loss have — pretraining computes it at every position of the text, SFT only over the assistant's reply (everything else is `-100`), which is why the architecture needs no change and SFT continues straight from the base. Then three implementation details: the **chat template** separates roles with special tokens that ordinary text cannot forge (prompt injection); **`<|im_end|>` is learned too**, or the model never knows where to stop; **the learning rate is an order of magnitude below pretraining** and the data is seen only two or three times, or the base model's knowledge is washed away. On the engineering side, add that the positions carrying loss are sparse, which is why samples are packed.

## Exercises {#练习}

**1. The share of system prompts.** Change the probability of mixing in a system prompt in `chat_data.py` from 0.2 to 0 and to 1.0, train each, and then ask questions both with and without a system prompt. What is the difference?

??? success "An approach"
    A model trained with a system prompt on every sample usually drops noticeably when asked without one — it has taken that turn to be part of the format. Trained without any, it never learns to use a system prompt. Mixing them means the model has seen both and can handle both. Real datasets mix in ten to thirty percent, and vary the wording of the prompts themselves.

**2. Learn only the last turn.** In multi-turn samples, label the earlier assistant replies `-100` as well and learn only the last one. How do the training loss and the accuracy change? Why do both approaches have users?

??? success "An approach"
    Learning only the last turn leaves fewer positions carrying loss, so the same number of steps learns less; but it matches the inference situation — answer given the full context — more closely, and is steadier when multi-turn data is of uneven quality. Learning all of them uses the samples better. Mainstream implementations (MiniMind included) learn every assistant turn by default, and fall back to the last turn only when the data quality is shaky.

**3. Packing.** Only a little over a tenth of the positions in a 128-token sequence currently carry loss. If several short samples are packed into one sequence, how high can that go, and what has to be taken care of?

??? success "An approach"
    Packing takes it above half, which multiplies training throughput several times over. The thing to take care of is that **samples must not see each other**: you need a segmented attention mask (or FlashAttention's variable-length `varlen` interface), or the second sample takes the first as its context. Position encodings also have to restart from 0 inside each sample. This is exactly what [the distributed training handbook's packing exercise](train://basics/overview/) asks you to do.

## Summary {#小结}

- [x] Pretraining and SFT share one objective (predict the next token); only the data and the span of the loss differ. The architecture needs no change, and training continues straight from the base model.
- [x] The chat template separates roles with **special tokens that cannot be forged**; `<|im_end|>` is learned too, so the model knows where to stop; role names have to be tokenizable.
- [x] Only the assistant's replies carry labels; the system turn, the user turn and the padding are all `-100` (cross-entropy's default `ignore_index`). With padding on the right, causal attention guarantees it cannot affect anything before it.
- [x] Extending the vocabulary means adding rows to the embedding and copying every other weight; the new tokens' representations are learned during fine-tuning.
- [x] SFT's learning rate is an order of magnitude below pretraining and the data is seen only two or three times; the validation loss bottoms out quickly, and the checkpoint is picked by it.
- [x] The positions carrying loss are sparse (a little over a tenth here); real training packs samples to raise that fraction.
