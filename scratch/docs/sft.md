# 从零训练（四）：从续写到对话

<p class="lead">前三章训出来的模型只会做一件事：顺着给定的文字往下写。要让它"听懂指令、回答问题"，还得再训一轮——这一轮叫指令微调（SFT）。这一章把一份指令数据从语料里造出来，写出聊天模板，把 loss 只算在"助手说的话"上，再用同一个 1.8M 的模型微调一遍：它会开始用对话的格式回答，而且在"这句话是谁说的"上从 0% 答到两成以上。流程和真实的 SFT 完全一样，只是数据少了几个数量级。</p>

!!! note "这一章跑什么"
    **主线**：`chat_data.py`（约 不到 1 秒）→ 指令数据三份 jsonl；`chat.py`（聊天模板，后面要 import）；
    `sft.py`（约 3.2 分钟）→ 产出 `sft.pt`、`tokenizer_chat.json`。跑完用 `talk.py` 跟它聊两句。

!!! question "自测：能答上来就可以跳过本章"
    1. 预训练和指令微调，训练目标有什么不同？模型的结构变了吗？
    2. 聊天模板是什么？为什么要加特殊 token，而不是直接写"用户："？
    3. SFT 的 loss 为什么只算助手回答的那一段？把用户的问题也算上会怎样？
    4. 给基座模型加新的特殊 token 之后，模型要怎么改？
    5. 一条样本补齐（padding）到定长，padding 的位置要怎么处理？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 目标函数是同一个——预测下一个 token。差别在数据和算 loss 的范围：预训练在海量正文上每个位置都算，SFT 在"指令 + 回答"上只算回答。模型结构完全不变，所以 SFT 可以直接从预训练的权重接着训。
    2. 聊天模板规定了多轮对话怎么拼成一串 token：谁在说话、一段话在哪里开始和结束。用特殊 token（`<|im_start|>` / `<|im_end|>`）是为了让这些标记**无法被普通文本伪造**——如果只写"用户："，用户在正文里打一句"用户："就能冒充新一轮对话，这正是提示注入的一种。
    3. 因为我们要教的是"给定问题，答什么"。把问题也算进 loss，相当于顺便让模型去学"怎么编问题"，稀释了梯度；多轮对话里更明显。做法是把不该学的位置标成 `-100`，交叉熵会忽略它们。
    4. 词嵌入（和共享权重的输出层）要加几行，新行随机初始化；其余权重照搬。新 token 的表示完全靠 SFT 学出来。
    5. padding 的位置标成 `-100`，不算 loss。补在右边时，因果注意力保证它们影响不到前面的真实 token，所以可以不额外做注意力掩码；补在左边就必须有掩码了。

## 先造一份指令数据

真实的 SFT 数据是人写的或者大模型合成的：几十万到几百万条"指令 + 回答"，覆盖问答、改写、代码、数学、工具调用。我们没有这些，但可以从同一本书里机械地造出三类任务，用来把整条流程走通：

```python title="chat_data.py"
"""从同一份语料机械地造出指令数据：续写、下一句、谁说的。真实的 SFT 数据是人写的或大模型合成的，格式是一样的"""
import collections
import json
import random
import re
from pathlib import Path

random.seed(0)
chapters = re.split(r"(?=^第.{1,4}回：)", Path("sanguo.txt").read_text(encoding="utf-8"), flags=re.M)[1:]
SAY = re.compile(r"([一-龥]{1,3})曰：「([^」]{6,28})」")       # 「某某曰：「……」」
STOP = {"问", "公", "众", "答", "又", "大", "曰", "言", "报", "或", "左右", "众人", "老人", "童子", "军士",
        "一人", "二人", "后人", "来人", "细作", "门吏", "近臣", "侍臣"}                # 「问曰」「众人曰」不是人名


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
val = samples(chapters[-12:], NAMES)                                    # 后 12 回：从没见过的情节
random.shuffle(train)
who = [t for t in train if "这句话是谁说" in t[0][0]][:600]              # 「谁说的」另留一份：后 12 回里的人物几乎全换了
train = [t for t in train if t not in who]
for pool, n in ((train, 400), (val, 20)):                               # 一部分首尾相接拼成两轮对话
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

```text title="输出"
说话最多的 12 个人：玄德、操、孔明、肃、权、瑜、布、关公、飞、懿、云长、绍
训练 3861 条（两轮的 400 条）、验证 324 条、「谁说的」留出 600 条（前 400 条给下一章的 LoRA，后 200 条当测试）
训练集三类各： 续写 1917、下一句 1486、谁说的 458
样本：[u] 接下来写：荆、襄之民，闻曹兵至，未战而胆先寒，安能与之敌哉？ → [a] 众视之，乃山阳高平人，姓王，名粲，字仲宣。
样本：[u] 「前队才行，两下火起，乃是马超伏兵追赶。」的下一句是什么？ → [a] 操令军士急行，晓夜奔走无停；直至京兆，方始安心。
```

- **三类任务共用一批答案**：「接下来写：X」和「「X」的下一句是什么？」的正确答案都是下一句，只是问法不同。这是故意的——模型必须读懂指令才知道要输出什么，而不是只看最后一句话；
- **「谁说的」另外留了 600 条**：后 12 回的人物几乎全换了（玄德、曹操、孔明都已不在），按章节切分会让这一类的验证集失去意义，所以从训练章节里额外留出一批，训练时不用，专门拿来量准确率；
- **两成样本带 system**：真实的 SFT 数据集会按一定比例掺入系统提示，模型才能学会"有没有系统提示都能答"；
- **指令用的字必须在词表里**。我们的分词器只见过《三国演义》，原文里写的是"著"、一次"着"都没有，所以"接着往下写"会被切成 `接 <unk> 往 下 写`。真实项目里要么用通用分词器，要么扩词表；这里换成了"接下来写"。

## 聊天模板：把对话拼成一串 token

模型眼里只有一串 token，多轮对话要先拼成这一串。业界通行的做法是 ChatML 这类格式：每一段用特殊 token 包起来，标明角色。

```python title="chat.py"
"""聊天模板：把多轮对话拼成一串 token，标出"哪些位置要算 loss"（只有助手说的话算），再按模板生成"""
import json

import torch
from tokenizers import Tokenizer

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
ROLE = {"system": "系统", "user": "用户", "assistant": "助手"}    # 角色名也要能被分词器切出来：我们的词表只见过中文
IGNORE = -100                                                 # F.cross_entropy 默认忽略的标签值


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
            put(f"{msg['content']}{IM_END}\n", learn=True)     # 连 <|im_end|> 一起学，模型才知道在哪停
        else:
            put(f"{IM_START}{ROLE[msg['role']]}\n{msg['content']}{IM_END}\n")
            if msg["role"] == "user":
                put(f"{IM_START}{ROLE['assistant']}\n")        # 助手段的开头是提示，不算 loss
    if add_generation_prompt:                                  # 只给提问、让模型接着说时用
        put(f"{IM_START}{ROLE['assistant']}\n")
    return ids, labels


def encode(tok, conversations, max_len=128):
    """补齐到定长，并把标签错开一位，可以直接喂给 model(x, y)"""
    ids, labels = chat_ids(tok, conversations)
    ids, labels = ids[:max_len], labels[:max_len]
    n = max_len - len(ids)
    x = torch.tensor(ids + [tok.token_to_id(IM_END)] * n)      # 右边补齐；因果注意力下后面的 padding 影响不到前面
    y = torch.tensor(labels[1:] + [IGNORE] * (n + 1))          # 第 i 个位置要预测第 i+1 个 token
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

```text title="输出"
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

读这段输出：

- **特殊 token 排在原词表后面**：`<|im_start|>` 是 8192、`<|im_end|>` 是 8193，词表从 8192 变成 8194。它们必须是**不可分割的单个 token**，普通文本切不出来，所以用户没法在正文里伪造一轮新的对话；
- **角色名也要能切出来**：我们的词表只见过中文，英文的 `system` / `user` / `assistant` 会变成 `<unk>`，所以角色写成"系统 / 用户 / 助手"。真实模型的词表里有这些英文词，才能直接用 ChatML 的原样；
- **标签错开一位**：第 $i$ 个位置的任务是预测第 $i+1$ 个 token，所以标签整体左移一位。表里"输入 `\n`、要预测 `孟获`"就是助手回答的第一个 token；
- **只有助手回答带标签**：系统段、用户段、以及 `<|im_start|>助手\n` 这个提示本身都是 `-100`。整条 128 个位置里只有 25 个要算 loss；
- **`<|im_end|>` 也要学**：它是"我说完了"的信号。不学它，生成时模型就不知道在哪停。

## 只在回答上算 loss

`-100` 是 PyTorch 交叉熵的默认 `ignore_index`，所以第二章写的模型一行都不用改——`F.cross_entropy(logits, targets)` 自动跳过这些位置。两个细节值得注意：

- **padding 补在右边**：补齐用的 token 标签同样是 `-100`，不产生梯度；而因果注意力让每个位置只能看到自己和左边，右边的 padding 影响不到前面的真实 token，所以不需要额外的注意力掩码。如果补在左边，就必须传掩码，否则真实 token 会"看到"padding；
- **算 loss 的位置很稀疏**：这一批数据里只有一成多的位置要算 loss，剩下的算力都花在"读题"上。真实的 SFT 会把多条短样本**打包**（packing）进同一条序列、配上分段的注意力掩码，把这个比例提上去。

## 微调

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
with torch.no_grad():                                         # 词表多了两个 token：词嵌入多两行，其余权重照搬
    state["embed.weight"] = torch.cat([state["embed.weight"], model.embed.weight[8192:].clone()])
    state["head.weight"] = state["embed.weight"]              # 输入和输出共享同一个矩阵
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

```text title="输出"
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

一条一条读：

- **学习率比预训练小一个量级**（1e-3 对 3e-3）：基座里的知识是预训练用几百倍算力学来的，微调只是"调姿势"，学习率太大会把它冲掉（灾难性遗忘）。真实项目里 SFT 的学习率通常是预训练峰值的 1/10 到 1/30；
- **微调前问它问题，它只会接着编**：而且两个不同的问题得到同一个答案——这不是 bug。贪心解码下，一个 1.8M 的模型会塌到一条"最可能"的路径上（"张嶷、张嶷、张嶷"是典型的贪心退化），它根本没把前面的文字当成一个问题；
- **验证 loss 第 100 步就到底了，之后反而上升**：4000 条指令数据对 400 步 × 32 条来说太少，模型开始背答案。真实的 SFT 也只训 2～3 个 epoch，并按验证 loss 选 checkpoint；我们这里故意训完，好让这条曲线看得见；
- **"谁说的"从 0% 到两成以上**：12 个人里随机猜是 8.3%。它确实学到了一点"谁爱说什么话"，但远谈不上可靠——1.8M 参数、几千条样本，能把格式学对已经是主要收获；
- **格式学得很牢，内容学不动**：答人名的题里 100% 给出了 4 个字以内的回答，续写的题里 94% 给出了完整的一句——它确实读懂了「这道题要短答还是长答」。但上面第二个例子（问「接下来写」，它答了「操」）就是剩下的那 6%，第三个例子格式对、内容是胡编的。**格式是便宜的，内容是贵的**：前者几百步就学会，后者要靠预训练的规模；
- **三类任务共用答案是故意的**：如果每一类的答案长相都不一样，模型只要看最后一个标点就能蒙对。让「接下来写」和「下一句是什么」的答案完全相同，才能确认它真的在读指令。

## 跟它说句话

训练脚本最后那几行只是替你问了两句。要自己跟它聊，写一个最小的命令行：

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
            logits = model(idx[:, -cfg.seq_len:])[:, -1] / 0.8          # 温度 0.8，比贪心活一点
            nxt = torch.multinomial(logits.softmax(-1), 1)
            if nxt.item() == tok.token_to_id(IM_END):
                break
            out.append(nxt.item())
            idx = torch.cat([idx, nxt], dim=1)
    reply = tok.decode(out, skip_special_tokens=False)
    print("模型：" + reply)
    history = (history + [{"role": "assistant", "content": reply}])[-4:]  # 上下文只有 128 个 token，只留最近两轮
```

它是交互式的，所以这一页不自动跑它。下面是一次真实的会话：

```text
它只读过《三国演义》，会做三件事：接下来写 X、「X」的下一句是什么、这句话是谁说的。Ctrl-C 退出。

你：这句话是谁说的：「吾自有计。」
模型：操

你：接下来写：玄德大喜，遂引兵而进。
模型：次日，人报魏兵溃散。

你：「操大怒，拔剑欲斩之。」的下一句是什么？
模型：孔明
```

答得对不对另说，但它确实在按对话的格式回答、而且知道在哪停——这就是这一章全部的目的。

## 真实的 SFT 数据长什么样

格式是一样的，内容复杂得多。以开源的 [MiniMind](https://github.com/jingyaogong/minimind)（一个 64M 参数、单卡两小时从零训到能对话的项目）为例，它的 SFT 数据就是逐行的 JSON：

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

多出来的几件事，机制和我们这一章完全一样，只是模板里多了几种角色和标签：

- **工具调用**：`system` 里列出可用的工具，助手输出一段 `tool_calls`（而不是自然语言），`tool` 角色把执行结果送回来，助手再据此作答。要学的仍然只有"助手说的话"，包括那段 JSON；
- **思考过程**：`<think> …… </think>` 把推理过程包起来，训练时它和答案一样要算 loss，推理时可以选择不展示。数据里通常掺入一部分"空思考"（`<think>\n\n</think>`）的样本，模型才学得会"简单问题不必想"；
- **掺系统提示的比例**、**多轮的比例**、**长度分布**都是要调的配方；
- **找答案段的方式**：用现成的 `apply_chat_template` 渲染时拿不到每一段的边界，常见做法是渲染成文本后再**搜索**`<|im_start|>助手\n` 和 `<|im_end|>`，把中间的位置解掩码。我们是边拼边标，更直接，也不容易错。

!!! interview "怎么讲清楚"
    讲"预训练和 SFT 的区别"：目标函数没变，变的是数据和算 loss 的范围——预训练在正文上每个位置都算，SFT 只算助手回答那一段（其余标成 `-100`），所以模型结构一行不改，直接从基座接着训。再讲三个实现细节：**聊天模板**用特殊 token 划分角色，普通文本伪造不了（防提示注入）；**`<|im_end|>` 要一起学**，否则不知道在哪停；**学习率比预训练低一个量级**，数据只训两三遍，否则会把基座的知识冲掉。讲到工程时可以补一句：算 loss 的位置很稀疏，所以要做样本打包。

## 练习

**1. system 的比例。** 把 `chat_data.py` 里掺 system 的概率从 0.2 改成 0 和 1.0，各训一遍，再分别用"带 system"和"不带 system"的方式提问。有什么差别？

??? success "参考思路"
    全都带 system 训出来的模型，在不带 system 提问时往往会掉一截——它把那一段当成了格式的一部分。全不带则学不会利用系统提示。掺着来，模型两种情况都见过，才都能应付。真实的数据集一般掺入一到三成，并且系统提示本身也要有多种写法。

**2. 只学最后一轮。** 多轮样本里，把前面几轮助手的回答也标成 `-100`，只学最后一轮。训练 loss 和准确率有什么变化？为什么两种做法都有人用？

??? success "参考思路"
    只学最后一轮时，能算 loss 的位置更少，同样的步数学得更慢；但它更贴近"按完整上下文回答"的推理场景，在多轮数据质量参差时更稳。都学则样本利用率高。主流实现（包括 MiniMind）默认全部助手段都学，数据质量不稳时才退回只学最后一轮。

**3. 样本打包。** 现在一条 128 的序列里只有一成多的位置要算 loss。把多条短样本拼进同一条序列，这个比例能提到多少？要注意什么？

??? success "参考思路"
    拼接之后能提到五六成以上，训练吞吐直接翻几倍。要注意的是**样本之间不能互相看见**：需要分段的注意力掩码（或者 FlashAttention 的变长接口 `varlen`），否则后一条样本会把前一条当上下文。位置编码也要在每条样本内部重新从 0 开始。这正是[分布式训练手册的序列打包练习](train://basics/overview/)要做的事。

## 小结

- [x] 预训练和 SFT 的目标函数是同一个（预测下一个 token），差别只在数据和算 loss 的范围；模型结构一行不改，直接从基座接着训。
- [x] 聊天模板用**不可伪造的特殊 token** 划分角色；`<|im_end|>` 也要学，模型才知道在哪停；角色名必须能被分词器切出来。
- [x] 只给助手的回答打标签，系统段、用户段和 padding 都标 `-100`（交叉熵的默认 `ignore_index`）；padding 补在右边时，因果注意力保证它影响不到前面。
- [x] 扩了词表就给词嵌入加几行，其余权重照搬；新 token 的表示靠微调学出来。
- [x] SFT 的学习率比预训练低一个量级、只训两三遍；验证 loss 很快见底，按它选 checkpoint。
- [x] 算 loss 的位置很稀疏（这里只有一成多），真实训练用样本打包把这个比例提上去。
