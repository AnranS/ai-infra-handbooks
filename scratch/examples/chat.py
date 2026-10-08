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
