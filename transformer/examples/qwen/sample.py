"""采样：从 logits 里挑下一个 token。温度、top-k、top-p、重复惩罚，以及什么时候停。"""
import torch

from qwen.cache import State


def sample(logits, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, presence_penalty=0.0, generated=()):
    """logits: [词表] 的分数 -> 一个 token id。temperature=0 就是贪心（永远取最高分）。"""
    logits = logits.float().clone()
    if generated and (repetition_penalty != 1.0 or presence_penalty):
        seen = torch.tensor(sorted(set(generated)), device=logits.device)
        s = logits[seen]
        s = torch.where(s > 0, s / repetition_penalty, s * repetition_penalty)             # 重复惩罚（乘法）：出现过的词正分压低、负分压得更低
        logits[seen] = s - presence_penalty                                                # 出现惩罚（减法）：出现过就统一扣分，Qwen 官方推荐用这个
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature                                                          # 温度越低，分布越尖
    if top_k > 0:
        kth = logits.topk(top_k).values[-1]
        logits[logits < kth] = float("-inf")                                               # 只留分数最高的 k 个
    if top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p                                            # 累计概率超过 p 之后的都去掉（刚好跨线的那个留下）
        sorted_logits[drop] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, order, sorted_logits)
    return int(torch.multinomial(logits.softmax(-1), 1))


def generate(model, ids, max_new_tokens=200, stop=(), on_token=None, **kw):
    """ids: [1, T] 的提示词 -> 生成的 token 列表。kw 原样传给 sample；每生成一个 token 调一次 on_token。"""
    state = State(model)
    logits = model(ids, state)[0, -1]                                                      # prefill：整段提示词一次算完
    out = []
    for _ in range(max_new_tokens):
        nxt = sample(logits, generated=out, **kw)
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if nxt in stop:
            break
        logits = model(torch.tensor([[nxt]], device=ids.device), state)[0, -1]              # decode：每次只算一个新 token
    return out
