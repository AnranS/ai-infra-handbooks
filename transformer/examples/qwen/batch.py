"""批处理：把几个长短不一的请求拼成一批，一次前向算所有人的下一个 token。

拼批要靠填充（padding）：短的请求左边补占位 token，mask 标出哪些位置是假的。
全注意力里假位置的 key 不许被看；线性注意力里假位置的输入清零，不写进记忆表；位置编号也要跳过假位置。
"""
import torch

from qwen.cache import State
from qwen.sample import sample


def pad_left(seqs, pad_id=0):
    """[[...], [...]] 长短不一的 token 列表 -> ids [B, T]（左边补 pad_id）和 mask [B, T]（True 是真 token）"""
    T = max(len(s) for s in seqs)
    ids = torch.tensor([[pad_id] * (T - len(s)) + s for s in seqs])
    mask = torch.tensor([[False] * (T - len(s)) + [True] * len(s) for s in seqs])
    return ids, mask


@torch.no_grad()
def generate_batch(model, seqs, max_new_tokens=100, stop=(), **kw):
    """seqs: 每个请求的提示词 token 列表 -> 每个请求生成的 token 列表（停止符之后的不算）"""
    ids, mask = pad_left(seqs)
    state = State(model, batch=len(seqs))
    logits = model(ids, state, mask)[:, -1]                                   # prefill：一次算整批
    outs, done = [[] for _ in seqs], [False] * len(seqs)
    for _ in range(max_new_tokens):
        nxt = [sample(logits[b], generated=outs[b], **kw) for b in range(len(seqs))]
        for b, t in enumerate(nxt):
            if not done[b]:
                outs[b].append(t)
                done[b] = t in stop
        if all(done):
            break
        logits = model(torch.tensor(nxt)[:, None], state)[:, -1]                 # decode：整批各进一个 token（做完的也陪跑）
    return outs
