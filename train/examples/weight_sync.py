import torch

torch.manual_seed(0)
H, HEADS, KV_HEADS, D = 64, 8, 2, 8               # 隐藏维度、Q 头数、KV 头数（GQA）、头维度
GROUP = HEADS // KV_HEADS                         # 每个 KV 头服务 4 个 Q 头
Wq, Wk, Wv = torch.randn(HEADS * D, H), torch.randn(KV_HEADS * D, H), torch.randn(KV_HEADS * D, H)
Wo = torch.randn(H, HEADS * D)
x = torch.randn(5, H)


def attention(q, k, v):
    """q: [T, 头数, D]，k/v: [T, KV 头数, D]；每 GROUP 个 Q 头共用一个 KV 头"""
    rep = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("thd,shd->hts", q, k) / D**0.5
    s = s.masked_fill(torch.ones(len(q), len(q)).triu(1).bool(), float("-inf"))
    return torch.einsum("hts,shd->thd", s.softmax(-1), v).reshape(len(q), -1)


ref = attention((x @ Wq.T).view(5, HEADS, D), (x @ Wk.T).view(5, KV_HEADS, D), (x @ Wv.T).view(5, KV_HEADS, D)) @ Wo.T

# ---- 训练端：Megatron 风格 TP=2，q/k/v 各自按头切（列并行），o 按输入维切（行并行）
TRAIN_TP = 2
train = [dict(q=Wq.chunk(TRAIN_TP)[r], k=Wk.chunk(TRAIN_TP)[r], v=Wv.chunk(TRAIN_TP)[r], o=Wo.chunk(TRAIN_TP, 1)[r])
         for r in range(TRAIN_TP)]


def gather(name, dim=0):
    return torch.cat([t[name] for t in train], dim)      # 第 1 步：把训练端的分片拼回完整权重


# ---- 推理端：q/k/v 融合成一个 qkv_proj，按 TP 切；KV 头比 TP 度数少时，每个 KV 头复制到多个 rank 上
def reshard(tp):
    q, k, v, o = gather("q"), gather("k"), gather("v"), gather("o", 1)
    q_per, kv_per = HEADS // tp, max(1, KV_HEADS // tp)
    shards = []
    for r in range(tp):
        h0 = r * q_per
        kv0 = h0 // GROUP                                 # 这几个 Q 头对应的第一个 KV 头
        shards.append(dict(qkv=torch.cat([q[h0 * D:(h0 + q_per) * D], k[kv0 * D:(kv0 + kv_per) * D],
                                          v[kv0 * D:(kv0 + kv_per) * D]]),
                           o=o[:, h0 * D:(h0 + q_per) * D]))
    return shards


def reshard_naive(tp):
    qkv = torch.cat([gather("q"), gather("k"), gather("v")])      # 先拼出完整的 qkv，再按行均分
    return [dict(qkv=w, o=o) for w, o in zip(qkv.chunk(tp), gather("o", 1).chunk(tp, 1))]


def infer_forward(shards):
    tp = len(shards)
    q_per, kv_per = HEADS // tp, max(1, KV_HEADS // tp)
    out = 0
    for w in shards:                                              # 每个 rank 算自己的头，最后 all-reduce（这里直接相加）
        q, k, v = (x @ w["qkv"].T).split([q_per * D, kv_per * D, kv_per * D], -1)
        out = out + attention(q.view(5, q_per, D), k.view(5, kv_per, D), v.view(5, kv_per, D)) @ w["o"].T
    return out


for tp in (2, 4):
    for name, fn in (("正确", reshard), ("朴素", reshard_naive)):
        try:
            err = f"最大误差 {(infer_forward(fn(tp)) - ref).abs().max():.1e}"
        except RuntimeError:
            err = "形状对不上，报错"
        print(f"推理 TP={tp}，{name}的重切分：{err}")

for r in range(4):                                                # 每个推理 rank 只需要一个训练 rank 的数据：可以点对点直传
    heads = range(r * 2, r * 2 + 2)
    src = sorted({h // (HEADS // TRAIN_TP) for h in heads})
    print(f"推理 rank {r} ← 训练 rank {src}：Q 头 {heads[0]}～{heads[-1]}，KV 头 {heads[0] // GROUP}")
