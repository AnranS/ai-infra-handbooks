def _heads_of(rank, tp, heads, kv_heads):
    """推理 rank 负责的 Q 头区间和 KV 头区间（按头编号）"""
    if heads % tp:
        raise ValueError("Q 头数必须能被 TP 整除")
    if kv_heads >= tp and kv_heads % tp or kv_heads < tp and tp % kv_heads:
        raise ValueError("KV 头数与 TP 不能互相整除")
    q_per = heads // tp
    kv_per = max(1, kv_heads // tp)
    q0 = rank * q_per
    kv0 = q0 // (heads // kv_heads)                  # 第一个 Q 头所属的 KV 头
    return (q0, q0 + q_per), (kv0, kv0 + kv_per)


def fused_qkv_rows(rank, tp, heads, kv_heads, d):
    (q0, q1), (k0, k1) = _heads_of(rank, tp, heads, kv_heads)
    return [("q", q0 * d, q1 * d), ("k", k0 * d, k1 * d), ("v", k0 * d, k1 * d)]


def train_sources(rank, infer_tp, train_tp, heads, kv_heads):
    if kv_heads % train_tp or heads % train_tp:
        raise ValueError("训练端的 TP 必须能整除 Q 头数和 KV 头数")
    (q0, q1), (k0, k1) = _heads_of(rank, infer_tp, heads, kv_heads)
    q_train, kv_train = heads // train_tp, kv_heads // train_tp
    src = {h // q_train for h in range(q0, q1)} | {h // kv_train for h in range(k0, k1)}
    return sorted(src)
