"""decode_gaps.py —— 把 decode 一步的时间拆成"读写数据"和"kernel 边界的空隙"，看空隙占多少。

假设：H100 的带宽 3.35 TB/s，kernel 能跑到其中的 85%；decode 的每个 kernel 都受访存限制，时间 = 读写字节数 / 有效带宽。
kernel 之间的空隙 g 取 0.5～3 µs 几档看敏感度；逐个启动（不用 CUDA Graph）时，CPU 发射每个 kernel 算 5 µs。
"""

BW = 3.35e12 * 0.85                  # 有效带宽，字节/秒
CPU_LAUNCH_US = 5.0
GAPS_US = (0.5, 1.0, 2.0, 3.0)
MODELS = {  # Qwen3 的配置（hidden、intermediate、层数、Q 头数、KV 头数、head_dim、词表）
    "Qwen3-0.6B": dict(hidden=1024, inter=3072, layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936),
    "Qwen3-8B": dict(hidden=4096, inter=12288, layers=36, heads=32, kv_heads=8, head_dim=128, vocab=151936),
}


def kernels(cfg, batch, context, nbytes=2):
    """decode 一步的 kernel 列表：(名字, 读写的字节数)。融合程度参照 vLLM：残差加 + RMSNorm、QK-Norm + RoPE 各是一个 kernel"""
    h, inter = cfg["hidden"], cfg["inter"]
    q, kv = cfg["heads"] * cfg["head_dim"], cfg["kv_heads"] * cfg["head_dim"]
    act = batch * h * nbytes
    layer = [
        ("add_rmsnorm", 4 * act),
        ("qkv_proj", h * (q + 2 * kv) * nbytes),
        ("qk_norm_rope", 2 * batch * (q + kv) * nbytes),
        ("kv_cache_write", 2 * batch * kv * nbytes),
        ("attention", batch * context * 2 * kv * nbytes),
        ("o_proj", q * h * nbytes),
        ("add_rmsnorm", 4 * act),
        ("gate_up_proj", h * 2 * inter * nbytes),
        ("silu_mul", 3 * batch * inter * nbytes),
        ("down_proj", inter * h * nbytes),
    ]
    tail = [("final_norm", 2 * act), ("lm_head", h * cfg["vocab"] * nbytes), ("sample", batch * cfg["vocab"] * 4)]
    return layer * cfg["layers"] + tail


def cell(text, width):
    """按显示宽度右对齐（汉字占两格）"""
    w = sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)
    return " " * (width - w) + text


print("CUDA Graph 下 decode 一步的时间（括号里是 kernel 边界的空隙所占的比例）")
print(cell("场景", 22) + cell("kernel 数", 10) + cell("读写数据", 10) + cell("逐个启动", 10)
      + "".join(cell(f"g = {g:g} µs", 16) for g in GAPS_US))
for name, batch, context in (("Qwen3-0.6B", 1, 1024), ("Qwen3-8B", 1, 1024), ("Qwen3-0.6B", 64, 4096)):
    ks = kernels(MODELS[name], batch, context)
    n = len(ks)
    t_data = sum(b for _, b in ks) / BW * 1e6                 # µs
    t_eager = max(n * CPU_LAUNCH_US, t_data + n * GAPS_US[0])  # CPU 发射跟不上时，GPU 等 CPU
    print(cell(f"{name[6:]} b={batch} ctx={context}", 22) + cell(str(n), 10) + cell(f"{t_data:.0f} µs", 10)
          + cell(f"{t_eager:.0f} µs", 10) + "".join(cell(f"{t_data + n * g:.0f} µs ({n * g / (t_data + n * g):.0%})", 16)
                                                    for g in GAPS_US))

ks = kernels(MODELS["Qwen3-0.6B"], 1, 1024)
small = sum(1 for _, b in ks if b / BW * 1e6 < 1.0)
print(f"Qwen3-0.6B b=1：{small} / {len(ks)} 个 kernel 读写数据的时间不到 1 µs，平均每个 {sum(b for _, b in ks) / BW * 1e6 / len(ks):.2f} µs")
