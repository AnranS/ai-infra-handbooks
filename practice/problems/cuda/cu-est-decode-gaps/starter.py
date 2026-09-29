BW = 3.35e12 * 0.85


def decode_kernels(hidden, inter, layers, heads, kv_heads, head_dim, vocab, batch=1, context=1024, nbytes=2):
    q, kv = heads * head_dim, kv_heads * head_dim
    layer = [
        ("qkv_proj", hidden * (q + 2 * kv) * nbytes),
        ("o_proj", q * hidden * nbytes),
        ("gate_up_proj", hidden * 2 * inter * nbytes),
        ("down_proj", inter * hidden * nbytes),
    ]                                                   # 只算了四个 GEMM，漏了逐元素的 kernel、注意力和最后三个
    return layer * layers


def step_time_us(kernels, gap_us, bw=BW):
    pass


def eager_time_us(kernels, cpu_launch_us=5.0, gap_us=0.5, bw=BW):
    pass


def fused_step_us(kernels, layers, per_layer_before, per_layer_after, gap_us, bw=BW):
    pass
