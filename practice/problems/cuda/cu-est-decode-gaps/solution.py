BW = 3.35e12 * 0.85


def decode_kernels(hidden, inter, layers, heads, kv_heads, head_dim, vocab, batch=1, context=1024, nbytes=2):
    q, kv = heads * head_dim, kv_heads * head_dim
    act = batch * hidden * nbytes
    layer = [
        ("add_rmsnorm", 4 * act),
        ("qkv_proj", hidden * (q + 2 * kv) * nbytes),
        ("qk_norm_rope", 2 * batch * (q + kv) * nbytes),
        ("kv_cache_write", 2 * batch * kv * nbytes),
        ("attention", batch * context * 2 * kv * nbytes),
        ("o_proj", q * hidden * nbytes),
        ("add_rmsnorm", 4 * act),
        ("gate_up_proj", hidden * 2 * inter * nbytes),
        ("silu_mul", 3 * batch * inter * nbytes),
        ("down_proj", inter * hidden * nbytes),
    ]
    tail = [("final_norm", 2 * act), ("lm_head", hidden * vocab * nbytes), ("sample", batch * vocab * 4)]
    return layer * layers + tail


def step_time_us(kernels, gap_us, bw=BW):
    t_data = sum(b for _, b in kernels) / bw * 1e6
    total = t_data + len(kernels) * gap_us
    return t_data, total, len(kernels) * gap_us / total


def eager_time_us(kernels, cpu_launch_us=5.0, gap_us=0.5, bw=BW):
    t_data = sum(b for _, b in kernels) / bw * 1e6
    return max(len(kernels) * cpu_launch_us, t_data + len(kernels) * gap_us)


def fused_step_us(kernels, layers, per_layer_before, per_layer_after, gap_us, bw=BW):
    t_data, total, _ = step_time_us(kernels, gap_us, bw)
    return total - layers * (per_layer_before - per_layer_after) * gap_us
