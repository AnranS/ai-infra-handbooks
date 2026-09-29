def train_flops(n_params, n_tokens):
    return 6 * n_params * n_tokens


def gpu_hours(total_flops, peak_tflops, mfu):
    return total_flops / (peak_tflops * 1e12 * mfu) / 3600


def flops_per_token(n_params, n_layers, d_attn, seq_len):
    return 6 * n_params + 12 * n_layers * d_attn * seq_len


def mfu(tokens_per_s, n_gpus, peak_tflops, n_params, n_layers, d_attn, seq_len):
    used = tokens_per_s * flops_per_token(n_params, n_layers, d_attn, seq_len)
    return used / (n_gpus * peak_tflops * 1e12)


def hfu(tokens_per_s, n_gpus, peak_tflops, n_params, n_layers, d_attn, seq_len):
    executed = tokens_per_s * (8 * n_params + 16 * n_layers * d_attn * seq_len)
    return executed / (n_gpus * peak_tflops * 1e12)
