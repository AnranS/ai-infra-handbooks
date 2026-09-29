def decode_step_ms(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp=1, n_allreduce=0, allreduce_us=0.0):
    per_gpu = (weight_bytes + batch * avg_ctx * kv_bytes_per_token) / tp
    return per_gpu / (bw_gbs * 1e9) * 1e3 + n_allreduce * allreduce_us / 1e3


def decode_tokens_per_s(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp=1, n_allreduce=0,
                        allreduce_us=0.0):
    step = decode_step_ms(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp, n_allreduce, allreduce_us)
    return batch / (step / 1e3)


def prefill_ms(n_params, prompt_len, n_layers, d_attn, peak_tflops, mfu=0.5, tp=1):
    flops = 2 * n_params * prompt_len + 2 * n_layers * d_attn * prompt_len ** 2
    return flops / (tp * peak_tflops * 1e12 * mfu) * 1e3
