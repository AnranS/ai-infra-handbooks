def decode_step_ms(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp=1, n_allreduce=0, allreduce_us=0.0):
    pass


def decode_tokens_per_s(weight_bytes, kv_bytes_per_token, batch, avg_ctx, bw_gbs, tp=1, n_allreduce=0,
                        allreduce_us=0.0):
    pass


def prefill_ms(n_params, prompt_len, n_layers, d_attn, peak_tflops, mfu=0.5, tp=1):
    pass
