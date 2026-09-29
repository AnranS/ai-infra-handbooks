import math


def max_batch_by_kv(kv_budget_bytes, avg_ctx, kv_bytes_per_token):
    pass


def step_ms(batch, weight_bytes, kv_bytes_per_token, avg_ctx, bw_gbs, overhead_ms=0.0):
    pass


def plan(qps, input_len, output_len, tpot_ms, weight_bytes, kv_bytes_per_token, gpus_per_instance, gpu_mem_gb,
         gpu_bw_gbs, mem_util=0.9, reserve_gb=4.0, overhead_ms=0.0, headroom=0.7):
    pass
