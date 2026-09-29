import math


def max_batch_by_kv(kv_budget_bytes, avg_ctx, kv_bytes_per_token):
    return max(0, int(kv_budget_bytes // (avg_ctx * kv_bytes_per_token)))


def step_ms(batch, weight_bytes, kv_bytes_per_token, avg_ctx, bw_gbs, overhead_ms=0.0):
    return (weight_bytes + batch * avg_ctx * kv_bytes_per_token) / (bw_gbs * 1e9) * 1e3 + overhead_ms


def plan(qps, input_len, output_len, tpot_ms, weight_bytes, kv_bytes_per_token, gpus_per_instance, gpu_mem_gb,
         gpu_bw_gbs, mem_util=0.9, reserve_gb=4.0, overhead_ms=0.0, headroom=0.7):
    avg_ctx = input_len + output_len / 2
    per_seq = avg_ctx * kv_bytes_per_token
    kv_budget = gpus_per_instance * (gpu_mem_gb * mem_util - reserve_gb) * 1e9 - weight_bytes
    bw = gpus_per_instance * gpu_bw_gbs
    by_kv = max_batch_by_kv(kv_budget, avg_ctx, kv_bytes_per_token)
    by_tpot = math.floor(((tpot_ms - overhead_ms) / 1e3 * bw * 1e9 - weight_bytes) / per_seq)
    batch = min(by_kv, by_tpot)
    if batch < 1:
        raise ValueError("一个实例连 1 个请求都满足不了：换更多卡或放宽 SLO")
    step = step_ms(batch, weight_bytes, kv_bytes_per_token, avg_ctx, bw, overhead_ms)
    rps = batch / (output_len * step / 1e3)
    instances = math.ceil(qps / (rps * headroom))
    return {"batch": batch, "step_ms": step, "rps_per_instance": rps, "instances": instances,
            "gpus": instances * gpus_per_instance}
