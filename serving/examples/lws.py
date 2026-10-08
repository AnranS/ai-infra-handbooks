"""多机多卡的副本组（LeaderWorkerSet 的模型）与 PD 分离的配比。

一个"副本"不再是一个 Pod，而是"1 个 leader + (size-1) 个 worker"的一组 Pod：
一起创建、一起就绪、一起重启、一起被扩缩容。
"""


def group_pods(name, replicas, size):
    """返回每个副本组的 Pod 名字：<name>-<组号> 是 leader，<name>-<组号>-<序号> 是 worker"""
    out = []
    for g in range(replicas):
        pods = [f"{name}-{g}"] + [f"{name}-{g}-{i}" for i in range(1, size)]
        out.append(pods)
    return out


def gpus_needed(replicas, size, gpus_per_pod):
    return replicas * size * gpus_per_pod


def restart_blast_radius(size, policy):
    """一个 worker 挂掉时受影响的 Pod 数：整组重启 vs 只重启这一个"""
    return size if policy == "RecreateGroupOnHostFailure" else 1


def pd_ratio(prefill_ms, decode_ms_per_token, out_tokens, prefill_gpus, decode_gpus,
             decode_batch=64, prefill_batch=1):
    """PD 分离的配比：让两边的"每秒能处理多少个请求"匹配。

    prefill 实例受算力限制，一次基本只服务一个请求，每个请求占用 prefill_ms；
    decode 实例受带宽限制，一步同时推进 decode_batch 个请求，所以每个请求的实际占用是
    out_tokens * decode_ms_per_token / decode_batch。两边每秒的处理能力相等时配比最优。
    """
    per_request_prefill = prefill_ms / prefill_batch
    per_request_decode = out_tokens * decode_ms_per_token / decode_batch
    ideal = per_request_decode / per_request_prefill
    actual = decode_gpus / prefill_gpus
    bottleneck = "prefill" if actual > ideal else "decode"
    return round(ideal, 2), round(actual, 2), bottleneck


def kv_transfer_ms(kv_bytes, link_gbs):
    """PD 分离要把 KV 从 prefill 实例搬到 decode 实例"""
    return kv_bytes / (link_gbs * 1e9) * 1000
