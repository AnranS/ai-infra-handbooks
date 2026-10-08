"""sim.py —— 推理服务的离散事件模拟器：连续批处理 + 分块 prefill，每一步的耗时用屋顶线模型估算。

它不运行模型，只模拟调度与时间，用来回答"在这个负载下 TTFT/TPOT 会是多少、一张卡能扛多少 QPS"。
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class Setup:
    params: float                  # 每个 token 参与计算的参数量（MoE 取激活参数）
    weight_bytes: float            # 每步要读的权重字节数
    kv_bytes_per_token: float
    kv_capacity_tokens: int        # 显存能放下的 KV token 总数
    peak_flops: float = 989e12     # H100 BF16 稠密峰值
    bandwidth: float = 3.35e12     # H100 HBM3
    mfu: float = 0.5               # prefill 能达到的算力利用率
    step_overhead: float = 0.5e-3  # 每步的固定开销（调度、kernel 启动，已用 CUDA Graph）
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256


@dataclass
class SimRequest:
    arrival: float
    input_len: int
    output_len: int
    computed: int = 0              # 已计算的提示词 token
    generated: int = 0
    first_token: float | None = None
    finish: float | None = None
    token_times: list = field(default_factory=list)


def step_time(s: Setup, num_tokens: int, context_tokens: int) -> float:
    compute = 2 * s.params * num_tokens / (s.peak_flops * s.mfu)
    memory = (s.weight_bytes + s.kv_bytes_per_token * context_tokens) / s.bandwidth
    return max(compute, memory) + s.step_overhead


def simulate(s: Setup, rate: float, num_requests: int, input_len: int, output_len: int, seed: int = 0):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for _ in range(num_requests):                                  # 泊松到达：间隔服从指数分布
        t += rng.expovariate(rate)
        reqs.append(SimRequest(t, input_len, output_len))
    now, i, waiting, running, kv_used = 0.0, 0, [], [], 0
    while i < len(reqs) or waiting or running:
        while i < len(reqs) and reqs[i].arrival <= now:
            waiting.append(reqs[i])
            i += 1
        if not waiting and not running:
            now = reqs[i].arrival                                  # 空闲：直接跳到下一个请求到达
            continue
        budget = s.max_num_batched_tokens
        batch = []                                                 # (请求, 本步 token 数)
        for r in running:                                          # 运行中的请求：decode 或继续分块 prefill
            n = 1 if r.computed == r.input_len else min(r.input_len - r.computed, budget)
            if n <= 0:
                break
            batch.append((r, n))
            budget -= n
        while waiting and budget > 0 and len(running) < s.max_num_seqs:
            r = waiting[0]
            if kv_used + r.input_len + r.output_len > s.kv_capacity_tokens:
                break                                              # 显存不够：按最终长度预留，不接收
            waiting.pop(0)
            running.append(r)
            kv_used += r.input_len + r.output_len
            n = min(r.input_len, budget)
            batch.append((r, n))
            budget -= n
        context = sum(r.computed + r.generated for r in running)
        now += step_time(s, sum(n for _, n in batch), context)
        for r, n in batch:
            if r.computed < r.input_len:
                r.computed += n
                if r.computed < r.input_len:
                    continue                                       # 分块 prefill 还没算完，不产生 token
            r.generated += 1
            r.token_times.append(now)
            if r.first_token is None:
                r.first_token = now
            if r.generated == r.output_len:
                r.finish = now
                running.remove(r)
                kv_used -= r.input_len + r.output_len
    return reqs


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(math.ceil(p / 100 * len(values))) - 1)]


def summarize(reqs, ttft_slo: float, tpot_slo: float) -> dict:
    ttft = [r.first_token - r.arrival for r in reqs]
    tpot = [(r.finish - r.first_token) / (r.output_len - 1) for r in reqs]
    duration = max(r.finish for r in reqs) - min(r.arrival for r in reqs)
    good = sum(a <= ttft_slo and b <= tpot_slo for a, b in zip(ttft, tpot))
    return {"ttft_p50": percentile(ttft, 50), "ttft_p99": percentile(ttft, 99),
            "tpot_p50": percentile(tpot, 50), "tpot_p99": percentile(tpot, 99),
            "output_tps": sum(r.output_len for r in reqs) / duration, "goodput": good / duration,
            "slo_ok": good / len(reqs)}
