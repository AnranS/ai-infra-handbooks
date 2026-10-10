"""阶段一的里程碑：同样的提示词、同样的 16 个 token，但骨架已经换成正式结构。

和第 0 步比两件事：单个请求的 tokens/s（正式结构在这里并不占便宜），
以及骨架根本做不到的——把 4 个请求放进一个 batch，一步算完。
"""

import time

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "models/Qwen3-0.6B"
engine = Engine(EngineConfig(model_path=path, tp_info=DistributedInfo(0, 1), dtype=torch.float32,
                             max_running_req=4, num_page_override=1024, max_seq_len_override=256))
tok = AutoTokenizer.from_pretrained(path)
print(f"正式结构：{engine.device}，注意力后端 {engine.config.attention_backend}，"
      f"KV 池 {engine.num_pages} 页，{torch.get_num_threads()} 个线程（Engine 自己设的）")


def make_reqs(prompts, n):
    reqs = []
    for row, p in enumerate(prompts):
        engine.page_table[row, :256] = torch.arange(row * 256, row * 256 + 256)   # 手工"分配"KV 位置
        reqs.append(Req(input_ids=torch.tensor(tok(p).input_ids, dtype=torch.int32), table_idx=row,
                        cached_len=0, output_len=n, uid=row, sampling_params=SamplingParams(max_tokens=n),
                        cache_handle=None))
    return reqs


def step(reqs, phase):
    """没有调度器：自己把 batch 的字段填好，前向 + 采样，再把新 token 写回每个请求。"""
    batch = Batch(reqs=reqs, phase=phase)
    engine.graph_runner.pad_batch(batch)
    batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).int()
    batch.input_ids = torch.cat([r.input_ids[r.cached_len:r.device_len] for r in reqs])
    batch.out_loc = torch.cat([engine.page_table[r.table_idx, r.cached_len:r.device_len] for r in reqs])
    engine.attn_backend.prepare_metadata(batch)
    out = engine.forward_batch(batch, engine.sampler.prepare(batch))
    for i, r in enumerate(reqs):
        r.append_host(out.next_tokens_cpu[i:i + 1])
    return out.next_tokens_cpu.tolist()


def generate(prompts, n=16):
    reqs = make_reqs(prompts, n)
    outs = [[] for _ in reqs]
    t0 = time.perf_counter()
    for i, t in enumerate(step(reqs, "prefill")):
        outs[i].append(t)
    t1 = time.perf_counter()
    for _ in range(n - 1):
        for i, t in enumerate(step(reqs, "decode")):
            outs[i].append(t)
    t2 = time.perf_counter()
    return outs, t1 - t0, (t2 - t1) / (n - 1)


# ------------------------------------------------------------------ 1. 和第 0 步同一个请求
prompt = "The capital of France is"
[out], t_prefill, t_step = generate([prompt])
print(f"单个请求：{tok.decode(out)!r}")
print(f"  prefill {t_prefill * 1000:.0f} ms，decode 每步 {t_step * 1000:.0f} ms（{1 / t_step:.1f} tokens/s）")

# ------------------------------------------------------------------ 2. 骨架做不到的：4 个请求一个 batch
prompts = [prompt, "List three prime numbers:", "def fibonacci(n):", "Once upon a time,"]
outs, t_prefill, t_step4 = generate(prompts)
print(f"4 个请求一个 batch：prefill {t_prefill * 1000:.0f} ms，decode 每步 {t_step4 * 1000:.0f} ms"
      f"（每步出 4 个 token，合计 {4 / t_step4:.1f} tokens/s；一步只比单请求慢 {(t_step4 / t_step - 1) * 100:.0f}%）")
written = sum(len(tok(p).input_ids) + 15 for p in prompts)               # 每个请求：提示词 + 15 个送回去的新 token
print(f"KV 池写了 {written} / {engine.num_pages} 页：每个 token 只写自己那一页，没有任何拷贝")

hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
for p, o in zip(prompts, outs):
    ids = tok(p, return_tensors="pt").input_ids
    with torch.no_grad():
        ref = hf.generate(ids, max_new_tokens=16, do_sample=False)[0, ids.shape[1]:].tolist()
    print(f"  {p!r:30} -> {tok.decode(o)[:32]!r:36} 与 HF 一致: {o == ref}")
print(f"单独跑和放进 4 个请求的 batch 里跑，第一个请求的 16 个 token 完全相同：{outs[0] == out}")
engine.shutdown()
