"""第 6 章：没有调度器，手工驱动 Engine 做 prefill + decode，与 HF generate 对比。"""

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "models/Qwen3-0.6B"
engine = Engine(EngineConfig(model_path=path, tp_info=DistributedInfo(0, 1), dtype=torch.float32,
                             max_running_req=4, num_page_override=1024, max_seq_len_override=256))
print(f"设备 {engine.device}，注意力后端 {engine.config.attention_backend}，KV 池 {engine.num_pages} 页，"
      f"page table {tuple(engine.page_table.shape)}")

tok = AutoTokenizer.from_pretrained(path)
prompts = ["The capital of France is", "def fibonacci(n):"]
reqs = []
for row, p in enumerate(prompts):
    ids = tok(p).input_ids
    engine.page_table[row, :256] = torch.arange(row * 256, row * 256 + 256)  # 手工"分配"KV 位置
    reqs.append(Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=row, cached_len=0,
                    output_len=8, uid=row, sampling_params=SamplingParams(max_tokens=8),
                    cache_handle=None))

outs = [[] for _ in reqs]
phase = "prefill"
for step in range(8):
    batch = Batch(reqs=reqs, phase=phase)
    engine.graph_runner.pad_batch(batch)
    batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).int()
    batch.input_ids = torch.cat([r.input_ids[r.cached_len:r.device_len] for r in reqs])
    batch.out_loc = torch.cat([engine.page_table[r.table_idx, r.cached_len:r.device_len] for r in reqs])
    engine.attn_backend.prepare_metadata(batch)
    out = engine.forward_batch(batch, engine.sampler.prepare(batch))  # 前向 + 采样，并推进 Req 状态
    for i, r in enumerate(reqs):
        r.append_host(out.next_tokens_cpu[i:i + 1])
        outs[i].append(int(out.next_tokens_cpu[i]))
    if step == 0:
        print("prefill 输入 token 数:", len(batch.input_ids), " 之后 decode 每轮:", len(reqs))
    phase = "decode"

hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
for p, o in zip(prompts, outs):
    ids = tok(p, return_tensors="pt").input_ids
    ref = hf.generate(ids, max_new_tokens=8, do_sample=False)[0, ids.shape[1]:].tolist()
    print(f"{p!r}: {tok.decode(o)!r}  与 HF 一致: {o == ref}")
engine.shutdown()
