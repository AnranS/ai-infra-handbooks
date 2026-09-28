"""第 8 章：KV 池只有 64 个 token 时，准入控制怎样让请求排队。"""

import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM

ENV.DISABLE_OVERLAP_SCHEDULING.value = True
llm = LLM("models/Qwen3-0.6B", dtype=torch.float32, max_running_req=8, num_page_override=64,
          max_seq_len_override=128, cache_type="naive")
prompts = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):",
           "Once upon a time, in a small village by the sea, there lived", "1 + 1 ="]
print("提示词长度:", [len(llm.tokenizer(p).input_ids) for p in prompts], " max_tokens = 8")
orig = llm._forward


def traced(fi):
    b = fi.batch
    out = orig(fi)
    cm = llm.cache_manager
    print(f"  {b.phase:<7} uids={[r.uid for r in b.reqs]!s:<16} 等待中={len(llm.prefill_manager.pending_list)} "
          f"空闲页={len(cm.free_slots):<3} 运行中预留={llm.decode_manager.inflight_tokens}")
    return out


llm._forward = traced
out = llm.generate(prompts, SamplingParams(max_tokens=8, ignore_eos=True))
llm.cache_manager.check_integrity()
print("全部完成，空闲页:", len(llm.cache_manager.free_slots), "/ 64，完整性检查通过")
llm.shutdown()
