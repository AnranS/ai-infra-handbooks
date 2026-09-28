"""第 10 章：max_extend_tokens=16 时，一个长提示词怎样被切块，以及它与其他请求的交错。"""

import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.llm import LLM
from minisgl.scheduler.prefill import ChunkedReq

ENV.DISABLE_OVERLAP_SCHEDULING.value = True
llm = LLM("models/Qwen3-0.6B", dtype=torch.float32, max_running_req=8, num_page_override=1024,
          max_seq_len_override=256, cache_type="radix", max_extend_tokens=16)
long_prompt = "Once upon a time, in a small village by the sea, there lived an old fisherman. " * 3
prompts = [long_prompt, "1 + 1 =", "The capital of France is"]
print("提示词长度:", [len(llm.tokenizer(p).input_ids) for p in prompts])
orig = llm._forward


def traced(fi):
    b = fi.batch
    parts = []
    for r in b.reqs:
        tag = "分块" if isinstance(r, ChunkedReq) else "完整"
        parts.append(f"uid{r.uid}[{r.cached_len}:{r.device_len}]{tag}" if b.is_prefill else f"uid{r.uid}")
    print(f"  {b.phase:<7} {len(b.positions):>2} 个 token: {', '.join(parts)}")
    return orig(fi)


llm._forward = traced
out = llm.generate(prompts, SamplingParams(max_tokens=3, ignore_eos=True))
llm.shutdown()
from transformers import AutoModelForCausalLM  # noqa: E402

hf = AutoModelForCausalLM.from_pretrained("models/Qwen3-0.6B", dtype=torch.float32)
refs = []
for p in prompts:
    ids = llm.tokenizer(p, return_tensors="pt").input_ids
    refs.append(hf.generate(ids, max_new_tokens=3, min_new_tokens=3, do_sample=False)[0, ids.shape[1]:].tolist())
print("与 HF（整段 prefill）的输出一致:", [o["token_ids"] for o in out] == refs)
