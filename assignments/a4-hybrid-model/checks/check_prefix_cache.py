"""检查四：radix 前缀缓存。混合模型的前缀缓存需要线性层的状态检查点——
要么在创建时明确拒绝（抛出 ValueError 或 NotImplementedError），要么命中前缀缓存之后结果仍然正确。"""
import torch
from common import MODEL, PROMPTS, first_diff, hf_greedy, report

from minisgl.core import SamplingParams
from minisgl.llm import LLM

N = 16
try:
    llm = LLM(MODEL, dtype=torch.float32, cache_type="radix", max_running_req=4, device="cpu")
except (ValueError, NotImplementedError) as e:
    report("prefix-cache", True, f"创建时明确拒绝了 radix 缓存：{e}")
first = PROMPTS[3]
second = first + " sold his boat"                            # 与第一个请求共享很长的前缀
try:
    a = llm.generate([first], SamplingParams(max_tokens=N, ignore_eos=True))[0]["token_ids"]
    b = llm.generate([second], SamplingParams(max_tokens=N, ignore_eos=True))[0]["token_ids"]
finally:
    llm.shutdown()
want_a, want_b = hf_greedy([first, second], N)
problems = [f"{name}从第 {first_diff(g, w)} 个 token 起不一致" for name, g, w in
            (("第一个请求", a, want_a), ("命中前缀缓存的第二个请求", b, want_b)) if g != w]
report("prefix-cache", not problems, "；".join(problems) or "命中前缀缓存后结果正确")
