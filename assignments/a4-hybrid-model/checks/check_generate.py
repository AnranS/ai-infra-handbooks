"""检查三：贪心解码与 Hugging Face 逐 token 一致——单个请求、变长批处理、分块 prefill、请求槽复用"""
import sys

from common import LONG_PROMPT, PROMPTS, first_diff, hf_greedy, minisgl_generate, report

N = 24
cases = {
    "单个请求": dict(prompts=PROMPTS[:1], kw={}),
    "5 个请求一起跑（提示词长短不一）": dict(prompts=PROMPTS, kw={}),
    "分块 prefill（每步最多 16 个 token）": dict(prompts=[LONG_PROMPT, PROMPTS[1]], kw={"max_extend_tokens": 16}),
    "请求槽复用（最多 2 个并发，5 个请求）": dict(prompts=PROMPTS, kw={"max_running_req": 2}),
}
only = sys.argv[1:]
problems = []
for name, case in cases.items():
    if only and not any(o in name for o in only):
        continue
    want = hf_greedy(case["prompts"], N)
    got = minisgl_generate(case["prompts"], N, **case["kw"])
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            problems.append(f"{name}：第 {i} 个请求从第 {first_diff(g, w)} 个 token 起不一致")
    print(f"  {'ok ' if not any(p.startswith(name) for p in problems) else 'BAD'} {name}", flush=True)
report("generate", not problems, "；".join(problems))
