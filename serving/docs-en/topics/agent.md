# Inference under agent workloads

<p class="lead">An agent lets the model call tools: the model stops in the middle, waits for the tool to return, then carries on. To the inference engine, one agent task is <strong>a chain of requests</strong>: each request's prompt is the whole previous request plus a short new piece (a tool result, a follow-up question). This chapter works through such a chain with Qwen3.5's real template: how prefill and decode alternate, how much the prefix cache saves and where it breaks, and what the engine and the router have to do about it.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. How many requests does the engine receive for an agent task that calls two tools? What does each request's prefill mainly compute?
    2. Within one round of tool calls, which parts of the previous request can the prefix cache reuse? And when the user's follow-up starts a new round?
    3. Between two requests the tool runs; what happens to the previous step's KV in the meantime? What can the engine do?
    4. What trouble do multi-instance deployment, PD disaggregation and hybrid models each add to cache reuse for agents?

??? success "Answers (try first, then expand to compare)"
    1. Three: one for the user's question and one each time a tool result comes back. The first request prefills the system prompt, the tool definitions and the question; every later request starts exactly like the previous one, and what really needs prefilling is only the tool result appended to it.
    2. Within a round it can reuse the previous request's whole sequence, including its prompt and the thinking and tool call it generated (in this chapter's simulation, 580 of 580). In a new round the template deletes the previous round's thinking from the history; the history changes, and the cache breaks at the start of the previous round's first assistant turn, so everything of that round after it is recomputed.
    3. When a request finishes, its KV stays in the radix tree, but no request references it any more; under memory pressure LRU may evict it, and then the next step recomputes it. SGLang lets requests carry a `session_id` and evicts KV without session references first; the KV can also be offloaded to host memory (tiered caching) and read back when needed.
    4. Multiple instances: a session has to come back to the machine that holds its cache, which takes cache-aware or session-affinity routing. PD disaggregation: the KV generated in the previous step lives on the decode instance, and the next step's prefill instance does not have it, so it must be sent back or kept in a shared KV store. Hybrid models: the linear-attention layers' state can only be reused at positions where a checkpoint was saved; when the hit point falls between two checkpoints, recomputation starts from the earlier one.

## One agent task = a chain of requests {#一次-agent-任务--一串请求}

Take "look up today's high temperature in Beijing and convert it to Fahrenheit". The framework and the engine go back and forth like this:

1. **The user asks**: the framework puts the system prompt, the tool definitions and the question through the template and sends request 1. The engine prefills the whole prompt, then decodes: the model thinks a little, then emits a `<tool_call>…</tool_call>`; it hits a stop token and the request ends.
2. **The tool runs**: the framework parses the tool name and arguments and does the search. Meanwhile the engine serves other requests.
3. **The tool result comes back**: the framework appends the model's reply and the tool result to the history and sends request 2. Its start is identical to request 1's whole sequence (prompt + generated tokens), so with a prefix cache only the newly appended tool result needs prefilling, and decoding continues.
4. This repeats until the model answers without calling a tool. When the user asks a follow-up, a new round begins.

So, "is a follow-up question appended to everything before it, prefilled again, then decoded?" Yes. Every time, the model sees the whole history; the only question is how much the engine can reuse.

![Figure: prefill and decode in an agent session: every model call is a new request that prefills the part not in the cache, then decodes and stops at a tool call; the prefix cache cuts the prefill of four requests from 3001 tokens to 1037](../assets/figures/agent-turns.svg){.aig-svg}

## Counting with the real template {#用真实模板算一遍}

No model needs to run: write one session as a script (the model's replies are written in advance), encode the prompt each step sends to the engine with Qwen3.5's chat template, and compare it with the previous request's full sequence; the part that starts the same is what the prefix cache can reuse:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B")
tools = [
    {"type": "function", "function": {"name": "web_search", "description": "搜索网页，返回前几条结果的标题和摘要",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "搜索词"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "在沙箱里执行一段 Python 代码，返回标准输出",
     "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "要执行的代码"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "读取工作区里的一个文件",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]
system = {"role": "system", "content": "你是一个能调用工具的助手。先想清楚再行动，需要时调用工具，拿到结果后继续推理，直到能回答用户。" * 3}
steps = [   # the script of one session: the model's replies are written in advance (no model is run; we only look at the tokens sent to the engine at each step)
    {"role": "user", "content": "北京今天最高气温是多少摄氏度？换算成华氏度是多少？"},
    {"role": "assistant", "reasoning_content": "用户要北京今天的最高气温，并换算成华氏度。我需要先搜索今天的天气。" * 2, "content": "",
     "tool_calls": [{"type": "function", "function": {"name": "web_search", "arguments": {"query": "北京 今天 天气 最高气温"}}}]},
    {"role": "tool", "content": "1. 北京今日天气：晴转多云，气温 12～23°C，北风 3 级，空气质量良。\n" * 6},
    {"role": "assistant", "reasoning_content": "搜索结果显示最高气温 23°C。华氏度 = 摄氏度 × 9/5 + 32，用代码算一下更稳妥。" * 2, "content": "",
     "tool_calls": [{"type": "function", "function": {"name": "run_python", "arguments": {"code": "print(23 * 9 / 5 + 32)"}}}]},
    {"role": "tool", "content": "73.4"},
    {"role": "assistant", "reasoning_content": "23°C 换算成华氏度是 73.4°F，可以回答了。", "content": "北京今天最高气温约 23°C，换算成华氏度是 73.4°F。"},
    {"role": "user", "content": "那明天呢？"},
]


def encode(messages, generation_prompt):
    text = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=generation_prompt, enable_thinking=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


history, done, checkpoints = [system], [], []
total = new = new_hybrid = 0
for i, msg in enumerate(steps):
    history.append(msg)
    if msg["role"] == "assistant":                     # a model reply is not a new request; it is what the previous request decoded
        continue
    prompt = encode(history, True)
    hit = common_prefix(prompt, done)                  # the part that starts the same as the previous request's full sequence (prompt + generated tokens)
    hit_hybrid = max((c for c in checkpoints if c <= hit), default=0)   # hybrid model: the linear layers' state can only be resumed at positions where it was saved
    reply = next((m for m in steps[i + 1:] if m["role"] == "assistant"), None)
    full = encode(history + [reply], False) if reply else prompt
    why = {"user": "用户提问" if i == 0 else "用户追问（新一轮）", "tool": "工具结果回来"}[msg["role"]]
    decode = len(full) - len(prompt) if reply else "…"
    print(f"{why}：提示词 {len(prompt)}，能复用 {hit}，要 prefill {len(prompt) - hit}（混合模型 {len(prompt) - hit_hybrid}），然后 decode {decode}")
    total, new, new_hybrid = total + len(prompt), new + len(prompt) - hit, new_hybrid + len(prompt) - hit_hybrid
    checkpoints += [len(prompt), len(full)]            # assumption: a copy of the linear layers' state is kept at the end of each request's prompt and at the end of its generation
    done = full
print(f"四次请求的提示词一共 {total} 个 token：有前缀缓存时只 prefill {new} 个，混合模型 {new_hybrid} 个")

a, b = (encode([system, {"role": "user", "content": q}], True) for q in ("北京今天会下雨吗？", "帮我算一下 2 的 30 次方。"))
print(f"两个不相干的会话，第一个请求开头一样的有 {common_prefix(a, b)} 个 token：工具定义 + 系统提示词")
for where in ("开头", "末尾"):
    a, b = (encode([{"role": "system", "content": f"当前时间：2026-10-20 09:00:0{s}。" + system["content"] if where == "开头" else system["content"] + f"当前时间：2026-10-20 09:00:0{s}。"},
                    {"role": "user", "content": q}], True) for s, q in ((1, "北京今天会下雨吗？"), (7, "帮我算一下 2 的 30 次方。")))
    print(f"系统提示词{where}加上当前时间：开头一样的变成 {common_prefix(a, b)} 个")
```

```text title="output"
用户提问：提示词 506，能复用 0，要 prefill 506（混合模型 506），然后 decode 74
工具结果回来：提示词 773，能复用 580，要 prefill 193（混合模型 193），然后 decode 107
工具结果回来：提示词 898，能复用 880，要 prefill 18（混合模型 18），然后 decode 47
用户追问（新一轮）：提示词 824，能复用 504，要 prefill 320（混合模型 824），然后 decode …
四次请求的提示词一共 3001 个 token：有前缀缓存时只 prefill 1037 个，混合模型 1541 个
两个不相干的会话，第一个请求开头一样的有 486 个 token：工具定义 + 系统提示词
系统提示词开头加上当前时间：开头一样的变成 418 个
系统提示词末尾加上当前时间：开头一样的变成 502 个
```

**Within a round, almost everything is reused.** Request 2 reuses 580 tokens, exactly request 1's prompt of 506 plus its 74 generated tokens; request 3 reuses 880 = 773 + 107. In other words, the thinking and tool calls the model decoded stay in the next step's prompt as they are, the prefix cache picks them all up, and only the tool results need prefilling: 193 and 18 tokens. The four prompts add up to 3001 tokens, of which only 1037 are actually prefilled.

**A new round breaks it.** The user's follow-up reuses only 504 tokens, breaking at the start of the previous round's first assistant turn: Qwen3.5's template keeps the thinking only "after the last user message", so once the user asks again, the previous round's thinking is deleted from the history. Everything the model said, the tools it called and the results it got move to new positions, and 320 tokens are recomputed. [Chapter 12 of Transformer by Hand](transformer://chat/#多轮对话第二轮能接上多少) shows the same thing with a plain question-and-answer chat.

**Hybrid models break harder.** The numbers in brackets are for a hybrid model like Qwen3.5: the linear-attention layers' state summarizes the whole prefix and can only be resumed at positions where a checkpoint was saved (see [the linear attention chapter](../frontier/linear-attn.md#前缀缓存要重做)). Assuming checkpoints are kept only at the end of each request's prompt and the end of its generation, the hit point within a round lands exactly on the end of the previous request, so everything is still reused; but the new round's break at 504 falls before the first checkpoint at 506, so all 824 tokens are prefilled from scratch.

**What sessions share is the start.** Two unrelated sessions have the same first 486 tokens in their first request: Qwen3.5's template puts the tool definitions first, then the system prompt. Put the current time at the start of the system prompt and what is shared shrinks to the 418 tokens of tool definitions; put it at the end and everything before it is kept.

## Where it breaks, and how to avoid it {#哪里会断怎么避免}

The prefix cache only recognizes "identical from the start". Check an agent's prompts against that:

- **Rewritten history**: the template deleting old thinking (measured above), the framework compressing long tool results into summaries, dropping turns in the middle: each breaks the cache at the point of change. The earlier the rewrite, the more is recomputed.
- **Something changing at the start**: timestamps, random request IDs, a tool list in a different order every time; put at the front of the prompt, they destroy what sessions share.
- **Changed tools**: the tool definitions come first, so adding or removing one tool makes the whole cache useless.

The principle is: **stable content first, changing content last, history only appended, never rewritten**. Deleting thinking is the template's choice: keeping it lets the cache connect, but the context gets longer and the format differs from what the model saw in training; deleting it keeps the context clean, at the price of recomputing the previous round in each new one.

## Between steps: the KV has to survive the tool call {#两步之间kv-要撑过工具执行}

When a request finishes, its KV is not freed right away; it is inserted back into the radix tree, waiting to be reused by the next request (see [the prefix cache chapter](../engine/prefix-cache.md#基数树sglang-的做法)). But by then no request references it, so to the engine it is just like any other cold data; a tool can run for seconds or even minutes, and under memory pressure LRU may well evict it first, so the next step comes back to a full recomputation.

SGLang adds a **session-aware radix cache** for this situation (`--enable-session-radix-cache`): requests carry the same `session_id`; when a request finishes, its reusable cache is registered under that session; eviction takes KV without session references first, then KV with fewer session references. This is soft protection, not pinning: if that is not enough, it is still evicted. When the session ends, call `/close_session` and its cache returns to the normal eviction order. The other route is offloading KV to host memory ([KV tiered caching and offloading](../distributed/kv-offload.md)): it takes no GPU memory while the tool runs and is read back when the next step arrives, which is cheaper than recomputing.

## Multiple instances: a session has to come back to the same machine {#多实例同一个会话要回到同一台机器}

With several instances, if request 2 goes to another machine, that machine does not have request 1's cache and starts from scratch. So routing must be either **cache-aware** (estimate which prefixes each machine holds; see [the prefix cache chapter](../engine/prefix-cache.md#缓存感知的调度与路由)) or **session-affine**: always send a session to the same machine. The `SessionAwarePolicy` of SGLang Model Gateway remembers which machine each session goes to and forgets sessions that stay idle too long. The cost is possible load imbalance: when the sessions on one machine all happen to be busy, you have to choose between "wait for it" and "switch machines and recompute"; [global scheduling for disaggregated serving](../frontier/disagg-sched.md#路由缓存与排队的权衡) compares several routing policies in simulation.

## PD disaggregation: the previous step's generated KV is on the decode side {#pd-分离上一步生成的-kv-在-decode-那边}

With PD disaggregation, each request first has its prompt computed on a prefill instance, then its KV is sent to a decode instance for generation. In the next step, the new prompt starts with the previous request's "prompt + generated tokens": the prefill instance has the KV of the prompt part, but the generated part was computed on the decode instance. In the simulation above, of the 580 tokens request 2 can reuse, the prefill instance itself holds only 506. To fill the gap, either the decode instance writes it back to a shared KV store ([KV transfer engines and distributed KV storage](../comm/kv-storage.md)), or the prefill instance simply recomputes these tens to hundreds of tokens. Each agent step decodes only a little, so recomputing is often cheap too.

## The decode side: stopping at a tool call {#decode-这一侧停在工具调用上}

- **Stopping**: when the model emits `</tool_call>` or `<|im_end|>`, the request ends; decode never waits for the tool.
- **Parsing**: the engine parses the tool name and arguments out of the generated text and returns them as OpenAI-style `tool_calls`. Every model family has its own format; Qwen3.5's is `<tool_call><function=…><parameter=…>`, the same as Qwen3-Coder's.
- **Constraining**: arguments that must be valid JSON or follow a schema can be guaranteed with constrained decoding; see [structured output and tool calling](structured-output.md).

!!! source "Source code"
    The paths below refer to SGLang commit `29f6d408c0` (2026-10-01):

    - The prefix cache interface is in `python/sglang/srt/mem_cache/base_prefix_cache.py`: `match_prefix` finds the longest hit, `insert_req` hands a finished request's KV to the tree, `cache_unfinished_req` handles the intermediate state of chunked prefill.
    - The session-aware radix cache: the switch is `--enable-session-radix-cache`, the reference counting lives in `mem_cache/unified_cache/session_ref_tracker.py`, and only the unified radix tree `UnifiedRadixCache` supports it; the documentation is `docs/docs/advanced_features/session_radix_cache.mdx`. `srt/session/streaming_session.py` is a different kind of "streaming session" that keeps a session's KV in a dedicated slot between turns.
    - Routing: `SessionAwarePolicy` in `experimental/sgl-router/src/policies/session_aware.rs`.
    - Tool-call parsing: `python/sglang/srt/function_call/`; `function_call_parser.py` picks the format by `--tool-call-parser`, and Qwen3.5's format is parsed by `qwen3_coder_detector.py`.
    - Checkpoints for hybrid models: `mem_cache/mamba_checkpoint_pool.py`.
    - Benchmarking: `benchmark/agentic-rollout/` uses synthetic multi-turn sessions (generate, wait for a simulated tool delay, append the result, generate again) to compare resending the full history, ordinary sessions and streaming sessions.

!!! interview "How to explain it"
    To explain inference for agents, first reduce it to a chain of requests: every model call is a new request whose prompt is "the whole previous request + the new tool result"; the engine prefills, then decodes, and stops at a tool call. Then the prefix cache: within a round, the previous step's prompt and generated tokens are all reused and only the tool result is prefilled; in a new round the template deletes the old thinking and the cache breaks at the start of the previous round. Then three engineering points: the KV must survive the tool call between steps (session-aware eviction, offloading to host memory), a session must come back to the same machine across instances (session affinity, cache-aware routing), and with PD disaggregation the previous step's generated KV is on the decode side. Finish with the principle: stable content first, changing content last, history only appended, never rewritten.

## Exercises {#练习}

1. In this chapter's simulation, if the template kept the previous round's thinking in a new round, how much could request 4 reuse? Estimate first, then change the code to check.

??? success "Answer"
    It could reuse request 3's whole sequence: 898 + 47 = 945 tokens. Request 4's prompt becomes 959 tokens, and only the new question and a few special tokens of the template need prefilling, 14 in total. The price is a prompt longer than 824 by the previous round's thinking. Qwen3.5's template has no switch for keeping old thinking, so check it by hand: take the text of request 3's whole sequence (rendered with `add_generation_prompt=False`) and append the new user message and `<|im_start|>assistant\n<think>\n` directly.

2. An agent's tools take 5 seconds per step on average, the engine handles 100 such session steps per second, and every finished step leaves 2000 tokens of KV behind (100 KB per token). With GPU memory alone (40 GB left for caching), can this KV survive the tool calls?

??? success "Answer"
    In 5 seconds 500 session steps finish, leaving 500 × 2000 × 100 KB ≈ 100 GB of KV, far beyond 40 GB, so LRU will certainly evict most of it before the tools return. Remedies: let session caches stay ahead of cold data (session-aware eviction), offload KV to host memory or SSD, or keep less per session (for example only the system prompt and the last few steps).

## Summary {#小结}

- [x] An agent task is a chain of requests: every model call prefills, then decodes, and stops at a tool call; the prompt is the whole previous request plus the new tool result.
- [x] Counted with Qwen3.5's real template: within a round the previous request's whole sequence is reused, and four requests with 3001 prompt tokens prefill only 1037; in a new round the template deletes the old thinking and the cache breaks at the start of the previous round, so 320 tokens are recomputed, and a hybrid model even has to recompute all 824 from scratch.
- [x] Stable content first, changing content last, history only appended, never rewritten; with a timestamp at the start of the system prompt, sessions share only the tool definitions.
- [x] What the engine and the router must do: keep KV alive through tool calls (session-aware eviction, offloading to host memory), route a session back to the same machine, and fill in the previous step's generated KV under PD disaggregation.
