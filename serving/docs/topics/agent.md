# Agent 负载下的推理

<p class="lead">Agent 让模型调工具：模型说到一半停下来，等工具返回，再接着往下说。对推理引擎来说，一次 agent 任务就是<strong>一串请求</strong>：每个请求的提示词，是上一个请求的全部内容再接上一小段新的（工具结果、用户追问）。这一章用 Qwen3.5 的真实模板把这串请求算一遍：prefill 和 decode 怎么交替，前缀缓存能省多少、在哪里会断，引擎和路由又要为它做什么。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一次调了两次工具的 agent 任务，引擎收到几个请求？每个请求的 prefill 主要算的是什么？
    2. 同一轮里调工具，前缀缓存能复用上一个请求的哪些部分？用户追问开启新一轮时呢？
    3. 两次请求之间要等工具执行，这段时间里上一步的 KV 会怎样？引擎能做什么？
    4. 多实例部署、PD 分离、混合模型，各给 agent 的缓存复用添了什么麻烦？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 三个：用户提问一个，每次工具结果回来各一个。第一个请求的 prefill 是系统提示词、工具定义和问题；之后每个请求，前面都和上一个请求一样，真正要 prefill 的只是新接上的工具结果。
    2. 同一轮里能复用上一个请求的整段序列，包括它的提示词和它生成的思考、工具调用（本章的模拟里 580 个复用了 580 个）。新一轮时，模板会把上一轮的思考过程从历史里删掉，历史变了，缓存断在上一轮第一个 assistant 回合的开头，后面那一轮的内容都要重算。
    3. 请求结束后，它的 KV 留在基数树里，但没有请求再引用它，显存紧张时可能按 LRU 被淘汰，下一步就得重算。SGLang 可以让请求带上 `session_id`，淘汰时先淘汰没有会话引用的 KV；也可以把 KV 卸载到内存里（分层缓存），回来时再读回显存。
    4. 多实例：同一个会话要回到存着它缓存的那台机器，要靠缓存感知或会话亲和的路由。PD 分离：上一步生成的那段 KV 在 decode 实例上，下一步的 prefill 实例没有，要传回来或放进共享的 KV 存储。混合模型：线性注意力层的状态只能在存过检查点的位置复用，命中点落在两个检查点之间时，要从前一个检查点重算。

## 一次 agent 任务 = 一串请求

以"查一下北京今天的最高气温，换算成华氏度"为例，框架和引擎之间是这样来回的：

1. **用户提问**：框架把系统提示词、工具定义、问题套上模板，发出第 1 个请求。引擎 prefill 整段提示词，然后 decode：模型先想一想，再吐出一段 `<tool_call>…</tool_call>`，遇到结束符，请求就结束了。
2. **执行工具**：框架解析出工具名和参数去搜索。这段时间引擎在服务别的请求。
3. **工具结果回来**：框架把模型刚才的回答和工具结果都拼进历史，发出第 2 个请求。它的开头和第 1 个请求的整段序列（提示词 + 生成）一模一样，有前缀缓存时只要 prefill 新接上的工具结果，然后接着 decode。
4. 如此往复，直到模型不再调工具、直接回答。用户再追问一句，就开始新一轮。

所以"追问是不是把问题拼到之前的内容后面，重新 prefill，再 decode"：是的。模型每一次看到的都是整段历史，区别只在引擎能复用多少。

![图：Agent 会话里 prefill 和 decode 怎么切换：每调一次模型就是一个新请求，先 prefill 没命中的部分再 decode，停在工具调用上；前缀缓存把四次请求 3001 个 token 的 prefill 省到 1037 个](../assets/figures/agent-turns.svg){.aig-svg}

## 用真实模板算一遍

不用跑模型：把一次会话写成剧本（模型的回答是写好的），用 Qwen3.5 的聊天模板把每一步要送进引擎的提示词编码出来，和上一个请求的完整序列比一比，开头一样的部分就是前缀缓存能复用的：

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
steps = [   # 一次会话的剧本：模型的回答是写好的（不跑模型，只看每一步送进引擎的 token）
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
    if msg["role"] == "assistant":                     # 模型的回答不是新请求，它是上一个请求 decode 出来的
        continue
    prompt = encode(history, True)
    hit = common_prefix(prompt, done)                  # 和上一个请求的完整序列（提示词 + 生成）开头一样的部分
    hit_hybrid = max((c for c in checkpoints if c <= hit), default=0)   # 混合模型：线性层的状态只能在存过的位置接着用
    reply = next((m for m in steps[i + 1:] if m["role"] == "assistant"), None)
    full = encode(history + [reply], False) if reply else prompt
    why = {"user": "用户提问" if i == 0 else "用户追问（新一轮）", "tool": "工具结果回来"}[msg["role"]]
    decode = len(full) - len(prompt) if reply else "…"
    print(f"{why}：提示词 {len(prompt)}，能复用 {hit}，要 prefill {len(prompt) - hit}（混合模型 {len(prompt) - hit_hybrid}），然后 decode {decode}")
    total, new, new_hybrid = total + len(prompt), new + len(prompt) - hit, new_hybrid + len(prompt) - hit_hybrid
    checkpoints += [len(prompt), len(full)]            # 假设：每个请求的提示词结尾、生成结尾各留一份线性层的状态
    done = full
print(f"四次请求的提示词一共 {total} 个 token：有前缀缓存时只 prefill {new} 个，混合模型 {new_hybrid} 个")

a, b = (encode([system, {"role": "user", "content": q}], True) for q in ("北京今天会下雨吗？", "帮我算一下 2 的 30 次方。"))
print(f"两个不相干的会话，第一个请求开头一样的有 {common_prefix(a, b)} 个 token：工具定义 + 系统提示词")
for where in ("开头", "末尾"):
    a, b = (encode([{"role": "system", "content": f"当前时间：2026-10-20 09:00:0{s}。" + system["content"] if where == "开头" else system["content"] + f"当前时间：2026-10-20 09:00:0{s}。"},
                    {"role": "user", "content": q}], True) for s, q in ((1, "北京今天会下雨吗？"), (7, "帮我算一下 2 的 30 次方。")))
    print(f"系统提示词{where}加上当前时间：开头一样的变成 {common_prefix(a, b)} 个")
```

```text title="输出"
用户提问：提示词 506，能复用 0，要 prefill 506（混合模型 506），然后 decode 74
工具结果回来：提示词 773，能复用 580，要 prefill 193（混合模型 193），然后 decode 107
工具结果回来：提示词 898，能复用 880，要 prefill 18（混合模型 18），然后 decode 47
用户追问（新一轮）：提示词 824，能复用 504，要 prefill 320（混合模型 824），然后 decode …
四次请求的提示词一共 3001 个 token：有前缀缓存时只 prefill 1037 个，混合模型 1541 个
两个不相干的会话，第一个请求开头一样的有 486 个 token：工具定义 + 系统提示词
系统提示词开头加上当前时间：开头一样的变成 418 个
系统提示词末尾加上当前时间：开头一样的变成 502 个
```

**同一轮里几乎全能复用。** 第 2 个请求能复用 580 个，正好是第 1 个请求的提示词 506 加上生成的 74；第 3 个请求复用 880 = 773 + 107。也就是说，模型自己 decode 出来的思考和工具调用，下一步原样留在了提示词里，前缀缓存全接得上，要 prefill 的只是工具结果：193 个和 18 个。四次请求的提示词一共 3001 个 token，实际只 prefill 1037 个。

**新一轮会断。** 用户追问时只能复用 504 个，断在上一轮第一个 assistant 回合的开头：Qwen3.5 的模板只保留"最后一条用户消息之后"的思考过程，用户一追问，上一轮的思考就从历史里删掉了。上一轮模型说过的话、调过的工具、拿到的结果都换了位置，320 个 token 要重算。[手写 Transformer 第 12 章](transformer://chat/#多轮对话第二轮能接上多少)用一问一答的聊天演示过同一件事。

**混合模型断得更狠。** 括号里是 Qwen3.5 这种混合模型的情况：线性注意力层的状态是整段前缀的汇总，只能在存过检查点的位置接着用（见[线性注意力一章](../frontier/linear-attn.md#前缀缓存要重做)）。假设检查点只留在每个请求提示词的结尾和生成的结尾，同一轮里命中点正好落在上一个请求的结尾，照样全接得上；新一轮的断点 504 却落在第一个检查点 506 之前，只能从头 prefill 824 个。

**跨会话共享的是开头。** 两个不相干的会话，第一个请求开头有 486 个 token 一样：Qwen3.5 的模板把工具定义放在最前面，然后是系统提示词。把当前时间写进系统提示词的开头，共享的就只剩工具定义那 418 个；写在末尾，前面的都保住了。

## 哪里会断，怎么避免

前缀缓存只认"开头一模一样"。按这个标准检查 agent 的提示词：

- **历史被改写**：模板删掉旧的思考过程（上面的实测）、框架把很长的工具结果压缩成摘要、删掉中间的几轮对话，都会让缓存从改动的地方断开。改写越靠前，重算越多。
- **开头有会变的东西**：时间戳、随机的请求编号、每次顺序都不一样的工具列表，放在提示词的前面，跨会话共享的部分就没了。
- **换了工具**：工具定义在最前面，增删一个工具，整段缓存都用不上。

原则是**稳定的放前面，会变的放后面，历史只追加不改写**。删思考过程是模板的选择：留着它，缓存能接上，但上下文更长，也和模型训练时见过的格式不一样；删掉它，上下文干净，代价是新一轮要重算上一轮。

## 两步之间：KV 要撑过工具执行

一个请求结束后，它的 KV 不会马上释放，而是插回基数树，等着被下一个请求复用（见[前缀缓存一章](../engine/prefix-cache.md#基数树sglang-的做法)）。可这时已经没有请求引用它了，在引擎眼里它和别的冷数据一样；工具一跑就是几秒甚至几分钟，显存紧张时，LRU 很可能先把它淘汰掉，下一步回来就得整段重算。

SGLang 给这种场景加了**会话感知的基数缓存**（`--enable-session-radix-cache`）：请求带上同一个 `session_id`，请求结束时，它可复用的那段缓存记在这个会话名下；淘汰时先淘汰没有会话引用的，再淘汰会话引用少的。这只是软保护，不是锁定，实在不够还是会被淘汰；会话结束要调用 `/close_session`，那段缓存回到普通的淘汰顺序。另一条路是把 KV 卸载到内存里（[KV 分层缓存与卸载](../distributed/kv-offload.md)），工具执行期间占不了显存，回来时再读回去，比重算便宜。

## 多实例：同一个会话要回到同一台机器

多个实例时，第 2 个请求要是被发到另一台机器上，那台机器没有第 1 个请求的缓存，只能从头算。所以路由要么做**缓存感知**（估计每台机器缓存了哪些前缀，见[前缀缓存一章](../engine/prefix-cache.md#缓存感知的调度与路由)），要么做**会话亲和**：同一个会话一直发往同一台机器，SGLang Model Gateway 的 `SessionAwarePolicy` 就是按会话记住发往哪台，空闲太久的会话再忘掉。代价是负载可能不均：某台机器上的会话正好都很忙，就要在"等它"和"换一台重算"之间权衡，[分离式架构的全局调度](../frontier/disagg-sched.md#路由缓存与排队的权衡)一章用模拟比较过几种路由。

## PD 分离：上一步生成的 KV 在 decode 那边

PD 分离时，每个请求先在 prefill 实例上算提示词，再把 KV 传给 decode 实例生成。到了下一步，新提示词的开头是上一个请求的"提示词 + 生成"：提示词那段的 KV prefill 实例上有，生成那段却是在 decode 实例上算出来的。上面的模拟里，第 2 个请求能复用的 580 个，prefill 实例自己只有 506 个。要补上这一段，要么让 decode 实例把它写回共享的 KV 存储（[KV 传输引擎与分布式 KV 存储](../comm/kv-storage.md)），要么干脆在 prefill 实例上重算这几十到几百个 token。Agent 的每一步 decode 都不长，重算往往也不贵。

## decode 这一侧：停在工具调用上

- **停止**：模型吐出 `</tool_call>` 或 `<|im_end|>` 就结束这个请求，不会在 decode 里等工具。
- **解析**：引擎从生成的文本里把工具名和参数解析出来，按 OpenAI 接口的 `tool_calls` 返回。各家模型的格式不同，Qwen3.5 是 `<tool_call><function=…><parameter=…>`，和 Qwen3-Coder 一样。
- **约束**：参数要是合法的 JSON 或者符合 schema，可以在 decode 时用约束解码保证，见[结构化输出与工具调用](structured-output.md)。

!!! source "源码对照"
    以下路径对应 SGLang 提交 `29f6d408c0`（2026-10-01）：

    - 前缀缓存的接口在 `python/sglang/srt/mem_cache/base_prefix_cache.py`：`match_prefix` 找最长命中，`insert_req` 把结束的请求的 KV 交给树，`cache_unfinished_req` 处理分块 prefill 的中间状态。
    - 会话感知的基数缓存：开关 `--enable-session-radix-cache`，引用计数在 `mem_cache/unified_cache/session_ref_tracker.py`，只有统一基数树 `UnifiedRadixCache` 支持；文档是 `docs/docs/advanced_features/session_radix_cache.mdx`。`srt/session/streaming_session.py` 是另一种"流式会话"，两轮之间把会话的 KV 留在一个专属的槽里。
    - 路由：`experimental/sgl-router/src/policies/session_aware.rs` 的 `SessionAwarePolicy`。
    - 工具调用解析：`python/sglang/srt/function_call/`，`function_call_parser.py` 按 `--tool-call-parser` 选格式，Qwen3.5 这种格式由 `qwen3_coder_detector.py` 解析。
    - 混合模型的检查点：`mem_cache/mamba_checkpoint_pool.py`。
    - 压测：`benchmark/agentic-rollout/` 用合成的多轮会话（生成、等一段模拟的工具延迟、接上结果、再生成）比较整段重发、普通会话和流式会话。

!!! interview "怎么讲清楚"
    讲 agent 场景的推理，先把它还原成一串请求：每调一次模型就是一个新请求，提示词是"上一个请求的全部 + 新的工具结果"，引擎先 prefill 再 decode，停在工具调用上。然后讲前缀缓存：同一轮里上一步的提示词和生成都能复用，只 prefill 工具结果；新一轮时模板删掉旧的思考，缓存断在上一轮开头。再讲工程上的三件事：两步之间 KV 要撑过工具执行（会话感知淘汰、卸载到内存），多实例时会话要回到同一台机器（会话亲和、缓存感知路由），PD 分离时上一步生成的 KV 在 decode 那边。最后给原则：稳定的放前面，会变的放后面，历史只追加不改写。

## 练习

1. 本章的模拟里，如果模板在新一轮时保留上一轮的思考过程，第 4 个请求能复用多少？试着估算，再改代码验证。

??? success "参考答案"
    能复用第 3 个请求的整段序列：898 + 47 = 945 个。第 4 个请求的提示词变成 959 个，只要 prefill 新的追问和模板的几个特殊 token，一共 14 个。代价是提示词比 824 个多出上一轮的思考过程。Qwen3.5 的模板没有保留旧思考的开关，验证的办法是自己拼：把第 3 个请求的整段序列（`add_generation_prompt=False` 渲染出来的文本）后面直接接上新的用户消息和 `<|im_start|>assistant\n<think>\n`。

2. 一个 agent 平均每步工具执行 5 秒，引擎每秒处理 100 个这样的会话步，每步结束后留下 2000 个 token 的 KV（每个 token 100 KB）。只靠显存（留给缓存 40 GB），这些 KV 能撑过工具执行吗？

??? success "参考答案"
    5 秒里会结束 500 个会话步，留下 500 × 2000 × 100 KB ≈ 100 GB 的 KV，远超 40 GB，LRU 一定会在工具返回前淘汰掉大部分。办法：让会话的缓存优先于冷数据留下（会话感知淘汰），把 KV 卸载到内存或 SSD，或者给每个会话少留一点（比如只保留系统提示词和最近几步）。

## 小结

- [x] 一次 agent 任务是一串请求：每次调模型都先 prefill、再 decode，停在工具调用上；提示词是上一个请求的全部加上新的工具结果。
- [x] 用 Qwen3.5 的真实模板算：同一轮里能复用上一个请求的整段序列，四次请求 3001 个 token 只 prefill 1037 个；新一轮时模板删掉旧的思考，缓存断在上一轮开头，320 个 token 要重算，混合模型更是要从头算 824 个。
- [x] 稳定的放前面、会变的放后面、历史只追加不改写；系统提示词开头放时间戳，跨会话共享就只剩工具定义。
- [x] 引擎和路由要做的事：KV 撑过工具执行（会话感知淘汰、卸载到内存）、同一个会话回到同一台机器、PD 分离时把上一步生成的 KV 补上。
