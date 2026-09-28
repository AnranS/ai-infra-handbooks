"""第 9 章：Radix Cache 的树形结构，以及端到端的前缀复用效果。"""

import torch
from minisgl.core import Context, SamplingParams, set_global_ctx
from minisgl.env import ENV
from minisgl.kvcache.radix_cache import RadixPrefixCache


def show(node, depth=0):
    for child in sorted(node.children.values(), key=lambda n: n._key.tolist()):
        print(f"{'    ' * depth}└─ key={child._key.tolist()} value={child.value.tolist()} ref={child.ref_count}")
        show(child, depth + 1)


def t(*xs):
    return torch.tensor(xs, dtype=torch.int32)


set_global_ctx(Context(page_size=1))
cache = RadixPrefixCache(torch.device("cpu"))
cache.insert_prefix(t(1, 2, 3, 4), t(10, 11, 12, 13))
cache.insert_prefix(t(1, 2, 5, 6), t(10, 11, 20, 21))
cache.insert_prefix(t(7, 8), t(30, 31))
print("插入 [1,2,3,4]、[1,2,5,6]、[7,8] 之后：")
show(cache.root_node)
handle = cache.match_prefix(t(1, 2, 3, 9)).cuda_handle
cache.lock_handle(handle)
print(f"匹配 [1,2,3,9]：命中 {handle.cached_len} 个，位置 {handle.get_matched_indices().tolist()}；加锁后：")
show(cache.root_node)
print("可淘汰 / 受保护:", tuple(cache.size_info))
print("淘汰 2 个 token，释放位置:", cache.evict(2).tolist())
print("再淘汰 1 个 token，释放位置:", cache.evict(1).tolist())
print("剩下的树：")
show(cache.root_node)
cache.check_integrity()

# 端到端：同一个长"系统提示词" + 不同问题
from minisgl.core import reset_global_ctx
from minisgl.llm import LLM

reset_global_ctx()  # 上面手工设置过全局上下文，引擎会重新设置

ENV.DISABLE_OVERLAP_SCHEDULING.value = True
system = "You are a helpful assistant. Answer concisely and accurately. " * 4
questions = ["What is 2 + 2?", "Name a primary color.", "What is the capital of Japan?", "Spell cat."]
for cache_type in ("naive", "radix"):
    llm = LLM("models/Qwen3-0.6B", dtype=torch.float32, max_running_req=8, num_page_override=1024,
              max_seq_len_override=256, cache_type=cache_type)
    computed = []
    orig = llm._forward
    llm._forward = lambda fi: (computed.append(sum(r.extend_len for r in fi.batch.reqs))
                               if fi.batch.is_prefill else None, orig(fi))[1]
    first = llm.generate([system + questions[0]], SamplingParams(max_tokens=4))  # 先来一个请求
    rest = llm.generate([system + q for q in questions[1:]], SamplingParams(max_tokens=4))
    total = sum(len(llm.tokenizer(system + q).input_ids) for q in questions)
    print(f"{cache_type:<5}: 提示词共 {total} 个 token，prefill 实际计算 {sum(computed)} 个")
    llm.shutdown()
