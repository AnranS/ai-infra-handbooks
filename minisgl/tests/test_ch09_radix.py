import torch
from minisgl.core import Context, set_global_ctx
from minisgl.env import ENV
from minisgl.kvcache.radix_cache import RadixPrefixCache

from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_llm, greedy


def _cache(page_size: int = 1) -> RadixPrefixCache:
    set_global_ctx(Context(page_size=page_size))
    return RadixPrefixCache(torch.device("cpu"))


def t(*xs):
    return torch.tensor(xs, dtype=torch.int32)


def test_insert_match_split():
    c = _cache()
    c.insert_prefix(t(1, 2, 3, 4), t(10, 11, 12, 13))
    c.insert_prefix(t(1, 2, 5), t(10, 11, 20))  # 与已有序列共享 [1, 2]：节点在 2 之后分裂
    root_children = list(c.root_node.children.values())
    assert len(root_children) == 1 and root_children[0].length == 2  # 公共前缀 [1, 2]
    assert sorted(ch.length for ch in root_children[0].children.values()) == [1, 2]  # [5] 和 [3, 4]
    handle = c.match_prefix(t(1, 2, 3, 9)).cuda_handle
    assert handle.cached_len == 3 and handle.get_matched_indices().tolist() == [10, 11, 12]
    # 匹配只到 3 为止，[3, 4] 这个节点被分裂成 [3] -> [4]：匹配不改变缓存的内容，但会改变树的形状
    assert handle.node.length == 1 and handle.node.children[4].length == 1
    assert c.size_info.evictable_size == 5
    c.check_integrity()


def test_lock_protects_from_eviction_and_lru_order():
    c = _cache()
    c.insert_prefix(t(1, 2, 3), t(10, 11, 12))
    c.insert_prefix(t(7, 8), t(20, 21))
    c.insert_prefix(t(4, 5), t(30, 31))
    handle = c.match_prefix(t(7, 8)).cuda_handle  # 刚访问过 [7, 8]
    c.lock_handle(handle)
    assert c.size_info == (5, 2)
    evicted = c.evict(3)  # 锁住的 [7, 8] 不能淘汰；另外两个里 [1, 2, 3] 最久没用
    assert evicted.tolist() == [10, 11, 12]
    c.lock_handle(handle, unlock=True)
    assert c.evict(2).tolist() == [30, 31]  # 解锁后，[4, 5] 比 [7, 8] 更久没用
    c.check_integrity()


def test_page_aligned_matching():
    c = _cache(page_size=4)
    r = c.insert_prefix(torch.arange(10, dtype=torch.int32), torch.arange(100, 110, dtype=torch.int32))
    assert r.handle.cached_len == 8  # 只缓存完整的页：最后 2 个 token 丢掉
    assert c.match_prefix(torch.arange(7, dtype=torch.int32)).cuda_handle.cached_len == 4


def test_shared_prefixes_are_reused_and_outputs_unchanged():
    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    llm = build_llm(QWEN3, cache_type="radix")
    computed = []
    orig = llm._forward

    def spy(fi):
        if fi.batch.is_prefill:
            computed.append(sum(r.extend_len for r in fi.batch.reqs))
        return orig(fi)

    llm._forward = spy
    first = llm.generate(PROMPTS, greedy(6))
    cold = sum(computed)
    computed.clear()
    second = llm.generate(PROMPTS, greedy(6))  # 同样的提示词再来一遍：几乎全部命中
    assert [o["token_ids"] for o in first] == [o["token_ids"] for o in second] == hf_greedy(QWEN3, PROMPTS, 6)
    assert sum(computed) == len(PROMPTS)  # 每个请求只需重算最后 1 个 token
    assert cold > 5 * len(PROMPTS)
    llm.cache_manager.check_integrity()
    llm.shutdown()
