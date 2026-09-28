import pytest
import torch
from minisgl.core import Context, Req, SamplingParams, set_global_ctx
from minisgl.env import ENV
from minisgl.scheduler.cache import CacheManager

from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_llm, greedy


def _cm(num_pages: int, page_size: int, cache: str = "radix") -> CacheManager:
    set_global_ctx(Context(page_size=page_size))
    page_table = torch.zeros((4, 64), dtype=torch.int32)
    return CacheManager(num_pages, page_size, page_table, type=cache)


def _req(table_idx, prompt_len, cached_len, handle, max_tokens=4):
    return Req(input_ids=torch.arange(prompt_len, dtype=torch.int32), table_idx=table_idx,
               cached_len=cached_len, output_len=max_tokens, uid=table_idx,
               sampling_params=SamplingParams(max_tokens=max_tokens), cache_handle=handle)


def test_allocate_paged_writes_page_aligned_token_positions():
    cm = _cm(num_pages=8, page_size=4, cache="naive")
    handle = cm.prefix_cache.match_prefix(torch.arange(1)).cuda_handle
    req = _req(table_idx=2, prompt_len=6, cached_len=0, handle=handle)
    cm.allocate_paged([req])  # 6 个 token 需要 2 页
    row = cm.page_table[2, :8].tolist()
    assert row == [0, 1, 2, 3, 4, 5, 6, 7] and len(cm.free_slots) == 6
    req.complete_one()  # device_len 6 -> 7：仍在第 2 页内，不需要新页
    cm.allocate_paged([req])
    assert len(cm.free_slots) == 6
    req.complete_one()
    req.complete_one()  # device_len 9：需要第 3 页
    cm.allocate_paged([req])
    assert len(cm.free_slots) == 5 and cm.page_table[2, 8].item() % 4 == 0


def test_integrity_check_detects_leak():
    cm = _cm(num_pages=8, page_size=1, cache="naive")
    cm.check_integrity()
    cm._allocate(3)  # 分配了却没有交给任何人：泄漏
    with pytest.raises(RuntimeError, match="integrity"):
        cm.check_integrity()


def test_admission_control_queues_requests_when_kv_is_short():
    """KV 池只有 64 个 token：每个请求最坏需要 ~20 个，一次只能放进几个，其余排队，结果仍然正确。"""
    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    llm = build_llm(QWEN3, cache_type="naive", num_page_override=64)
    max_batch = 0
    orig = llm._forward

    def spy(fi):
        nonlocal max_batch
        max_batch = max(max_batch, fi.batch.size)
        return orig(fi)

    llm._forward = spy
    out = llm.generate(PROMPTS, greedy(8))
    assert [o["token_ids"] for o in out] == hf_greedy(QWEN3, PROMPTS, 8)
    assert max_batch < len(PROMPTS)
    llm.cache_manager.check_integrity()
    llm.shutdown()
