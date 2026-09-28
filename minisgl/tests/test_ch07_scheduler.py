from minisgl.env import ENV

from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_llm, greedy


def test_continuous_batching_matches_hf():
    """最朴素的配置：naive 缓存、不分块、不重叠。5 个请求一起跑，每个都与单独生成一致。"""
    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    llm = build_llm(QWEN3, cache_type="naive")
    out = llm.generate(PROMPTS, greedy(12))
    assert [o["token_ids"] for o in out] == hf_greedy(QWEN3, PROMPTS, 12)
    llm.shutdown()


def test_requests_with_different_lengths_leave_and_join():
    """max_tokens 各不相同：短的先结束离开 batch，其余继续；新一轮 generate 的请求随后加入。"""
    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    llm = build_llm(QWEN3, cache_type="naive")
    lengths = [3, 12, 7, 1, 9]
    out = llm.generate(PROMPTS, [greedy(n) for n in lengths])
    ref = hf_greedy(QWEN3, PROMPTS, 12)
    assert [o["token_ids"] for o in out] == [r[:n] for r, n in zip(ref, lengths)]
    assert llm.table_manager.available_size == 8  # 所有行都已归还
    llm.shutdown()
