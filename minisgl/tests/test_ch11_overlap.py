import pytest
import torch
from minisgl.core import SamplingParams
from minisgl.env import ENV

from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_llm, greedy


@pytest.mark.parametrize("cache,chunk,page_size", [("radix", 8192, 1), ("naive", 5, 1), ("radix", 7, 4)])
def test_overlap_equals_normal_and_hf(cache, chunk, page_size):
    results = {}
    for overlap in (True, False):
        ENV.DISABLE_OVERLAP_SCHEDULING.value = not overlap
        llm = build_llm(QWEN3, cache_type=cache, max_extend_tokens=chunk, page_size=page_size,
                        num_page_override=1024 // page_size)
        results[overlap] = [o["token_ids"] for o in llm.generate(PROMPTS, greedy(10))]
        llm.cache_manager.check_integrity()
        llm.shutdown()
    assert results[True] == results[False] == hf_greedy(QWEN3, PROMPTS, 10)


def test_overlap_respects_max_tokens_exactly():
    """重叠调度下 req.can_decode 已被下一轮提前推进；按它判断结束，finished 标记会提前一个 token。

    在线服务的前端收到第一条 finished 就结束响应，于是会少返回一个 token。这里检查发给
    detokenizer 的消息：每个请求恰好 max_tokens 条，只有最后一条带 finished。
    """
    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    llm = build_llm(QWEN3)
    replies = []
    orig = llm.offline_send_result
    llm.send_result = lambda reply: (replies.extend(reply), orig(reply))
    lengths = (1, 2, 5)
    llm.generate(PROMPTS[:3], [greedy(n) for n in lengths])
    for uid, n in enumerate(lengths):
        flags = [m.finished for m in replies if m.uid == uid]
        assert flags == [False] * (n - 1) + [True]
    llm.shutdown()


def test_stale_result_after_eos_is_dropped():
    """遇到 EOS 结束的请求在重叠调度下会被多跑一轮；那一轮的结果必须丢弃。

    为了让"EOS"确定地出现，把 eos_token_id 临时设成 HF 贪心输出的第 3 个 token。
    """
    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    ref = hf_greedy(QWEN3, PROMPTS[:1], 8)[0]
    llm = build_llm(QWEN3)
    llm.eos_token_id = ref[2]
    replies = []
    orig = llm.offline_send_result
    llm.send_result = lambda reply: (replies.extend(reply), orig(reply))
    out = llm.generate(PROMPTS[:1], SamplingParams(max_tokens=8))
    assert [m.next_token for m in replies] == ref[:3] and replies[-1].finished
    assert sum(m.finished for m in replies) == 1  # 结束之后没有再发任何消息
    assert out[0]["token_ids"] == ref[:2]  # EOS 本身不计入输出
    llm.shutdown()


def test_request_slot_is_not_reused_while_a_batch_using_it_is_in_flight():
    """遇到 EOS 结束的请求还在在途的 batch 里时，它的请求槽不能马上分给新请求。

    CPU 上没有真正的并发，这里直接检查不变量：每次分配请求槽时，它不属于任何在途 batch 中的请求。
    """
    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    ref = hf_greedy(QWEN3, PROMPTS[:1], 8)[0]
    llm = build_llm(QWEN3, max_running_req=1)  # 只有 1 个请求槽，逼出复用
    llm.eos_token_id = ref[2]
    inflight = {"batch": None}
    orig_loop, orig_alloc = llm.overlap_loop, llm.table_manager.allocate

    def loop(last_data):
        inflight["batch"] = None if last_data is None else last_data[0].batch
        return orig_loop(last_data)

    def allocate():
        slot = orig_alloc()
        b = inflight["batch"]
        assert b is None or slot not in {r.table_idx for r in b.reqs}, "slot reused while in flight"
        return slot

    llm.overlap_loop, llm.table_manager.allocate = loop, allocate
    out = llm.generate(PROMPTS[:2], SamplingParams(max_tokens=8))
    assert out[0]["token_ids"] == ref[:2]
    assert llm.table_manager.available_size == 1
    llm.shutdown()


def test_abort_while_prefill_is_in_flight():
    """prefill 已经发射、结果还没处理时收到 abort：请求被释放，在途 batch 的结果必须丢弃。"""
    from minisgl.message import AbortBackendMsg

    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    llm = build_llm(QWEN3, cache_type="radix")
    calls = {"n": 0}
    orig = llm.offline_receive_msg

    def receive(blocking=False):
        calls["n"] += 1
        if calls["n"] == 2:  # 第二轮开始时（prefill 正在"GPU 上"）中止它
            return [AbortBackendMsg(uid=0)]
        return orig(blocking)

    llm.receive_msg = receive
    replies = []
    llm.send_result = lambda reply: replies.extend(reply)
    llm.generate(PROMPTS[:1], greedy(8))
    assert replies == []  # 被中止的请求没有任何输出
    llm.cache_manager.check_integrity()
    assert llm.table_manager.available_size == 8
    llm.shutdown()
