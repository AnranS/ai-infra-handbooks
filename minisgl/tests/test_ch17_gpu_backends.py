import pytest
import torch
from minisgl.env import ENV

from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_engine, build_llm, forward, greedy, identity_page_table, new_req


@pytest.mark.parametrize("backend", ["fi", "fa"])
def test_backend_logits_match_torch_backend(fake_kernels, backend):
    """同一个 batch（一个命中前缀的 prefill + 一个 decode），FlashInfer/FlashAttention 与参考后端一致。"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3)
    ids = [tok(p).input_ids for p in PROMPTS[:2]]
    logits = {}
    for name in ("torch", backend):
        engine = build_engine(QWEN3, attention_backend=name)
        identity_page_table(engine, 2)
        forward(engine, [new_req(ids[0][:3], 0), new_req(ids[1], 1)], "prefill")  # 先把 KV 填好
        a = new_req(ids[0], 0, cached_len=3)  # 命中前 3 个 token，再算剩下的
        b = new_req(ids[1] + [100], 1, cached_len=len(ids[1]))  # decode 一步
        logits[name] = forward(engine, [a, b], "prefill")
        engine.shutdown()
    assert (logits["torch"] - logits[backend]).abs().max().item() < 1e-4


def test_flashinfer_plans_once_per_batch(fake_kernels):
    engine = build_engine(QWEN3, attention_backend="fi")
    identity_page_table(engine, 1)
    forward(engine, [new_req([1, 2, 3, 4], 0)], "prefill")
    assert engine.attn_backend.prefill_wrapper.plan_count == 1  # 28 层只 plan 一次
    engine.shutdown()


@pytest.mark.parametrize("backend,page_size", [("fi", 1), ("fa", 4), ("fa,fi", 1)])
def test_end_to_end_with_gpu_backends(fake_kernels, backend, page_size):
    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    llm = build_llm(QWEN3, attention_backend=backend, page_size=page_size,
                    num_page_override=1024 // page_size, max_extend_tokens=9)
    out = llm.generate(PROMPTS, greedy(8))
    assert [o["token_ids"] for o in out] == hf_greedy(QWEN3, PROMPTS, 8)
    llm.shutdown()
