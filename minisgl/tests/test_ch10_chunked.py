from minisgl.env import ENV

from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_llm, greedy


def test_chunked_prefill_is_exact_and_bounded():
    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    llm = build_llm(QWEN3, cache_type="radix", max_extend_tokens=6)
    sizes = []
    orig = llm._forward

    def spy(fi):
        if fi.batch.is_prefill:
            sizes.append(sum(r.extend_len for r in fi.batch.reqs))
        return orig(fi)

    llm._forward = spy
    out = llm.generate(PROMPTS, greedy(8))
    assert [o["token_ids"] for o in out] == hf_greedy(QWEN3, PROMPTS, 8)
    total = sum(len(llm.tokenizer(p).input_ids) for p in PROMPTS)
    # 每个 prefill batch 恰好用满 6 个 token 的预算（最后一个除外），所有提示词 token 各算一次
    assert sum(sizes) == total and all(s == 6 for s in sizes[:-1]) and sizes[-1] <= 6
    llm.cache_manager.check_integrity()
    llm.shutdown()
