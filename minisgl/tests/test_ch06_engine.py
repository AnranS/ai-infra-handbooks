import torch
from minisgl.core import Batch, SamplingParams
from minisgl.engine import Sampler
from transformers import AutoTokenizer

from conftest import PROMPTS, QWEN3, hf_greedy, hf_model
from helpers import build_engine, forward, identity_page_table, new_req


def test_prefill_logits_match_hf():
    engine = build_engine(QWEN3)
    identity_page_table(engine, 2)
    tok = AutoTokenizer.from_pretrained(QWEN3)
    ids = [tok(p).input_ids for p in PROMPTS[:2]]
    logits = forward(engine, [new_req(ids[0], 0), new_req(ids[1], 1)], "prefill")
    with torch.no_grad():
        ref = [hf_model(QWEN3)(torch.tensor([x])).logits[0, -1] for x in ids]
    assert logits.shape == (2, 151936)
    assert max((logits[i] - ref[i]).abs().max().item() for i in range(2)) < 1e-4
    engine.shutdown()


def test_manual_greedy_generation_matches_hf():
    """不用调度器，手工走 prefill + decode 循环，与 HF generate 逐 token 相同。"""
    engine = build_engine(QWEN3)
    identity_page_table(engine, 3)
    tok = AutoTokenizer.from_pretrained(QWEN3)
    reqs = [new_req(tok(p).input_ids, i, max_tokens=8) for i, p in enumerate(PROMPTS[:3])]
    outs = [[] for _ in reqs]
    phase = "prefill"
    for _ in range(8):
        logits = forward(engine, reqs, phase)
        next_tokens = logits.argmax(-1)
        for i, r in enumerate(reqs):
            r.complete_one()
            r.append_host(next_tokens[i:i + 1].to(torch.int32))
            outs[i].append(int(next_tokens[i]))
        phase = "decode"
    assert outs == hf_greedy(QWEN3, PROMPTS[:3], 8)
    engine.shutdown()


def _batch(params):
    reqs = [new_req([0], i) for i in range(len(params))]
    for r, p in zip(reqs, params):
        r.sampling_params = p
    return Batch(reqs=reqs, phase="decode")


def test_sampler_greedy_and_filters():
    sampler = Sampler(torch.device("cpu"), vocab_size=6)
    assert sampler.prepare(_batch([SamplingParams(), SamplingParams()])).temperatures is None
    logits = torch.tensor([[5.0, 4.0, 3.0, 0.0, 0.0, 0.0]] * 3)
    args = sampler.prepare(_batch([SamplingParams(temperature=1.0, top_k=2),
                                   SamplingParams(temperature=1.0, top_p=0.5),
                                   SamplingParams()]))  # 第三个请求是贪心
    torch.manual_seed(0)
    draws = torch.stack([sampler.sample(logits, args) for _ in range(2000)])
    assert set(draws[:, 0].tolist()) == {0, 1}  # top-k=2：只可能是前两个
    assert set(draws[:, 1].tolist()) == {0}  # 第一名概率约 0.67 > 0.5，top-p 只留它
    assert set(draws[:, 2].tolist()) == {0}  # 混合 batch 里的贪心请求
    p0 = (draws[:, 0] == 0).float().mean().item()  # top-k 后重新归一化：e^5/(e^5+e^4) = 0.731
    assert abs(p0 - 0.731) < 0.03
