import pytest
import torch
from minisgl.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx


def make_req(prompt_len: int, max_tokens: int, cached_len: int = 0) -> Req:
    return Req(input_ids=torch.arange(prompt_len, dtype=torch.int32), table_idx=0,
               cached_len=cached_len, output_len=max_tokens, uid=0,
               sampling_params=SamplingParams(max_tokens=max_tokens), cache_handle=None)


def test_req_lengths_through_prefill_and_decode():
    req = make_req(prompt_len=5, max_tokens=3, cached_len=2)  # 前 2 个 token 命中缓存
    assert (req.cached_len, req.device_len, req.extend_len, req.remain_len) == (2, 5, 3, 3)
    req.complete_one()  # prefill 结束：5 个 token 进了缓存，下一轮要算第 6 个位置
    assert (req.cached_len, req.device_len, req.extend_len, req.remain_len) == (5, 6, 1, 2)
    req.complete_one()
    req.complete_one()
    assert req.remain_len == 0 and not req.can_decode


def test_sampling_params_greedy():
    assert SamplingParams().is_greedy
    assert SamplingParams(temperature=0.7, top_k=1).is_greedy
    assert not SamplingParams(temperature=0.7).is_greedy
    assert not SamplingParams(temperature=0.0, top_p=0.9).is_greedy


def test_context_forward_batch_is_scoped_and_not_nested():
    ctx = Context(page_size=1)
    set_global_ctx(ctx)
    batch = Batch(reqs=[make_req(3, 2)], phase="prefill")
    with get_global_ctx().forward_batch(batch):
        assert get_global_ctx().batch is batch
        with pytest.raises(AssertionError):
            with ctx.forward_batch(batch):
                pass
    with pytest.raises(AssertionError):
        _ = ctx.batch  # 离开上下文后没有"当前 batch"
