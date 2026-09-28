import pytest
from minisgl.core import Batch
from minisgl.engine.graph import GraphCaptureBuffer, determine_cuda_graph_bs
from minisgl.env import ENV

import torch
from conftest import PROMPTS, QWEN3, hf_greedy
from helpers import build_llm, greedy


def test_graph_batch_sizes():
    cpu = torch.device("cpu")
    assert determine_cuda_graph_bs(None, None, 0, cpu) == []  # CPU 默认关闭
    assert determine_cuda_graph_bs(None, 24, 0, cpu) == [1, 2, 4, 8, 16, 24]


@pytest.mark.parametrize("backend", ["torch", "fi", "fa"])
def test_emulated_graph_replay_matches_hf(fake_kernels, backend):
    ENV.DISABLE_OVERLAP_SCHEDULING.value = False
    llm = build_llm(QWEN3, attention_backend=backend, cuda_graph_max_bs=4)
    runner = llm.engine.graph_runner
    replays = []
    orig = runner.replay
    runner.replay = lambda batch: (replays.append((batch.size, batch.padded_size)), orig(batch))[1]
    out = llm.generate(PROMPTS[:3], greedy(6))
    assert [o["token_ids"] for o in out] == hf_greedy(QWEN3, PROMPTS[:3], 6)
    assert (3, 4) in replays  # 3 个请求的 decode 补齐到 4，走 graph
    llm.shutdown()


def test_forgetting_to_copy_an_input_breaks_replay(monkeypatch):
    """反例：replay 前漏拷 positions（graph 里 RoPE 读的还是捕获时的位置），结果就错了。"""
    ENV.DISABLE_OVERLAP_SCHEDULING.value = True

    def buggy_copy_from(self, batch: Batch) -> None:
        s = slice(batch.padded_size)
        self.input_ids[s] = batch.input_ids
        self.out_loc[s] = batch.out_loc  # 漏掉了 positions

    monkeypatch.setattr(GraphCaptureBuffer, "copy_from", buggy_copy_from)
    llm = build_llm(QWEN3, cuda_graph_max_bs=4)
    out = llm.generate(PROMPTS[:3], greedy(6))
    assert [o["token_ids"] for o in out] != hf_greedy(QWEN3, PROMPTS[:3], 6)
    llm.shutdown()
