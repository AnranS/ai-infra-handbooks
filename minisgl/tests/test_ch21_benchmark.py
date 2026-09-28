import torch
from minisgl.benchmark.offline import run

from conftest import QWEN3


def test_offline_benchmark_runs():
    r = run(QWEN3, num_seqs=6, max_input_len=40, max_output_len=16, dtype=torch.float32,
            max_running_req=8, num_page_override=1024, max_seq_len_override=128)
    assert r["output_tokens"] > 6 and r["throughput"] > 0
