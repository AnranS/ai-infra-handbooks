"""第 18 章：CPU 上的 CUDA Graph 仿真——补齐、replay，以及漏拷一个输入的后果。"""

import torch
from minisgl.core import SamplingParams
from minisgl.engine.graph import GraphCaptureBuffer
from minisgl.env import ENV
from minisgl.llm import LLM

ENV.DISABLE_OVERLAP_SCHEDULING.value = True
PROMPTS = ["The capital of France is", "List three prime numbers:", "def fibonacci(n):"]
KW = dict(dtype=torch.float32, max_running_req=8, num_page_override=512, max_seq_len_override=128)


def run(cuda_graph_max_bs):
    llm = LLM("models/Qwen3-0.6B", cuda_graph_max_bs=cuda_graph_max_bs, **KW)
    runner = llm.engine.graph_runner
    log = []
    orig = runner.replay
    runner.replay = lambda b: (log.append(f"{b.size}->{b.padded_size}"), orig(b))[1]
    out = llm.generate(PROMPTS, SamplingParams(max_tokens=6, ignore_eos=True))
    sizes = sorted(runner.graph_map)
    llm.shutdown()
    return [o["token_ids"] for o in out], sizes, log


eager, _, _ = run(0)
graph, sizes, log = run(4)
print("已录制的批大小:", sizes)
print("每轮 decode 的 replay（实际批大小->补齐后）:", log)
print("与不用 graph 的输出一致:", graph == eager)

orig_copy = GraphCaptureBuffer.copy_from


def buggy_copy_from(self, batch):  # 漏掉了 positions
    s = slice(batch.padded_size)
    self.input_ids[s] = batch.input_ids
    self.out_loc[s] = batch.out_loc


GraphCaptureBuffer.copy_from = buggy_copy_from
wrong, _, _ = run(4)
GraphCaptureBuffer.copy_from = orig_copy
print("漏拷 positions 之后与正确输出一致:", wrong == eager)
print("  正确:", eager[0])
print("  出错:", wrong[0])
