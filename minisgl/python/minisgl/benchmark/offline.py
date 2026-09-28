"""离线吞吐测试（参考官方 benchmark/offline/bench.py，沿自 nano-vllm）。

随机生成 num_seqs 个请求，输入、输出长度各自在范围内均匀随机，全部交给 LLM.generate，
统计输出 token 的吞吐。用环境变量 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 可以做重叠调度的消融。

    python -m minisgl.benchmark.offline --model Qwen/Qwen3-0.6B --num-seqs 256
"""

from __future__ import annotations

import argparse
import time
from random import randint, seed

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def run(model: str, num_seqs: int, max_input_len: int, max_output_len: int,
        dtype: torch.dtype = torch.bfloat16, **llm_kwargs) -> dict:
    seed(0)
    llm = LLM(model, dtype=dtype, **llm_kwargs)
    prompts = [[randint(0, 10000) for _ in range(randint(max_input_len // 10, max_input_len))]
               for _ in range(num_seqs)]
    params = [SamplingParams(temperature=0.6, ignore_eos=True,
                             max_tokens=randint(max_output_len // 10, max_output_len))
              for _ in range(num_seqs)]
    llm.generate(["Benchmark: "], SamplingParams(max_tokens=4))  # 预热
    t = time.perf_counter()
    outputs = llm.generate(prompts, params)
    t = time.perf_counter() - t
    total = sum(len(o["token_ids"]) for o in outputs)
    llm.shutdown()
    return {"output_tokens": total, "seconds": t, "throughput": total / t}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--max-input-len", type=int, default=1024)
    p.add_argument("--max-output-len", type=int, default=1024)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--cache-type", default="radix")
    p.add_argument("--page-size", type=int, default=1)
    p.add_argument("--cuda-graph-max-bs", type=int, default=None)
    args = p.parse_args()
    r = run(args.model, args.num_seqs, args.max_input_len, args.max_output_len,
            dtype=getattr(torch, args.dtype), cache_type=args.cache_type, page_size=args.page_size,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            max_seq_len_override=args.max_input_len + args.max_output_len)
    print(f"Total: {r['output_tokens']} tok, Time: {r['seconds']:.2f}s, "
          f"Throughput: {r['throughput']:.2f} tok/s")


if __name__ == "__main__":
    main()
