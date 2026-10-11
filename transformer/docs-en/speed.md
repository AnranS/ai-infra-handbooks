# 15. How fast this machine can go

<p class="lead">The first fourteen chapters only cared about "is it right". This chapter measures "how fast": the number of threads, recurrent versus chunked, fp32 versus bf16, and the differences between a CPU, Apple's MPS and an NVIDIA GPU. The script picks the device automatically; run it once on your machine and you get the same table — the one pasted below is from a 32-core Xeon.</p>

!!! note "What this chapter runs"
    `ch15_speed.py`, about 2 minutes on a CPU (the thread sweep takes most of it).

## Measure four things {#量四件事}

```python title="ch15_speed.py" ci="loose"
"""第 15 章：这台机器上能跑多快。线程数、递推 vs 分块、fp32 vs bf16、CPU / MPS / CUDA。"""
import os
import platform
import time

import torch

from qwen.cache import State
from qwen.chat import Chat, build_prompt
from qwen.chunk import chunked_delta_rule
from qwen.deltanet import delta_rule
from qwen.model import Model
from qwen.weights import find_model

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"{platform.machine()} / {device} / torch {torch.__version__} / {os.cpu_count()} 个逻辑核")


def decode_speed(model, ids, n=32):
    state = State(model)
    model(ids, state)
    tok = ids[:, -1:]
    t = time.time()
    for _ in range(n):
        tok = model(tok, state)[:, -1:].argmax(-1)
    if device == "cuda":
        torch.cuda.synchronize()
    return n / (time.time() - t)


path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)
ids = chat.encode(build_prompt([{"role": "user", "content": "写一段 200 字的短文，介绍长江。"}]))

if device == "cpu":                                                              # 1. threads: more is not better
    for n in sorted({1, 2, 4, 8, os.cpu_count() // 2, os.cpu_count()}):
        torch.set_num_threads(n)
        print(f"{n:>3} 线程：decode {decode_speed(model, ids):.1f} tok/s")
    torch.set_num_threads(max(1, os.cpu_count() // 2))

torch.manual_seed(0)                                                             # 2. recurrent vs. chunked: prefill of 512 tokens
B, T, H, d = 1, 512, 16, 128
q, k, v = (torch.randn(B, T, H, d, device=device) for _ in range(3))
g, beta = -torch.rand(B, T, H, device=device) * 0.5, torch.rand(B, T, H, device=device)
for name, fn in (("递推", delta_rule), ("分块", chunked_delta_rule)):
    t = time.time()
    out, _ = fn(q, k, v, g, beta)
    print(f"一层线性注意力，{T} 个 token：{name} {(time.time() - t) * 1000:.0f} ms")

for dtype in (torch.float32, torch.bfloat16):                                    # 3. fp32 vs bf16
    m = Model.from_pretrained(path, dtype=dtype).to(device)
    x = ids.to(device)
    t = time.time()
    m(x, State(m))
    pre = (time.time() - t) * 1000
    print(f"{str(dtype).split('.')[-1]:>8}：prefill {ids.shape[1]} token {pre:.0f} ms，decode {decode_speed(m, x):.1f} tok/s")
```

```text title="output"
x86_64 / cpu / torch 2.14.0+cpu / 32 个逻辑核
  1 线程：decode 4.3 tok/s
  2 线程：decode 7.4 tok/s
  4 线程：decode 10.6 tok/s
  8 线程：decode 15.9 tok/s
 16 线程：decode 18.4 tok/s
 32 线程：decode 0.3 tok/s
一层线性注意力，512 个 token：递推 129 ms
一层线性注意力，512 个 token：分块 11 ms
 float32：prefill 25 token 157 ms，decode 18.4 tok/s
bfloat16：prefill 25 token 327 ms，decode 18.0 tok/s
```

| What | Finding |
| --- | --- |
| **Threads** | 16 threads are fastest, and 32 (every logical core) is 60× slower — hyper-threading plus contention for memory bandwidth; PyTorch's default (all cores) is a disaster on this machine. That is why every script in the book calls `set_num_threads(cpu_count() // 2)`. No need to bother on a Mac: PyTorch on macOS uses only the performance cores by default. |
| **Recurrent vs. chunked** | one linear-attention layer over 512 tokens: 129 ms vs. 11 ms. Prefill must use the chunked version; decode handles one token at a time and the recurrence is faster. |
| **fp32 vs. bf16** | bf16 buys nothing on a CPU: decode is level (the weights are half the size, but every operator converts to fp32 first), prefill is twice as slow (this CPU has no bf16 matrix instructions). On a GPU bf16 is twice as fast across the board and is the default precision for deployment. |
| **Devices** | the script uses MPS / CUDA automatically when available. |

!!! tip "On your machine"
    `python ch15_speed.py` runs directly on a Mac (MPS) or an NVIDIA GPU. The recurrent linear attention suffers especially on a GPU (hundreds of tiny kernel launches), so the chunked version's advantage is larger than on a CPU; decode tok/s depends mostly on memory bandwidth.

## Why these numbers {#为什么是这些数字}

The floor for decoding one token is **reading the weights once**: 752M fp32 parameters = 3 GB, and this machine's memory bandwidth is about 100 GB/s, so the theoretical ceiling is thirty-odd tok/s; we reach 18, the rest being the scheduling overhead of 24 layers × dozens of small operators and the Python loop. The same arithmetic on a GPU: 1.5 GB in bf16, and the 896 GB/s of a 5070 Ti corresponds to several hundred tok/s, but the launch overhead of small operators becomes the bottleneck first — which is why CUDA Graphs and operator fusion exist (covered in *mini-sglang from Scratch*).

Prefill is compute-bound instead: the matrix multiplications for T tokens are T × 1.5 GFLOP, and on long inputs the GPU's TFLOPS decide.

!!! interview "How to explain it"
    To explain inference performance, first separate the two phases, prefill (compute-bound) and decode (memory-bound), then give the floor formula for decode: weight bytes / bandwidth. Then the three cheapest optimizations: bf16 (half the weights), batching (one read of the weights yields B tokens), speculative decoding (one read yields more than one token). The thread-count lesson is CPU-specific, but it illustrates the general point that "the default configuration is not necessarily the best".

## Summary {#小结}

- [x] CPU: using every thread is dozens of times slower; fp32 decode is about 18 tok/s, and bf16 buys nothing on a CPU.
- [x] Chunked for prefill, recurrent for decode; on long inputs chunking is over 10× faster.
- [x] The theoretical ceiling for decode = weight bytes / memory bandwidth.
