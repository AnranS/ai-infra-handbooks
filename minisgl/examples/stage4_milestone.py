"""阶段四的里程碑：同一份代码，切成两个 rank 跑，或者录成 graph 回放，输出一个 token 都不变。

CPU 上这一阶段只能验证"对"：张量并行走 gloo、CUDA Graph 走仿真，都快不起来。
"快"的数字要上真卡：根目录的 gpu_check.py minisgl，以及第 21 章的基准表。
"""

import multiprocessing as mp

import torch

PATH = "models/Qwen3-0.6B"
PROMPT_IDS = [785, 6722, 315, 9625, 374]       # "The capital of France is"
STEPS = 4


def run_tp(rank: int, size: int, out: mp.Queue) -> None:
    """一个 rank：加载自己那一份权重，手工驱动 prefill + 3 步 decode（和阶段一的 step() 一样）。"""
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    engine = Engine(EngineConfig(model_path=PATH, tp_info=DistributedInfo(rank, size), dtype=torch.float32,
                                 max_running_req=2, num_page_override=256, max_seq_len_override=128,
                                 distributed_port=29610))
    engine.page_table[0, :128] = torch.arange(128)
    req = Req(input_ids=torch.tensor(PROMPT_IDS, dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=STEPS, uid=0, sampling_params=SamplingParams(max_tokens=STEPS), cache_handle=None)
    tokens = []
    for phase in ["prefill"] + ["decode"] * (STEPS - 1):
        batch = Batch(reqs=[req], phase=phase)
        engine.graph_runner.pad_batch(batch)
        batch.positions = torch.arange(req.cached_len, req.device_len).int()
        batch.input_ids = req.input_ids[req.cached_len:req.device_len]
        batch.out_loc = engine.page_table[0, req.cached_len:req.device_len]
        engine.attn_backend.prepare_metadata(batch)
        o = engine.forward_batch(batch, engine.sampler.prepare(batch))
        req.append_host(o.next_tokens_cpu[:1])
        tokens.append(int(o.next_tokens_cpu[0]))
    qkv = engine.model.model.layers.op_list[0].self_attn.qkv_proj.weight
    out.put((rank, tokens, tuple(qkv.shape), tuple(engine.kv_cache.k_cache(0).shape)))
    engine.shutdown()


def run_graph(cuda_graph_max_bs: int):
    """CPU 上的 CUDA Graph 仿真：录制固定批大小的 graph，decode 走 replay。"""
    from minisgl.core import SamplingParams
    from minisgl.env import ENV
    from minisgl.llm import LLM

    ENV.DISABLE_OVERLAP_SCHEDULING.value = True
    llm = LLM(PATH, dtype=torch.float32, cuda_graph_max_bs=cuda_graph_max_bs, max_running_req=8,
              num_page_override=512, max_seq_len_override=128)
    replays = []
    orig = llm.engine.graph_runner.replay
    llm.engine.graph_runner.replay = lambda b: (replays.append(f"{b.size}->{b.padded_size}"), orig(b))[1]
    out = llm.generate(["The capital of France is", "List three prime numbers:", "def fibonacci(n):"],
                       SamplingParams(max_tokens=6, ignore_eos=True))
    llm.shutdown()
    return [o["token_ids"] for o in out], replays


if __name__ == "__main__":
    mp.set_start_method("spawn")
    results = {}
    for size in (1, 2):                                       # 先 TP=1 当基准，再 TP=2
        q: mp.Queue = mp.Queue()
        procs = [mp.Process(target=run_tp, args=(r, size, q)) for r in range(size)]
        for p in procs:
            p.start()
        results[size] = sorted((q.get() for _ in range(size)), key=lambda x: x[0])
        for p in procs:
            p.join()
    base = results[1][0]
    print(f"TP=1：qkv_proj {base[2]}，KV 池每层 {base[3]}，生成 {base[1]}")
    for rank, tokens, qkv_shape, k_shape in results[2]:
        print(f"TP=2 rank {rank}：qkv_proj {qkv_shape}（一半），KV 池每层 {k_shape}（KV 头也减半），"
              f"生成 {tokens}，与 TP=1 一致：{tokens == base[1]}")

    eager, _ = run_graph(0)
    graph, replays = run_graph(4)
    print(f"CUDA Graph 仿真：3 个请求的 decode 每轮补齐后 replay（{replays[0]} x {len(replays)} 轮），与不用 graph 的输出一致：{graph == eager}")
    print("CPU 上到此为止——快不快要上真卡：.venv-gpu/bin/python gpu_check.py minisgl，以及第 21 章的基准表")
