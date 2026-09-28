"""第 16 章：张量并行下每个 rank 拿到的权重形状；两个进程（gloo）的前向与单卡对比，并统计通信次数。"""

import multiprocessing as mp

import torch

PATH = "models/Qwen3-0.6B"
PROMPT_IDS = [785, 6722, 315, 9625, 374]  # "The capital of France is"


def show_shapes() -> None:
    import safetensors
    from minisgl.models import shard_tensor

    names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
             "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight",
             "model.layers.0.mlp.down_proj.weight", "model.embed_tokens.weight",
             "model.layers.0.input_layernorm.weight"]
    with safetensors.safe_open(f"{PATH}/model.safetensors", "pt") as f:
        print(f"{'权重':<42} {'完整':<14} {'TP=2 每个 rank':<16} TP=16 每个 rank")
        for n in names:
            w = f.get_tensor(n)
            s2 = tuple(shard_tensor(n, w, 0, 2, num_kv_heads=8).shape)
            s16 = tuple(shard_tensor(n, w, 0, 16, num_kv_heads=8).shape)
            print(f"{n.removeprefix('model.'):<42} {str(tuple(w.shape)):<14} {str(s2):<16} {s16}")


def run(rank: int, size: int, out: mp.Queue) -> None:
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedCommunicator, DistributedInfo
    from minisgl.engine import Engine, EngineConfig

    engine = Engine(EngineConfig(model_path=PATH, tp_info=DistributedInfo(rank, size), dtype=torch.float32,
                                 max_running_req=2, num_page_override=256, max_seq_len_override=128,
                                 distributed_port=29600))
    counts = {"all_reduce": 0, "all_gather": 0}
    impl = DistributedCommunicator.plugins[-1]
    for op in counts:
        fn = getattr(impl, op)

        def wrapped(x, fn=fn, op=op):
            counts[op] += 1
            return fn(x)

        setattr(impl, op, wrapped)
    n = len(PROMPT_IDS)
    engine.page_table[0, :n] = torch.arange(n)
    req = Req(input_ids=torch.tensor(PROMPT_IDS, dtype=torch.int32), table_idx=0, cached_len=0,
              output_len=4, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.input_ids, batch.positions = req.input_ids.clone(), torch.arange(n, dtype=torch.int32)
    batch.out_loc = engine.page_table[0, :n]
    engine.attn_backend.prepare_metadata(batch)
    with engine.ctx.forward_batch(batch):
        logits = engine.model.forward()
    qkv = engine.model.model.layers.op_list[0].self_attn.qkv_proj.weight
    out.put((rank, logits[0].numpy().copy(), tuple(qkv.shape), counts, tuple(engine.kv_cache.k_cache(0).shape)))
    engine.shutdown()


if __name__ == "__main__":
    show_shapes()
    mp.set_start_method("spawn")
    results = {}
    for size in (1, 2):
        q: mp.Queue = mp.Queue()
        procs = [mp.Process(target=run, args=(r, size, q)) for r in range(size)]
        for p in procs:
            p.start()
        results[size] = sorted((q.get() for _ in range(size)), key=lambda x: x[0])
        for p in procs:
            p.join()
    base = torch.from_numpy(results[1][0][1])
    for rank, logits, qkv_shape, counts, k_shape in results[2]:
        print(f"TP=2 rank {rank}: qkv_proj {qkv_shape}，KV 池每层 {tuple(k_shape)}，"
              f"一次前向 all_reduce {counts['all_reduce']} 次、all_gather {counts['all_gather']} 次，"
              f"logits 与 TP=1 最大误差 {(torch.from_numpy(logits) - base).abs().max().item():.2e}")
