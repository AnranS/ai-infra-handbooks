"""pp.py —— 流水线并行的最小实现：两个进程各持有一半的层，点对点传递隐藏状态。

用法：torchrun --standalone --nproc-per-node 2 pp.py
stage 0：嵌入 + 前一半的层；stage 1：后一半的层 + 最终归一化 + 输出层。
每个 stage 只存自己那些层的权重和 KV Cache；每生成一个 token，隐藏状态从 stage 0 发到 stage 1，
采样出的 token 再发回 stage 0。
"""

import os

import torch
import torch.distributed as dist

from mini_llm import KVCache, Transformer, rope_cos_sin


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    full = Transformer.from_pretrained(os.environ.get("MODEL", "models/Qwen3-0.6B"))
    cfg = full.cfg
    L = cfg.num_hidden_layers
    my_layers = range(0, L // 2) if rank == 0 else range(L // 2, L)
    cache = KVCache(L)                                   # 只会用到自己那些层的槽位
    prompt = [151644, 872, 198, 105043, 100165, 11319, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
    ids, generated, sent_bytes = torch.tensor([prompt]), [], 0

    with torch.no_grad():
        for step in range(20):
            T = ids.shape[1]
            first = cache.k[my_layers[0]]                # 本 stage 第一层的 KV Cache 长度就是已处理的 token 数
            start = 0 if first is None else first.shape[2]
            cos, sin = rope_cos_sin(torch.arange(start, start + T), cfg.hd, cfg.rope_theta)
            if rank == 0:
                x = full.embed_tokens(ids)
            else:
                x = torch.empty(1, T, cfg.hidden_size)
                dist.recv(x, src=0)                      # 收到上一个 stage 的隐藏状态
            for i in my_layers:
                x = full.layers[i](x, cos, sin, cache)
            if rank == 0:
                dist.send(x, dst=1)                      # 发给下一个 stage：[T, hidden]
                sent_bytes += x.numel() * 2              # 按 BF16 计算实际部署时的通信量
                nxt = torch.empty(1, dtype=torch.long)
                dist.recv(nxt, src=1)                    # 等最后一个 stage 采样出的 token
            else:
                nxt = full.lm_head(full.norm(x[:, -1])).argmax(-1)
                dist.send(nxt, dst=0)
            generated.append(nxt.item())
            ids = nxt.view(1, 1)

    if rank == 0:
        ref = []
        with torch.no_grad():
            ref_cache, x = KVCache(L), torch.tensor([prompt])
            for _ in range(20):
                nxt = full(x, ref_cache)[0, -1].argmax().item()
                ref.append(nxt)
                x = torch.tensor([[nxt]])
        stage_layers = [f"{r.start}～{r.stop - 1}" for r in (range(0, L // 2), range(L // 2, L))]
        print(f"PP=2：stage 0 负责第 {stage_layers[0]} 层，stage 1 负责第 {stage_layers[1]} 层")
        print(f"生成 20 个 token，stage 之间共传输 {sent_bytes / 1024:.0f} KB 隐藏状态")
        print(f"与单进程一致：{generated == ref}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
