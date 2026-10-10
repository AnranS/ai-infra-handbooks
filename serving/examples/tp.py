"""tp.py —— 从零实现张量并行（Megatron 式），在 CPU 上用 gloo 后端跑多进程。

用法：torchrun --nproc-per-node 2 --master-addr 127.0.0.1 --master-port 29500 tp.py
每个进程只持有 1/tp 的权重和 KV Cache，每层两次 all-reduce；rank 0 与单进程的完整模型比较输出。
"""

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from mini_llm import KVCache, RMSNorm, Transformer, apply_rope, rope_cos_sin

NUM_ALL_REDUCE = 0


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    global NUM_ALL_REDUCE
    NUM_ALL_REDUCE += 1
    dist.all_reduce(x)                     # 默认求和：把各 rank 的部分结果加起来
    return x


class ColumnParallelLinear(nn.Module):
    """按输出维切分：每个 rank 算输出的一段，不需要通信。"""

    def __init__(self, full: nn.Linear, rank: int, world: int, rows: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(full.weight[rows].clone())
        self.bias = nn.Parameter(full.bias[rows].clone()) if full.bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """按输入维切分：每个 rank 用自己那段输入算出一个"部分和"，all-reduce 后才是完整结果。"""

    def __init__(self, full: nn.Linear, rank: int, world: int, cols: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(full.weight[:, cols].clone())
        assert full.bias is None

    def forward(self, x):
        return all_reduce(F.linear(x, self.weight))


class VocabParallelEmbedding(nn.Module):
    """词表切分：每个 rank 只存一段词表；不在本段的 token 查出 0，all-reduce 后得到完整的嵌入。"""

    def __init__(self, full: nn.Embedding, rank: int, world: int):
        super().__init__()
        per = (full.num_embeddings + world - 1) // world
        self.start, self.end = rank * per, min((rank + 1) * per, full.num_embeddings)
        self.weight = nn.Parameter(full.weight[self.start:self.end].clone())

    def forward(self, ids):
        mask = (ids < self.start) | (ids >= self.end)
        out = F.embedding((ids - self.start).masked_fill(mask, 0), self.weight)
        return all_reduce(out.masked_fill(mask[..., None], 0.0))


class TPAttention(nn.Module):
    """按头切分：每个 rank 负责 nh/tp 个 query 头和 nkv/tp 个 KV 头，KV Cache 也只存自己的头。"""

    def __init__(self, attn, cfg, rank: int, world: int):
        super().__init__()
        hd = cfg.hd
        self.nh, self.nkv, self.hd = cfg.num_attention_heads // world, cfg.num_key_value_heads // world, hd
        q_rows = torch.arange(rank * self.nh * hd, (rank + 1) * self.nh * hd)
        kv_rows = torch.arange(rank * self.nkv * hd, (rank + 1) * self.nkv * hd)
        self.q_proj = ColumnParallelLinear(attn.q_proj, rank, world, q_rows)
        self.k_proj = ColumnParallelLinear(attn.k_proj, rank, world, kv_rows)
        self.v_proj = ColumnParallelLinear(attn.v_proj, rank, world, kv_rows)
        self.o_proj = RowParallelLinear(attn.o_proj, rank, world, q_rows)
        self.q_norm, self.k_norm = attn.q_norm, attn.k_norm     # QK-Norm 按头做、权重只有 head_dim 维：每个 rank 存完整的一份
        self.layer = attn.layer

    def forward(self, x, cos, sin, cache):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = cache.update(self.layer, k, v)
        S = k.shape[2]
        rep = self.nh // self.nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.nh * self.hd))


class TPMLP(nn.Module):
    """gate/up 按列切分，down 按行切分：中间的 SiLU × up 在各 rank 本地完成，整个 MLP 只需一次 all-reduce。"""

    def __init__(self, mlp, rank: int, world: int):
        super().__init__()
        n = mlp.gate_proj.out_features // world
        rows = torch.arange(rank * n, (rank + 1) * n)
        self.gate_proj = ColumnParallelLinear(mlp.gate_proj, rank, world, rows)
        self.up_proj = ColumnParallelLinear(mlp.up_proj, rank, world, rows)
        self.down_proj = RowParallelLinear(mlp.down_proj, rank, world, rows)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TPTransformer(nn.Module):
    def __init__(self, full: Transformer, rank: int, world: int):
        super().__init__()
        cfg = self.cfg = full.cfg
        assert cfg.num_attention_heads % world == 0 and cfg.num_key_value_heads % world == 0
        self.world = world
        self.embed_tokens = VocabParallelEmbedding(full.embed_tokens, rank, world)
        self.layers = nn.ModuleList()
        for layer in full.layers:
            new = nn.Module()
            new.input_layernorm, new.post_attention_layernorm = layer.input_layernorm, layer.post_attention_layernorm
            new.self_attn = TPAttention(layer.self_attn, cfg, rank, world)
            new.mlp = TPMLP(layer.mlp, rank, world)
            self.layers.append(new)
        self.norm = full.norm
        # 输出层按词表切分（与嵌入共享权重时就是嵌入的那一段），各 rank 算出一段 logits 再 all-gather
        self.lm_head_weight = self.embed_tokens.weight if cfg.tie_word_embeddings else \
            nn.Parameter(full.lm_head.weight[self.embed_tokens.start:self.embed_tokens.end].clone())

    def forward(self, ids, cache):
        start = cache.length
        cos, sin = rope_cos_sin(torch.arange(start, start + ids.shape[1]), self.cfg.hd, self.cfg.rope_theta)
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = x + layer.self_attn(layer.input_layernorm(x), cos, sin, cache)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        local = F.linear(self.norm(x[:, -1:]), self.lm_head_weight)       # 只算最后一个位置
        parts = [torch.empty_like(local) for _ in range(self.world)]
        dist.all_gather(parts, local)
        return torch.cat(parts, dim=-1)[..., : self.cfg.vocab_size]


def main():
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.set_num_threads(max(1, int(os.environ.get("THREADS", "8"))))
    full = Transformer.from_pretrained(os.environ.get("MODEL", "models/Qwen3-0.6B"))
    ids = torch.tensor([[151644, 872, 198, 105043, 100165, 11319, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]])  # "你是谁？"的对话模板（关闭思考）
    model = TPTransformer(full, rank, world)
    n_local = sum(p.numel() for p in model.parameters())

    global NUM_ALL_REDUCE
    with torch.no_grad():
        cache, tokens = KVCache(full.cfg.num_hidden_layers), []
        logits = model(ids, cache)
        prefill_all_reduce = NUM_ALL_REDUCE
        for _ in range(20):
            nxt = logits[0, -1].argmax().item()
            tokens.append(nxt)
            logits = model(torch.tensor([[nxt]]), cache)
    kv_heads_local = cache.k[0].shape[1]

    if rank == 0:
        with torch.no_grad():
            ref_cache, ref_tokens = KVCache(full.cfg.num_hidden_layers), []
            ref_logits = full(ids, ref_cache)[:, -1:]
            for _ in range(20):
                nxt = ref_logits[0, -1].argmax().item()
                ref_tokens.append(nxt)
                ref_logits = full(torch.tensor([[nxt]]), ref_cache)[:, -1:]
        n_full = sum(p.numel() for p in full.parameters())
        print(f"TP={world}：每个 rank 持有 {n_local / 1e6:.0f}M 参数（完整模型 {n_full / 1e6:.0f}M），"
              f"每层 KV Cache 存 {kv_heads_local} 个 KV 头（共 {full.cfg.num_key_value_heads} 个）")
        print(f"prefill 一次前向的 all-reduce 次数：{prefill_all_reduce}（{full.cfg.num_hidden_layers} 层 × 2 + 嵌入 1）")
        print(f"最后一步 logits 与单进程的最大误差：{(logits[0, -1] - ref_logits[0, -1]).abs().max().item():.1e}")
        print(f"贪心生成 20 个 token 与单进程一致：{tokens == ref_tokens}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
