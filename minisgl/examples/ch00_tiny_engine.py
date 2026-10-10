"""第 0 步：一个能跑的最小推理引擎。

不 import minisgl 里的任何东西。读 config.json 和 safetensors，手写 Qwen3 的前向，
KV cache 用最朴素的方式存，贪心解码一个提示词，和 Hugging Face 逐 token 对答案。
整本书后面的 25 章，都是在这一份代码上做加法。
"""

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL = Path("models/Qwen3-0.6B")
torch.manual_seed(0)

# ------------------------------------------------------------------ 1. 配置与权重
cfg = json.loads((MODEL / "config.json").read_text())
H, HQ, HKV, D = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
L, EPS, THETA = cfg["num_hidden_layers"], cfg["rms_norm_eps"], cfg["rope_theta"]

with safe_open(MODEL / "model.safetensors", "pt") as f:
    W = {k: f.get_tensor(k).float() for k in f.keys()}       # bf16 -> fp32：为了和 HF 的 fp32 逐位对齐
if cfg.get("tie_word_embeddings"):                            # 输出层和词嵌入共用一份权重
    W["lm_head.weight"] = W["model.embed_tokens.weight"]
n_params = sum(v.numel() for k, v in W.items() if k != "lm_head.weight")
print(f"Qwen3-0.6B：{L} 层，hidden {H}，{HQ} 个 query 头 / {HKV} 个 KV 头，head_dim {D}，{n_params / 1e6:.0f}M 参数")


# ------------------------------------------------------------------ 2. 四个算子
def rmsnorm(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS))


def rope(x, pos):
    """旋转位置编码（前后两半配对）。x: [T, 头数, D]，pos: [T]"""
    inv_freq = 1.0 / (THETA ** (torch.arange(0, D, 2).float() / D))      # [D/2]
    ang = pos[:, None].float() * inv_freq[None, :]                       # [T, D/2]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]              # 广播到头维
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def attention(q, k, v):
    """q: [T, HQ, D]；k、v: [S, HKV, D]，S 是缓存里的总长度（含本轮）。返回 [T, HQ * D]"""
    T, S = q.shape[0], k.shape[0]
    k = k.repeat_interleave(HQ // HKV, dim=1)                 # GQA：每个 KV 头服务 HQ/HKV 个 query 头
    v = v.repeat_interleave(HQ // HKV, dim=1)
    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(D)
    causal = torch.arange(S)[None, :] <= torch.arange(S - T, S)[:, None]  # 第 t 个 query 的绝对位置是 S-T+t
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.einsum("hts,shd->thd", scores.softmax(-1), v)
    return out.reshape(T, HQ * D)


def mlp(x, p):
    return F.linear(F.silu(F.linear(x, W[p + "gate_proj.weight"])) * F.linear(x, W[p + "up_proj.weight"]),
                    W[p + "down_proj.weight"])


# ------------------------------------------------------------------ 3. 一层、整个模型
def layer(x, i, pos, cache):
    """x: [T, H]。cache[i] 是这一层已经算过的 (k, v)，形状 [S_old, HKV, D]；本轮算完追加进去。"""
    p = f"model.layers.{i}."
    T = x.shape[0]
    h = rmsnorm(x, W[p + "input_layernorm.weight"])
    q = F.linear(h, W[p + "self_attn.q_proj.weight"]).view(T, HQ, D)
    k = F.linear(h, W[p + "self_attn.k_proj.weight"]).view(T, HKV, D)
    v = F.linear(h, W[p + "self_attn.v_proj.weight"]).view(T, HKV, D)
    q = rope(rmsnorm(q, W[p + "self_attn.q_norm.weight"]), pos)         # Qwen3：q、k 各过一次 RMSNorm 再旋转
    k = rope(rmsnorm(k, W[p + "self_attn.k_norm.weight"]), pos)
    if cache[i] is not None:                                             # 把历史 KV 接在前面
        k, v = torch.cat([cache[i][0], k]), torch.cat([cache[i][1], v])
    cache[i] = (k, v)
    x = x + F.linear(attention(q, k, v), W[p + "self_attn.o_proj.weight"])
    return x + mlp(rmsnorm(x, W[p + "post_attention_layernorm.weight"]), p + "mlp.")


def forward(ids, pos, cache):
    """ids、pos: [T]。返回最后一个位置的 logits：[vocab]"""
    x = W["model.embed_tokens.weight"][ids]
    for i in range(L):
        x = layer(x, i, pos, cache)
    x = rmsnorm(x[-1:], W["model.norm.weight"])
    return F.linear(x, W["lm_head.weight"])[0]


# ------------------------------------------------------------------ 4. 生成：prefill 一次，decode 一个 token 一步
def generate(ids, n):
    cache = [None] * L
    out = []
    t0 = time.perf_counter()
    logits = forward(torch.tensor(ids), torch.arange(len(ids)), cache)  # prefill：整段提示词一起算
    t1 = time.perf_counter()
    for step in range(n):                                               # decode：每步只算 1 个新 token
        nxt = int(logits.argmax())                                      # 贪心：取最大的那个
        out.append(nxt)
        logits = forward(torch.tensor([nxt]), torch.tensor([len(ids) + step]), cache)
    t2 = time.perf_counter()
    return out, t1 - t0, t2 - t1


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402  只用来分词和对答案

tok = AutoTokenizer.from_pretrained(MODEL)
prompt = "The capital of France is"
ids = tok(prompt).input_ids
print(f"提示词 {prompt!r}（{len(ids)} 个 token）")
out, t_prefill, t_decode = generate(ids, 16)
print(f"生成：{tok.decode(out)!r}")
print(f"默认 {torch.get_num_threads()} 个线程：prefill {t_prefill * 1000:.0f} ms，"
      f"decode 16 步 {t_decode:.2f} s（{16 / t_decode:.1f} tokens/s）")
torch.set_num_threads(max(1, os.cpu_count() // 2))          # 逻辑核数的一半 ≈ 物理核数：decode 的矩阵乘太小，线程多了反而慢
out2, t_prefill, t_decode = generate(ids, 16)
assert out2 == out
print(f"改成 {torch.get_num_threads()} 个线程：prefill {t_prefill * 1000:.0f} ms，"
      f"decode 16 步 {t_decode:.2f} s（{16 / t_decode:.1f} tokens/s）")

hf = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
with torch.no_grad():
    ref = hf.generate(torch.tensor([ids]), max_new_tokens=16, do_sample=False)[0, len(ids):].tolist()
print(f"与 Hugging Face 逐 token 一致：{out == ref}")
