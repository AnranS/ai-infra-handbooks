"""几个尺寸怎么选：词表、深浅、注意力头的 KV——先把账算出来，再决定"""
from model import GPT, GPTConfig


def breakdown(cfg):
    m = GPT(cfg)
    embed = cfg.vocab_size * cfg.d_model                      # 输入输出共享，只算一次
    return m.num_params(), embed


print("词表大小对一个 d=128 的小模型意味着什么（输入输出共享词嵌入）：")
print("  词表    总参数   词嵌入占比")
for v in (4096, 8192, 16384, 32768):
    total, embed = breakdown(GPTConfig(vocab_size=v))
    print(f"{v:6d}   {total / 1e6:5.2f}M   {embed / total:8.0%}")

print("\n同样约 1.3～1.5M 的非词嵌入参数，摊在宽度上还是深度上：")
print("   d  层数   非词嵌入   总参数   每层参数")
for d, n_layer in ((320, 1), (192, 3), (128, 8), (96, 13)):
    total, embed = breakdown(GPTConfig(d_model=d, n_layer=n_layer, n_head=max(1, d // 32)))
    print(f"{d:4d}  {n_layer:3d}    {(total - embed) / 1e6:5.2f}M   {total / 1e6:5.2f}M   {(total - embed) / n_layer / 1e3:7.0f}K")

print("\n推理时每个 token 的 KV cache（d=768、28 层、fp16，按一个 token 算）：")
print("  查询头  KV 头   每 token 的 KV   相对 MHA")
d_model, n_layer, n_head = 768, 28, 12
for kv in (12, 4, 2, 1):
    per_token = 2 * n_layer * kv * (d_model // n_head) * 2    # K 和 V，各 2 字节
    print(f"{n_head:6d}  {kv:5d}   {per_token / 1024:11.1f} KB   {kv / n_head:8.0%}")
