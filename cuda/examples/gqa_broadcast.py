import torch

torch.manual_seed(0)
b, hq, hkv, s, d = 1, 8, 2, 16, 64
g = hq // hkv
q = torch.randn(b, hq, 1, d)        # decode：每个头一个 query
k = torch.randn(b, hkv, s, d)

k_rep = k.repeat_interleave(g, dim=1)                    # 写法一：KV 复制成 8 个头
s1 = q @ k_rep.transpose(-1, -2)
s2 = (q.view(b, hkv, g, 1, d) @ k.view(b, hkv, 1, s, d).transpose(-1, -2)).view(b, hq, 1, s)   # 写法二：按组广播
print("两种写法结果一致：", torch.allclose(s1, s2, atol=1e-5))
print("复制出来的 K 是原来的", k_rep.untyped_storage().nbytes() // k.untyped_storage().nbytes(), "倍")
