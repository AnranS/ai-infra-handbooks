import torch
import torch.distributed as dist
import torch.nn.functional as F

from tp_ops import CopyToTP, GatherSeq, ReduceFromTP, ScatterSeq

dist.init_process_group("gloo")
rank, tp = dist.get_rank(), dist.get_world_size()
S, H, FFN = 8, 16, 64                                   # 序列长度、隐藏维度、FFN 中间维度

torch.manual_seed(0)                                    # 所有 rank 生成同样的完整权重和输入
x_full = torch.randn(S, H)
ln_w = torch.randn(H)
w1 = torch.randn(FFN, H) / H ** 0.5                     # 升维：[FFN, H]
w2 = torch.randn(H, FFN) / FFN ** 0.5                   # 降维：[H, FFN]
cols = slice(rank * FFN // tp, (rank + 1) * FFN // tp)  # 本 rank 负责的 FFN 中间维度
rows = slice(rank * S // tp, (rank + 1) * S // tp)      # 序列并行时本 rank 负责的 token


def block(x, ln_w, w1, w2):                             # 单进程参照：LayerNorm → 升维 → GELU → 降维
    h = F.layer_norm(x, (H,), weight=ln_w)
    return F.linear(F.gelu(F.linear(h, w1)), w2)


ref_in = [t.clone().requires_grad_() for t in (x_full, ln_w, w1, w2)]
ref_out = block(*ref_in)
ref_out.square().sum().backward()
r_x, r_ln, r_w1, r_w2 = (t.grad for t in ref_in)

# ---- 张量并行：w1 按行切（输出列切分），w2 按列切（输入行切分），中间的 GELU 各算各的
x = x_full.clone().requires_grad_()
lw = ln_w.clone().requires_grad_()
w1_s = w1[cols].clone().requires_grad_()
w2_s = w2[:, cols].clone().requires_grad_()
h = CopyToTP.apply(F.layer_norm(x, (H,), weight=lw))    # f
out = ReduceFromTP.apply(F.linear(F.gelu(F.linear(h, w1_s)), w2_s))   # g：部分和相加
out.square().sum().backward()
tp_ok = [torch.allclose(out, ref_out, atol=1e-5), torch.allclose(w1_s.grad, r_w1[cols], atol=1e-5),
         torch.allclose(w2_s.grad, r_w2[:, cols], atol=1e-5), torch.allclose(x.grad, r_x, atol=1e-5)]

# ---- 序列并行：LayerNorm 只处理本 rank 的那一段序列；进入 TP 区域前 all-gather，出来时 reduce-scatter
x = x_full[rows].clone().requires_grad_()
lw = ln_w.clone().requires_grad_()
w1_s = w1[cols].clone().requires_grad_()
w2_s = w2[:, cols].clone().requires_grad_()
h = GatherSeq.apply(F.layer_norm(x, (H,), weight=lw))   # 激活在 LayerNorm 处只有 S/tp 行
out = ScatterSeq.apply(F.linear(F.gelu(F.linear(h, w1_s)), w2_s))
out.square().sum().backward()
dist.all_reduce(lw.grad)                                 # LayerNorm 的权重被各段序列共用：梯度要在 TP 组内求和
sp_ok = [torch.allclose(out, ref_out[rows], atol=1e-5), torch.allclose(x.grad, r_x[rows], atol=1e-5),
         torch.allclose(w1_s.grad, r_w1[cols], atol=1e-5), torch.allclose(lw.grad, r_ln, atol=1e-5)]

flags = torch.tensor([int(all(tp_ok)), int(all(sp_ok))])
dist.all_reduce(flags, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"TP={tp}：输出、w1 分片梯度、w2 分片梯度、输入梯度都与单进程一致：", bool(flags[0]))
    print(f"TP={tp} + 序列并行：输出分片、输入分片梯度、w1 分片梯度、LayerNorm 权重梯度都一致：", bool(flags[1]))
dist.destroy_process_group()
