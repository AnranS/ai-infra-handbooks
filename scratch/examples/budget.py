# 一张 16 GB 的卡：多大的模型放得下，训完要多久
GB = 1024 ** 3
VRAM = 16 * GB
RESERVED = 1.2 * GB                      # 驱动、CUDA context、cuBLAS workspace 和碎片，先扣掉


def states(params):
    """混合精度 AdamW 的常驻显存 = 16Ψ 字节：
    bf16 参数 2Ψ + bf16 梯度 2Ψ + fp32 参数副本 4Ψ + Adam 的 m、v 各 4Ψ"""
    return 16 * params


def activations(layers, d, seq, batch, recompute):
    """激活。不重计算时每层每 token 大约 34d 字节（bf16，含注意力里的中间量）；
    整层重计算只留每层的输入，约 2d 字节，代价是多做一次前向（约 +30% 时间）。"""
    return layers * seq * batch * (2 if recompute else 34) * d


print("模型能不能放进 16 GB（seq 1024，micro-batch 8，整层重计算）")
for name, layers, d, params in [
    ("30M", 6, 384, 30e6), ("124M / GPT-2", 12, 768, 124e6), ("350M", 24, 1024, 350e6),
    ("770M", 24, 1536, 770e6), ("1.4B", 24, 2048, 1.4e9),
]:
    st, act = states(params), activations(layers, d, 1024, 8, True)
    total = st + act + RESERVED
    print(f"  {name:<14} {layers:>2} 层 × {d:<5} 权重+优化器 {st / GB:>5.1f} GB  "
          f"激活 {act / GB:>4.1f} GB  合计 {total / GB:>5.1f} GB  "
          f"{'放得下' if total < VRAM else '放不下，要更小的 batch 或 LoRA'}")

print()
print("不重计算的话激活是多少（同样 seq 1024、micro-batch 8）")
for name, layers, d in [("124M / GPT-2", 12, 768), ("350M", 24, 1024)]:
    a1 = activations(layers, d, 1024, 8, False)
    a2 = activations(layers, d, 1024, 8, True)
    print(f"  {name:<14} 不重计算 {a1 / GB:>5.1f} GB，整层重计算 {a2 / GB:>4.1f} GB，省了 {a1 / a2:>4.0f} 倍")

print()
print("训完要多久（Chinchilla 的 20 token / 参数，6ΨN FLOP）")
for name, params, tflops in [("30M", 30e6, 60), ("124M / GPT-2", 124e6, 90), ("350M", 350e6, 110)]:
    tokens = 20 * params
    hours = 6 * params * tokens / (tflops * 1e12) / 3600
    print(f"  {name:<14} {tokens / 1e9:>5.1f}B token，按实测有效算力 {tflops} TFLOPS 约 {hours:>5.1f} 小时")
