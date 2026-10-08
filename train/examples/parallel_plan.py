from dataclasses import dataclass
from itertools import product

GiB = 2**30
PEAK, EFF = 989e12, 0.6                 # H100 BF16 稠密峰值；大矩阵乘实际能达到的比例
NVLINK, NET = 450e9, 50e9               # 每张卡单方向带宽：节点内 NVLink、节点间网卡（400 Gb/s）
NODE, MEM = 8, 80e9 * 0.9               # 每节点 8 卡；每卡 80 GB，留 10% 给碎片和临时缓冲
MIN_CP_CHUNK = 4096                     # 每个 CP rank 至少分到这么多 token，否则 ring 的每一步太小


@dataclass
class Model:
    N: float     # 参数量（含词表）
    L: int       # 层数
    H: int       # 隐藏维度
    KV: int      # K 或 V 一行的宽度（GQA：KV 头数 × 头维度）
    V: int       # 词表大小


def bandwidth(inner, size):
    """维度由内到外依次是 TP → CP → DP → PP：一个并行组的跨度不超过一个节点就走 NVLink"""
    return NVLINK if inner * size <= NODE else NET


def plan(m, seq, gbs, gpus, tp, cp, pp, mbs=1):
    if tp > NODE or gpus % (tp * cp * pp) or m.L % pp or (cp > 1 and seq // cp < MIN_CP_CHUNK):
        return None
    dp = gpus // (tp * cp * pp)
    if gbs % (dp * mbs):
        return None
    micro = gbs // (dp * mbs)                          # 每条流水线每步要跑的 micro-batch 数
    s, layers = seq // cp, m.L // pp                   # 每张卡上的序列长度和层数

    # ---- 显存：第一个 stage 最紧（参数最多、同时保存 pp 个 micro-batch 的激活）
    psi = m.N / (tp * pp)
    states = 4 * psi + 12 * psi / (dp * cp)            # ZeRO-1：优化器状态在 DP×CP 上切分
    sbh = s * mbs * m.H / tp
    for rc in range(layers + 1):                       # 从 0 开始加全量重计算的层数，直到放得下
        act = (34 * (layers - rc) + 2 * rc) * sbh * min(pp, micro)
        if states + act <= MEM:
            break
    else:
        return None

    # ---- 时间：一个 micro-batch 在最慢的 stage（最后一个，多了 LM head）上要多久
    f_layer = 6 * (m.N - 2 * m.V * m.H) / m.L + 6 * seq * m.H     # 每个 token 每层：线性层 + 因果注意力
    f_head = 6 * m.V * m.H
    sec = s * mbs / tp / (PEAK * EFF)                  # 每 FLOP/token 在这张卡上要多少秒
    avg = (m.L * f_layer + f_head) / pp * sec          # 各 stage 平均分到的计算
    recompute = rc * f_layer / 3 * sec                 # 重算的层多做一次前向（前向是前向 + 反向的 1/3）
    slow = layers * f_layer * sec + recompute + f_head * sec
    act_bytes = s * mbs * m.H * 2
    tp_c = layers * 4 * 2 * (tp - 1) / tp * act_bytes / bandwidth(1, tp)            # 每层前向 2 次、反向 2 次，不重叠
    ring = layers * 3 * (cp - 1) * (2 * s * mbs * m.KV * 2) / bandwidth(tp, cp)      # ring 传 KV（反向再传 dKV）
    cp_c = max(0.0, ring - layers * 6 * seq * m.H * sec)                           # 与注意力计算重叠，只算露出来的部分
    pp_c = 2 * act_bytes / tp / bandwidth(tp * cp * dp, pp) * 0.5 if pp > 1 else 0  # 激活和梯度的点对点，一半藏住
    t_mb = slow + tp_c + cp_c + pp_c
    dpc = dp * cp
    dp_c = 2 * (dpc - 1) / dpc * 2 * psi / bandwidth(tp, dpc) * 0.2                  # 梯度 RS + 参数 AG，80% 与反向重叠
    parts = {"计算": micro * avg, "重算": micro * recompute, "不均衡": micro * (slow - recompute - avg),
             "TP": micro * tp_c, "CP": micro * cp_c, "PP": micro * pp_c, "气泡": (pp - 1) * t_mb, "DP": dp_c}
    step = sum(parts.values())
    model_flops = (m.L * f_layer + f_head) * gbs * seq          # 不含重计算：MFU 只认模型本身要做的运算
    return dict(tp=tp, cp=cp, pp=pp, dp=dp, mfu=model_flops / (step * gpus * PEAK),
                mem=(states + act) / GiB, rc=rc, layers=layers, step=step, parts=parts)


def show(p):
    share = " ".join(f"{k} {v / p['step']:.0%}" for k, v in p["parts"].items() if v / p["step"] >= 0.005)
    rc = f"，重算 {p['rc']}/{p['layers']} 层" if p["rc"] else ""
    print(f"TP={p['tp']} CP={p['cp']} PP={p['pp']} DP={p['dp']}：MFU {p['mfu']:.1%}，"
          f"{p['mem']:.0f} GiB/卡{rc}｜{share}")


def search(title, m, seq, gbs, gpus, top=3, extra=()):
    plans = [p for tp, cp, pp in product((1, 2, 4, 8), (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16))
             if (p := plan(m, seq, gbs, gpus, tp, cp, pp))]
    plans.sort(key=lambda p: -p["mfu"])
    print(f"== {title}：{len(plans)} 种配置放得下")
    for p in plans[:top]:
        show(p)
    for tp, cp, pp in extra:                           # 再看几种"直觉配置"排在哪里
        rank, p = next((i, q) for i, q in enumerate(plans, 1) if (q["tp"], q["cp"], q["pp"]) == (tp, cp, pp))
        print(f"  第 {rank} 名：", end="")
        show(p)


llama70b = Model(N=70.6e9, L=80, H=8192, KV=1024, V=128256)
llama8b = Model(N=8.03e9, L=32, H=4096, KV=1024, V=128256)

if __name__ == "__main__":
    search("70B，64 卡，8K 序列，每步 4M token", llama70b, 8192, 512, 64, extra=[(8, 1, 1), (8, 1, 4)])
    search("70B，64 卡，128K 序列，每步 4M token", llama70b, 131072, 32, 64, extra=[(8, 1, 1)])
    search("8B，8 卡，8K 序列，每步 0.5M token", llama8b, 8192, 64, 8, extra=[(1, 1, 1)])
