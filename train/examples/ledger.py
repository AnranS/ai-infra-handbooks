GiB = 2**30

# Llama-3-70B 量级的稠密模型
PSI, LAYERS, HIDDEN = 70.6e9, 80, 8192
SEQ, MBS = 8192, 1            # 序列长度、micro-batch 大小


def model_states(psi, zero, dp):
    """混合精度 + Adam：bf16 参数 2 + bf16 梯度 2 + fp32 主参数/一阶矩/二阶矩 12 = 16 字节/参数"""
    return {0: 16 * psi, 1: 4 * psi + 12 * psi / dp, 2: 2 * psi + 14 * psi / dp, 3: 16 * psi / dp}[zero]


def activations(layers, tp=1, sp=False, recompute=False, inflight=1):
    """每层激活（bf16，用 FlashAttention）：不切分 34·sbh；TP 不开 SP 时 sbh·(10 + 24/t)；开 SP 时 34·sbh/t；全量重计算 2·sbh/t"""
    sbh = SEQ * MBS * HIDDEN
    if recompute:
        per = 2 * sbh / tp
    elif sp:
        per = 34 * sbh / tp
    else:
        per = sbh * (10 + 24 / tp)
    return per * layers * inflight


configs = [
    ("DP=64，不切分", dict(zero=0, dp=64, tp=1, pp=1)),
    ("DP=64，ZeRO-3", dict(zero=3, dp=64, tp=1, pp=1)),
    ("DP=64，ZeRO-3 + 全量重计算", dict(zero=3, dp=64, tp=1, pp=1, recompute=True)),
    ("TP=8（SP）× PP=4 × DP=2，ZeRO-1", dict(zero=1, dp=2, tp=8, pp=4, sp=True)),
]
for name, c in configs:
    psi_local = PSI / (c["tp"] * c["pp"])                  # 每张卡负责的参数
    states = model_states(psi_local, c["zero"], c["dp"])
    layers_local = LAYERS // c["pp"]
    inflight = c["pp"]                                     # 1F1B：第一个 stage 最多同时保存 pp 个 micro-batch 的激活
    act = activations(layers_local, c["tp"], c.get("sp", False), c.get("recompute", False), inflight)
    total = (states + act) / GiB
    print(f"{name}：模型状态 {states / GiB:.1f} GiB，激活 {act / GiB:.1f} GiB，合计 {total:.1f} GiB，"
          f"{'放得下' if total < 80 * 0.9 else '放不下'}")

TOKENS = 1e9
flops = 6 * PSI * TOKENS
seconds = flops / (64 * 989e12 * 0.40)
print(f"训练 10 亿 token：{flops:.2e} FLOPs，64 张 H100、MFU 40% 需要 {seconds / 3600:.1f} 小时")
