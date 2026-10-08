import torch

torch.manual_seed(0)
# 模拟一批回答：用旧策略采样，更新几步之后，每个 token 的 log 比率 log(π/π_old) 带一点噪声
# （MoE 模型里，同一个 token 在新旧策略下可能被路由到不同的专家，噪声更大）
for sigma in (0.02, 0.2):
    print(f"每个 token 的 log 比率 ~ N(0, {sigma}²)：")
    for L in (100, 1000, 4000):
        log_r = torch.randn(512, L) * sigma                   # 512 个回答，每个长 L
        clipped = ((log_r.exp() < 0.8) | (log_r.exp() > 1.28)).float().mean()   # 逐 token 裁剪（ε_low 0.2、ε_high 0.28）
        product = log_r.sum(1).exp()                          # 严格的序列级重要性权重：逐 token 比率的乘积
        geo = log_r.mean(1).exp()                             # GSPO：几何平均，按长度归一化
        print(f"  L={L:4d}：超出裁剪范围的 token {clipped:5.1%}；比率的乘积在 [{product.min():.0e}, {product.max():.0e}]；"
              f"几何平均在 [{geo.min():.4f}, {geo.max():.4f}]")
