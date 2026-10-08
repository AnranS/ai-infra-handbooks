import numpy as np

GPUS_PER_NODE = 8
TP, CP, DP, PP = 2, 2, 2, 2                    # 16 张卡：2 个节点

# 由外到内 PP → DP → CP → TP：最内层的维度 rank 号相邻，落在同一个节点里
mesh = np.arange(PP * DP * CP * TP).reshape(PP, DP, CP, TP)
axes = {"TP": 3, "CP": 2, "DP": 1, "PP": 0}

for name, axis in axes.items():
    groups = np.moveaxis(mesh, axis, -1).reshape(-1, mesh.shape[axis])   # 沿这一维的每一条"线"是一个通信组
    local = all(len({r // GPUS_PER_NODE for r in g}) == 1 for g in groups)
    print(f"{name} 组：{' '.join(str(g.tolist()) for g in groups[:4])} …共 {len(groups)} 组，"
          f"{'都在节点内' if local else '跨节点'}")

rank = 13
pp, dp, cp, tp = (int(i[0]) for i in np.nonzero(mesh == rank))
print(f"rank {rank} 的坐标：PP={pp} DP={dp} CP={cp} TP={tp}")
