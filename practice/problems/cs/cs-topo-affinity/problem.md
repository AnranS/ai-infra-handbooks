---
title: 拓扑亲和性：给每张卡挑网卡
chapter: arch/multi-gpu.md
difficulty: 中等
tags: [拓扑, NUMA, GPUDirect, NCCL]
---
`nvidia-smi topo -m` 给出每两个设备之间的连接方式。按从快到慢排序：`NV#`（# 条 NVLink）> `PIX`（同一个 PCIe 交换芯片）> `PXB`（跨交换芯片）> `PHB`（经过根桥）> `NODE`（同一个 NUMA 节点内）> `SYS`（跨 CPU 插槽）。实现三个函数：

1. `link_rank(link)`：把连接方式变成一个可比较的整数，越小越快。`NV#` 一律排在最前（不同的 # 之间按链路数多的更快），其余按上面的顺序；无法识别的记为最慢；
2. `pick_nics(topo, gpus, nics)`：`topo[a][b]` 是设备 a 到设备 b 的连接方式。给每张 GPU 选一张网卡：**每张网卡最多给一张 GPU 用**，让所有 GPU 的连接总代价（`link_rank` 之和）最小；代价相同时，按 GPU 顺序优先选编号靠前的网卡。返回 `{gpu: nic}`；
3. `all_nvlink(topo, gpus)`：这组 GPU 两两之间是否都是 NVLink。

```python
topo = {"GPU0": {"NIC0": "PIX", "NIC1": "SYS"}, "GPU1": {"NIC0": "PIX", "NIC1": "NODE"}}
pick_nics(topo, ["GPU0", "GPU1"], ["NIC0", "NIC1"])   # {"GPU0": "NIC0", "GPU1": "NIC1"}
```

<!-- 题解 -->
`link_rank` 用一张顺序表；`NV#` 解析出链路数，用一个比所有其他类型都小的值减去它，保证 `NV18` 比 `NV2` 更靠前。

`pick_nics` 是一个小规模的指派问题。GPU 和网卡通常都只有 8 个，直接对网卡的全排列取最小总代价就可以（`itertools.permutations`），代价相同时 `permutations` 的字典序保证优先选编号靠前的。贪心（每张 GPU 各自挑最好的没被占用的网卡）会在"两张 GPU 抢同一张网卡"时给出更差的结果：例子里 GPU0 只有 NIC0 是近的，GPU1 到两张网卡分别是 `PIX` 和 `NODE`，贪心让 GPU0 先挑走 NIC0 是对的，但如果顺序反过来，贪心就会让 GPU1 占走 NIC0，把 GPU0 逼到 `SYS`。

实际用处：跨机的 EP、PD 分离的 KV 传输都要 GPUDirect RDMA，网卡和 GPU 在同一个 PCIe 交换芯片下（`PIX`）时数据只在交换芯片里走一跳；隔了插槽（`SYS`）就要经过插槽间的互联，带宽和延迟都变差。NCCL 的 `NCCL_IB_HCA` 可以显式指定每个 rank 用哪张网卡。
