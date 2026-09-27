# 硬件与生态速查

<p class="lead">推理优化的一切估算都从硬件参数开始：显存容量决定能放下什么，带宽决定 decode 有多快，算力决定 prefill 有多快，互连决定能怎样并行。这一页整理常见推理硬件的关键参数、由它们推出的推理特性，以及推理软件生态的地图，方便估算与面试时查阅。</p>

!!! warning "关于数字"
    表中是各厂商公开资料中常见的**稠密**（非稀疏）峰值，不同形态（SXM、PCIe、NVL）和不同批次的产品可能不同。做正式评估时，请以官方数据手册和实测为准。

## GPU 参数

| 型号 | 显存 | 带宽 | BF16 稠密 | FP8 稠密 | 互连（每卡） | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| A100 SXM 80GB | 80 GB HBM2e | 2.0 TB/s | 312 TFLOPS | 不支持 | NVLink 600 GB/s | 无 FP8 Tensor Core，量化多用 W8A16/W4A16 |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | 1979 TFLOPS | NVLink 900 GB/s | 本书估算的默认硬件 |
| H200 | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 1979 TFLOPS | NVLink 900 GB/s | 算力同 H100，显存和带宽更大，decode 更快 |
| H20 | 96 GB HBM3 | 4.0 TB/s | 148 TFLOPS | 296 TFLOPS | NVLink 900 GB/s | 带宽高、算力低，适合 decode，不适合 prefill |
| L40S | 48 GB GDDR6 | 0.86 TB/s | 362 TFLOPS | 733 TFLOPS | 仅 PCIe | 推理用的性价比卡，带宽低 |
| B200 | 约 180～192 GB HBM3e | 8 TB/s | 约 2250 TFLOPS | 约 4500 TFLOPS | NVLink 1.8 TB/s | 原生 FP4（NVFP4/MXFP4） |
| MI300X | 192 GB HBM3 | 5.3 TB/s | 1307 TFLOPS | 2615 TFLOPS | Infinity Fabric | 大显存，ROCm 生态 |

另外值得了解的：GB200/GB300 NVL72 把 72 张 Blackwell GPU 放进一个 NVLink 域，大规模 EP 可以全部走 NVLink；国产加速卡（如华为昇腾）有自己的软件栈（CANN、MindIE），vLLM 和 SGLang 都有对应的硬件后端（例如 vLLM-Ascend 插件、SGLang 中的 Ascend 支持），国内岗位经常涉及。

## 由参数推出的推理特性

```python
gpus = {  # 名称: (显存 GB, 带宽 TB/s, BF16 TFLOPS, FP8 TFLOPS 或 None)
    "A100": (80, 2.0, 312, None), "H100": (80, 3.35, 989, 1979), "H200": (141, 4.8, 989, 1979),
    "H20": (96, 4.0, 148, 296), "L40S": (48, 0.86, 362, 733), "B200": (180, 8.0, 2250, 4500),
    "MI300X": (192, 5.3, 1307, 2615),
}
params = 8.03e9                                                   # LLaMA-3-8B，BF16 权重 16 GB
kv_per_token = 131072                                             # 128 KB/token（BF16）
print("型号     屋脊点(FLOP/B)  decode 下限(ms)  prefill 4K(ms)  可放下的 KV(token)")
for name, (mem, bw, bf16, fp8) in gpus.items():
    ridge = bf16 / bw                                             # TFLOPS / (TB/s) = FLOP/字节
    decode = params * 2 / (bw * 1e12) * 1e3                       # batch=1：读一遍权重
    prefill = 2 * params * 4096 / (bf16 * 1e12 * 0.5) * 1e3       # MFU 50%
    kv_tokens = (mem * 1e9 * 0.9 - params * 2) / kv_per_token
    print(f"{name:7s} {ridge:12.0f} {decode:15.1f} {prefill:15.0f} {kv_tokens / 1e3:14.0f}K")
```

```text
型号     屋脊点(FLOP/B)  decode 下限(ms)  prefill 4K(ms)  可放下的 KV(token)
A100             156             8.0             422            427K
H100             295             4.8             133            427K
H200             206             3.3             133            846K
H20               37             4.0             889            537K
L40S             421            18.7             363            207K
B200             281             2.0              58           1113K
MI300X           247             3.0             101           1196K
```

怎样读这张表：

- **屋脊点**越低，越容易达到计算受限，decode 越"划算"：H20 只有 37，意味着较小的 batch 就能把它的算力用满；它的带宽比 H100 还高，所以 decode 比 H100 快，而 prefill 慢好几倍。这种卡最适合 PD 分离中的 decode 实例；
- **decode 下限**只看带宽：H200 比 H100 快 40%，B200 快一倍多；
- **prefill** 只看算力：H20、A100 明显慢，FP8 能让支持它的卡再快一倍；
- **KV 容量**决定并发与长上下文：大显存的 H200、MI300X、B200 能容纳多几倍的请求。

## 互连

| 互连 | 带宽（单向、每卡或每端口） | 用途 |
| --- | --- | --- |
| NVLink 4（H100） | 450 GB/s（双向合计 900） | 机内 TP、EP |
| NVLink 5（B200） | 900 GB/s（双向合计 1.8 TB/s） | 机内、NVL72 机柜内 |
| PCIe 5.0 x16 | 约 64 GB/s | CPU 与 GPU 之间（KV 卸载、权重加载） |
| InfiniBand NDR 400 Gb/s | 约 50 GB/s | 机间 EP、PD 分离的 KV 传输、PP |
| RoCE 400/800 Gb/s | 约 50/100 GB/s | 以太网上的 RDMA，常见于云上 |

机内与机间的带宽差近一个数量级，这就是 TP 限于机内、跨机用 PP/EP/DP 的根本原因（见[张量并行](../distributed/tensor-parallel.md#通信的代价)）。

## 软件生态地图

| 类别 | 代表项目 | 说明 |
| --- | --- | --- |
| 推理引擎 | vLLM、SGLang、TensorRT-LLM、LMDeploy | 前两者开源社区最活跃；TensorRT-LLM 是 NVIDIA 官方，性能强、灵活性较低 |
| 端侧与本地 | llama.cpp、Ollama、MLC-LLM | CPU/Apple Silicon/移动端，GGUF 量化格式 |
| 注意力与算子库 | FlashAttention、FlashInfer、FlashMLA、CUTLASS、DeepGEMM | 推理引擎的底层 kernel |
| kernel 编程 | Triton、TileLang、CuTe DSL | 用高层语言写高性能 kernel |
| 通信 | NCCL、DeepEP、NIXL、Mooncake Transfer Engine | 集合通信、EP 的 all-to-all、KV 传输 |
| 编排与调度 | NVIDIA Dynamo、llm-d、SGLang Model Gateway、vLLM production stack | 多实例路由、PD 分离、KV 感知调度 |
| KV 存储 | Mooncake Store、LMCache、3FS | 分布式 KV 缓存与卸载 |
| 量化工具 | LLM Compressor、NVIDIA ModelOpt、GPTQModel、AutoAWQ | 产出各引擎可加载的量化 checkpoint |
| 评测 | lm-evaluation-harness、OpenCompass、各引擎自带的 benchmark | 精度与性能评测 |
| RL 框架 | verl、OpenRLHF、slime、AReaL | 训练与推理引擎的组合 |

## 小结

- [x] decode 看带宽、prefill 看算力、并发与长上下文看显存、并行方式看互连。
- [x] 屋脊点低的卡（如 H20）适合 decode，算力强的卡适合 prefill，这是异构 PD 分离的依据。
- [x] 机内 NVLink 与机间 RDMA 的带宽差近一个数量级，决定了并行策略的边界。
- [x] 生态地图：引擎、算子库、通信库、编排、KV 存储、量化与评测工具，面试时能说出各层的代表项目。
