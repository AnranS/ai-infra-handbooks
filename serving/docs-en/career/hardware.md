# Hardware and ecosystem cheat sheet

<p class="lead">Every estimate in inference optimization starts from hardware parameters: memory capacity decides what fits, bandwidth decides how fast decode goes, compute decides how fast prefill goes, and interconnect decides how you can parallelize. This page collects the key parameters of common inference hardware, the inference characteristics they imply, and a map of the inference software ecosystem, for reference when estimating and in interviews.</p>

!!! warning "About the numbers"
    The table lists the **dense** (non-sparse) peaks commonly found in vendors' public materials; different form factors (SXM, PCIe, NVL) and different production batches may differ. For a formal evaluation, go by the official datasheets and your own measurements.

## GPU parameters {#gpu-参数}

| Model | Memory | Bandwidth | BF16 dense | FP8 dense | Interconnect (per GPU) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| A100 SXM 80GB | 80 GB HBM2e | 2.0 TB/s | 312 TFLOPS | not supported | NVLink 600 GB/s | no FP8 Tensor Cores, so quantization mostly uses W8A16/W4A16 |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | 1979 TFLOPS | NVLink 900 GB/s | the default hardware for this book's estimates |
| H200 | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 1979 TFLOPS | NVLink 900 GB/s | the same compute as H100 with more memory and bandwidth, so decode is faster |
| H20 | 96 GB HBM3 | 4.0 TB/s | 148 TFLOPS | 296 TFLOPS | NVLink 900 GB/s | high bandwidth, low compute; suits decode, not prefill |
| L40S | 48 GB GDDR6 | 0.86 TB/s | 362 TFLOPS | 733 TFLOPS | PCIe only | a cost-effective inference card with low bandwidth |
| B200 | about 180–192 GB HBM3e | 8 TB/s | about 2250 TFLOPS | about 4500 TFLOPS | NVLink 1.8 TB/s | native FP4 (NVFP4/MXFP4) |
| MI300X | 192 GB HBM3 | 5.3 TB/s | 1307 TFLOPS | 2615 TFLOPS | Infinity Fabric | lots of memory, the ROCm ecosystem |

Also worth knowing: GB200/GB300 NVL72 puts 72 Blackwell GPUs in one NVLink domain, so large-scale EP can run entirely over NVLink; domestic accelerators (such as Huawei Ascend) have their own software stacks (CANN, MindIE), and both vLLM and SGLang have corresponding hardware backends (the vLLM-Ascend plugin, Ascend support in SGLang), which comes up often in roles in China.

## Inference characteristics implied by the parameters {#由参数推出的推理特性}

```python
gpus = {  # name: (memory GB, bandwidth TB/s, BF16 TFLOPS, FP8 TFLOPS or None)
    "A100": (80, 2.0, 312, None), "H100": (80, 3.35, 989, 1979), "H200": (141, 4.8, 989, 1979),
    "H20": (96, 4.0, 148, 296), "L40S": (48, 0.86, 362, 733), "B200": (180, 8.0, 2250, 4500),
    "MI300X": (192, 5.3, 1307, 2615),
}
params = 8.03e9                                                   # LLaMA-3-8B, 16 GB of BF16 weights
kv_per_token = 131072                                             # 128 KB/token (BF16)
print("型号     屋脊点(FLOP/B)  decode 下限(ms)  prefill 4K(ms)  可放下的 KV(token)")
for name, (mem, bw, bf16, fp8) in gpus.items():
    ridge = bf16 / bw                                             # TFLOPS / (TB/s) = FLOP/byte
    decode = params * 2 / (bw * 1e12) * 1e3                       # batch=1: read the weights once
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

How to read this table:

- **The lower the ridge point**, the easier it is to become compute-bound and the better the deal decode is: H20's is only 37, meaning a fairly small batch saturates its compute; its bandwidth is even higher than H100's, so decode is faster than on an H100 while prefill is several times slower. Such a card suits the decode instances of PD disaggregation best;
- **The decode floor** depends only on bandwidth: H200 is 40% faster than H100, and B200 more than twice as fast;
- **Prefill** depends only on compute: H20 and A100 are clearly slower, and FP8 doubles the speed again on cards that support it;
- **KV capacity** decides concurrency and long contexts: the large-memory H200, MI300X and B200 hold several times more requests.

## Interconnect {#互连}

| Interconnect | Bandwidth (one way, per GPU or per port) | Use |
| --- | --- | --- |
| NVLink 4 (H100) | 450 GB/s (900 both ways) | TP and EP within a machine |
| NVLink 5 (B200) | 900 GB/s (1.8 TB/s both ways) | within a machine, and within an NVL72 rack |
| PCIe 5.0 x16 | about 64 GB/s | between CPU and GPU (KV offloading, loading weights) |
| InfiniBand NDR 400 Gb/s | about 50 GB/s | cross-machine EP, KV transfer in PD disaggregation, PP |
| RoCE 400/800 Gb/s | about 50/100 GB/s | RDMA over Ethernet, common in the cloud |

Bandwidth within a machine and between machines differs by nearly an order of magnitude, which is the fundamental reason TP stays within a machine while PP/EP/DP go across machines (see [tensor parallelism](../distributed/tensor-parallel.md#通信的代价)).

## A map of the software ecosystem {#软件生态地图}

| Category | Representative projects | Notes |
| --- | --- | --- |
| Inference engines | vLLM, SGLang, TensorRT-LLM, LMDeploy | the first two have the most active open-source communities; TensorRT-LLM is NVIDIA's official framework, strong on performance and lower on flexibility |
| On-device and local | llama.cpp, Ollama, MLC-LLM | CPU/Apple Silicon/mobile, with the GGUF quantization format |
| Attention and operator libraries | FlashAttention, FlashInfer, FlashMLA, CUTLASS, DeepGEMM | the kernels underneath inference engines |
| Kernel programming | Triton, TileLang, CuTe DSL | writing high-performance kernels in higher-level languages |
| Communication | NCCL, DeepEP, NIXL, Mooncake Transfer Engine | collectives, EP's all-to-all, KV transfer |
| Orchestration and scheduling | NVIDIA Dynamo, llm-d, SGLang Model Gateway, the vLLM production stack | multi-instance routing, PD disaggregation, KV-aware scheduling |
| KV storage | Mooncake Store, LMCache, 3FS | distributed KV caching and offloading |
| Quantization tools | LLM Compressor, NVIDIA ModelOpt, GPTQModel, AutoAWQ | producing quantized checkpoints the engines can load |
| Evaluation | lm-evaluation-harness, OpenCompass, each engine's own benchmarks | accuracy and performance evaluation |
| RL frameworks | verl, OpenRLHF, slime, AReaL | combinations of training and inference engines |

## Summary {#小结}

- [x] Decode depends on bandwidth, prefill on compute, concurrency and long contexts on memory, and parallelism on interconnect.
- [x] Cards with a low ridge point (such as H20) suit decode and cards with strong compute suit prefill, which is the basis for heterogeneous PD disaggregation.
- [x] Bandwidth within a machine (NVLink) and between machines (RDMA) differs by nearly an order of magnitude, which sets the boundaries of parallel strategies.
- [x] The ecosystem map: engines, operator libraries, communication libraries, orchestration, KV storage, quantization and evaluation tools, with representative projects per layer to name in an interview.
