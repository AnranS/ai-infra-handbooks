# 手写 mini-sglang

<p class="lead">mini-sglang 是 SGLang 团队写的"教学版 SGLang"：核心约 5000 行 Python，却包含了一个现代推理引擎的全部要素——Radix Cache、分块 prefill、重叠调度、张量并行、FlashInfer / FlashAttention、CUDA Graph，以及 OpenAI 兼容的在线服务。这本手册带你按官方的模块划分，从一个空目录开始，一章实现一个模块，最后得到一个完整可用、接口与官方一致的 mini-sglang。每一步都有测试，输出与 Hugging Face transformers 逐 token 对齐。</p>

## 学完能做到

- 说清 mini-sglang 的进程结构：API Server、tokenizer、detokenizer、每个 TP rank 一个调度器进程，以及它们之间用 ZMQ 传递的每一种消息；
- 独立实现引擎的计算侧：不依赖 `nn.Module` 的算子与权重体系、流式权重加载、分页 KV 池、page table 与 token pool、可插拔的注意力后端、采样器；
- 独立实现调度侧：连续批处理、准入控制、页分配与回收、Radix Cache（匹配、插入、分裂、LRU 淘汰、引用计数）、分块 prefill、重叠调度；
- 讲清并亲手验证三个"让它变快"的机制：张量并行的四种线性层切法、FlashInfer 的 plan/run 与 FlashAttention 的分页接口、CUDA Graph 的固定缓冲区约定；
- 读懂官方仓库的每一个文件，并能讲清复刻时发现并修正的五个问题：为什么会出现、怎样复现、怎样修。

## 学习路线

八本手册合在一起的逐章路线见[学习路线图](root://roadmap/)。这本书建议放在[推理系统手册](serving://)的"从零写一个推理引擎"之后：那里的 nano_engine 帮你建立概念，这里对照一个真实、完整、性能达到正式版水平的代码库，把每个细节落实到代码。

<div class="roadmap" markdown>

| 部分 | 章节 | 目标 | 建议用时 |
| --- | --- | --- | --- |
| 全景 | [导读](overview/architecture.md) | 看懂进程、数据流和模块依赖，搭好环境 | 半天 |
| 算得对 | [核心数据结构](compute/core.md) · [算子层](compute/layers.md) · [模型与权重](compute/models.md) · [KV 池](compute/kvcache.md) · [注意力后端](compute/attention.md) · [Engine 与采样](compute/engine.md) | 手工组 batch 跑通 prefill + decode，与 HF 逐 token 一致 | 4～5 天 |
| 排得好 | [调度器骨架](schedule/scheduler.md) · [CacheManager](schedule/cache-manager.md) · [Radix Cache](schedule/radix-cache.md) · [分块 prefill](schedule/chunked-prefill.md) · [重叠调度](schedule/overlap.md) | 多请求连续批处理，前缀复用，CPU 开销被藏起来 | 1 周 |
| 服务化 | [消息与 ZMQ](serve/message.md) · [Tokenizer](serve/tokenizer.md) · [调度器收发](serve/scheduler-io.md) · [API Server](serve/api-server.md) | 多进程的 OpenAI 兼容服务，支持流式与断连中止 | 3～4 天 |
| 更快、更大 | [张量并行](perf/tensor-parallel.md) · [GPU 注意力](perf/gpu-attention.md) · [CUDA Graph](perf/cuda-graph.md) · [CUDA kernel](perf/kernels.md) · [MoE](perf/moe.md) · [基准测试](perf/benchmark.md) | 多卡、GPU kernel、MoE 模型 | 1 周 |
| 收尾 | [与 SGLang 的差距](wrap/next-steps.md) | 知道还缺什么，选一个方向继续做 | 按需 |
| 大作业 | [GPU 性能门槛](wrap/assignment.md) · [接入混合架构模型 Qwen3.5](wrap/assignment-hybrid.md) | 上 GPU 跑到正式版的 60%；给引擎加上"每个请求带状态"的线性注意力层 | 各 1～2 周 |

</div>

## 这本书的做法

**同名同接口。** 我们的包也叫 `minisgl`，目录结构、类名、函数名、方法签名都与官方一致（官方仓库固定在 2026-05-17 的提交 `9a91cfa`）。每章开头列出本章要写的文件，正文里的 !!! upstream 提示框给出官方对应代码的精确位置（带行号的链接），读完一章可以直接对照官方代码逐行比较。

**CPU 上也能验证每一步。** 官方实现只支持 NVIDIA GPU。为了让任何一台机器都能跟着做、并且每一步都能验证，我们做了三件事：

1. 加了一层很薄的设备抽象（`minisgl/utils/device.py`）：CUDA 上是真的 stream、event 和锁页内存，CPU 上退化成空操作；
2. 每个 GPU 算子都有一个 PyTorch 参考实现：注意力后端多一个 `torch`，采样、RMSNorm、RoPE、写 KV 缓存等都有 CPU 版本；
3. 只能在 GPU 上运行的部分，用"同接口的替身"在 CPU 上验证逻辑：FlashInfer 和 FlashAttention 各有一个同接口的 PyTorch 假实现，CUDA Graph 有一个 CPU 仿真（它强制所有输入都走固定缓冲区，漏拷一个输入结果就会出错），CUDA kernel 在[CUDA 手册](cuda://)的 CPU 模拟器上运行，Triton kernel 用解释器模式运行。

在真正的 GPU 上，同一份代码会自动切换到 FlashInfer、FlashAttention、CUDA Graph 和自定义 kernel。

**每一步都与 Hugging Face 对齐。** 除了单元测试，每章都有端到端的检查：用 float32 在 Qwen3-0.6B 上跑，贪心解码的每个 token 都与 `transformers` 的 `generate` 相同。到最后，同一套检查覆盖了 Radix Cache、分块 prefill、重叠调度、page size 大于 1、张量并行（TP=2、TP=4）、三种注意力后端、CUDA Graph，以及 Qwen2.5、Llama 3（长上下文 RoPE）、Qwen3-MoE 三种结构。

!!! diff "我们与官方不同的地方"
    除了上面的 CPU 支持，还有几处差异，都在对应章节的"与官方的差异"提示框里说明。其中五处是对官方问题的修正，每处都有测试（改回官方写法测试就会失败）：重叠调度下 finished 标记提前一个 token、EOS 之后多发一条过期消息、请求槽在 batch 仍在 GPU 上时被复用、prefill 在途时 abort 导致 KV 页被重复释放（第 11 章），以及 PUB/SUB 广播没有等待订阅者（第 12 章）。其余是简化：模型定义合并成一个通用的 decoder 文件；张量并行只用 `torch.distributed`，没有实现官方基于 tvm-ffi 的 PyNCCL；fused MoE 的 Triton kernel 是简化版。

## 代码与运行

全部代码在仓库的 [`minisgl/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/minisgl) 目录：

```text
minisgl/
├── python/minisgl/      我们的实现，与官方 python/minisgl/ 一一对应
├── tests/               每章一个测试文件（test_ch07_scheduler.py ……），外加 GPU 库的假实现和 CUDA 自检程序
├── examples/            正文里每段"运行结果"对应的脚本
├── tools/check.py       一条命令跑完所有验证：示例、nvcc 编译、CPU 模拟器、pytest
├── tools/diagrams.py    生成书中的 15 张架构图（SVG，随明暗主题变色）
├── tools/videos/        6 段教学动画的场景脚本与渲染器（配音 edge-tts，逐帧渲染后 ffmpeg 编码）
└── docs/                本手册
```

```bash
cd minisgl
pip install -e ".[dev]"                                   # 依赖：torch、transformers、pyzmq、fastapi 等
python -m minisgl --model Qwen/Qwen3-0.6B                 # 起一个 OpenAI 兼容服务（GPU 或 CPU 都行）
python -m minisgl --model Qwen/Qwen3-0.6B --shell        # 交互式对话
PYTHONPATH=python:tests pytest -q tests                   # 全部测试（需要下载 Qwen3-0.6B 到 models/）
```

每章的"本章要写的文件"就是你要亲手敲的内容。建议的节奏：先读原理、自己写一版，再对照本书的实现和官方实现，最后跑本章的测试。答不出章首自测题的，回到[推理系统手册](serving://)对应的概念章节补一补。

书里有 15 张架构图和 6 段带配音、字幕的动画（每段 1.5～2 分钟），适合在读一章之前先看一遍建立直觉：[一个请求的一生](overview/architecture.md)、[连续批处理与准入控制](schedule/scheduler.md)、[Radix Cache](schedule/radix-cache.md)、[重叠调度](schedule/overlap.md)、[张量并行](perf/tensor-parallel.md)、[CUDA Graph](perf/cuda-graph.md)。
