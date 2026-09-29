# AI Infra 学习手册

七本互相衔接的中文学习手册 + 148 道配套练习题，面向大模型推理岗位（推理框架、推理优化、推理平台）。每本手册都是一个 MkDocs Material 站点，示例代码都经过自动验证。

| 手册 | 目录 | 内容 | 验证方式 |
| --- | --- | --- | --- |
| **Python 进阶手册** | [`python/`](python/) | 对象模型、迭代器与生成器、装饰器、类型标注与协议、元编程、工程化与测试、并发与性能（23 页） | 所有 `python` 代码块在 Python 3.14 上运行，`>>>` 示例用 doctest 逐字核对，测试章节的示例用 pytest 实际运行 |
| **C++ 进阶手册** | [`cpp/`](cpp/) | 面向 AI Infra 的现代 C++：编译模型与未定义行为、值语义与 RAII、移动语义、所有权、模板与编译期派发、标准库的性能视角；对象布局与缓存、分配器与内存池（arena、KV 块分配器、缓存分配器）；线程与条件变量、atomic 与内存序、无锁队列与线程池；CMake 与 sanitizer、pybind11 与 PyTorch C++ 扩展、以 vLLM `csrc/` 为例读懂推理基础库、面试高频题（15 页） | 每个程序用 g++ 12（C++20，`-Wall -Wextra -Werror`）编译，在 ASan + UBSan 下运行，并发章节在 TSan 下运行，输出与页面逐行比对；故意演示的错误必须被 sanitizer 抓到；CMake 工程、pybind11 与 PyTorch 扩展实际构建运行 |
| **大模型原理手册** | [`llm/`](llm/) | 语言模型、分词；数学基础（线性代数与低秩、概率与采样、信息论与 KL、反向传播与 GPTQ、浮点误差、屋顶线与排队论）；Transformer 各组件、从零实现 LLaMA 结构、GQA/MLA、MoE、训练与对齐、采样、KV Cache、估算、量化、推理服务；剪枝、2:4 稀疏与蒸馏；大作业：从零训练一个小语言模型（31 页） | 代码在 CPU 版 PyTorch 上实际运行；数学章节的结论都在真实模型上测量（SVD 能量、量化后的 KL、draft 接受率、GPTQ 与 RTN 对比等）；自己实现的模型加载真实的 Qwen2.5-0.5B 权重，与 Hugging Face 官方实现逐项对比；知名模型的参数量用官方配置实际构建模型核对 |
| **CUDA 进阶手册** | [`cuda/`](cuda/) | GPU 架构、内存与执行模型、经典算子（归约、转置、GEMM、Softmax/归一化、scan）、Tensor Core、Hopper、FlashAttention、量化 GEMV、Nsight、CUDA Graphs、NCCL、Triton；框架与编译器（张量的内存模型、autograd、dispatcher 与自定义算子、PyTorch 的 CUDA 运行时、torch.compile、AI 编译器全景）；面试题（29 页） | 每个 `.cu` 用 nvcc 12.9 与 13.4 编译检查；kernel 在自制的 CPU 模拟器上执行自检；Triton 示例在解释器模式下运行；框架与编译器几章的 PyTorch 示例用 `tools/check_torch.py` 在 CPU 版 PyTorch 上实跑、输出逐行比对 |
| **分布式训练手册** | [`train/`](train/) | 显存账本与时间模型、集合通信原语；DDP（分桶与通信重叠）、ZeRO 与 FSDP2；Megatron 张量并行与序列并行、流水线并行（GPipe、1F1B、交错与零气泡）、上下文并行（Ring Attention、Ulysses）、MoE 与专家并行（含无辅助损失均衡）；混合精度与 FP8 细粒度缩放、3D / 5D 并行的组合与配置搜索、训练框架、分布式 checkpoint 与 RL 训练系统；大作业：DDP、ZeRO-1 与激活重计算（13 页） | 所有脚本用 `tools/check_code.py` 在 CPU 版 PyTorch 上实跑，输出与页面逐行比对；多进程示例用 `torchrun` + gloo 启动，每种并行都与单进程的前向、损失和梯度逐项对齐 |
| **推理系统手册** | [`serving/`](serving/) | 一个请求的一生；从零写推理引擎（分页 KV、变长批处理、调度器、前缀缓存、采样与流式 API、CUDA Graphs）；vLLM V1 与 SGLang 源码导读；张量/专家/流水线/上下文并行、PD 分离、KV 分层缓存；通信与存储（NVLink 与 RDMA 网络、NCCL 算法与定制 all-reduce、RDMA 编程模型、NVSHMEM 与 DeepEP、KV 传输引擎与分布式 KV 存储）；压测与容量规划、Profiling、量化部署；投机解码、长上下文、结构化输出、多模态、RL 中的推理；前沿专题：大规模 MoE 推理（MLA 与 FlashMLA、FP8 细粒度量化与 DeepGEMM、分层 EPLB 与双 batch 重叠、MTP 与稀疏注意力、公开系统复盘）；前沿专题：分离式架构的全局调度、线性注意力与混合架构、低比特推理与 QAT、异步 RL 与权重同步；生产与生态（部署与运维、框架选型、多 LoRA、端侧推理）；面试题库、手撕代码、系统设计与 10 题参考答案、模拟面试套卷、作品集与作品指南（50 页） | 迷你引擎的输出与逐个生成逐 token 比较；通信与存储篇、两个前沿专题的模型与模拟输出与页面逐行比对；TP/EP/PP/PD 用 torch.distributed 多进程在 CPU 上与单进程核对；源码导读基于 vLLM 0.30.0 与 SGLang 0.5.20 核对 |
| **手写 mini-sglang** | [`minisgl/`](minisgl/) | 按官方 mini-sglang 的模块划分，从零实现完整的推理引擎：核心数据结构、BaseOP 算子体系、流式权重加载、分页 KV 池与 page table、注意力后端；调度器、CacheManager 与准入控制、Radix Cache、分块 prefill、重叠调度；消息与 ZMQ、增量反分词、多 rank 同步、OpenAI 兼容服务；张量并行、FlashInfer / FlashAttention、CUDA Graph、自定义 CUDA kernel、fused MoE、基准测试；大作业：GPU 性能门槛（25 页） | 代码在 `minisgl/python/minisgl/`，与官方同名同接口；63 个 pytest 测试，贪心输出与 HF transformers 逐 token 对齐（覆盖 Radix、分块、重叠调度、TP=2/4、三种注意力后端、CUDA Graph 仿真、Qwen2.5/Llama3/Qwen3-MoE）；GPU 库用同接口的假实现验证，CUDA kernel 用 nvcc 编译并在 CPU 模拟器上自检，Triton 用解释器模式运行 |

推荐顺序：**Python → 大模型原理 → CUDA（同时学 C++）→ 推理系统（分布式推理时对照学分布式训练）→ 手写 mini-sglang**。网站上的[学习路线图](https://anrans.github.io/ai-infra-handbooks/roadmap/)把 181 章排成 12 周，标出每章是必学还是选学、不同岗位方向的重点和跨书的知识依赖，还能记录进度；推理系统手册的[作品集与学习计划](serving/docs/career/projects.md)一章给出了 12 周的具体安排；[ROADMAP.md](ROADMAP.md) 是一页纸的推理引擎学习路线摘要。手册之间有交叉链接（比如大模型手册讲到 FlashAttention 时，会链接到 CUDA 手册中对应的 kernel 实现）。

除了七本手册，站点上还有：

- **[练习题](https://anrans.github.io/ai-infra-handbooks/practice/)**（[`practice/`](practice/)）：每章配套的编程题，在浏览器里写代码、一键判题（Pyodide）；C++ 题在本地用 g++ + sanitizer 判题；CUDA 题用 GPU 模拟器检查越界、竞争、合并访存和 bank conflict，也可以在 macOS 与 WSL2 + NVIDIA GPU 上本地判题，见 [practice/README.md](practice/README.md)。
- **[大作业](assignments/README.md)**：参考 CS336 的做法，只给接口、测试和评分脚本，不给骨架：从零训练一个小语言模型、DDP + ZeRO-1 + 重计算的训练系统、推理引擎的 GPU 性能门槛。
- **[17 周冲刺计划](https://anrans.github.io/ai-infra-handbooks/plan/)**（[`portal/plan/`](portal/plan/)）：面向推理系统岗的求职计划，逐周对应到章节、练习题和验收清单，打卡记录保存在浏览器里。
- **[全站搜索](https://anrans.github.io/ai-infra-handbooks/search/)**：在七本手册、练习题和学习路线里一起搜索。

## 在线阅读

<https://anrans.github.io/ai-infra-handbooks/>

仓库配置了 GitHub Actions：推送到 `main` 后会自动构建七本手册、练习题和全站搜索索引，并部署到 GitHub Pages（Settings → Pages 中的 Source 需要设为 “GitHub Actions”）。

## 本地构建

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-docs.txt
MKDOCS=.venv/bin/mkdocs ./build.sh                  # 输出到 _site/：总入口页、学习路线图、冲刺计划、练习题、全站搜索 + 七本手册
python3 -m http.server 8000 --directory _site         # 打开 http://localhost:8000
```

只写某一本时，可以用 `mkdocs serve` 实时预览，比如 `.venv/bin/mkdocs serve -f llm/mkdocs.yml`。单独预览时，跨手册的链接会指向上一级目录中的其他手册，只有在 `_site/` 中才能跳转成功。

## 目录结构

```text
.
├── python/ cpp/ llm/ cuda/ train/ serving/ minisgl/   七本手册：各自的 mkdocs.yml、docs/（正文）、tools/（代码校验脚本）、README.md
├── assignments/             大作业：只给接口、测试和评分脚本（从零训练小语言模型、训练系统、推理引擎的 GPU 门槛）
├── portal/                  总入口页、学习路线图（roadmap/）、冲刺计划（plan/）、全站搜索（search/）
├── practice/                练习题：题目（problems/）、浏览器判题（app/、runtime/）、本地判题（judge.py）
├── theme/                   七本手册共用的 MkDocs Material 主题覆盖：顶栏与手册切换、页面样式
├── hooks/                   crosslinks.py 改写 cuda://、llm:// 等跨手册链接；practice.py 在每章末尾列出本章练习题；fence_attrs.py 去掉代码块上给校验工具看的属性
├── tools/search_index.py    合并各手册的搜索索引，生成全站搜索用的 search/index.json
├── build.sh                 构建全部手册、练习题和搜索索引到 _site/
├── ROADMAP.md               推理引擎（vLLM / SGLang）学习路线
└── .github/workflows/       GitHub Pages 部署
```

## 校验示例代码

各手册的 `tools/` 下是校验脚本，需要各自的运行环境（Python 3.14、CPU 版 PyTorch 与 Qwen2.5-0.5B / Qwen2.5-VL-3B 权重、CUDA 工具链等），这些环境和模型文件不在仓库中。具体见各手册的 README。

练习题的参考解答和测试用 `python practice/judge.py check` 校验：每道题的参考解答必须通过、初始模板必须不通过。

## 参与贡献

欢迎通过 Issue 反馈错误、提出想看的内容，也欢迎直接提 PR：

- 改正文：改对应手册的 `docs/` 下的 Markdown，提交前用 `./build.sh` 构建一遍（各手册都以 `--strict` 模式构建，坏链接会直接报错）；如果改了示例代码，请跑一下该手册 `tools/` 下的校验脚本。
- 加练习题：在 `practice/problems/<手册>/<题目>/` 下放 `problem.md`、`starter.py`、`solution.py`、`test.py`，再跑 `python practice/judge.py check`，格式见 [practice/README.md](practice/README.md)。
- 讨论内容规划：[冲刺计划页](https://anrans.github.io/ai-infra-handbooks/plan/#build)的「配套内容建设」列出了还在写的部分。

## 许可

文档内容采用 [CC BY-NC-SA 4.0](LICENSE-docs.md)，代码采用 [MIT](LICENSE)；引用的上游代码片段按原项目的许可使用，详见 [LICENSE-docs.md](LICENSE-docs.md)。

## 致谢

站点的视觉风格参考了 [AIInfraGuide](https://caomaolufei.github.io/AIInfraGuide/)；源码导读基于 [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang) 与 [mini-sglang](https://github.com/sgl-project/mini-sglang)。
