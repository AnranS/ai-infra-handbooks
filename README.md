# AI Infra 学习手册

四本互相衔接的中文学习手册，面向大模型推理岗位（推理框架、推理优化、推理平台）。每本都是一个 MkDocs Material 站点，示例代码都经过自动验证。

| 手册 | 目录 | 内容 | 验证方式 |
| --- | --- | --- | --- |
| **Python 进阶手册** | [`python/`](python/) | 对象模型、迭代器与生成器、装饰器、类型标注与协议、元编程、工程化与测试、并发与性能（23 页） | 所有 `python` 代码块在 Python 3.14 上运行，`>>>` 示例用 doctest 逐字核对，测试章节的示例用 pytest 实际运行 |
| **大模型原理手册** | [`llm/`](llm/) | 语言模型、分词、Transformer 各组件、从零实现 LLaMA 结构、GQA/MLA、MoE、训练与对齐、采样、KV Cache、估算、量化、推理服务（23 页） | 代码在 CPU 版 PyTorch 上实际运行；自己实现的模型加载真实的 Qwen2.5-0.5B 权重，与 Hugging Face 官方实现逐项对比；知名模型的参数量用官方配置实际构建模型核对 |
| **CUDA 进阶手册** | [`cuda/`](cuda/) | GPU 架构、内存与执行模型、经典算子（归约、转置、GEMM、Softmax/归一化、scan）、Tensor Core、Hopper、FlashAttention、量化 GEMV、Nsight、CUDA Graphs、NCCL、Triton、面试题（23 页） | 每个 `.cu` 用 nvcc 12.9 与 13.4 编译检查；kernel 在自制的 CPU 模拟器上执行自检；Triton 示例在解释器模式下运行 |
| **推理系统手册** | [`serving/`](serving/) | 一个请求的一生；从零写推理引擎（分页 KV、变长批处理、调度器、前缀缓存、采样与流式 API、CUDA Graphs）；vLLM V1 与 SGLang 源码导读；张量/专家/流水线/上下文并行、PD 分离、KV 分层缓存；压测与容量规划、Profiling、量化部署；投机解码、长上下文、结构化输出、多模态、RL 中的推理；面试题库、手撕代码、系统设计、作品集（28 页） | 迷你引擎的输出与逐个生成逐 token 比较；TP/EP/PP/PD 用 torch.distributed 多进程在 CPU 上与单进程核对；源码导读基于 vLLM 0.30.0 与 SGLang 0.5.20 核对 |

推荐顺序：**Python → 大模型原理 → CUDA → 推理系统**，推理系统手册的[作品集与学习计划](serving/docs/career/projects.md)一章给出了 12 周的具体安排；[ROADMAP.md](ROADMAP.md) 是一页纸的推理引擎学习路线摘要。手册之间有交叉链接（比如大模型手册讲到 FlashAttention 时，会链接到 CUDA 手册中对应的 kernel 实现）。

## 在线阅读

<https://anrans.github.io/ai-infra-handbooks/>

仓库配置了 GitHub Actions：推送到 `main` 后会自动构建四本手册，并部署到 GitHub Pages（Settings → Pages 中的 Source 需要设为 “GitHub Actions”）。

## 本地构建

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-docs.txt
MKDOCS=.venv/bin/mkdocs ./build.sh                  # 输出到 _site/：总入口页 + python/、llm/、cuda/、serving/
python3 -m http.server 8000 --directory _site         # 打开 http://localhost:8000
```

只写某一本时，可以用 `mkdocs serve` 实时预览，比如 `.venv/bin/mkdocs serve -f llm/mkdocs.yml`。单独预览时，跨手册的链接会指向上一级目录中的其他手册，只有在 `_site/` 中才能跳转成功。

## 目录结构

```text
.
├── python/ llm/ cuda/ serving/   四本手册：各自的 mkdocs.yml、docs/（正文）、tools/（代码校验脚本）、README.md
├── portal/                  总入口页
├── theme/                   三本手册共用的 MkDocs Material 主题覆盖：顶栏与手册切换、页面样式
├── hooks/crosslinks.py      把 cuda://、llm://、python:// 形式的跨手册链接改写成相对链接
├── build.sh                 构建全部手册到 _site/
├── ROADMAP.md               推理引擎（vLLM / SGLang）学习路线
└── .github/workflows/       GitHub Pages 部署
```

## 校验示例代码

各手册的 `tools/` 下是校验脚本，需要各自的运行环境（Python 3.14、CPU 版 PyTorch 与 Qwen2.5-0.5B / Qwen2.5-VL-3B 权重、CUDA 工具链等），这些环境和模型文件不在仓库中。具体见各手册的 README。
