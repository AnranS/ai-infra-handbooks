# AI Infra 学习手册

五本互相衔接的中文学习手册 + 125 道配套练习题，面向大模型推理岗位（推理框架、推理优化、推理平台）。每本手册都是一个 MkDocs Material 站点，示例代码都经过自动验证。

| 手册 | 目录 | 内容 | 验证方式 |
| --- | --- | --- | --- |
| **Python 进阶手册** | [`python/`](python/) | 对象模型、迭代器与生成器、装饰器、类型标注与协议、元编程、工程化与测试、并发与性能（23 页） | 所有 `python` 代码块在 Python 3.14 上运行，`>>>` 示例用 doctest 逐字核对，测试章节的示例用 pytest 实际运行 |
| **大模型原理手册** | [`llm/`](llm/) | 语言模型、分词；数学基础（线性代数与低秩、概率与采样、信息论与 KL、反向传播与 GPTQ、浮点误差、屋顶线与排队论）；Transformer 各组件、从零实现 LLaMA 结构、GQA/MLA、MoE、训练与对齐、采样、KV Cache、估算、量化、推理服务（29 页） | 代码在 CPU 版 PyTorch 上实际运行；数学章节的结论都在真实模型上测量（SVD 能量、量化后的 KL、draft 接受率、GPTQ 与 RTN 对比等）；自己实现的模型加载真实的 Qwen2.5-0.5B 权重，与 Hugging Face 官方实现逐项对比；知名模型的参数量用官方配置实际构建模型核对 |
| **CUDA 进阶手册** | [`cuda/`](cuda/) | GPU 架构、内存与执行模型、经典算子（归约、转置、GEMM、Softmax/归一化、scan）、Tensor Core、Hopper、FlashAttention、量化 GEMV、Nsight、CUDA Graphs、NCCL、Triton、面试题（23 页） | 每个 `.cu` 用 nvcc 12.9 与 13.4 编译检查；kernel 在自制的 CPU 模拟器上执行自检；Triton 示例在解释器模式下运行 |
| **推理系统手册** | [`serving/`](serving/) | 一个请求的一生；从零写推理引擎（分页 KV、变长批处理、调度器、前缀缓存、采样与流式 API、CUDA Graphs）；vLLM V1 与 SGLang 源码导读；张量/专家/流水线/上下文并行、PD 分离、KV 分层缓存；压测与容量规划、Profiling、量化部署；投机解码、长上下文、结构化输出、多模态、RL 中的推理；面试题库、手撕代码、系统设计、作品集（28 页） | 迷你引擎的输出与逐个生成逐 token 比较；TP/EP/PP/PD 用 torch.distributed 多进程在 CPU 上与单进程核对；源码导读基于 vLLM 0.30.0 与 SGLang 0.5.20 核对 |
| **手写 mini-sglang** | [`minisgl/`](minisgl/) | 按官方 mini-sglang 的模块划分，从零实现完整的推理引擎：核心数据结构、BaseOP 算子体系、流式权重加载、分页 KV 池与 page table、注意力后端；调度器、CacheManager 与准入控制、Radix Cache、分块 prefill、重叠调度；消息与 ZMQ、增量反分词、多 rank 同步、OpenAI 兼容服务；张量并行、FlashInfer / FlashAttention、CUDA Graph、自定义 CUDA kernel、fused MoE、基准测试（24 页） | 代码在 `minisgl/python/minisgl/`，与官方同名同接口；63 个 pytest 测试，贪心输出与 HF transformers 逐 token 对齐（覆盖 Radix、分块、重叠调度、TP=2/4、三种注意力后端、CUDA Graph 仿真、Qwen2.5/Llama3/Qwen3-MoE）；GPU 库用同接口的假实现验证，CUDA kernel 用 nvcc 编译并在 CPU 模拟器上自检，Triton 用解释器模式运行 |

推荐顺序：**Python → 大模型原理 → CUDA → 推理系统 → 手写 mini-sglang**。网站上的[学习路线图](https://anrans.github.io/ai-infra-handbooks/roadmap/)把 122 章排成 12 周，标出每章是必学还是选学、不同岗位方向的重点和跨书的知识依赖，还能记录进度；推理系统手册的[作品集与学习计划](serving/docs/career/projects.md)一章给出了 12 周的具体安排；[ROADMAP.md](ROADMAP.md) 是一页纸的推理引擎学习路线摘要。手册之间有交叉链接（比如大模型手册讲到 FlashAttention 时，会链接到 CUDA 手册中对应的 kernel 实现）。

除了五本手册，站点上还有：

- **[练习题](https://anrans.github.io/ai-infra-handbooks/practice/)**（[`practice/`](practice/)）：每章配套的编程题，在浏览器里写代码、一键判题（Pyodide）；CUDA 题用 GPU 模拟器检查越界、竞争、合并访存和 bank conflict，也可以在 macOS 与 WSL2 + NVIDIA GPU 上本地判题，见 [practice/README.md](practice/README.md)。
- **[17 周冲刺计划](https://anrans.github.io/ai-infra-handbooks/plan/)**（[`portal/plan/`](portal/plan/)）：面向推理系统岗的求职计划，逐周对应到章节、练习题和验收清单，打卡记录保存在浏览器里。
- **[全站搜索](https://anrans.github.io/ai-infra-handbooks/search/)**：在五本手册、练习题和学习路线里一起搜索。

## 在线阅读

<https://anrans.github.io/ai-infra-handbooks/>

仓库配置了 GitHub Actions：推送到 `main` 后会自动构建五本手册、练习题和全站搜索索引，并部署到 GitHub Pages（Settings → Pages 中的 Source 需要设为 “GitHub Actions”）。

## 本地构建

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-docs.txt
MKDOCS=.venv/bin/mkdocs ./build.sh                  # 输出到 _site/：总入口页、学习路线图、冲刺计划、练习题、全站搜索 + 五本手册
python3 -m http.server 8000 --directory _site         # 打开 http://localhost:8000
```

只写某一本时，可以用 `mkdocs serve` 实时预览，比如 `.venv/bin/mkdocs serve -f llm/mkdocs.yml`。单独预览时，跨手册的链接会指向上一级目录中的其他手册，只有在 `_site/` 中才能跳转成功。

## 目录结构

```text
.
├── python/ llm/ cuda/ serving/ minisgl/   五本手册：各自的 mkdocs.yml、docs/（正文）、tools/（代码校验脚本）、README.md
├── portal/                  总入口页、学习路线图（roadmap/）、冲刺计划（plan/）、全站搜索（search/）
├── practice/                练习题：题目（problems/）、浏览器判题（app/、runtime/）、本地判题（judge.py）
├── theme/                   五本手册共用的 MkDocs Material 主题覆盖：顶栏与手册切换、页面样式
├── hooks/                   crosslinks.py 改写 cuda://、llm:// 等跨手册链接；practice.py 在每章末尾列出本章练习题
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
