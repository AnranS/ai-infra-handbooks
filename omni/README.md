# SGLang-Omni 源码导读

十二章读懂 [SGLang-Omni](https://github.com/sgl-project/sglang-omni)——SGLang 社区为全模态、TTS、ASR 模型写的多阶段推理运行时：请求怎么流过 Coordinator 和 Stage、`OmniScheduler` 怎么组合 SGLang 的调度器、stage 之间怎么按边选传输、配置与部署怎么规划；精读 Qwen3-TTS、Qwen3-Omni 和 Rust router；最后是测试、CI 与"从读代码到提 MR"。

## 目录

- `docs/`：正文十二章与首页
- `tools/check_code.py`：重跑正文里所有的实验、按提交号重新截取所有引用的源码，与页面逐行比对
- `tools/setup_env.sh`：准备环境（克隆 sglang-omni 与 SGLang、建 Python 环境、装 Rust 工具链）
- `requirements-check.txt`：本书 Python 环境的依赖（CPU 版 PyTorch 和 `--no-deps` 的 SGLang 另装，见首页）
- `examples/`：正文里的代码导出成的真实文件（`tools/export_examples.py` 生成）

## 校验

```bash
bash tools/setup_env.sh                  # 第一次：克隆两个仓库、建 .venv-omni、装 Rust
python3 tools/check_code.py              # 全部页面，约 10 分钟（第一次会构建 router）
python3 tools/check_code.py docs/journey/stage.md
```

全书基于 sglang-omni 2026-10-10 的提交 `921ea2c8`（`REF` 可以换）。实验不需要 GPU：omni 的 Coordinator、Stage、ZMQ 控制面、SHM 数据面和配置规划都不依赖 GPU，书里用 CPU 版 PyTorch 和 SGLang 的 Python 部分就能起真实的多进程流水线。在 Linux x86-64 上验证过；macOS 没有验证。

构建站点：在仓库根目录运行 `./build.sh`，或单独预览 `mkdocs serve -f omni/mkdocs.yml`。
