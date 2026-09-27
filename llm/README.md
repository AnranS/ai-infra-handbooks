# 大模型原理手册

推理优化的前置知识：从语言模型、Transformer 到推理服务。MkDocs Material 站点，源文件在 `docs/`。

- 构建：`.venv/bin/mkdocs build --strict`（依赖见 `requirements-docs.txt`）
- 校验全部代码：`.venv-llm/bin/python tools/check_code.py`（需要 CPU 版 PyTorch、transformers 与 `models/Qwen2.5-0.5B-Instruct`；全部通过后会打包 `docs/assets/llm-code.tar.gz`）
- 三本手册合并构建：在仓库根目录运行 `./build.sh`
