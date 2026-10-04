# 大模型原理手册

推理优化的前置知识：从语言模型、Transformer 到推理服务。MkDocs Material 站点，源文件在 `docs/`。用到的数学（线性代数与低秩、概率与采样、信息论、反向传播、浮点误差、屋顶线与排队论）在单独的[数学基础手册](../math/)里，正文用到时链接过去。

- 构建：在仓库根目录运行 `.venv/bin/mkdocs build --strict -f llm/mkdocs.yml`（依赖见 `requirements-docs.txt`）
- 校验全部代码：`.venv-llm/bin/python tools/check_code.py`（需要 CPU 版 PyTorch、transformers 与 `models/Qwen3-0.6B`；全部通过后会打包 `docs/assets/llm-code.tar.gz`）
- 数学基础手册的例子也用这套环境：在仓库根目录运行 `llm/.venv-llm/bin/python math/tools/check_code.py`
- 全部手册合并构建：在仓库根目录运行 `./build.sh`
