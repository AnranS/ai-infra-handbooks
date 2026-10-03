# 数学基础手册

读大模型和推理系统时用到的数学，单独成一本、用到再查：线性代数与低秩、概率与采样、信息论、反向传播与二阶信息、浮点误差、屋顶线与排队论。MkDocs Material 站点，源文件在 `docs/`。

- 构建：`.venv/bin/mkdocs build --strict -f math/mkdocs.yml`（依赖见仓库根目录的 `requirements-docs.txt`）
- 校验全部代码：`llm/.venv-llm/bin/python math/tools/check_code.py`。例子用大模型原理手册的环境（`mini_llm`、`llm/models/Qwen3-0.6B`、冻结的样本文本），由 `llm/tools/check_code.py` 在 `llm/` 目录下运行，输出与页面逐行比对
- 示意图由 `tools/figures.py` 生成（`@figure("math", ...)`）
- 全部手册合并构建：在仓库根目录运行 `./build.sh`
