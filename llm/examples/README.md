<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《大模型原理手册》正文里的代码

这里的 7 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `bpe.py` | [basics/tokenization.md](https://anrans.github.io/ai-infra-handbooks/llm/basics/tokenization) |  |
| `estimate.py` | [inference/estimation.md](https://anrans.github.io/ai-infra-handbooks/llm/inference/estimation) |  |
| `mini_llm.py` | [transformer/build-llm.md](https://anrans.github.io/ai-infra-handbooks/llm/transformer/build-llm) |  |
| `moe.py` | [transformer/moe.md](https://anrans.github.io/ai-infra-handbooks/llm/transformer/moe) |  |
| `quant.py` | [inference/quantization.md](https://anrans.github.io/ai-infra-handbooks/llm/inference/quantization) |  |
| `rope.py` | [transformer/position.md](https://anrans.github.io/ai-infra-handbooks/llm/transformer/position) |  |
| `sampling.py` | [inference/decoding.md](https://anrans.github.io/ai-infra-handbooks/llm/inference/decoding) |  |
