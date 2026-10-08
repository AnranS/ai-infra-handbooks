<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《从零训练一个小模型》正文里的代码

这里的 27 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `bench_gpu.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) | 需要 GPU，手册里只做语法检查 |
| `budget.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) | 覆盖了 scale.md 里的同名文件 |
| `chat.py` | [sft.md](https://anrans.github.io/ai-infra-handbooks/scratch/sft) |  |
| `chat_data.py` | [sft.md](https://anrans.github.io/ai-infra-handbooks/scratch/sft) |  |
| `ddp_train.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/scratch/scale) |  |
| `distill.py` | [align.md](https://anrans.github.io/ai-infra-handbooks/scratch/align) |  |
| `dpo.py` | [align.md](https://anrans.github.io/ai-infra-handbooks/scratch/align) |  |
| `export.py` | [align.md](https://anrans.github.io/ai-infra-handbooks/scratch/align) |  |
| `flow2d.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) |  |
| `grad_accum.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/scratch/scale) |  |
| `lora.py` | [align.md](https://anrans.github.io/ai-infra-handbooks/scratch/align) |  |
| `model.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/scratch/model) |  |
| `prepare.py` | [data.md](https://anrans.github.io/ai-infra-handbooks/scratch/data) |  |
| `resume.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/scratch/model) |  |
| `sample_image.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) | 需要 GPU，手册里只做语法检查 |
| `sampling.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/scratch/model) |  |
| `scaling.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/scratch/scale) |  |
| `serve.py` | [align.md](https://anrans.github.io/ai-infra-handbooks/scratch/align) | 需要 GPU，手册里只做语法检查 |
| `sft.py` | [sft.md](https://anrans.github.io/ai-infra-handbooks/scratch/sft) |  |
| `sizes.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/scratch/model) |  |
| `speed.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/scratch/scale) |  |
| `talk.py` | [sft.md](https://anrans.github.io/ai-infra-handbooks/scratch/sft) | 需要 GPU，手册里只做语法检查 |
| `train.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/scratch/model) |  |
| `train_gpu.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/scratch/scale) | 需要 GPU，手册里只做语法检查 |
| `train_image.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) | 需要 GPU，手册里只做语法检查 |
| `train_text.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) | 需要 GPU，手册里只做语法检查 |
| `unet.py` | [one-gpu.md](https://anrans.github.io/ai-infra-handbooks/scratch/one-gpu) | 需要 GPU，手册里只做语法检查 |
