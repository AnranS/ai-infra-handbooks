# 从零训练一个小模型

六章，从一个空目录开始，在自己的电脑上训出一个会写《三国演义》的小语言模型：准备语料、训练 BPE 分词器、写小 GPT 与完整的训练循环、续训与采样，再用梯度累积、`torch.compile`、混合精度和 DDP 训得更快更大；然后做指令微调（聊天模板、只在助手回答上算 loss）与 LoRA、DPO、白盒蒸馏，并导出成 LLaMA 的权重布局；最后换到一张 16 GB 的消费级卡，算显存账本、训一晚上的文本模型和一个图像生成小模型。

## 快速开始

主线五个脚本，CPU 上约 6 分钟：`prepare.py` → `train.py` → `chat_data.py` → `sft.py` → `export.py`，然后 `python talk.py` 跟它说话。
其余脚本是各章的实验（测速、梯度累积、DDP、规模、LoRA、DPO、蒸馏），不产出主线要用的文件。各脚本的耗时与产出见[首页的脚本清单](docs/index.md)。

## 目录

- `docs/`：正文六章（`data.md`、`model.md`、`scale.md`、`sft.md`、`align.md`、`one-gpu.md`）与首页
- `docs/assets/data/sanguo.txt`：语料（《三国演义》，Project Gutenberg #23950，公有领域，已转成简体并整理排版）
- `docs-en/`：英文译文（`tools/i18n.py` 构建到站点的 `en/scratch/`）
- `tools/check_code.py`：运行正文里所有标了文件名的脚本，确认页面上的输出与实际运行一致

## 校验

```bash
# 需要 CPU 版 PyTorch（2.4 以上）、numpy 与 tokenizers；默认使用 ../cpp/.venv-py（见 C++ 手册的 README），也可以用环境变量 PYTHON 指定解释器
python3 tools/check_code.py                         # 全部六章
python3 tools/check_code.py docs/model.md           # 只到某一章为止（仍从第一章开始接力）
```

约定和分布式训练手册一样（规则直接复用 `../train/tools/check_code.py`）：

- ```` ```python title="x.py" ```` 是完整脚本，紧跟的 ```` ```text title="输出" ```` 必须与标准输出逐行一致；
- `torchrun="N"` 的脚本用 `torchrun --standalone --nproc-per-node N` 启动，通信后端是 gloo；
- `run="no"` 的脚本需要 GPU，只做语法检查（最后一章整章如此，页面上的数字标明了测量它的卡）；
- 六章共用一个工作目录 `build/examples/tutorial/`，按 `data.md → model.md → scale.md → sft.md → align.md → one-gpu.md` 的顺序接力（语料、分词器、token 文件、checkpoint 留给后一章），检查其中任何一章都会从第一章跑起（约 20 分钟）。

构建站点：在仓库根目录运行 `./build.sh`，或单独预览 `mkdocs serve -f scratch/mkdocs.yml`。
