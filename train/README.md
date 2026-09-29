# 分布式训练手册

先用三章"从零训练一个小模型"在 CPU 上训出一个会写《三国演义》的小 GPT（语料、分词器、训练循环、续训、多卡与规模估算），再从显存账本和时间模型出发，把数据并行（DDP、ZeRO、FSDP2）、张量并行与序列并行、流水线并行、上下文并行和专家并行逐个从零实现，再讲混合精度与 FP8、3D / 5D 并行的组合与配置搜索、训练框架、分布式 checkpoint 与 RL 训练系统。

## 目录

- `docs/`：正文，按部分分为 `scratch/`（从零训练）、`basics/`、`data/`、`model/`、`practice/`、`algo/`（优化器、稳定性、RL 算法）
- `docs/assets/data/sanguo.txt`：从零训练用的语料（《三国演义》，Project Gutenberg #23950，公有领域，已转成简体并整理排版）
- `tools/check_code.py`：运行正文里所有标了文件名的脚本，确认页面上的输出与实际运行一致

## 校验

```bash
# 需要 CPU 版 PyTorch（2.4 以上）、numpy 与 tokenizers；默认使用 ../cpp/.venv-py（见 C++ 手册的 README），也可以用环境变量 PYTHON 指定解释器
python3 tools/check_code.py                         # 所有页面
python3 tools/check_code.py docs/model/*.md         # 指定页面
```

约定（详见 `tools/check_code.py` 的说明）：

- ```` ```python title="x.py" ```` 是完整脚本，紧跟的 ```` ```text title="输出" ```` 必须与标准输出逐行一致；
- `torchrun="N"` 的脚本用 `torchrun --standalone --nproc-per-node N` 启动，通信后端是 gloo；
- `run="no"` 的脚本需要 GPU，只做语法检查；
- 同一页的脚本写在同一个目录里、按顺序运行，可以互相 import，也可以读前一个脚本写下的文件；
- `scratch/` 下的三页共用一个工作目录，按文件名顺序接力（分词器、token 文件、checkpoint 留给后一页），检查其中任何一页都会把三页从头跑一遍（约 6 分钟）。

构建站点：在仓库根目录运行 `./build.sh`，或单独预览 `mkdocs serve -f train/mkdocs.yml`。
