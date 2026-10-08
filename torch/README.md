# PyTorch 速成

八章把 PyTorch 用熟：张量与形状、索引与掩码、autograd、nn.Module、数据与训练循环、保存与复现、调试与提速。只讲**怎么用**；内部实现见 CUDA 手册的「框架与编译器」几章，完整训练一个模型见《从零训练一个小模型》。

## 目录

- `docs/`：正文八章与首页
- `docs-en/`：英文译文（`tools/i18n.py` 构建到站点的 `en/torch/`）
- `examples/`：正文里的代码导出成的真实文件（`tools/export_examples.py` 生成）
- `tools/check_code.py`：运行正文里所有标了文件名的脚本，确认页面上的输出与实际运行一致

## 校验

```bash
# 需要 CPU 版 PyTorch（2.4 以上）与 numpy；默认使用 ../cpp/.venv-py（见 C++ 手册的 README），也可以用环境变量 PYTHON 指定解释器
python3 tools/check_code.py                      # 全部八章，约半分钟
python3 tools/check_code.py docs/autograd.md     # 指定页面
```

约定和分布式训练手册一样（规则直接复用 `../train/tools/check_code.py`）：每一页是独立的，页内的脚本写进同一个目录按顺序运行；`ci="loose"` 的脚本（训练 loss、profiler 排序这类随机器变的输出）在 CI 里照常运行但不比对输出。

构建站点：在仓库根目录运行 `./build.sh`，或单独预览 `mkdocs serve -f torch/mkdocs.yml`。
