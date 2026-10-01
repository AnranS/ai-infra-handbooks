# 图像与视频生成推理手册

扩散 / 流匹配模型（SD、SDXL、SD3、FLUX、CogVideoX、HunyuanVideo、Wan）的推理与服务：pipeline 解剖、算力与显存账、少步生成、特征缓存、量化、多卡并行、视频推理的瓶颈，以及生成服务的调度与部署。面向已经懂 LLM 推理的读者。

在线阅读：<https://anrans.github.io/ai-infra-handbooks/media/>

## 代码怎么验证

正文里没有标题的 `python` 代码块会被 `tools/check_code.py` 按顺序执行，紧跟其后的「输出」块必须和实际输出逐行一致（规则与大模型原理手册相同）。所有模型都用 diffusers 的最小配置随机初始化构建，**不需要下载任何权重，CPU 就能跑**：

```bash
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -p .venv/bin/python diffusers transformers accelerate safetensors numpy pillow sentencepiece
.venv/bin/python tools/check_code.py                      # 全部页面
.venv/bin/python tools/check_code.py docs/basics/vae-latent.md   # 只查一页
```

换了库版本之后用 `python ../tools/refresh_outputs.py media docs/<page>.md --write` 把输出块刷新成实际结果。

真实模型的参数量、FLOP、显存和时延在正文里标明来源或"估算"，没有在真卡上实测的数字都写明了算法。
