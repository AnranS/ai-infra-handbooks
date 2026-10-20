<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《手写 Transformer》正文里的代码

这里的 41 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `ch01_weights.py` | [weights.md](https://anrans.github.io/ai-infra-handbooks/transformer/weights) |  |
| `ch02_embedding.py` | [embedding.md](https://anrans.github.io/ai-infra-handbooks/transformer/embedding) |  |
| `ch03_norm.py` | [norm.md](https://anrans.github.io/ai-infra-handbooks/transformer/norm) |  |
| `ch04_mlp.py` | [mlp.md](https://anrans.github.io/ai-infra-handbooks/transformer/mlp) |  |
| `ch05_rope.py` | [rope.md](https://anrans.github.io/ai-infra-handbooks/transformer/rope) |  |
| `ch06_attention.py` | [attention.md](https://anrans.github.io/ai-infra-handbooks/transformer/attention) |  |
| `ch06_gate_sgd.py` | [attention.md](https://anrans.github.io/ai-infra-handbooks/transformer/attention) |  |
| `ch06_gate_toy.py` | [attention.md](https://anrans.github.io/ai-infra-handbooks/transformer/attention) |  |
| `ch06_toy.py` | [attention.md](https://anrans.github.io/ai-infra-handbooks/transformer/attention) |  |
| `ch07_deltanet.py` | [deltanet.md](https://anrans.github.io/ai-infra-handbooks/transformer/deltanet) |  |
| `ch07_toy.py` | [deltanet.md](https://anrans.github.io/ai-infra-handbooks/transformer/deltanet) |  |
| `ch08_chunk.py` | [chunk.md](https://anrans.github.io/ai-infra-handbooks/transformer/chunk) |  |
| `ch08_toy.py` | [chunk.md](https://anrans.github.io/ai-infra-handbooks/transformer/chunk) |  |
| `ch09_inside.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/transformer/model) |  |
| `ch09_model.py` | [model.md](https://anrans.github.io/ai-infra-handbooks/transformer/model) |  |
| `ch10_cache.py` | [cache.md](https://anrans.github.io/ai-infra-handbooks/transformer/cache) |  |
| `ch11_sample.py` | [sample.md](https://anrans.github.io/ai-infra-handbooks/transformer/sample) |  |
| `ch12_chat.py` | [chat.md](https://anrans.github.io/ai-infra-handbooks/transformer/chat) |  |
| `ch12_turns.py` | [chat.md](https://anrans.github.io/ai-infra-handbooks/transformer/chat) |  |
| `ch13_batch.py` | [batch.md](https://anrans.github.io/ai-infra-handbooks/transformer/batch) |  |
| `ch14_mtp.py` | [mtp.md](https://anrans.github.io/ai-infra-handbooks/transformer/mtp) |  |
| `ch15_speed.py` | [speed.md](https://anrans.github.io/ai-infra-handbooks/transformer/speed) |  |
| `ch16_scale.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/transformer/scale) |  |
| `main.py` | [main.md](https://anrans.github.io/ai-infra-handbooks/transformer/main) |  |
| `map.py` | [main.md](https://anrans.github.io/ai-infra-handbooks/transformer/main) |  |
| `qwen/attention.py` | [cache.md](https://anrans.github.io/ai-infra-handbooks/transformer/cache) | 覆盖了 attention.md 里的同名文件 |
| `qwen/batch.py` | [batch.md](https://anrans.github.io/ai-infra-handbooks/transformer/batch) |  |
| `qwen/cache.py` | [cache.md](https://anrans.github.io/ai-infra-handbooks/transformer/cache) |  |
| `qwen/chat.py` | [chat.md](https://anrans.github.io/ai-infra-handbooks/transformer/chat) |  |
| `qwen/chunk.py` | [chunk.md](https://anrans.github.io/ai-infra-handbooks/transformer/chunk) |  |
| `qwen/deltanet.py` | [cache.md](https://anrans.github.io/ai-infra-handbooks/transformer/cache) | 覆盖了 deltanet.md 里的同名文件 |
| `qwen/embedding.py` | [embedding.md](https://anrans.github.io/ai-infra-handbooks/transformer/embedding) |  |
| `qwen/layer.py` | [cache.md](https://anrans.github.io/ai-infra-handbooks/transformer/cache) | 覆盖了 model.md 里的同名文件 |
| `qwen/mlp.py` | [mlp.md](https://anrans.github.io/ai-infra-handbooks/transformer/mlp) |  |
| `qwen/model.py` | [cache.md](https://anrans.github.io/ai-infra-handbooks/transformer/cache) | 覆盖了 model.md 里的同名文件 |
| `qwen/moe.py` | [scale.md](https://anrans.github.io/ai-infra-handbooks/transformer/scale) |  |
| `qwen/mtp.py` | [mtp.md](https://anrans.github.io/ai-infra-handbooks/transformer/mtp) |  |
| `qwen/norm.py` | [norm.md](https://anrans.github.io/ai-infra-handbooks/transformer/norm) |  |
| `qwen/rope.py` | [rope.md](https://anrans.github.io/ai-infra-handbooks/transformer/rope) |  |
| `qwen/sample.py` | [sample.md](https://anrans.github.io/ai-infra-handbooks/transformer/sample) |  |
| `qwen/weights.py` | [weights.md](https://anrans.github.io/ai-infra-handbooks/transformer/weights) |  |
