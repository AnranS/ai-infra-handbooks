<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《分布式训练手册》正文里的代码

这里的 45 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `adamw_decay.py` | [algo/optimizer.md](https://anrans.github.io/ai-infra-handbooks/train/algo/optimizer) |  |
| `balance_bias.py` | [model/moe-ep.md](https://anrans.github.io/ai-infra-handbooks/train/model/moe-ep) |  |
| `collectives.py` | [basics/collectives.md](https://anrans.github.io/ai-infra-handbooks/train/basics/collectives) |  |
| `commtime.py` | [basics/no-multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/train/basics/no-multi-gpu) |  |
| `cp_balance.py` | [model/context.md](https://anrans.github.io/ai-infra-handbooks/train/model/context) |  |
| `cp_common.py` | [model/context.md](https://anrans.github.io/ai-infra-handbooks/train/model/context) |  |
| `dcp_load.py` | [practice/frameworks-rl.md](https://anrans.github.io/ai-infra-handbooks/train/practice/frameworks-rl) |  |
| `dcp_save.py` | [practice/frameworks-rl.md](https://anrans.github.io/ai-infra-handbooks/train/practice/frameworks-rl) |  |
| `ddp_check.py` | [data/ddp.md](https://anrans.github.io/ai-infra-handbooks/train/data/ddp) |  |
| `distributed_muon.py` | [algo/optimizer.md](https://anrans.github.io/ai-infra-handbooks/train/algo/optimizer) |  |
| `fp4_training.py` | [algo/stability.md](https://anrans.github.io/ai-infra-handbooks/train/algo/stability) |  |
| `fp8_scaling.py` | [practice/mixed-precision.md](https://anrans.github.io/ai-infra-handbooks/train/practice/mixed-precision) |  |
| `fsdp2_check.py` | [data/zero-fsdp.md](https://anrans.github.io/ai-infra-handbooks/train/data/zero-fsdp) |  |
| `grpo_toy.py` | [practice/frameworks-rl.md](https://anrans.github.io/ai-infra-handbooks/train/practice/frameworks-rl) |  |
| `hyper_connections.py` | [algo/stability.md](https://anrans.github.io/ai-infra-handbooks/train/algo/stability) |  |
| `kl_estimators.py` | [algo/rl-algorithms.md](https://anrans.github.io/ai-infra-handbooks/train/algo/rl-algorithms) |  |
| `ledger.py` | [basics/overview.md](https://anrans.github.io/ai-infra-handbooks/train/basics/overview) |  |
| `logit_growth.py` | [algo/stability.md](https://anrans.github.io/ai-infra-handbooks/train/algo/stability) |  |
| `loss_aggregation.py` | [algo/rl-algorithms.md](https://anrans.github.io/ai-infra-handbooks/train/algo/rl-algorithms) |  |
| `mesh.py` | [practice/strategy.md](https://anrans.github.io/ai-infra-handbooks/train/practice/strategy) |  |
| `mismatch.py` | [algo/rl-algorithms.md](https://anrans.github.io/ai-infra-handbooks/train/algo/rl-algorithms) |  |
| `moe_ep.py` | [model/moe-ep.md](https://anrans.github.io/ai-infra-handbooks/train/model/moe-ep) |  |
| `muon.py` | [algo/optimizer.md](https://anrans.github.io/ai-infra-handbooks/train/algo/optimizer) |  |
| `muon_vs_adamw.py` | [algo/optimizer.md](https://anrans.github.io/ai-infra-handbooks/train/algo/optimizer) |  |
| `my_ddp.py` | [data/ddp.md](https://anrans.github.io/ai-infra-handbooks/train/data/ddp) |  |
| `optimizer_state.py` | [algo/optimizer.md](https://anrans.github.io/ai-infra-handbooks/train/algo/optimizer) |  |
| `orthogonalize.py` | [algo/optimizer.md](https://anrans.github.io/ai-infra-handbooks/train/algo/optimizer) |  |
| `parallel_plan.py` | [practice/strategy.md](https://anrans.github.io/ai-infra-handbooks/train/practice/strategy) |  |
| `pp_1f1b.py` | [model/pipeline.md](https://anrans.github.io/ai-infra-handbooks/train/model/pipeline) |  |
| `pp_sim.py` | [model/pipeline.md](https://anrans.github.io/ai-infra-handbooks/train/model/pipeline) |  |
| `precision.py` | [practice/mixed-precision.md](https://anrans.github.io/ai-infra-handbooks/train/practice/mixed-precision) |  |
| `probe.py` | [basics/no-multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/train/basics/no-multi-gpu) |  |
| `qk_clip.py` | [algo/stability.md](https://anrans.github.io/ai-infra-handbooks/train/algo/stability) |  |
| `ratios.py` | [algo/rl-algorithms.md](https://anrans.github.io/ai-infra-handbooks/train/algo/rl-algorithms) |  |
| `ring_allreduce.py` | [basics/collectives.md](https://anrans.github.io/ai-infra-handbooks/train/basics/collectives) |  |
| `ring_attention.py` | [model/context.md](https://anrans.github.io/ai-infra-handbooks/train/model/context) |  |
| `scale_out.py` | [practice/strategy.md](https://anrans.github.io/ai-infra-handbooks/train/practice/strategy) |  |
| `tp_check.py` | [basics/no-multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/train/basics/no-multi-gpu) |  |
| `tp_mlp_check.py` | [model/tensor-sequence.md](https://anrans.github.io/ai-infra-handbooks/train/model/tensor-sequence) |  |
| `tp_ops.py` | [model/tensor-sequence.md](https://anrans.github.io/ai-infra-handbooks/train/model/tensor-sequence) |  |
| `ulysses.py` | [model/context.md](https://anrans.github.io/ai-infra-handbooks/train/model/context) |  |
| `weight_sync.py` | [practice/frameworks-rl.md](https://anrans.github.io/ai-infra-handbooks/train/practice/frameworks-rl) |  |
| `z_loss.py` | [algo/stability.md](https://anrans.github.io/ai-infra-handbooks/train/algo/stability) |  |
| `zero_adam.py` | [data/zero-fsdp.md](https://anrans.github.io/ai-infra-handbooks/train/data/zero-fsdp) |  |
| `zero_check.py` | [data/zero-fsdp.md](https://anrans.github.io/ai-infra-handbooks/train/data/zero-fsdp) |  |
