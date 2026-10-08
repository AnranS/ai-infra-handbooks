<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《推理系统手册》正文里的代码

这里的 25 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `api_server.py` | [engine/sampler-api.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/sampler-api) |  |
| `constrained.py` | [topics/structured-output.md](https://anrans.github.io/ai-infra-handbooks/serving/topics/structured-output) |  |
| `detokenizer.py` | [engine/sampler-api.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/sampler-api) |  |
| `ep.py` | [distributed/expert-parallel.md](https://anrans.github.io/ai-infra-handbooks/serving/distributed/expert-parallel) |  |
| `formats.py` | [perf/quantization-deploy.md](https://anrans.github.io/ai-infra-handbooks/serving/perf/quantization-deploy) |  |
| `gpualloc.py` | [k8s/gpu.md](https://anrans.github.io/ai-infra-handbooks/serving/k8s/gpu) |  |
| `isvc_controller.py` | [k8s/operator-ops.md](https://anrans.github.io/ai-infra-handbooks/serving/k8s/operator-ops) |  |
| `lws.py` | [k8s/multi-node.md](https://anrans.github.io/ai-infra-handbooks/serving/k8s/multi-node) |  |
| `mini_platform.py` | [ops/platforms.md](https://anrans.github.io/ai-infra-handbooks/serving/ops/platforms) |  |
| `nano_engine.py` | [engine/scheduler.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/scheduler) |  |
| `paged.py` | [engine/paged-kv.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/paged-kv) |  |
| `pd.py` | [distributed/pd-disagg.md](https://anrans.github.io/ai-infra-handbooks/serving/distributed/pd-disagg) |  |
| `pdmux_policy.py` | [frontier/pd-multiplex.md](https://anrans.github.io/ai-infra-handbooks/serving/frontier/pd-multiplex) |  |
| `pp.py` | [distributed/pp-cp.md](https://anrans.github.io/ai-infra-handbooks/serving/distributed/pp-cp) |  |
| `prefix_cache.py` | [engine/prefix-cache.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/prefix-cache) |  |
| `radix.py` | [engine/prefix-cache.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/prefix-cache) |  |
| `reconcile.py` | [k8s/basics.md](https://anrans.github.io/ai-infra-handbooks/serving/k8s/basics) |  |
| `rollout.py` | [k8s/deploy-scale.md](https://anrans.github.io/ai-infra-handbooks/serving/k8s/deploy-scale) |  |
| `runner.py` | [engine/batch-layout.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/batch-layout) |  |
| `sampler.py` | [engine/sampler-api.md](https://anrans.github.io/ai-infra-handbooks/serving/engine/sampler-api) |  |
| `sched.py` | [k8s/scheduling.md](https://anrans.github.io/ai-infra-handbooks/serving/k8s/scheduling) |  |
| `sim.py` | [perf/benchmark.md](https://anrans.github.io/ai-infra-handbooks/serving/perf/benchmark) |  |
| `tiered.py` | [distributed/kv-offload.md](https://anrans.github.io/ai-infra-handbooks/serving/distributed/kv-offload) |  |
| `tp.py` | [distributed/tensor-parallel.md](https://anrans.github.io/ai-infra-handbooks/serving/distributed/tensor-parallel) |  |
| `tree_spec.py` | [topics/speculative.md](https://anrans.github.io/ai-infra-handbooks/serving/topics/speculative) |  |
