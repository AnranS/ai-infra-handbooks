<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《SGLang 设计演进》正文里的代码

这里的 79 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `adapted-from-vllm.sh` | [service/borrow-vllm.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/borrow-vllm) |  |
| `attention-backends.sh` | [perf/restructure.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/restructure) |  |
| `backend-commits.sh` | [scale/attention-backends.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/attention-backends) |  |
| `backend-interface.sh` | [scale/attention-backends.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/attention-backends) |  |
| `backends-per-tag.sh` | [scale/attention-backends.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/attention-backends) |  |
| `by-month.py` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `compile-commits.sh` | [perf/mla-compile.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/mla-compile) |  |
| `constrained-dir.sh` | [origins/fsm-jump.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/fsm-jump) |  |
| `diffusion-import.sh` | [platform/multimodal-diffusion.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/multimodal-diffusion) |  |
| `diffusion-size.sh` | [platform/multimodal-diffusion.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/multimodal-diffusion) |  |
| `eagle-commits.sh` | [scale/eagle.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/eagle) |  |
| `entrypoints-dir.sh` | [platform/entrypoints.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/entrypoints) |  |
| `eplb-files.sh` | [scale/large-ep.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/large-ep) |  |
| `first-appearance.sh` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `first-commits.sh` | [origins/paper.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/paper) |  |
| `function-call.sh` | [platform/entrypoints.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/entrypoints) |  |
| `gateway-size.sh` | [platform/gateway.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/gateway) |  |
| `gateway-timeline.sh` | [platform/gateway.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/gateway) |  |
| `gateway-tree.sh` | [platform/gateway.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/gateway) |  |
| `hardware-first.sh` | [scale/attention-backends.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/attention-backends) |  |
| `hicache-commits.sh` | [scale/hicache.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/hicache) |  |
| `hicache-storage.sh` | [scale/hicache.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/hicache) |  |
| `inspect.sh` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `jump-forward-timeline.sh` | [origins/fsm-jump.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/fsm-jump) |  |
| `keyword-first.sh` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `lang-size.sh` | [origins/frontend.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/frontend) |  |
| `large-ep-commits.sh` | [scale/large-ep.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/large-ep) |  |
| `managers-tree.sh` | [service/processes.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/processes) |  |
| `mla-commits.sh` | [perf/mla-compile.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/mla-compile) |  |
| `models-per-tag.sh` | [service/api-multimodal.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/api-multimodal) |  |
| `module-quarter.py` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `moe-dirs.sh` | [scale/large-ep.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/large-ep) |  |
| `multi-gpu-commits.sh` | [perf/multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/multi-gpu) |  |
| `multimodal-commits.sh` | [service/api-multimodal.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/api-multimodal) |  |
| `multimodal-growth.sh` | [platform/multimodal-diffusion.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/multimodal-diffusion) |  |
| `new-dirs-2026.sh` | [platform/codebase-2026.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/codebase-2026) |  |
| `one-page-timeline.sh` | [method/interview.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/interview) |  |
| `openai-refactor.sh` | [platform/entrypoints.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/entrypoints) |  |
| `openai-serving.sh` | [platform/entrypoints.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/entrypoints) |  |
| `overlap-commits.sh` | [perf/overlap.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/overlap) |  |
| `overlap-files.sh` | [perf/overlap.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/overlap) |  |
| `page-size-stat.sh` | [scale/attention-backends.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/attention-backends) |  |
| `paper-vs-code.py` | [origins/paper.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/paper) |  |
| `pd-commits.sh` | [scale/pd.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/pd) |  |
| `pd-dir.sh` | [scale/pd.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/pd) |  |
| `pd-initial.sh` | [scale/pd.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/pd) |  |
| `pickaxe.sh` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `pin-ref.sh` | [method/archaeology.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/archaeology) |  |
| `principles-anchors.sh` | [method/principles.md](https://anrans.github.io/ai-infra-handbooks/sglang/method/principles) |  |
| `process-commits.sh` | [service/processes.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/processes) |  |
| `radix-bug.py` | [origins/radix-v1.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/radix-v1) |  |
| `radix-evolution.sh` | [origins/radix-v1.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/radix-v1) |  |
| `radix-fix.sh` | [origins/radix-v1.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/radix-v1) |  |
| `release-size.sh` | [origins/paper.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/paper) |  |
| `reliability-commits.sh` | [platform/codebase-2026.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/codebase-2026) |  |
| `remove-jump-forward.sh` | [origins/fsm-jump.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/fsm-jump) |  |
| `remove-vllm-commits.sh` | [service/borrow-vllm.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/borrow-vllm) |  |
| `restructure-commits.sh` | [perf/restructure.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/restructure) |  |
| `rl-commits.sh` | [platform/rl.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/rl) |  |
| `router-size.sh` | [perf/multi-gpu.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/multi-gpu) |  |
| `routes-and-grpc.sh` | [platform/entrypoints.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/entrypoints) |  |
| `routes-per-tag.sh` | [service/api-multimodal.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/api-multimodal) |  |
| `rust-crates.sh` | [platform/gateway.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/gateway) |  |
| `sgl-kernel-commits.sh` | [perf/sgl-kernel.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/sgl-kernel) |  |
| `sgl-kernel-dirs.sh` | [perf/sgl-kernel.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/sgl-kernel) |  |
| `sgl-kernel-releases.sh` | [perf/sgl-kernel.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/sgl-kernel) |  |
| `spec-dir.sh` | [scale/eagle.md](https://anrans.github.io/ai-infra-handbooks/sglang/scale/eagle) |  |
| `srt-dirs-2026.sh` | [platform/codebase-2026.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/codebase-2026) |  |
| `srt-dirs.sh` | [perf/restructure.md](https://anrans.github.io/ai-infra-handbooks/sglang/perf/restructure) |  |
| `srt-lines.py` | [origins/first-commit.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/first-commit) |  |
| `then-and-now.sh` | [origins/first-commit.md](https://anrans.github.io/ai-infra-handbooks/sglang/origins/first-commit) |  |
| `update-weights-api.sh` | [platform/rl.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/rl) |  |
| `v02-tag.sh` | [service/v02.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/v02) |  |
| `v020-commits.sh` | [service/v02.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/v02) |  |
| `v020-core-lines.sh` | [service/v02.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/v02) |  |
| `verl-engine.sh` | [platform/rl.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/rl) |  |
| `vllm-imports.sh` | [service/borrow-vllm.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/borrow-vllm) |  |
| `vllm-pins.sh` | [service/borrow-vllm.md](https://anrans.github.io/ai-infra-handbooks/sglang/service/borrow-vllm) |  |
| `yearly-stats.sh` | [platform/codebase-2026.md](https://anrans.github.io/ai-infra-handbooks/sglang/platform/codebase-2026) |  |
