<!-- 这个目录由 tools/export_examples.py 生成，不要手改；改正文里的代码块 -->

# 《SGLang-Omni 源码导读》正文里的代码

这里的 51 个文件逐字取自正文的代码块，页面是唯一的源；`tools/export_examples.py` 负责导出，构建时会检查两边一致。

怎么跑见每一页正文；需要的环境见仓库根目录的 `env/setup-macos.sh`（Mac）或 `setup-gpu.sh`（NVIDIA 显卡）。

| 文件 | 出自 | 备注 |
| --- | --- | --- |
| `architectures.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `builder-hooks.sh` | [journey/embed-sglang.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/embed-sglang) |  |
| `bumps.sh` | [journey/embed-sglang.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/embed-sglang) |  |
| `ch10_build.sh` | [cases/router.md](https://anrans.github.io/ai-infra-handbooks/omni/cases/router) |  |
| `ch10_fleet.py` | [cases/router.md](https://anrans.github.io/ai-infra-handbooks/omni/cases/router) |  |
| `ch10_tests.sh` | [cases/router.md](https://anrans.github.io/ai-infra-handbooks/omni/cases/router) |  |
| `ch11_layout.sh` | [contrib/testing.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/testing) |  |
| `ch11_lint.sh` | [contrib/testing.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/testing) |  |
| `ch11_unit.sh` | [contrib/testing.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/testing) |  |
| `ch11_workflows.sh` | [contrib/testing.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/testing) |  |
| `ch3_abort.py` | [journey/coordinator.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/coordinator) |  |
| `ch3_endpoints.py` | [journey/coordinator.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/coordinator) |  |
| `ch3_stages.py` | [journey/coordinator.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/coordinator) |  |
| `ch3_terminals.py` | [journey/coordinator.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/coordinator) |  |
| `ch4_batch.py` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `ch4_batch_stages.py` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `ch4_fanin.py` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `ch4_fanin_stages.py` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `ch4_stream.py` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `ch4_stream_stages.py` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `ch5_compose.py` | [journey/embed-sglang.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/embed-sglang) |  |
| `ch5_methods.py` | [journey/embed-sglang.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/embed-sglang) |  |
| `ch6_pipeline.py` | [journey/comm.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/comm) | 需要 GPU，手册里只做语法检查 |
| `ch6_stages.py` | [journey/comm.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/comm) |  |
| `ch6_trace.py` | [journey/comm.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/comm) |  |
| `ch7_resolve.sh` | [journey/config.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/config) |  |
| `ch7_topology.py` | [journey/config.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/config) |  |
| `ch8_chunks.py` | [cases/qwen3-tts.md](https://anrans.github.io/ai-infra-handbooks/omni/cases/qwen3-tts) |  |
| `ch9_routing.py` | [cases/qwen3-omni.md](https://anrans.github.io/ai-infra-handbooks/omni/cases/qwen3-omni) |  |
| `check.sh` | [index.md](https://anrans.github.io/ai-infra-handbooks/omni/) | 需要 GPU，手册里只做语法检查 |
| `docs-tests-ci.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `imports.sh` | [overview/position.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/position) |  |
| `inline-origin.sh` | [journey/comm.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/comm) |  |
| `mlx-fast.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `multimodal-gen.sh` | [overview/position.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/position) |  |
| `no-sglang-models.sh` | [overview/position.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/position) |  |
| `pace.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `package.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `pins.sh` | [overview/position.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/position) |  |
| `pr-1628.sh` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `qwen3-tts-files.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `relay-kinds.sh` | [journey/comm.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/comm) |  |
| `scan_issues.py` | [contrib/first-mr.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/first-mr) | 需要 GPU，手册里只做语法检查 |
| `schedulers.sh` | [journey/stage.md](https://anrans.github.io/ai-infra-handbooks/omni/journey/stage) |  |
| `setup-manual.sh` | [index.md](https://anrans.github.io/ai-infra-handbooks/omni/) | 需要 GPU，手册里只做语法检查 |
| `setup.sh` | [index.md](https://anrans.github.io/ai-infra-handbooks/omni/) | 需要 GPU，手册里只做语法检查 |
| `stale-issues.sh` | [contrib/first-mr.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/first-mr) |  |
| `todos.sh` | [contrib/first-mr.md](https://anrans.github.io/ai-infra-handbooks/omni/contrib/first-mr) |  |
| `toolbox.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) | 需要 GPU，手册里只做语法检查 |
| `top-level.sh` | [overview/repo-map.md](https://anrans.github.io/ai-infra-handbooks/omni/overview/repo-map) |  |
| `tts-16g.sh` | [cases/qwen3-tts.md](https://anrans.github.io/ai-infra-handbooks/omni/cases/qwen3-tts) | 需要 GPU，手册里只做语法检查 |
