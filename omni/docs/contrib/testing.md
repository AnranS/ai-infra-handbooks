# 测试、CI 与代码规范：一个 PR 要过哪些关

<p class="lead">在 sglang-omni 提 PR，代码本身往往只是一小部分：仓库要求每个 if 都有 else、不许用下划线开头的名字、import 必须放在模块顶层；PR 要写清动机、改动、关联 issue、精度和性能数据；CI 跑在自托管的 H100 上，要维护者打上 <code>run-ci</code> 标签才会触发。这一章把这些关一道道列出来，并且演示在一台没有 GPU 的机器上，提交之前能先验证到什么程度——这决定了你的 PR 是"一次过"还是"来回改五轮"。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 仓库的测试分成哪几条"车道"？一个新的单测文件应该放在哪里？
    2. 没有 GPU 的机器上，怎么只跑不需要加速器的测试？跑 omni 的单测时最常见的一个环境问题是什么？
    3. 代码规范里有哪几条是"硬要求"、由 pre-commit 钩子强制检查的？
    4. 外部贡献者的 PR，GPU CI 是怎么被触发的？
    5. PR 描述的模板有哪几节？

??? success "自测参考答案（先自己答，再展开对照）"
    1. `tests/unit_test/`（单测）、`tests/test_model/`（加载真实模型的 CI 测试）、`tests/test_ci/`，外加 `tests/utils/` 放工具。`test-layout` 工作流会拒绝放在别处的测试文件。新单测按被测模块放进 `tests/unit_test/<模块>/`。
    2. 用 pytest 标记：`-m "not benchmark and not accelerator"`（CI 的 CPU 那一半就是这么选的）。最常见的环境问题是 Unix 域套接字的路径长度上限：pytest 的临时目录如果太长（用户名长、路径深），会报 "IPC endpoint path would exceed the Unix-domain socket path limit"，加 `--basetemp=/tmp/短路径` 就好。
    3. 每个 `if` 必须有 `else`（`scripts/check_if_else.py`）、类 / 函数 / 属性名不能以下划线开头（`scripts/check_leading_underscore.py`），外加 isort、black、autoflake、ruff、rustfmt 等通用格式化。"名字要用完整的词、import 必须在模块顶层"是规范文档里的硬要求，由 review 把关。
    4. 维护者给 PR 打 `run-ci` 标签，或者有权限的人评论 `/tag-and-rerun-ci <模型>`（例如 `qwen3-tts`、`fun-asr`、`qwen3-omni`）选择要跑的模型 CI；打上标签之后每次推送都会重跑，草稿 PR 不跑。
    5. Motivation、Modifications、Related Issues、Accuracy Test、Benchmark & Profiling、Checklist。

## 测试的三条车道

```bash title="ch11_layout.sh"
for lane in unit_test test_model test_ci; do
  n=$(git ls-tree -r --name-only "$REF" "tests/$lane/" | grep -c '/test_.*\.py$')
  printf '%-10s %4d 个测试文件\n' "$lane" "$n"
done
echo "单测按模块分的目录（前 12 个，按文件数）："
git ls-tree -r --name-only "$REF" tests/unit_test/ | grep '/test_.*\.py$' | cut -d/ -f3 | grep -v '\.py$' \
  | sort | uniq -c | sort -rn | head -12 | awk '{printf "  %-24s %3d\n", $2, $1}'
```

```text title="输出"
unit_test   511 个测试文件
test_model   37 个测试文件
test_ci       6 个测试文件
单测按模块分的目录（前 12 个，按文件数）：
  qwen3_omni                43
  serve                     35
  pipeline                  34
  ming_omni                 25
  benchmarks                25
  scheduling                23
  minicpm_o                 18
  fun_cosyvoice3            17
  dots_tts                  16
  model_runner              15
  higgs_tts                 15
  qwen3_tts                 14
```

- **`unit_test/`**：不加载真实模型权重，用假实现（`tests/unit_test/fixtures/`、`fakes.py`）替身。大部分能在 CPU 上跑；需要 GPU 的打 `@pytest.mark.accelerator`。
- **`test_model/`**：加载真实模型、起真实服务，量 WER / 延迟 / 吞吐，跑在 CI 的 H100 上。
- **`test_ci/`**：CI 基础设施本身的测试。

`test-layout` 工作流强制这个分层：

```yaml title=".github/workflows/test-layout.yaml @ 921ea2c8 L18-36"
      - uses: actions/checkout@v4

      - name: Check test file layout
        shell: bash
        run: |
          set -euo pipefail

          invalid_files=()
          while IFS= read -r path; do
            case "$path" in
              tests/__init__.py|tests/utils.py|tests/utils/*) ;;
              tests/unit_test/*|tests/test_model/*|tests/test_ci/*) ;;
              *) invalid_files+=("$path") ;;
            esac
          done < <(find tests -type f -name '*.py' | sort)

          if (( ${#invalid_files[@]} )); then
            echo "Python test files must live in an explicit test lane."
            echo "Use tests/unit_test/, tests/test_model/, tests/test_ci/, or tests/utils/."
```

PR 测试在 CI 里分成两半，用 pytest 标记区分：

```yaml title=".github/workflows/test.yaml @ 921ea2c8 L50-54,80-84"
        run: |
          source omni/bin/activate
          export PYTHONPATH=$PWD
          bash .github/scripts/run_flaky_pytest.sh \
            pytest tests/ -v -m "not benchmark and not accelerator" -x
...
        run: |
          source omni/bin/activate
          export PYTHONPATH=$PWD
          bash .github/scripts/run_flaky_pytest.sh \
            pytest tests/ -v -m "accelerator and not benchmark" -x
```

## 在没有 GPU 的机器上跑单测

本书的环境（CPU 版 PyTorch + `--no-deps` 安装的 SGLang，见[首页](../index.md#环境)）已经能跑相当一部分单测。在基准提交的源码树里，挑几个和本书第二部分对应的目录：

```bash title="ch11_unit.sh"
cd "$OMNI_TREE"
for d in pipeline config relay; do
  line=$("$PYTHON" -m pytest -q -p no:cacheprovider --basetemp="/tmp/omni-ut-$d" \
           -m "not benchmark and not accelerator" "tests/unit_test/$d" 2>/dev/null | tail -1)
  printf '%-10s %s\n' "$d" "$(echo "$line" | sed -E 's/ in [0-9.]+s.*//; s/, [0-9]+ warnings?//')"
done
```

```text title="输出"
pipeline   699 passed, 3 skipped, 13 deselected
config     146 passed
relay      13 passed, 3 deselected
```

两点经验：

- **`--basetemp` 一定要短。** omni 的单测会真的创建 ZMQ 的 `ipc://` 端点，Unix 域套接字的路径上限是 107 个字符。pytest 默认的临时目录是 `/tmp/pytest-of-<用户名>/pytest-<N>/<测试名>0`，用户名一长就超了——这台开发机的用户名就让 `tests/unit_test/pipeline/` 里有 10 个测试因此失败，加上 `--basetemp` 之后全部通过；
- **缺依赖就补依赖。** 有的模型目录的单测要 `torchaudio`、`sentencepiece`、`silero_vad`……报 `ModuleNotFoundError` 时按名字装上即可（本书的 `requirements-check.txt` 已经包含了跑这几个目录需要的全部）。

改了哪个模块，就先把那个模块的单测在本地跑绿，再提 PR。GPU 相关的部分（`accelerator` 标记、`test_model/`）只能交给 CI。

## 代码规范：两条由钩子强制的硬规则

规范写在 `.claude/skills/code-review/coding-style.md`（给人和 AI 共用），其中两条由 pre-commit 钩子强制：

```yaml title=".pre-commit-config.yaml @ 921ea2c8 L82-97"
      - id: leading-underscore-names
        name: leading-underscore class, function, and attribute names
        entry: python3 scripts/check_leading_underscore.py
        language: system
        types: [python]
        files: ^(sglang_omni|tests)/
        exclude: ^sglang_omni/vendor/
      # Check only. Do not pass --fix: else: pass is a last resort, not the
      # rewrite this hook should apply on commit.
      - id: if-else-required
        name: if statements must have else
        entry: python3 scripts/check_if_else.py
        language: system
        types: [python]
        files: ^sglang_omni/
        exclude: ^sglang_omni/vendor/
```

看看它们会怎么报错。在源码树里临时放一个"普通 Python 程序员会写"的文件：

```bash title="ch11_lint.sh"
cd "$OMNI_TREE"
demo=sglang_omni/lint_demo.py
trap 'rm -f "$demo"' EXIT
cat > "$demo" <<'EOF'
class _Cache:
    def get(self, key):
        if key is None:
            return None
        return key
EOF
python3 scripts/check_if_else.py "$demo" 2>&1 || true
python3 scripts/check_leading_underscore.py "$demo" 2>&1 || true
echo "== --fix 之后的 if："
python3 scripts/check_if_else.py --fix "$demo" > /dev/null || true
sed -n '3,7p' "$demo"
```

```text title="输出"
sglang_omni/lint_demo.py:3:8: if without else. At least use `else: pass` to fix this lint
1 if statement(s) without else in sglang_omni/. At least use `else: pass` to fix this lint, or run `python scripts/check_if_else.py --fix`.
sglang_omni/lint_demo.py:1:0: leading-underscore class '_Cache'; nested functions may keep '_'; run with --fix, or use a public name / '# noqa: leading-underscore'
1 leading-underscore class/function/attribute name(s) in sglang_omni/ and tests/
== --fix 之后的 if：
        if key is None:
            return None
        else:
            pass
        return key
```

规范文档里对"每个 if 都要有 else"的解释是"把互斥的分支写全，主流程放在最后一个分支里"，并且明确说"短的 if、只有 return / raise 的 if 也不例外"。对习惯了"卫语句"写法的人这很别扭，但这是这个仓库的规矩：**提交之前在本地跑 `pre-commit run --all-files`**，别让 CI 替你发现。其他几条硬要求（由 review 把关）：

```text title=".claude/skills/code-review/coding-style.md @ 921ea2c8 L9-24"
## Principles

Write clean, professional, maintainable code. Match the surrounding codebase's conventions
where they exist; where they don't, follow these rules.

The overriding goal is simplicity: fewer, smaller files and fewer functions. Avoid
speculative generality.

Naming clarity is part of that bar, not a polish pass. A reader must know the
unit of every name without opening the function body. Field jargon and short
forms do not meet it. The NAMING section below is the rule.

Imports are part of that bar too. A name used by a module is imported at
module scope, where a reader can see it. An import inside a function is not
a way to defer a heavy dependency or to paper over a cycle. The IMPORTS
section below is the rule.
```

- **名字用完整的词**：读者不打开函数体就要知道每个名字的单位（`timeout_ms` 而不是 `t`）；
- **import 放在模块顶层**：不许用函数内 import 来推迟重依赖或绕开循环依赖（你会在老代码里看到很多函数内 import，它们是规范出现之前写的）；
- **少文件、少函数、不做投机的通用化**。

Rust router 那边是另一套：`cargo fmt`、`clippy -D warnings`，`Cargo.toml` 的 lint 段禁止 `unwrap` / `expect` / `panic`（第十章）。

## CI：标签、斜杠命令和门禁

```bash title="ch11_workflows.sh"
for f in $(git ls-tree --name-only "$REF" .github/workflows/); do
  printf '%-40s %s\n' "$(basename "$f")" "$(git show "$REF:$f" | grep -m1 '^name:' | cut -c7-)"
done
```

```text title="输出"
cancel-pr-workflow-on-merge.yaml         Cancel PR Workflow Runs
cleanup-pr-ci-home-on-close.yaml         Cleanup PR CI Home On Close
docs-check.yaml                          Docs Check
lint.yaml                                Lint
omni-ci.yaml                             Omni CI
omni-cpu-ci.yaml                         CPU CI
omni-xpu-ci.yaml                         XPU CI
omni-xpu-docker-release.yaml             XPU Docker Release
publish-docs.yaml                        Docs Publish
release-docker-npu-nightly.yml           Release Docker Images Nightly (NPU)
release-docker-npu.yml                   Release Docker Images (NPU)
rust-router.yml                          Rust Router
slash-command-handler.yml                Slash Command Handler
test-asr-ci.yaml                         ASR CI
test-layout.yaml                         Test Layout
test-qwen3-omni-ci.yaml                  Omni model CI
test-tts-ci.yaml                         TTS CI
test.yaml                                PR Test
voxt-mac-ci.yaml                         Voxt Mac CI
```

一个外部贡献者的 PR 会经历：

1. **自动跑的轻量检查**：`Lint`（pre-commit）、`Test Layout`、`Docs Check`；改了 router 的话还有 `Rust Router`（在 GitHub 托管的 Ubuntu 上跑 fmt / clippy / test）。
2. **GPU CI 需要标签**：`Omni CI` 是总入口，检查 PR 有没有 `run-ci` 标签；有权限打标签、重跑的人是维护者和 `.github/CI_PERMISSIONS.json` 里登记的少数几位。标签打上之后，`PR Test`（单测，两半）、`TTS CI`、`ASR CI`、`Omni model CI` 在自托管的 H100 上跑；`pr_ci_gate.py` 先确认 lint 已经通过、PR 可以合并，才放行昂贵的 GPU 任务。
3. **用斜杠命令选模型**：每类 CI 默认随机挑一个模型（TTS 从 Higgs、MOSS、Qwen3-TTS、CosyVoice3 里挑），评论 `/tag-and-rerun-ci qwen3-tts fun-asr` 可以指定。

所以 PR 描述里写清"这个改动影响哪个模型"很重要——维护者据此决定跑哪个 CI。

## PR 描述：模板与真实的样子

```markdown title=".github/pull_request_template.md @ 921ea2c8 L3-29"
## Motivation

<!-- Explain the purpose of this PR and the goals it aims to achieve. -->

## Modifications

<!-- Describe the changes made in this PR. -->

## Related Issues

<!-- Link to any related issues here. e.g. "Fixes #123" or "Closes #456" -->

## Accuracy Test

<!-- If this PR affects model-side code (e.g., kernels, model architecture), please provide accuracy test results. Ref: https://docs.sglang.io/docs/developer_guide/evaluating_new_models -->

## Benchmark & Profiling

<!-- If this PR is expected to impact performance, please provide benchmark and profiling results. Ref: https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling -->

## Checklist

- [ ] Format your code according with pre-commit.
- [ ] Add unit tests.
- [ ] Update documentation / docstrings / example tutorials as needed.
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.
```

模板之外，这个仓库的 issue 和 PR 有一种很固定的写法，读几个合入的 PR 就能看出来——第四章读过的 #1628 的提交说明就是典型：

- **Why**：现象 + 数字（"每个请求白等 50 ms，约占语音首包时间的三分之一"）；
- **Scope**：做什么、明确不做什么；
- **Approach**：具体改哪几个文件、什么思路；
- **Verification**：哪些测试、哪些评测、前后对比的数字；行为不变的改动要说明"默认行为不变"。

第十二章会按这个结构写一个完整的 PR 描述。

## 练习

**1. 跑你关心的模块。** 选一个第二部分讲过的模块（`scheduling`、`comm`、`serve`……），在本书环境里跑它的单测，记录通过、失败、跳过的数量；对每个失败，判断是"缺依赖 / 环境问题"还是"需要 GPU"。

??? success "参考思路"
    `cd "$OMNI_TREE" && "$PYTHON" -m pytest -q --basetemp=/tmp/x -m "not benchmark and not accelerator" tests/unit_test/scheduling`。看失败的报错：`ModuleNotFoundError` → 补依赖；`torch.cuda` / `CUDA` 相关 → 需要 GPU，本地跳过即可；路径太长 → `--basetemp`。如果某个测试既不需要 GPU、也不是环境问题却失败了，那可能是一个真实的问题。

**2. 找一个违反规范的老代码。** 用 `git grep` 找出 `sglang_omni/` 里仍然在函数内 import 的地方，挑一个说明它为什么还在（规范之前写的？为了避免循环依赖？为了不在没有 GPU 的平台上导入 CUDA 模块？）。

??? success "参考思路"
    `git grep -nE '^\s{8,}(from|import) ' 921ea2c8 -- 'sglang_omni/*.py' | head`。很多是平台相关的延迟导入（例如只在 CUDA 平台上才 import 的模块），或者在 `TYPE_CHECKING` 之外为了避免循环依赖。规范要求新代码不要这样写；顺手把一个能安全挪到顶层的 import 挪上去，是一个合格的小清理 PR——前提是你确认挪上去之后在所有平台上都能导入。

**3. 读一次 CI 的失败。** 在 GitHub 上找一个最近 `TTS CI` 失败后又修好的 PR，读失败的日志，说明失败是代码问题、阈值问题还是基础设施问题。

??? success "参考思路"
    在 PR 列表里按标签 `run-ci` 筛选，打开 Checks 页。这个仓库的 CI 有不少"性能阈值"类的门禁（例如第二章提到的 #1202：p95 只用 9 个样本，实际上等于最大值），区分"代码变慢了"和"门禁本身不稳"是 review 时的常见讨论。

!!! interview "怎么讲清楚"
    讲在一个活跃开源项目里提 PR 的流程，按"本地 → 轻量 CI → 重量 CI → review"讲：本地先跑改动模块的单测（CPU 上用 `-m "not accelerator"`，注意 `--basetemp`）和 `pre-commit run --all-files`；推上去后 lint、测试布局、文档检查自动跑；GPU CI 要维护者打 `run-ci` 标签，用斜杠命令选模型；PR 描述按 Why / Scope / Approach / Verification 写，有数字。再提一句这个仓库特有的规矩：每个 if 必须有 else、不许下划线前缀、import 在模块顶层，都有钩子或 review 把关。

## 小结

- [x] 测试三条车道：`unit_test/`、`test_model/`、`test_ci/`，`test-layout` 强制；GPU 测试打 `accelerator` 标记。
- [x] CPU 上跑单测：`-m "not benchmark and not accelerator"`，`--basetemp` 要短（Unix 域套接字 107 字符上限），缺依赖按名补。
- [x] 钩子强制：每个 if 有 else、无下划线前缀名，外加 isort / black / autoflake / ruff / rustfmt；review 把关：完整词命名、模块顶层 import。
- [x] GPU CI 要 `run-ci` 标签（维护者或登记的少数人），`/tag-and-rerun-ci <模型>` 选模型，`pr_ci_gate.py` 先确认 lint 和可合并。
- [x] PR 描述：模板六节 + 仓库惯用的 Why / Scope / Approach / Verification，带数字。
