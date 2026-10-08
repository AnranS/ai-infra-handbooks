# A snapshot of the codebase in 2026

<p class="lead">The baseline commit <code>29f6d408c0</code>'s <code>srt/</code> has 1979 Python files and v0.5.21's runtime is over 840 thousand lines; the first nine months of 2026 had more than 1200 authors and over ten thousand commits, 1200 a month on average, nearly 9 times 2024's monthly rate. This chapter is not about a feature but about reading today's codebase as a map: which directories only exist after the second half of 2025, what kind of problem each class of directory solves, how the roadmap's "compatibility and reliability" shows up in the commits, and what keeps a project releasing every two or three weeks from falling apart.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What order of magnitude are the commit and author counts for 2024, 2025 and 2026? What phase does that put the project in?
    2. What are the largest directories at the baseline commit? Which of them only appeared after the second half of 2025?
    3. The Q3 2025 roadmap shifted the focus from features to compatibility and reliability; what in the commits corresponds to that?
    4. What mechanisms keep a repository releasing every two or three weeks, with over a thousand contributors, maintainable?

??? success "Answers for the self-test (answer first, then open this)"
    1. 2024: about 1600 commits and nearly 190 authors; 2025: about 6800 and nearly 800; 2026 to early October: about 11 thousand and over 1200. From "a few authors' paper code" to a community project to a platform-scale project.
    2. `layers/` (400 files), `models/` (287), `mem_cache/` (156), `multimodal/` and `hardware_backend/` (95 each), `debug_utils/` (91), `configs/` and `arg_groups/` (79 each). Appearing after 2025-10: `compilation/`, `elastic_ep/`, `checkpoint_engine/`, `dllm/`, `hardware_backend/`, `observability/`, `arg_groups/`, `scheduler_components/`, `kv_canary/`, `weight_cache/`, `beam_search/`, `rust_extensions/` and `kernels/`.
    3. The allocation logic split out of the scheduler (#11313), piecewise CUDA graphs (#10062 and what followed), the v2 of overlapped scheduling and speculative decoding, elastic EP and failed ranks (#10423 and others), `debug_utils/`'s tensor comparison tools, `kv_canary/`'s cache-consistency checks and `observability/`'s tidying — all of it making the system stable and diagnosable across more combinations.
    4. CI triggered by directory, module boundaries (chapter 10's layering), mixins and pluggability (the attention backends, the speculative-method registry, the eviction policies, the transfer backends), argument grouping (`arg_groups/`), parity tests (the Python and Rust trees) and the rhythm of a tag per release.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/codebase-2026.webp is in Chinese; put it back once the English version exists -->

## Three years, three orders of magnitude {#三年的量级}

```bash title="yearly-stats.sh"
REF=${REF:-29f6d408c0}
for y in 2024 2025 2026; do
  printf '%s  提交 %5d  作者 %4d\n' "$y" "$(git log --date=short --format=%ad "$REF" | grep -c "^$y")" "$(git log --date=short --format='%ad %aN' "$REF" | grep "^$y" | cut -c12- | sort -u | wc -l)"
done
echo "累计作者：$(git log --format=%aN "$REF" | sort -u | wc -l)"
echo "srt/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- python/sglang/srt | grep -c '\.py$')；test/ 的 .py 文件：$(git ls-tree -r --name-only "$REF" -- test | grep -c '\.py$')"
echo "2026 年的版本 tag（不含网关）：$(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -vc gateway) 个，从 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -v gateway | head -1) 到 $(git for-each-ref --sort=creatordate --format='%(creatordate:short) %(refname:short)' refs/tags | grep '^2026' | grep -v gateway | tail -1)"
```

```text title="output"
2024  提交  1619  作者  189
2025  提交  6796  作者  798
2026  提交 10831  作者 1214
累计作者：1877
srt/ 的 .py 文件：1979；test/ 的 .py 文件：2515
2026 年的版本 tag（不含网关）：21 个，从 2026-01-01 v0.5.7 到 2026-09-30 v0.5.21
```

The timeline widget on [the home page](../index.md) draws exactly this curve: tens to a couple of hundred a month in 2024, four hundred to nine hundred in 2025, and a thousand to sixteen hundred in 2026. More test files than runtime files is this phase's most visible feature.

## A map of the directories {#目录地图}

```bash title="srt-dirs-2026.sh"
REF=${REF:-29f6d408c0}
git ls-tree -r --name-only "$REF" -- python/sglang/srt | grep '\.py$' | awk -F/ 'NF > 4 {print $4}' | sort | uniq -c | sort -rn | head -24 | awk '{printf "%4d  %s\n", $1, $2}'
```

```text title="output"
 400  layers
 287  models
 156  mem_cache
  95  multimodal
  95  hardware_backend
  91  debug_utils
  79  configs
  79  arg_groups
  66  entrypoints
  65  utils
  62  speculative
  58  model_executor
  53  managers
  50  kv_canary
  47  function_call
  46  lora
  36  disaggregation
  30  distributed
  23  model_loader
  15  parser
  15  observability
  14  eplb
  13  compilation
  12  sampling
```

Tagging each directory with its year of birth by this book's periods:

```bash title="new-dirs-2026.sh"
REF=${REF:-29f6d408c0}
for d in compilation elastic_ep checkpoint_engine dllm hardware_backend observability arg_groups kv_canary weight_cache beam_search rust_extensions; do
  printf '%-18s %s\n' "$d" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/$d | head -1 | cut -c1-78)"
done
printf '%-18s %s\n' "kernels (aot)" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/kernels | head -1 | cut -c1-78)"
printf '%-18s %s\n' "scheduler_components" "$(git log --reverse --date=short --format='%ad  %h  %s' "$REF" -- python/sglang/srt/managers/scheduler_components | head -1 | cut -c1-78)"
```

```text title="output"
compilation        2025-10-13  932e263725  Compilation Folder Reset (#11539)
elastic_ep         2025-10-22  904655c5fd  [2/N] Added the core structure of elastic EP and the e
checkpoint_engine  2025-10-24  96a5e4dd79  [Feature] Support loading weights from ckpt engine wor
dllm               2025-11-26  21b0582d4b  [feature] Initial block diffusion language model supp
hardware_backend   2025-12-04  894c0dc57c  [NPU][1/N] NPU basic functions refactor and new models
observability      2026-02-25  3b89302277  Refactor: observability code cleanup (#17862)
arg_groups         2026-05-03  00d620b77d  introduce arg_groups/ with nemotron_h hook (#24328)
kv_canary          2026-05-31  11391b2a1c  Add the KV-canary core: data layer, MHA KV-pool patche
weight_cache       2026-07-25  f9c14e6bd4  [FEAT] Support fast engine recovery through weight cac
beam_search        2026-08-26  ec4bdbfa4a  [Feature] Beam search support (#31626)
rust_extensions    2026-08-16  67e12131df  Build Rust extensions on demand in source checkouts (#
kernels (aot)      2026-07-10  6ed9843b57  [Kernel] Introduce sglang.kernels namespace and migrat
scheduler_components 2026-05-18  062f6f7ae8  Move get_draft_kv_pool to mem_cache.kv_cache_builder (
```

![Figure: srt's directories coloured by the period they were born in](../assets/figures/sgl-codebase-2026.svg){.aig-svg}

Four classes:

- **The skeleton settled in 2024**: `managers/`, `mem_cache/`, `model_executor/`, `layers/`, `models/`, `sampling/`, `constrained/` — [chapter 10](../perf/restructure.md)'s product, and still among the largest directories today.
- **The scaling of H1 2025**: `speculative/`, `disaggregation/`, `distributed/`, `eplb/`, `lora/`, `entrypoints/`, `function_call/`, `platforms/`.
- **The reliability and observability of H2 2025**: `compilation/` (where piecewise CUDA graphs and the torch.compile backend ended up, #11539 "Compilation Folder Reset"), `elastic_ep/` (failed ranks and elastic scaling), `checkpoint_engine/`, `debug_utils/` (tools for comparing internal activation tensors, #7976), `hardware_backend/` (each vendor's adaptation gathered, starting with the NPU restructuring of 2025-12).
- **The platform-building of 2026**: `arg_groups/` (several hundred startup arguments tidied into groups), `scheduler_components/` (splitting the nearly 6000-line `scheduler.py` further), `kv_canary/` (a "canary" consistency check in the KV cache), `observability/`, `weight_cache/` (fast engine recovery), `beam_search/`, `rust_extensions/` (Rust components compiled on demand, [chapter 21](gateway.md)) and `kernels/` (sgl-kernel moved into the main package, [chapter 14](../perf/sgl-kernel.md)).

## From features to reliability {#从功能到可靠性}

The Q3 2025 roadmap (issue #7736) is titled "compatibility and reliability", and the code corresponds:

```bash title="reliability-commits.sh"
REF=${REF:-29f6d408c0}
for h in b36afed4a7 4ac8e09df0 1801cd199f a40229f6f8 904655c5fd 932e263725 ed0fdbf35b 11391b2a1c 00d620b77d 062f6f7ae8; do
  git log -1 --date=short --format='%ad  %h  %s' $h | cut -c1-96
done
```

```text title="output"
2025-10-10  b36afed4a7  Separate allocation logic from scheduler (#11313)
2025-10-11  4ac8e09df0  Piecewise CUDA Graph Support & Torch Compile Backend (#10062)
2025-10-23  1801cd199f  support more model in piecewise cuda graph (#11745)
2025-10-15  a40229f6f8  [1/N] Introduce Mooncake Backend and Mooncake EP to Support Elastic EP (
2025-10-22  904655c5fd  [2/N] Added the core structure of elastic EP and the eplb algorithm with
2025-10-13  932e263725  Compilation Folder Reset (#11539)
2025-07-27  ed0fdbf35b  Tool to dump and compare internal activation tensors (#7976)
2026-05-31  11391b2a1c  Add the KV-canary core: data layer, MHA KV-pool patcher, and per-forward
2026-05-03  00d620b77d  introduce arg_groups/ with nemotron_h hook (#24328)
2026-05-18  062f6f7ae8  Move get_draft_kv_pool to mem_cache.kv_cache_builder (#25602)
```

- **Split the large files.** #11313 took the allocation logic out of the scheduler; `scheduler_components/` and the various mixins (output handling, PP, control messages) stop `scheduler.py` swelling.
- **Make CUDA graphs cover more shapes.** Piecewise CUDA graphs (from #10062) cut the forward pass into independently capturable segments so that prefill's non-attention parts can be graphed too; `compilation/` became the common home for that and torch.compile.
- **Let the system run through failures.** Elastic EP (#10423, the #11837 series) allows a rank in an EP group to drop out and allows scaling while running.
- **Make problems locatable.** `debug_utils/`'s tensor dumps and comparisons, `kv_canary/` planting check values in the KV pool, and `observability/`'s tidied metrics and tracing.
- **Make the arguments manageable.** `arg_groups/` groups `server_args.py`'s several hundred arguments by function, each group validating itself.

None of this work appears in a release blog's headline, yet it is the precondition for a thousand people changing one repository at once.

## The release rhythm and the contributors {#发布节奏与贡献者}

The v0.5 series ran from rc0 in August 2025 to v0.5.21 on 30 September 2026, 50 tags, 21 of them in 2026, about one a fortnight; the gateway has 12 more `gateway-v*` tags since July 2025. The contributors' composition changed too: in 2024 the first few authors made most of the commits ([chapter nine](../service/v02.md)'s `v02-tag.sh`), while among 2026's 1200-odd authors no one is more than a few percent. The roadmap went from an issue written by one person (#634 of Q3 2024) to several issues divided by direction (Diffusion's #12799 belongs to another maintainer).

## Design trade-offs {#设计取舍}

- **Splitting rather than rewriting.** The nearly 6000-line `scheduler.py` was not torn down and rebuilt but moved out piece by piece with mixins, components and separate modules — unlike the wholesale reorganisation of July 2024 ([chapter 10](../perf/restructure.md)), because a large repository cannot afford a global refactor.
- **Pluggability for extensibility.** The attention backends, the speculative methods, the eviction policies, the transfer backends, the hardware backends and the tool parsers are all a registry plus implementation files; the price is more concepts, and a newcomer has to learn the registries first.
- **More tests than code.** Over 2500 test files, CI triggered by directory, and parity tests between Python and Rust.
- **Instructions in the repository written for machines.** `multimodal_gen/` has `.agents` and `.claude` directories — project instructions for AI coding assistants, one of the new features of a large open-source project in 2026.

## What happened afterwards {#后来怎么样了}

This book's baseline stops at 2026-10-02. What came after you can run for yourself with [chapter 26](../method/archaeology.md)'s scripts: change the `REF` and every number updates.

## Exercises {#练习}

**1. The busiest directories.** Use `git log --since=2026-01-01 --format= --name-only "$REF" | grep '^python/sglang/srt/' | cut -d/ -f4 | sort | uniq -c | sort -rn | head` to find the srt subdirectories changed most in 2026, and compare with the ones holding the most files.

??? success "A way to approach it"
    The most changed are usually `layers/`, `models/`, `managers/` and `mem_cache/`; `models/` has many files but each changes little (a new model is added in one go) while `managers/` has few files changed often.

**2. One mixin's origin.** Take `managers/scheduler_pp_mixin.py` and use `git log --follow` to see when it was split out of `scheduler.py`, and what `scheduler.py`'s line count was before and after.

??? success "A way to approach it"
    Having found the splitting commit, compare `git show <commit>^:python/sglang/srt/managers/scheduler.py | wc -l` with `git show <commit>:... | wc -l`.

**3. The authorship.** Use `git shortlog -sn --since=2026-01-01 "$REF" | head -20` to see what fraction of the year's commits the top 20 authors make, and compare with 2024.

??? success "A way to approach it"
    In 2024 the top 3 made over half; in 2026 the top 20 together make less than half — the project went from being driven by a core group to being driven by a community.

!!! interview "How to explain it"
    "How do you get to grips with an inference framework of several hundred thousand lines quickly?" — Use this map: learn the skeleton directories settled in 2024 first (scheduling, caching, execution, layers, models), then see by year what each new directory solves (scaling → reliability → platform), and finally find the registries (backends, speculative methods, eviction policies) and follow them. Being able to state the engineering facts — more tests than code, large files split with mixins, arguments managed in groups — shows more judgement than reciting a feature list.

## Summary {#小结}

- [x] Three years, three orders of magnitude: 1600 commits and 190 authors in 2024 → 6800 and 800 in 2025 → 11 thousand and 1200 in the first nine months of 2026.
- [x] The directories fall into four classes by the period they were born in: 2024's skeleton, H1 2025's scaling, H2 2025's reliability and observability, and 2026's platform-building.
- [x] The roadmap turned from features to compatibility and reliability: splitting large files, widening CUDA graphs' coverage, elastic EP, debugging and consistency tools, and argument grouping.
