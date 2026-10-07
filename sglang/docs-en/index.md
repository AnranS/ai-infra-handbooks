# SGLang's design evolution

<p class="lead">In reading a large project's source, the hardest part is not understanding some function but working out why it looks the way it does. This book treats the SGLang repository's more than 19,000 commits as primary sources: starting from the 2023 paper and the first commit of January 2024, it goes through every key design in chronological order — the problem it answered, the commit that introduced it, what it replaced and how it evolved afterwards. Every chapter lands on specific commit ids, PR numbers and dates, and every command and number can be reproduced on your own clone.</p>

## What you will be able to do {#学完能做到}

- Take any directory in SGLang as it is today (`mem_cache/`, `disaggregation/`, `speculative/`, `sgl-router/` …) and say when it appeared, what problem it solved, and what its first version looked like.
- Explain RadixAttention, zero-overhead scheduling, PD disaggregation, large-scale expert parallelism and HiCache — the designs interviews always ask about — as "the problem → the first version → the trade-off → what came after", rather than reciting today's code.
- Command a method for doing source archaeology with git: commits per month, a path's first appearance, a keyword's first appearance, a module-by-quarter heat table, `git log -S`, `--follow`, and reading PRs and roadmap issues.
- Have a quantitative sense of how an industrial inference engine grows from ten thousand lines to over eight hundred thousand: which phases added features, which paid down technical debt, and which decisions ran through all of it.

## The route through {#学习路线}

This book is the sequel to [the inference-systems handbook](serving://)'s "reading the source: SGLang" and to [writing mini-sglang](minisgl://): the first reads today's code, the second reimplements today's modules, and this one tells how those modules grew one step at a time. It is in six parts, in chronological order:

<div class="roadmap" markdown>

| Part | Chapters | Goal | Suggested time |
| --- | --- | --- | --- |
| Origins (2023-10 → 2024-02) | [the paper](origins/paper.md) · [the first commit](origins/first-commit.md) · [RadixAttention's first version](origins/radix-v1.md) · [the compressed FSM and jump-forward decoding](origins/fsm-jump.md) · [the frontend language](origins/frontend.md) | understand the authors' original problem statement and every design in ten thousand lines of code | 2 days |
| From research code to a usable service (H1 2024) | [the process model's restructuring](service/processes.md) · [becoming a service](service/api-multimodal.md) · [borrowing from vLLM](service/borrow-vllm.md) · [v0.2](service/v02.md) | watch a research prototype turn into something deployable | 1 day |
| Performance engineering and zero-overhead scheduling (H2 2024) | [the big directory reorganisation](perf/restructure.md) · [MLA and torch.compile](perf/mla-compile.md) · [overlapped scheduling](perf/overlap.md) · [several cards](perf/multi-gpu.md) · [sgl-kernel](perf/sgl-kernel.md) | understand how "fast" was stacked up layer by layer | 2 days |
| Speculative decoding, PD disaggregation and large-scale EP (H1 2025) | [EAGLE](scale/eagle.md) · [HiCache](scale/hicache.md) · [PD disaggregation](scale/pd.md) · [large-scale EP](scale/large-ep.md) · [the attention backends](scale/attention-backends.md) | understand every new directory of the scaling phase | 2 days |
| From engine to platform (H2 2025 → 2026) | [the entrypoints layer](platform/entrypoints.md) · [the Rust gateway](platform/gateway.md) · [the RL loop](platform/rl.md) · [multimodal and diffusion](platform/multimodal-diffusion.md) · [a snapshot of 2026](platform/codebase-2026.md) | see what the engineering focuses on in a mature project | 1 day |
| Method and summary | [ten design decisions](method/principles.md) · [the git toolbox](method/archaeology.md) · [how to tell it in an interview](method/interview.md) | tell the evolution as a story, and dig for yourself | half a day |

</div>

The whole book's timeline first: the commits per month, the releases and the events, where clicking an event jumps to its chapter.

<div class="aig-widget" data-widget="sgl-timeline"></div>

## How this book works {#这本书的做法}

**Primary sources, a fixed baseline.** Every statistic is computed on a full clone of SGLang's official repository, with the baseline fixed at the `main` commit `29f6d408c0` of 2026-10-02 (the scripts in the book refer to it as `$REF`). As of that commit the repository has 19,247 commits and 165 tags; every number on a page comes from the command on that page. Change the `REF` and the numbers change, the method does not.

**Code is cut verbatim by commit id.** All historical code quoted in the book states its path, commit and line numbers, such as lines 113 to 140 of `python/sglang/srt/managers/router/radix_cache.py @ 22085081bb`. The checking script re-cuts it with `git show commit:path` and compares it character by character, so a quotation cannot drift through paraphrase. While reading, it helps to open that version in a local clone alongside:

```bash
git clone https://github.com/sgl-project/sglang.git ~/sglang-src
cd ~/sglang-src
git show 22085081bb:python/sglang/srt/managers/router/radix_cache.py | sed -n '113,140p'
```

**The official account as evidence.** For "what the authors were thinking at the time" at each stage, official material is quoted wherever possible: the paper, the LMSYS blog, roadmap issues, PR descriptions and commit messages. Where something is inferred, it says so.

**Each chapter's form.** A self-test at the head → a six-panel strip → the body (the problem at the time, the first version's code, the design trade-offs) → "what happened afterwards" → exercises (all archaeology questions answerable with git, with answers) → how to answer in an interview.

!!! note "Versions and naming"
    SGLang's directories were reorganised several times over three years. The book writes paths as they were at the time (the first version's `srt/managers/router/`, for instance) and gives today's equivalent at the end of each chapter. Commit ids are always the first 10 characters and PR numbers are written `#1234`; both open directly on GitHub: `https://github.com/sgl-project/sglang/commit/<commit id>` and `https://github.com/sgl-project/sglang/pull/<PR number>`.

## The code and running it {#代码与运行}

All of the scripts are in the repository's [`sglang/`](https://github.com/AnranS/ai-infra-handbooks/tree/main/sglang) directory:

<!-- i18n:diagram 096a4a53d6 -->
```text
sglang/
├── docs/                the text; each chapter's ```bash title="x.sh" / ```python title="x.py" blocks are its archaeology scripts
├── tools/check_code.py  re-runs every script on a local clone, re-cuts every quoted piece of code and compares it line by line with the page
└── build/code/          the script files written out while checking
```

How the checking works:

```bash
git clone https://github.com/sgl-project/sglang.git ~/sglang-src      # a full clone is about 400 MB (commands like git log -S read every commit, so do not use --filter=blob:none); add --no-checkout if you do not want a working tree
cd ai-infra-handbooks/sglang
python3 tools/check_code.py                      # run every page; SGLANG_SRC=/path points at the clone and REF=... changes the baseline commit
python3 tools/check_code.py docs/origins/radix-v1.md   # check one chapter only
```

The scripts use only `git` and Python's standard library: no GPU and no SGLang installation.
