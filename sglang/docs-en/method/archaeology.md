# Doing source archaeology yourself: this book's git toolbox

<p class="lead">Every conclusion in this book comes from a few git commands. This chapter gathers them into a reusable method: how to pin a baseline, how to count commits, how to find when a directory or a concept first appeared, how to follow a file through a rename, how to search for a string among 19,000 commits, and how to read PRs and roadmaps. Every script runs on your own clone, and changing the <code>REF</code> updates this book's numbers to today — or change the repository and go digging in another project.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why pin a baseline commit? Why does this book not use a tag as the baseline?
    2. What is the pitfall in finding "the first appearance" with `--diff-filter=A`? How do you work around it?
    3. How do `git log -S` and `git log -G` differ? When do you use each?
    4. How far apart can the PR number in a commit's title and the PR's actual time be? How do you tell when a feature was started and when it merged?

??? success "Answers for the self-test (answer first, then open this)"
    1. `main` moves every day, and without a pinned baseline no number is reproducible; this book uses `29f6d408c0` (main at 2026-10-02). A tag will not do because SGLang's release tags are cut on release branches and are not necessarily ancestors of main (`git merge-base --is-ancestor v0.5.21 main` returns false), so counting commits from a tag misses part of main.
    2. When a file enters a new path by a rename, `--diff-filter=A` does not see it (it is an R), so "the first appearance" comes out empty or later than the truth. The remedies: add `-M` (detect renames) and use `--diff-filter=AR`, or use `git log --follow` for a single file, or query the first appearance by directory (a directory-level `git log -- dir/` is unaffected by renames).
    3. `-S` finds the commits where a string's number of occurrences changed (additions and deletions count, moving it within a file does not), which suits finding an identifier's introduction and removal; `-G` matches a regex against each commit's diff, which is broader and slower and suits finding every commit that touched some pattern.
    4. A PR number is assigned when the PR is opened and only enters the history when it merges, so a large feature can be months apart (EAGLE's #2150 was opened in 2024-11 and merged 2025-01-02; HiCache's #2693 opened in 2024-12 and merged 2025-02-23). "Started" comes from the dates of the commits with neighbouring PR numbers, or the PR page's creation time; "merged" from the commit date; and the difference is the development period.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/archaeology.webp is in Chinese; put it back once the English version exists -->

## Pinning the baseline {#固定基准}

```bash title="pin-ref.sh"
REF=${REF:-29f6d408c0}
echo "基准：$(git log -1 --date=short --format='%H %ad' "$REF")"
echo "到基准为止的提交数：$(git rev-list --count "$REF")，tag 数：$(git tag | wc -l)"
echo "v0.5.21 是 main 的祖先吗：$(git merge-base --is-ancestor v0.5.21 "$REF" && echo 是 || echo 否)"
echo "第一个提交：$(git log --reverse --date=short --format='%ad %h %an %s' "$REF" | head -1 | cut -c1-70)"
```

```text title="output"
基准：29f6d408c01c457113cbecd8773aff64398fb7c2 2026-10-02
到基准为止的提交数：19247，tag 数：165
v0.5.21 是 main 的祖先吗：否
第一个提交：2023-10-09 f6d40df0ee Ying Sheng Initial commit
```

Every script in this book opens with `REF=${REF:-29f6d408c0}`: the baseline unless you pass one. The checking script (`sglang/tools/check_code.py`) re-runs each script on a clone and compares the output line by line; a quoted code block states its source as `title="path @ commit L<start>-<end>"`, and the script re-cuts it by commit id to check.

## Counting commits: by month, by year, by author {#数提交按月按年按作者}

```python title="by-month.py"
import collections, os, subprocess
REF = os.environ.get("REF", "29f6d408c0")
dates = subprocess.run(["git", "log", "--date=short", "--format=%ad", REF], capture_output=True, text=True).stdout.split()
by_month = collections.Counter(d[:7] for d in dates)
by_year = collections.Counter(d[:4] for d in dates)
print("按年：", dict(sorted(by_year.items())))
peak = max(by_month.items(), key=lambda kv: kv[1])
print(f"按月：共 {len(by_month)} 个月，最多的是 {peak[0]}（{peak[1]} 个）")
for m in ("2024-01", "2024-07", "2025-01", "2025-07", "2026-01", "2026-07"):
    print(f"  {m}: {by_month.get(m, 0):5d} {'#' * (by_month.get(m, 0) // 40)}")
```

```text title="output"
按年： {'2023': 1, '2024': 1619, '2025': 6796, '2026': 10831}
按月：共 35 个月，最多的是 2026-09（1642 个）
  2024-01:    88 ##
  2024-07:   223 #####
  2025-01:   305 #######
  2025-07:   433 ##########
  2026-01:   954 #######################
  2026-07:  1238 ##############################
```

`git shortlog -sn A..B` ranks the authors between two tags ([chapter nine](../service/v02.md) used it); for commits and authors by year this book uses `--date=short --format='%ad %aN'` and groups by the date's prefix ([chapter 24](../platform/codebase-2026.md) used it) rather than `--since` / `--until`: those are interpreted in the local time zone, and with a date and no time they mean "the current moment on that day" (`--since=2025-01-01` gives different results in the morning and in the evening), so the numbers drift with when you run them. Mind the merge style: SGLang squash-merges, one commit per PR, so "the commit count" is about "the PR count"; a repository that rebase-merges needs `--first-parent` first.

## Finding the first appearance {#找第一次出现}

A directory's or a file's first appearance:

```bash title="first-appearance.sh"
REF=${REF:-29f6d408c0}
first() { git log --reverse --date=short --format='%ad %h %s' "$REF" -- "$1" | head -1 | cut -c1-80; }
printf '%-46s %s\n' python/sglang/srt/mem_cache/ "$(first python/sglang/srt/mem_cache)"
printf '%-46s %s\n' python/sglang/srt/disaggregation/ "$(first python/sglang/srt/disaggregation)"
printf '%-46s %s\n' python/sglang/srt/mem_cache/radix_cache.py "$(first python/sglang/srt/mem_cache/radix_cache.py)"
echo "-- 用 --follow 追踪 radix_cache.py 的改名："
git log --follow --reverse --date=short --format='%ad %h %s' "$REF" -- python/sglang/srt/mem_cache/radix_cache.py | head -1 | cut -c1-80
```

```text title="output"
python/sglang/srt/mem_cache/                   2024-07-29 cdcbde5fc3 Code structure refactor (#807)
python/sglang/srt/disaggregation/              2025-03-21 c7c7dbebbe [PD] Release initial code (#4654)
python/sglang/srt/mem_cache/radix_cache.py     2024-07-29 cdcbde5fc3 Code structure refactor (#807)
-- 用 --follow 追踪 radix_cache.py 的改名：
2024-07-29 cdcbde5fc3 Code structure refactor (#807)
```

A directory-level `git log -- dir/` is unaffected by renames and is the most reliable "when did this directory appear"; a single file needs `--follow` — `radix_cache.py`'s first commit inside `mem_cache/` is the directory restructuring of 2024-07-29, and only `--follow` traces it back to the first code release of 2024-01-08. The analysis scripts written for this book started by scanning every commit with `--diff-filter=A`, and a dozen or so files that entered a new path by a rename showed as "never appeared" — which is the pitfall of self-test question 2 above.

A concept's first appearance (searching the commit titles for a keyword):

```bash title="keyword-first.sh"
REF=${REF:-29f6d408c0}
for kw in 'chunk(ed)? prefill' 'overlap' 'xgrammar' 'disagg' 'deepep' 'eplb' 'diffusion'; do
  printf '%-22s %s\n' "$kw" "$(git log --reverse --date=short --format='%ad %h %s' "$REF" | grep -iE "$kw" | head -1 | cut -c1-78)"
done
```

```text title="output"
chunk(ed)? prefill     2024-07-29 2ec39ab712 Chunked prefill support (#797)
overlap                2024-10-16 dbec2f1847 Launch a thread to overlap CPU and GPU (#1687)
xgrammar               2024-10-26 b77a02cdfd [Performance] Support both xgrammar and outlines for con
disagg                 2025-04-11 c35dcfdb30 [PD] fix: skip warmup request in disaggregation mode to 
deepep                 2025-03-19 f44db16c8e [Feature] Integrate DeepEP into SGLang (#4232)
eplb                   2025-05-20 f0653886a5 Expert distribution recording without overhead for EPLB 
diffusion              2025-11-07 32f7982800 sglang diffusion announcement (#12856)
```

Searching the titles is cheap and good enough, but it only works for a concept whose name is stable; searching the code is more accurate:

```bash title="pickaxe.sh"
REF=${REF:-29f6d408c0}
echo "-- -S：引入和删除 'import rpyc' 的提交"
git log --reverse --date=short --format='%ad %h %s' -S'import rpyc' "$REF" -- python/sglang/srt | cut -c1-80
echo "-- -S：future_token_ids_map 第一次出现"
git log --reverse --date=short --format='%ad %h %s' -S'future_token_ids_map' "$REF" | head -1 | cut -c1-80
```

```text title="output"
-- -S：引入和删除 'import rpyc' 的提交
2024-01-08 22085081bb release initial code
2024-05-27 0463f7fb52 Support data parallelism (static) (#480)
2024-07-18 d774acad5c Remove the dependency of rpyc (#646)
-- -S：future_token_ids_map 第一次出现
2024-10-16 dbec2f1847 Launch a thread to overlap CPU and GPU (#1687)
```

`-S`'s output is usually two or three lines — introduced, (possibly) moved, removed — a concept's life.

## Looking at one commit, one version {#看一个提交一个版本}

```bash title="inspect.sh"
echo "-- 一个提交改了哪些文件（-M 识别改名）："
git show --stat=90 -M --format='%ad %an %s' --date=short cdcbde5fc3 | sed -n '1p;/=>/p' | head -8 | cut -c1-90
echo "-- 某个版本的某个文件有多少行："
printf '%s\n' "$(git show v0.2.0:python/sglang/srt/managers/controller/tp_worker.py | wc -l) 行（tp_worker.py @ v0.2.0）"
echo "-- 两个版本之间某目录的提交数："
echo "$(git rev-list --count v0.4.0..v0.4.6 -- python/sglang/srt/mem_cache) 个（mem_cache/，v0.4.0 → v0.4.6）"
```

```text title="output"
-- 一个提交改了哪些文件（-M 识别改名）：
2024-07-29 Liangsheng Yin Code structure refactor (#807)
 .../{controller/manager_multi.py => controller_multi.py}  |  2 +-
 .../manager_single.py => controller_single.py}            |  2 +-
 .../schedule_heuristic.py => policy_scheduler.py}         | 24 +++++-----
 .../{controller/infer_batch.py => schedule_batch.py}      |  4 +-
 python/sglang/srt/managers/{controller => }/tp_worker.py  | 26 +++++------
 python/sglang/srt/{ => mem_cache}/flush_cache.py          |  2 +-
 python/sglang/srt/{ => mem_cache}/memory_pool.py          |  0
-- 某个版本的某个文件有多少行：
800 行（tp_worker.py @ v0.2.0）
-- 两个版本之间某目录的提交数：
46 个（mem_cache/，v0.4.0 → v0.4.6）
```

`git show commit:path` reads a file at any version without a checkout, which is how every piece of code in this book was cut; `git show --stat -M` shows what a restructuring moved (`=>` is a rename); and `git rev-list --count A..B -- path` counts a module's activity over a period. [Chapter three](../origins/radix-v1.md) used `git show <commit> -- path` to read a bug fix's diff, and [chapter 24](../platform/codebase-2026.md) used `git log --format= --name-only` to count each directory's changes.

## A module-by-time heat table {#模块--时间的热力表}

```python title="module-quarter.py"
import collections, os, re, subprocess
REF = os.environ.get("REF", "29f6d408c0")
MODS = [("managers", r"^python/sglang/srt/managers/"), ("mem_cache", r"^python/sglang/srt/mem_cache/"), ("layers/attention", r"^python/sglang/srt/layers/attention"),
        ("speculative", r"^python/sglang/srt/speculative/"), ("disaggregation", r"^python/sglang/srt/disaggregation/"), ("sgl-kernel/kernels", r"^sgl-kernel/|^python/sglang/kernels/"),
        ("router/gateway", r"^rust/|^sgl-router/|^sgl-model-gateway/"), ("multimodal_gen", r"^python/sglang/multimodal_gen/")]
pats = [(n, re.compile(p)) for n, p in MODS]
table, cur, touched = collections.defaultdict(collections.Counter), None, set()
out = subprocess.run(["git", "log", "--date=short", "--format=@%ad", "--name-only", REF], capture_output=True, text=True).stdout
for line in out.splitlines():
    if line.startswith("@"):
        for m in touched: table[cur][m] += 1
        d = line[1:]; cur = f"{d[:4]}Q{(int(d[5:7]) - 1) // 3 + 1}"; touched = set()
    elif line.strip():
        for n, p in pats:
            if p.search(line): touched.add(n)
for m in touched: table[cur][m] += 1
qs = sorted(q for q in table if q >= "2024Q3")
print("模块 \\ 季度".ljust(20) + "".join(q.rjust(7) for q in qs))
for n, _ in MODS:
    print(n.ljust(20) + "".join(str(table[q][n] or "·").rjust(7) for q in qs))
```

```text title="output"
模块 \ 季度              2024Q3 2024Q4 2025Q1 2025Q2 2025Q3 2025Q4 2026Q1 2026Q2 2026Q3 2026Q4
managers                184    193    141    207    243    339    276    499    516      4
mem_cache                18     21     31     43    110    123    109    269    500     14
layers/attention         13     40     75     88    103    158    173    291    481     15
speculative               ·      1     36     38     40     81     41    227    268      4
disaggregation            ·      ·      2     95     51     78    107    181    278      6
sgl-kernel/kernels        ·     23    188    145    193    136     79     79    556     16
router/gateway            ·     53     14     17    213    410    148     29     95      4
multimodal_gen            ·      ·      ·      ·      ·    191    366    345    590      2
```

A commit counts once for a module if it touched a file under it. This table is the basis of this book's periodisation: whichever directory suddenly becomes active in a quarter, go to that quarter for the key commits. `git log --name-only` over every commit takes about a minute — it needs only the tree objects, so a `--filter=blob:none` clone can run it.

## Reading what people wrote {#读人写的东西}

Commands can only tell you when and what changed; the why has to be read from what people wrote:

- **Commit messages and PR descriptions**: `git show commit --format=%B -s` for the full message; the PR page `https://github.com/sgl-project/sglang/pull/<number>` has the discussion and the review. This book estimates development periods from the gap between a PR's opening and its merge.
- **Roadmap issues**: SGLang has one a quarter (#634, #1487, #4042, #7736, #12799…) listing the planned features and their owners, the best source for "what they meant to do at the time"; against the commit history it shows what went to plan and what was deferred or dropped.
- **Blog posts and the paper**: a feature's motivation and numbers (the LMSYS blog posts and the paper this book cites), remembering that they are the account at release time and that the code is the authority on details.
- **File headers**: a comment like `Adapted from ...` tells you where the code came from ([chapter eight](../service/borrow-vllm.md) counted 129 of them).

## Applying the method to another repository {#把方法用到别的仓库}

None of this depends on SGLang. Changing repository means changing three things: the baseline commit, the path patterns in `MODS` and the keyword list. vLLM, TensorRT-LLM and llama.cpp can all have their own heat tables and first-appearance lists drawn with the same scripts — [the inference-systems handbook's source reading](serving://source/vllm/) covers vLLM's structure today, and this method fills in its history.

![Figure: this book's archaeology workflow](../assets/figures/sgl-archaeology.svg){.aig-svg}

## Exercises {#练习}

**1. Update this book.** Change `REF` to today's `main` (after a `git fetch`), rerun `by-month.py` and `module-quarter.py`, and see which module is busiest after Q4 2026.

??? success "A way to approach it"
    `REF=origin/main python3 by-month.py`; the heat table gains new quarter columns.

**2. Dig into one directory.** Take `python/sglang/srt/lora/`: its first appearance, the quarter with the most commits, the largest single change (by `--shortstat`'s insertions and deletions) and today's file count.

??? success "A way to approach it"
    `git log --reverse ... -- python/sglang/srt/lora | head -1`; `git log --format='%ad' --date=short -- dir | cut -c1-7 | sort | uniq -c`; `git log --shortstat --format='%h %ad %s' -- dir` to find the largest.

**3. Change repository.** Clone vLLM, change `MODS` to `vllm/v1/core/`, `vllm/v1/worker/`, `vllm/attention/` and `csrc/`, draw its heat table, and find the commit where the V1 scheduler (`vllm/v1/core/sched/`) first appeared.

??? success "A way to approach it"
    The method is the same; note that vLLM's merge style and tagging may differ, so `--first-parent` and "is the tag on main" have to be checked again.

!!! interview "How to answer in an interview"
    "How do you read a large project's source?" — Give a procedure, not a feeling: pin a baseline → find the active periods from the commits per month → locate the directories with a module-by-quarter heat table → find the first appearance of the directories and the keywords → read the version of the day with `git show commit:path` → read the PRs, the roadmaps and the blog posts for the why. Then give one concrete conclusion you reached this way (that SGLang's jump-forward was deleted in 2025-03 and the feature moved into xgrammar, say), and the interviewer will know you have actually done it.

## Summary {#小结}

- [x] Pin a baseline commit (not a tag on a release branch) and every number is reproducible.
- [x] Four kinds of question, four kinds of command: counting commits (`log --format` plus counting), the first appearance (directory-level `log -- dir`, file-level `--follow`, grepping the titles, `-S`), looking at one version (`show commit:path`, `--stat -M`) and the module-by-time heat table (`--name-only`).
- [x] The why has to be read from what people wrote: commit messages, PRs, roadmap issues, blog posts and file headers; and the method carries to any repository.
