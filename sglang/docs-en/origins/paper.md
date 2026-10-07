# From Chatbot Arena to a paper: the problem SGLang set out to solve

<p class="lead">SGLang was not born as "yet another inference engine". The paper of late 2023 defines it as a language plus a runtime: the language is for writing the "LM programs" made of several model calls, control flow and structured output, and the runtime uses those programs' structure (shared prefixes, fixed formats, parallel branches) to run them faster. This chapter reads the paper and the first two commits to establish the authors' original problem statement, then maps each of the paper's sections onto a directory of the initial code. Every evolution the later chapters describe starts from this point.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What is an "LM program" as the paper means it? Which two difficulties does it bring an inference system that traditional engines did not address?
    2. Which structure of an LM program does each of the runtime's three techniques exploit?
    3. Why is SGLang called a co-design of a frontend language and a runtime, rather than an engine with an API added afterwards?
    4. Which directories of the initial code correspond to which sections of the paper? What in the paper faded away later?

??? success "Answers for the self-test (answer first, then open this)"
    1. A program made of several LLM calls, Python control flow and structured input and output (a multi-turn conversation, an agent, a tree of thoughts, JSON extraction…). The two difficulties: it is tedious to write (string concatenation, prompt debugging, fragile output parsing, multimodal input, parallelism by hand), and it runs inefficiently (the engine sees only independent requests and does not know that they share prefixes or that the output has a fixed format).
    2. RadixAttention exploits the **shared prefixes** between calls (reusing the KV cache); the compressed finite-state machine exploits the **deterministic stretches** in structured output (decoding several tokens in one forward pass); API speculative execution exploits the **continuity** between calls to an API model (generating a few extra tokens for the calls that follow).
    3. The information the runtime needs (which requests share a prefix, what format the output must match, which branches can run in parallel) is known only at the program level; the language expresses it explicitly (`gen`'s `regex`, `fork`/`join`, `select`) and the runtime optimises on that basis. In the first commit the language (`lang/`) and the runtime (`srt/`) appear together, and the runtime's HTTP interface is designed around the language's needs (a request with `max_new_tokens=0` warms a prefix up).
    4. `lang/` is the frontend language with its interpreter and compiler; `srt/managers/router/radix_cache.py` and `scheduler.py` are RadixAttention and cache-aware scheduling; `srt/constrained/` is the compressed FSM; `backend/openai.py` is API speculative execution. What faded later is the frontend language and the compiler: the runtime became the main act and the frontend was reduced to an optional component in 2025.

![Figure: the four months from an empty repository to the first code](../assets/figures/sgl-origins-timeline.svg){.aig-svg}

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/paper.webp is in Chinese; put it back once the English version exists -->

## Four months: an empty repository, a paper, the code, a blog post {#四个月空仓库论文代码博客}

Look at the repository's first two commits with git. Every command in this book runs at the root of the SGLang repository, and `$REF` is the fixed baseline commit (see [the home page](../index.md)):

```bash title="first-commits.sh"
REF=${REF:-29f6d408c0}
git log --reverse --date=short --format='%ad  %h  %<(15)%an %s' "$REF" | head -4
```

```text title="output"
2023-10-09  f6d40df0ee  Ying Sheng      Initial commit
2024-01-08  22085081bb  Lianmin Zheng   release initial code
2024-01-09  ead5b39f82  Liangsheng Yin  Add flashinfer && Oultines (#1)
2024-01-08  93eeb543ba  Lianmin Zheng   Update readme.md
```

The first commit, of 9 October 2023, has three files: `.gitignore`, an Apache 2.0 `LICENSE` and a README of one line, `# sglang`. The actual code arrived all at once three months later on 8 January 2024 (`22085081bb`, "release initial code"), with the paper's first version going up on arXiv (2312.07104) on 12 December in between, LMSYS publishing an introductory blog post on 17 January, and the first tag `v0.1.5` cut the same day (the earlier `v0.1.3` on the 16th). So the code did not grow bit by bit in a public repository: it was written internally until it could run the paper's experiments, then released whole. The first code release's size:

```bash title="release-size.sh"
git show --stat=10 --format= 22085081bb | tail -1
git ls-tree -r --name-only 22085081bb | cut -d/ -f1 | sort | uniq -c | sort -rn
```

```text title="output"
 145 files changed, 17802 insertions(+), 2 deletions(-)
     56 benchmark
     52 python
     18 test
     11 examples
      2 playground
      2 docs
      1 format.sh
      1 README.md
      1 LICENSE
      1 .gitmodules
      1 .gitignore
```

145 files and 17.8 thousand lines, of which `python/`'s 52 files are the entire implementation, `benchmark/`'s 56 files are the paper's experiments (MMLU, HellaSwag, generative agents, a tree of thoughts, JSON decoding, multi-turn conversation, LLaVA… each directory with a `bench_sglang.py` and a `bench_other.py`), and `test/` has 18. That ratio says where the project came from: it was written to back up a paper's claims.

## The paper's problem statement: LM programs {#论文的问题定义lm-程序}

The paper (v2's title is *SGLang: Efficient Execution of Structured Language Model Programs*, v1 was *Efficiently Programming Large Language Models using SGLang*, and the initial README cites v1's title) opens by defining an "LM program":

> Applications are going from a single call to programs made of **several LLM calls, control flow and structured input and output**.

The typical cases: a multi-turn conversation (every turn carrying the whole history), an agent (called round after round in a ReAct loop), a tree of thoughts or skeleton-of-thought (one prefix branching several ways), LLM-as-judge (the same question scored several times), JSON extraction (the output has to match a schema), few-shot evaluation (hundreds or thousands of questions sharing one block of examples). The paper names two difficulties:

| Difficulty | How it shows | The paper's answer |
| --- | --- | --- |
| Hard to **write** | string concatenation, prompts debugged over and over, fragile output parsing, multimodal input, parallelism implemented by hand | a frontend language embedded in Python: `gen`, `select`, `fork`/`join`, `image`, role markers |
| Slow to **run** | the engine optimises throughput and latency per independent request and does not know that requests share prefixes or that the output has a fixed format | the runtime's three pieces: RadixAttention, the compressed FSM, API speculative execution |

That definition is what separates SGLang from the contemporary vLLM: vLLM starts from "how is one request served more efficiently" (PagedAttention solving KV memory fragmentation) while SGLang starts from "how does a set of structured requests run faster". The two absorbed each other later (vLLM added prefix caching, SGLang added paging), but their starting points differ, and many early designs only make sense when put back at this starting point.

## The runtime's three ideas {#运行时的三个想法}

**RadixAttention: treat the KV cache as a tree.** The conventional approach frees the KV when a request finishes; the paper organises every request's token sequence into a radix tree whose nodes hold the KV cache's location in memory. A new request first matches its longest prefix on the tree, and the part that hits is not computed again. Several accompanying decisions come straight from the "LM program" setting:

- Nodes are evicted LRU by last access time, leaves first, and a node referenced by a running request (a reference count above 0) cannot be evicted.
- Scheduling prefers the request with the **longest matching prefix** (longest-prefix-first), and the paper proves that processing a batch of requests in the tree's depth-first order achieves the optimal hit rate (theorem 3.1, provided the cache is at least as large as the longest request).
- Under multi-card data parallelism, the router keeps a "meta tree" recording each worker's subtree and sends a request to the worker whose prefix matches most.

The paper reports hit rates of 50% to 99% across the benchmarks; on LMSYS's own Chatbot Arena production traffic, LLaVA-NeXT-34B's hit rate is 52.4% and Vicuna-33B's 74.1%, with the first token's latency 1.7 times lower on average. The CPU overhead of managing the tree is small: 100 requests with no reuse take 74.3 seconds in all, of which the tree's operations are 0.2. [The next chapter](first-commit.md) shows that the first version's KV pool is **paged by token** (one token per page), exactly so that the tree can be cut at any point.

**A compressed finite-state machine: several tokens in one forward pass.** The usual way to do regex-constrained decoding (Outlines) compiles the regex into an FSM and allows only the tokens the FSM's current state accepts at each step. The paper observes that long stretches of a JSON schema are deterministic (key names, quotes, colons), corresponding to a run of states in the FSM with only one outgoing edge, and compresses such a run into a single edge that "jumps" over the whole deterministic string at once — the paper calls it jump-forward. The price is that the string jumped over has to be **retokenized** together with the text before it, or the token boundaries will not match what the model saw in training. This is [chapter four](fsm-jump.md)'s subject, and its later fate is the most winding of the three.

**API speculative execution: saving calls even to a closed model.** For an API-only model like OpenAI's, the runtime has one call generate a few extra tokens (ignoring the stop condition), and if the next `gen` happens to continue from there it is reused, saving one call's latency and input cost. The idea is implemented purely in the frontend and has nothing to do with the runtime.

## The frontend: two ways to execute {#前端两种执行方式}

The paper's frontend has two execution modes, both present in the initial code:

- **The interpreter**: each primitive in the program (`+=` text, `gen`, `select`) is submitted to a background thread and executed asynchronously, so the Python code runs on without waiting and blocks only when a variable is read — this is `StreamExecutor` in `lang/interpreter.py`.
- **The compiler and tracer**: run the program once with dummy arguments, record the graph the primitives form, optimise it and then execute. The paper's example optimisation is code movement: GPT-4 moves the sharable constant text to the front, which in 12 of 15 templates lengthened the sharable prefix by 60 tokens on average. This is `lang/tracer.py` and `lang/compiler.py`.

[Chapter five](frontend.md) reads that code. For now, remember one conclusion: one practical use of the tracer is to **extract a program's constant prefix** and send it to the runtime to warm the cache before a batch run.

## The evaluation tells you what the authors cared about {#评估告诉我们作者在乎什么}

The paper uses Llama-7B on an A10G (24 GB), tensor parallelism for the larger models, and compares against Guidance v0.1.8, vLLM v0.2.5 and LMQL v0.7.3. The list of workloads is telling: 5-shot MMLU, 20-shot HellaSwag, a ReAct agent, generative agents, a tree of thoughts (GSM-8K), skeleton-of-thought, LLM-as-judge, JSON decoding, multi-turn conversation, DSPy RAG, LLaVA image and video tasks, plus ShareGPT as an "unstructured" control. The headline numbers: up to 6.4 times the throughput and up to 3.7 times lower latency; the compressed FSM raises JSON decoding's throughput 1.6 times, provided the FSM's preprocessing can be amortised over a batch of requests — otherwise it is 2.4 times slower instead.

Those workloads all became directory names under `benchmark/` in the repository later, and they explain much of the first version's lopsidedness: no preemption, no chunked prefill, a scheduler of only 70 lines, but prefix matching, tree eviction, regex constraints, the normalised log probabilities `select` uses and caching for multimodal input, every one of them present — because the paper's experiments needed them.

## Mapping the paper onto the code {#把论文对应到代码}

A script counts the size of each module in the first code release (Python files and lines):

```python title="paper-vs-code.py"
import collections, subprocess
REF0 = "22085081bb"
files = subprocess.run(["git", "ls-tree", "-r", "--name-only", REF0, "--", "python/sglang"], capture_output=True, text=True).stdout.split()
GROUPS = [("lang/ 前端语言", "python/sglang/lang/"), ("backend/ 外部后端", "python/sglang/backend/"),
          ("srt/managers/router/ 调度与执行", "python/sglang/srt/managers/router/"), ("srt/managers/ 分词与进程", "python/sglang/srt/managers/"),
          ("srt/constrained/ 约束解码", "python/sglang/srt/constrained/"), ("srt/layers/ 注意力算子", "python/sglang/srt/layers/"),
          ("srt/models/ 模型", "python/sglang/srt/models/"), ("srt/ 其他（内存池、参数、服务）", "python/sglang/srt/"), ("顶层（api、test、utils）", "python/sglang/")]
count = collections.Counter(); lines = collections.Counter()
for f in files:
    if not f.endswith(".py"):
        continue
    n = subprocess.run(["git", "show", f"{REF0}:{f}"], capture_output=True, text=True).stdout.count("\n")
    group = next(name for name, prefix in GROUPS if f.startswith(prefix))
    count[group] += 1; lines[group] += n
print(f"{'模块':<34}{'文件':>4}{'行数':>7}")
for name, _ in GROUPS:
    print(f"{name:<34}{count[name]:>4}{lines[name]:>7}")
print(f"{'合计':<34}{sum(count.values()):>4}{sum(lines.values()):>7}")
```

```text title="output"
模块                                  文件     行数
lang/ 前端语言                           6   1841
backend/ 外部后端                        7   1082
srt/managers/router/ 调度与执行           6   1645
srt/managers/ 分词与进程                  4    404
srt/constrained/ 约束解码                4   1278
srt/layers/ 注意力算子                    6   1190
srt/models/ 模型                       3    907
srt/ 其他（内存池、参数、服务）                   7    960
顶层（api、test、utils）                   8    906
合计                                  51  10213
```

Against the paper:

| Section of the paper | The initial code | Where it is now |
| --- | --- | --- |
| 2 the frontend language, interpreter and compiler | `lang/` (1655 lines), `api.py`, `backend/` | `python/sglang/lang/`, an optional component since 2025-08 |
| 3.1 RadixAttention and cache-aware scheduling | `srt/managers/router/radix_cache.py`, `scheduler.py`, `srt/memory_pool.py`, `srt/layers/radix_attention.py` | `srt/mem_cache/`, `srt/managers/schedule_policy.py`, `srt/layers/radix_attention.py` |
| 3.2 the compressed FSM | `srt/constrained/` (FSM code adapted from Outlines) | `srt/constrained/` with several backends (Outlines, xgrammar, llguidance) |
| 3.3 API speculative execution | `backend/openai.py` (#48, added 2024-01-25) | `python/sglang/lang/backend/openai.py` |
| 5 the evaluation | 14 directories under `benchmark/` | `benchmark/`, most of the directories still there |

The runtime's three kinds of process (tokenizing, routing and scheduling, detokenizing) and the ZMQ communication between them are in the first code release too, which is [the next chapter](first-commit.md)'s material.

## The design trade-off: why co-design {#设计取舍为什么协同设计}

Putting the language and the runtime in one repository, rather than doing only a frontend like LangChain or only an engine like vLLM, is the first version's largest decision. It brings three things:

1. **Information crosses the layers.** The frontend knows which text is a constant prefix (the tracer), which `gen` has a format constraint (`regex`) and which branches share a context (`fork`), and all of it becomes a signal the runtime can use. The first version's HTTP interface has a request with `max_new_tokens=0` (warm the cache without generating) and `/concate_and_append_request` (splice a forked branch's KV back into the trunk), both interfaces in the frontend's service.
2. **The unit of measurement is a program, not a request.** Every number in the paper is "how long one LM program takes to finish", and throughput and latency are defined to follow the program. That gives a technique like RadixAttention, which only pays when there is shared structure, the stage it needs.
3. **The price is maintaining the frontend.** A language needs an interpreter, a tracer, a compiler, several backends and chat templates; once the runtime was what users actually wanted, that part became a burden.

!!! note "The authors' own words"
    The initial README defines SGLang as "a structured generation language designed for large language models… making interactions faster and more controllable by co-designing the frontend language and the runtime", and notes at the end that it "learned from the design of Guidance, vLLM and LightLLM and reused some of their code". The README's roadmap lists five items: function call, constrained decoding, quantization, S-LoRA, more models — and constrained decoding already existed in the code in its regex form, so what is meant here is fuller JSON and grammar support.

## What happened afterwards {#后来怎么样了}

- **The definition changed.** At the baseline commit `29f6d408c0`, the README's first sentence is "SGLang is an open-source inference framework for LLMs and multimodal models, optimized for agentic workloads, RL rollouts, and large-scale serving". The "language" is gone and "inference framework" is the identity; but "agentic workloads" is still a continuation of the paper's "LM programs".
- **The frontend faded.** "Simplify frontend language (#9029)" of 2025-08-10 moved `api.py` into `lang/` and made the frontend an optional install; the `lang/` directory has around 160 commits over three years while `srt/managers/` has more than 2700.
- **The three ideas fared differently.** RadixAttention became the whole system's foundation and grew into a tiered cache and distributed storage ([chapter 16](../scale/hicache.md)); the compressed FSM's jump-forward was deleted from the scheduler in 2025-03 and the functionality moved inside grammar libraries like xgrammar ([chapter four](fsm-jump.md)); API speculative execution stayed in the frontend and has barely been touched since.
- **The unit of measurement went back to the request.** The later blog posts compare `bench_serving.py`'s throughput and latency, counted per request as vLLM does; evaluation from the "program" point of view faded along with the frontend.

## Exercises {#练习}

**1. The gap between the paper and the code.** Use git to find the number of commits and the authors between the first code release and the first tag, and judge whether "written internally first, then released" holds up.

??? success "Answer"
    `git log --reverse --date=short --format='%ad %h %an %s' v0.1.3`: between `22085081bb` and `v0.1.3` (2024-01-16) there are only a dozen or so commits, by Lianmin Zheng, Liangsheng Yin, Ying Sheng and the paper's other authors, covering the README, the installation, a few examples and `fix radix cache match (#7)`. The core code arrived complete in one go and public iteration followed, so the claim holds.

**2. What the paper's benchmarks left in the repository.** List the subdirectories of `benchmark/` in the first code release, then see which are still there at the baseline commit and which are gone.

??? success "Answer"
    `git ls-tree --name-only 22085081bb benchmark/` gives 14 directories (dspy, generative_agents, gsm8k, hellaswag, latency_throughput, line_retrieval, llava_bench, llm_judge, long_json_decode, mmlu, mtbench, multi_chain_reasoning, react, tree_of_thought and so on); in `git ls-tree --name-only 29f6d408c0 benchmark/` most of them are still there, with kernels, deepseek, gpt-oss and other directories tied to kernels and particular models added. The benchmarks clearly extended from "LM programs" to "operators and models".

**3. The positioning's change, read from the README.** Use `git log -S` to find the commit where the README's first sentence went from "structured generation language" to "fast serving framework", and read that commit's message.

??? success "A way to approach it"
    `git log --date=short --format='%ad %h %s' -S'structured generation language' -- README.md` lists the commits that introduced and removed the sentence; the last one to remove it is when the positioning changed (mid-2024, close to the v0.2 release). Reading its diff shows the new self-description.

!!! interview "How to answer in an interview"
    "How do SGLang's and vLLM's design starting points differ?" — Give the starting points first: vLLM starts from one request's memory efficiency (PagedAttention) and SGLang from a set of structured requests (LM programs: shared prefixes, fixed formats, parallel branches), which is why it had a radix-tree prefix cache, per-token paging and regex-constrained decoding on day one. Then the convergence: each later absorbed the other's core (vLLM's prefix caching, SGLang's paging and preemption), and today the difference is more in the engineering details than in the philosophy. Being able to tell "different starting points, similar destinations" is more convincing than reciting a feature list.

## Summary {#小结}

- [x] SGLang was born out of a paper: it takes the "LM program" made of several calls, control flow and structured output as the thing to optimise, with the language and the runtime co-designed.
- [x] Each of the runtime's three ideas matches one structure of a program: RadixAttention the shared prefix, the compressed FSM the deterministic stretch, API speculative execution the continuity between calls.
- [x] The code was released all at once on 2024-01-08 (145 files, 17.8 thousand lines), and every section of the paper has its counterpart in the initial directories.
- [x] The runtime became the main act later and the frontend faded; of the three ideas RadixAttention grew throughout, jump-forward left the scheduler, and API speculative execution stayed exactly where it was.
