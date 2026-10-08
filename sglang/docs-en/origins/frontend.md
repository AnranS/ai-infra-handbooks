# The frontend language: the interpreter, the tracer and the compiler

<p class="lead">SGLang has "Lang" in its name. The first version's <code>lang/</code> directory is 1655 lines: a program is defined with a Python decorator, the primitives <code>gen</code>, <code>select</code> and <code>fork</code> form an IR, the interpreter executes asynchronously on a background thread, the tracer runs the program once with dummy arguments to extract the constant prefix, and the compiler topologically sorts the traced graph and executes it. This chapter reads that code and works out the few interfaces to the runtime that exist for the frontend's sake (the <code>max_new_tokens=0</code> warm-up request, <code>select</code>'s normalised log probabilities, the <code>concate_and_append</code> that splices a branch's KV back into the trunk), then looks at why it was reduced to an optional component in 2025.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What does `s += gen("answer")` do inside a function decorated with `@sgl.function`? When does `s["answer"]` block?
    2. How is `select` implemented on the runtime side? Why send a `max_new_tokens=0` request first?
    3. What is the tracer for? Why call `pin_program` before `run_batch`?
    4. What happened to the frontend in 2025? Is it still there today?

??? success "Answers for the self-test (answer first, then open this)"
    1. `s` is a `ProgramState` and `+=` submits the expression to its `StreamExecutor`: it goes into a queue to be executed in order by a background thread (which calls the backend to generate), and the Python main thread returns immediately. `s["answer"]` calls `get_var` and waits on that variable's `threading.Event` until the background thread has written the generated result into `variables`.
    2. It first sends the current text to the runtime as a `max_new_tokens=0` request (prefill only, which puts the prefix into the cache and returns `prompt_tokens`), then sends "the text plus each choice" as a batch of requests with `return_normalized_logprob=True` and `normalized_logprob_start_len=prompt_tokens`, and takes the choice with the highest normalised log probability. The warm-up request makes sure the choices' common prefix is computed once and lets `start_len` be exactly the prefix's length.
    3. It executes the program with dummy arguments (`SglArgument(name, None)`) and records the sequence of primitives as a graph; `extract_prefix_by_tracing` takes the run of constant text at the graph's head as the "program prefix", and `pin_program` sends it to the runtime to warm up (`cache_prefix`), so that every instance of a batch run hits that prefix.
    4. `Simplify frontend language (#9029)` of 2025-08-10 moved `api.py` into `lang/`, removed some dependencies and made the frontend an optional install. The directory is still there (14 files), but there have been few commits over three years and the runtime has become the main act.

A six-panel strip first, then the chapter:

<!-- comic ../assets/comics/frontend.webp is in Chinese; put it back once the English version exists -->

## The primitives and the IR {#原语与-ir}

The public API is in `api.py`:

```python title="python/sglang/api.py @ 22085081bb L23-28,31-45"
def function(func: Callable):
    return SglFunction(func)


def set_default_backend(backend: BaseBackend):
    global_config.default_backend = backend
...
def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    dtype: Optional[type] = None,
    choices: Optional[List[str]] = None,
    regex: Optional[str] = None,
):
    if choices:
        return SglSelect(name, choices, temperature)
```

`gen(choices=...)` becomes an `SglSelect` outright and `gen(regex=...)` hands the regex to the runtime (which is how [the previous chapter](fsm-jump.md)'s constrained decoding is triggered). Every primitive is a subclass of `SglExpr` (`ir.py`): `SglConstantText`, `SglGen`, `SglSelect`, `SglImage`, `SglRoleBegin`/`SglRoleEnd`, `SglFork`, `SglVariable`… and expressions concatenate with `+` into an `SglExprList`. `@function` returns an `SglFunction`, which requires the first argument to be called `s`:

```python title="python/sglang/lang/ir.py @ 22085081bb L69-78,182-190"
class SglFunction:
    def __init__(self, func, bind_arguments=None):
        self.func = func
        self.bind_arguments = bind_arguments or {}
        self.pin_prefix_rid = None

        # Parse arguments
        argspec = inspect.getfullargspec(func)
        assert argspec.args[0] == "s", 'The first argument must be "s"'
        self.arg_names = argspec.args[1:]
...
    def __call__(self, *args, **kwargs):
        from sglang.lang.tracer import TracingScope

        tracing_scope = TracingScope.get_current_scope()
        if tracing_scope is None:
            return self.run(*args, **kwargs)
        else:
            kwargs["backend"] = tracing_scope.tracer_state.backend
            return self.trace(*args, **kwargs)
```

The branch in `__call__` is the key: inside a tracing scope (`TracingScope`), a call traces rather than executes — which lets one SGLang function both run directly and be traced as a subprogram of another.

![Figure: the frontend's four layers: the program, the IR, the executor, the backend](../assets/figures/sgl-frontend-stack.svg){.aig-svg}

## The interpreter: one background thread and a pile of events {#解释器一个后台线程一堆事件}

`run_program` creates a `StreamExecutor` and a `ProgramState` wrapping it, then calls the user's function directly. Every `s += ...` inside it ends at `submit`:

```python title="python/sglang/lang/interpreter.py @ 22085081bb L200-215" linenums="200"
    def submit(self, expr: SglExpr):
        if isinstance(expr, (SglGen, SglSelect, SglVarScopeBegin)):
            self.variable_event[expr.name] = threading.Event()
            if self.stream:
                self.stream_var_event[expr.name] = threading.Event()
        elif isinstance(expr, SglExprList):
            for e in expr.expr_list:
                if isinstance(e, (SglGen, SglSelect, SglVarScopeBegin)):
                    self.variable_event[e.name] = threading.Event()
                    if self.stream:
                        self.stream_var_event[e.name] = threading.Event()

        if self.use_thread:
            self.queue.put(expr)
        else:
            self._execute(expr)
```

Every expression that produces a variable (`gen`, `select`, a variable scope) registers a `threading.Event` first and then goes into the queue. The background thread `_thread_worker_func` takes them out in order: constant text is appended to `text_` directly, `gen` calls the backend to generate and `set`s the event, `select` calls the backend to score. The user's code blocks on the event when it reads a variable. This is the paper's "execute asynchronously, synchronise on demand": three `gen`s in a row are issued one after another as fast as they can be while the Python main thread has long since run ahead.

`fork` copies the executor:

```python title="python/sglang/lang/interpreter.py @ 22085081bb L232-256" linenums="232"
    def fork(self, number: int, position_ids_offset: Optional[List[int]] = None):
        if number > 1:
            self.submit(SglCommitLazy())
            self.sync()

        number = int(number)

        exes = [
            StreamExecutor(
                self.backend,
                self.arguments,
                self.default_sampling_para,
                self.chat_template,
                self.stream,
            )
            for _ in range(number)
        ]
        for i in range(number):
            exes[i].variables = dict(self.variables)
            exes[i].text_ = str(self.text_)
            exes[i].messages_ = list(self.messages_)
            exes[i].cur_role = self.cur_role
            exes[i].fork_start_text_pos = len(self.text_)

        return exes
```

Before forking it submits an `SglCommitLazy` and synchronises — making the backend prefill the current text once (`commit_lazy_operations` also sends a `max_new_tokens=0` request) so that every branch hits that prefix. Each branch has its own thread and queue, and `join` collects the variables the branches produced back into the trunk (`gather_variable`), or splices the branches' KV back into the trunk (`concate_and_append`).

`_execute_gen`'s non-streaming part is short:

```python title="python/sglang/lang/interpreter.py @ 22085081bb L347-359" linenums="347"
    def _execute_gen(self, expr: SglGen):
        sampling_params = self._resolve_sampling_params(expr.sampling_params)
        name = expr.name

        if not self.stream:
            comp, meta_info = self.backend.generate(
                self, sampling_params=sampling_params
            )
            self.text_ += comp

            self.variables[name] = comp
            self.meta_info[name] = meta_info
            self.variable_event[name].set()
```

## The backend: runtime interfaces tailored for the frontend {#后端为前端定制的运行时接口}

`backend/runtime_endpoint.py` is the adapter from the frontend to SRT, and it exposes several runtime interfaces only the frontend uses:

```python title="python/sglang/backend/runtime_endpoint.py @ 22085081bb L35-39,131-159,161-166"
    def cache_prefix(self, prefix_str: str):
        res = http_request(
            self.base_url + "/generate",
            json={"text": prefix_str, "sampling_params": {"max_new_tokens": 0}},
        )
...
    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
    ):
        assert temperature <= 1e-5

        # Cache common prefix
        data = {"text": s.text_, "sampling_params": {"max_new_tokens": 0}}
        self._add_images(s, data)
        res = http_request(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        prompt_len = res.json()["meta_info"]["prompt_tokens"]

        # Compute logprob
        data = {
            "text": [s.text_ + c for c in choices],
            "sampling_params": {"max_new_tokens": 0},
            "return_normalized_logprob": True,
            "normalized_logprob_start_len": prompt_len,
        }
        self._add_images(s, data)
        res = http_request(self.base_url + "/generate", json=data)
        assert res.status_code == 200
        logps = [r["meta_info"]["normalized_logprob"] for r in res.json()]

        decision = choices[np.argmax(logps)]
        return decision, logps
...
    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        res = http_request(
            self.base_url + "/concate_and_append_request",
            json={"src_rids": src_rids, "dst_rid": dst_rid},
        )
        assert res.status_code == 200
```

- `cache_prefix` and `commit_lazy_operations`: a `/generate` with `max_new_tokens=0` — prefill without generating, whose side effect is putting the prefix into the radix tree. This is the most direct example of "information the frontend knows passed to the runtime": the runtime needs no new interface at all, one parameter is enough.
- `select`: warm the common prefix up first to get `prompt_tokens`, then send each choice appended to it as a batch, requesting `return_normalized_logprob` and starting the count at `prompt_tokens` — the runtime's `normalized_logprob_start_len` parameter (and that "at least two tokens" special case in [chapter two](first-commit.md)'s admission logic) exists for it.
- `concatenate_and_append`: the `/concate_and_append_request` interface splices several branches' KV back into the trunk, the runtime support for the paper's "parallel branches then merge"; it requires the branches to start at the same point (`fork_start_text_pos == self_len`) and is controlled by `global_config.enable_parallel_encoding`.

For the OpenAI backend, `select` degenerates into several calls comparing log probabilities; `support speculative execution for openai API (#48)` of 2024-01-25 added the paper's API speculative execution: one `gen` asks for extra tokens and a later `gen` that continues from there reuses them directly.

## The tracer and the compiler {#追踪器与编译器}

The tracer reuses the interpreter's `ProgramState` interface, but its `_execute` does not execute: it only chains the expressions together (`prev_node`):

```python title="python/sglang/lang/tracer.py @ 22085081bb L32-53" linenums="32"
def extract_prefix_by_tracing(program, backend):
    # Create dummy arguments
    dummy_arguments = {name: SglArgument(name, None) for name in program.arg_names}
    arguments = dummy_arguments
    arguments.update(program.bind_arguments)

    # Trace
    tracer = TracerProgramState(backend, arguments, only_trace_prefix=True)
    try:
        with TracingScope(tracer):
            tracer.ret_value = program.func(tracer, **arguments)
    except StopTracing:
        pass

    # Run and cache prefix
    prefix = ""
    for expr in tracer.flatten_nodes():
        if isinstance(expr, SglConstantText):
            prefix += expr.value
        else:
            break
    return prefix
```

`extract_prefix_by_tracing` only traces up to the first non-constant expression (with `only_trace_prefix=True` a `gen` raises `StopTracing`) and concatenates the constant text before it into a prefix. `pin_program` calls it before a batch run:

```python title="python/sglang/lang/interpreter.py @ 22085081bb L126-136" linenums="126"
def pin_program(program, backend):
    if global_config.enable_prefix_sharing and program.pin_prefix_rid is None:
        # TODO: handle multiple backends
        from sglang.lang.tracer import extract_prefix_by_tracing

        prefix = extract_prefix_by_tracing(program, backend)
        if prefix and len(prefix) > 64:
            prefix_rid = backend.cache_prefix(prefix)
            program.pin_prefix_rid = prefix_rid
            return prefix_rid
    return None
```

A prefix is only worth warming up above 64 characters. `run_batch` runs every instance concurrently with a thread pool and each one's prefix hits — this is the mechanism behind the high hit rates in the paper's few-shot evaluations like MMLU and HellaSwag.

The compiler builds the fully traced chain into a graph, sorts it topologically, and submits it in order to (possibly several) executors:

```python title="python/sglang/lang/compiler.py @ 22085081bb L72-87,95-123"
    def topological_sort(self):
        prevd = {}
        cand = Queue()
        for x in self.nodes:
            prevd[x] = (x.prev_node is not None) + (x.source_node is not None)
            if prevd[x] == 0:
                cand.put(x)
        new_list = []
        while cand.qsize() > 0:
            head = cand.get()
            new_list.append(head)
            for x in head.next_nodes:
                prevd[x] -= 1
                if prevd[x] == 0:
                    cand.put(x)
        self.nodes = new_list
...
    def run_internal(
        self,
        backend,
        kwargs,
        default_sampling_para,
    ):
        stream_executor_ids = set([x.expr.pid for x in self.nodes])
        stream_executors = {}
        for x in stream_executor_ids:
            arguments = kwargs if x == self.last_node.expr.pid else {}
            stream_executors[x] = StreamExecutor(
                backend, arguments, default_sampling_para, None, False
            )
        for node in self.nodes:
            se_id = node.expr.pid
            expr = node.expr
            if isinstance(expr, SglVariable):
                # Make a copy for SglVariable
                expr = SglVariable(expr.name, expr.source)
                expr.source_stream_executor = stream_executors[
                    node.source_node.expr.pid
                ]
            elif isinstance(expr, SglArgument):
                # Substitute SglArgument
                expr = kwargs[expr.name]
            stream_executors[se_id].submit(expr)
        for stream_executor in stream_executors.values():
            stream_executor.end()
        return ProgramState(stream_executors[self.last_node.expr.pid])
```

Each branch produced by a `fork` has its own `pid`, and `run_internal` creates one `StreamExecutor` per `pid`; when an `SglVariable` refers to another branch's variable, it fetches the value across executors through `source_stream_executor`. This is the paper's "graph execution", but the paper's code-movement optimisation (having GPT-4 rearrange the templates) never entered the repository — all the compiler actually does is execute in dependency order.

## Design trade-offs {#设计取舍}

- **Threads rather than asyncio.** The user's function is ordinary synchronous Python and `+=` cannot be an `await`; so the executor uses a background thread and events, and streaming output polls the events through `text_iter`. The price is one thread per program instance (and per forked branch), so a batch run has as many threads as its concurrency. `Increase interpreter parallelism (#46)` of 2024-01-18 and `Reduce overhead when fork(1) (#375)` of 2024-04 both work on that price.
- **Add as little to the runtime's interface as possible.** Warming up uses `max_new_tokens=0` and `select` uses normalised log probabilities; only `concate_and_append` is a dedicated interface. The benefit is a runtime that stays general, the cost is `select` taking two round trips.
- **Tracing by executing with dummy arguments.** It does not parse the Python source, it simply runs it — which is simple, but if the program's control flow depends on an argument's value (`if len(question) > 100`), the tracing takes the wrong branch; `SglArgument.__format__` raises outright for the same reason, to stop an argument being formatted into a string where the tracing cannot see it.

## What happened afterwards {#后来怎么样了}

```bash title="lang-size.sh"
REF=${REF:-29f6d408c0}
lines() { git ls-tree -r --name-only "$1" -- "$2" | grep '\.py$' | while read -r f; do git show "$1:$f"; done | wc -l; }
echo "lang/ 行数：初版 $(lines 22085081bb python/sglang/lang)，今天 $(lines "$REF" python/sglang/lang)"
echo "srt/ 行数：初版 $(lines 22085081bb python/sglang/srt)，今天 $(lines "$REF" python/sglang/srt)"
for y in 2024 2025 2026; do
  echo "$y 年改动 lang/ 的提交：$(git log --date=short --format=%ad "$REF" -- python/sglang/lang python/sglang/api.py | grep -c "^$y")，改动 srt/managers/ 的：$(git log --date=short --format=%ad "$REF" -- python/sglang/srt/managers | grep -c "^$y")"
done
```

```text title="output"
lang/ 行数：初版 1841，今天 4650
srt/ 行数：初版 6384，今天 851793
2024 年改动 lang/ 的提交：111，改动 srt/managers/ 的：510
2025 年改动 lang/ 的提交：42，改动 srt/managers/ 的：930
2026 年改动 lang/ 的提交：18，改动 srt/managers/ 的：1296
```

The frontend's size has barely changed in three years while the runtime has grown more than a hundredfold. A few milestones in between:

- 2024-02 → 2024-05: backends for Gemini / VertexAI, Anthropic and LiteLLM, and several ways to score `choices` (`Feat: add alternative choices selection methods (#835)`, 2024-08).
- The v0.2 blog post of 2024-07-25 starts calling SGLang an inference engine centred on "SGLang Runtime (SRT)", with the frontend in second place.
- `Add generator-style run_batch function (#2513)` of 2025-01-06, the last large change to the batch interface.
- `Simplify frontend language (#9029)` of 2025-08-10: `api.py` moved into `lang/api.py`, `Engine` in `sglang/__init__.py` imported from `srt.entrypoints.engine` instead, and the frontend's dependencies removed from `all` — installing SGLang no longer means installing the frontend.

Today's `python/sglang/lang/` still has 14 files, `sglang.function`, `gen` and `select` still work, and the official documentation keeps it in a "Frontend Language" section alongside the OpenAI-compatible interface and the native `/generate`. But the new features (function calling, reasoning models' thinking mode, multimodal input of every kind) are all built in the runtime and the gateway, and the frontend is only kept working.

## Exercises {#练习}

**1. How asynchronous is it.** In a program with `s += gen("a"); s += gen("b"); print(s["a"])` in a row, on which thread and when does each of the three run? And if `print` moves to the front?

??? success "Answer"
    Both `gen`s go into the queue immediately and the main thread does not wait; the background thread executes `gen("a")` first (calling the backend and setting the event on return) and then `gen("b")`. `print(s["a"])` blocks the main thread until `a`'s event is set, at which point `b` may still be generating. Moving `print` to the front gives a KeyError — `a` is not in `variable_event` yet (before its `submit`).

**2. Can `select`'s two round trips be avoided.** Read `RuntimeEndpoint.select` and think about what would happen without the `max_new_tokens=0` request first; what would the runtime need for one round trip to be enough?

??? success "A way to approach it"
    Without the warm-up, the choices' common prefix is computed separately within the one batch (the first version's tree only inserts when a request finishes, see [the RadixAttention chapter](radix-v1.md)), and `normalized_logprob_start_len` needs the prefix's token count, which takes a tokenization anyway. One round trip would need the runtime to support sharing a prefix within one batch (the later #2442) and determining the start position from the text prefix automatically.

**3. The frontend's last large change.** Use `git show b58ae7a2a0 --stat` to see which files #9029 touched, and explain why `Engine`'s import in `sglang/__init__.py` had to move.

??? success "A way to approach it"
    The changes concentrate in `pyproject.toml` (dependency groups), `__init__.py` (imports) and `api.py`'s move; importing `Engine` straight from `srt.entrypoints.engine` lets a user who only wants the runtime import `sglang` without loading the frontend's code, while the frontend API stays in `lang/api.py` to be imported on demand.

!!! interview "How to explain it"
    To explain what SGLang's frontend language is and why it is rarely mentioned now, put it like this: it is half of the paper, using a decorator and primitives to write a program of several calls as a Python function, with an interpreter executing asynchronously, a tracer extracting the shared prefix and `select` scoring by log probability; it asks very little of the runtime (a warm-up request, normalised log probabilities, branch merging), which is why the runtime could stand alone as a general engine. Users ended up wanting an OpenAI-compatible engine, and the frontend was reduced to an optional component in 2025. That states both co-design's value and its limits.

## Summary {#小结}

- [x] The frontend = an IR of primitives + an interpreter on a background thread + a tracer that executes with dummy arguments + a compiler that sorts topologically.
- [x] It asks the runtime for only three things: a `max_new_tokens=0` warm-up, a `select` by normalised log probability, and an interface for merging branches.
- [x] The constant prefix the tracer extracts plus the warm-up before a batch run are where the paper's high hit rates in few-shot evaluation come from.
- [x] The frontend's size barely changed in three years and it was reduced to an optional component in 2025-08; the runtime grew more than a hundredfold.
