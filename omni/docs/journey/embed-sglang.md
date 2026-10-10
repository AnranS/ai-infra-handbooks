# 把 SGLang 嵌进来：OmniScheduler、两个 ModelRunner 与引擎构建器

<p class="lead">上一章说，所有调度器对 Stage 都是同一个 inbox / outbox 接口。自回归 stage 的调度器 <code>OmniScheduler</code> 在这个接口后面藏了一整个 SGLang：连续批处理、RadixCache、prefill / decode 调度、CUDA Graph，全部原样复用。它的做法很特别——不继承 SGLang 的 <code>Scheduler</code>，而是用 <code>__getattr__</code> 把上游类的方法"借"过来绑到自己身上；输出则通过替换上游的一个组件截获下来，送进 outbox。这一章把这套嫁接拆开：组合是怎么工作的、和 SGLang 原本的事件循环差在哪、为什么禁掉了重叠调度、两个同名的 ModelRunner 各管什么，以及一个自回归 stage 是怎么被"构建器"一步步搭起来的。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `OmniScheduler` 和 SGLang 的 `Scheduler` 是什么关系？`isinstance(omni_scheduler, Scheduler)` 的结果是什么？
    2. SGLang 的调度器算完一批，本来会把结果发给反分词进程。omni 是怎么把结果截下来的？
    3. omni 为什么拒绝运行 SGLang 的重叠事件循环？它自己用什么来重叠 CPU 和 GPU？
    4. 代码里有两个叫 `ModelRunner` 的类，分别来自哪里、谁调用谁？
    5. 每两周一次的"Bump SGLang"PR，主要在改什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 组合关系：`OmniScheduler` 不继承 `Scheduler`，在 `__getattr__` 里从上游类上取方法，用 `types.MethodType` 绑到自己身上调用；自己定义的同名方法优先。所以 `isinstance` 是 `False`，上游的 `__init__` 也从不执行。
    2. SGLang 0.5.21 把调度器拆成了组件，结果由 `SchedulerBatchResultProcessor` 通过 `output_streamer.stream_output(...)` 发出去。omni 在 `init_upstream_scheduler_components` 里自己构造这些组件，并把 `output_streamer` 换成一个 `SimpleNamespace`，它的 `stream_output` 指向 omni 自己的方法——后者把完成的请求放进 outbox。
    3. omni 的模型 runner 在前向时读 `Req.inflight_middle_chunks`，要求它和 `process_batch_result` 在同一轮更新；重叠循环里这个计数晚一轮，分块 prefill 的最后一块会被当成中间块，TTS 模型会悄悄地切错 chunk 边界。所以 `event_loop_overlap` 直接抛 `NotImplementedError`。omni 自己的重叠是 `event_loop_async_decode`：先发射当前 decode 步，再处理上一步的结果。
    4. `sglang_omni.model_runner.base.ModelRunner`（omni 的，负责前向路径上的钩子：构造 ForwardBatch、注入多模态 embedding、采样、整理输出）和 `sglang.srt.model_executor.model_runner.ModelRunner`（SGLang 的，负责加载权重、KV 池、CUDA Graph、真正的前向）。调用链是 `OmniScheduler → omni 的 ModelRunner → ModelWorker → SGLModelRunner（继承 SGLang 的 ModelRunner）→ 模型`。
    5. 适配 SGLang 内部接口的变化：组合方式意味着上游的每次重构（改方法签名、把方法挪进组件、新增 `__init__` 里设置的属性）都要在 omni 这边手工跟进，同时同步 torch、transformers、flashinfer 的版本。

## 全景

![图：一个自回归 stage 的内部分层](../assets/figures/omni-ar-stage.svg){.aig-svg}

从上往下：

| 层 | 类 | 来自 | 负责 |
| --- | --- | --- | --- |
| 调度 | `OmniScheduler` | omni，组合 SGLang 的 `Scheduler` | 收 inbox、把 `StagePayload` 变成 SGLang 的 `Req`、选下一批、把完成的请求放进 outbox |
| 前向路径 | `ModelRunner` 及子类（`ThinkerModelRunner`、`Qwen3TTSModelRunner`……） | omni | 构造 `ForwardBatch`、prefill / decode 前后的钩子、采样、整理每个请求的输出 |
| worker | `ModelWorker` | omni，替代 SGLang 的 `TpModelWorker` | 初始化分布式环境、模型配置、持有下面的 runner |
| 执行 | `SGLModelRunner` | omni，**继承** SGLang 的 `ModelRunner` | 加载权重、分配 KV 池、捕获 CUDA Graph、执行前向 |
| 模型 | 例如 `Qwen3OmniTalker` | omni，用 SGLang 的并行层写 | 网络结构本身 |

把这些层搭起来的是一个**引擎构建器**（`SGLangGenerationEngineBuilder`），本章最后讲。

## 组合：不继承，借方法

`OmniScheduler` 的模块文档把思路写得很清楚：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L1-12"
# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler — stage-facing AR scheduler using composition.

Uses SGLang's batch selection and result processing logic via **unbound
method calls** on the upstream ``Scheduler`` class.  No inheritance.

When an upstream method (e.g. ``get_next_batch_to_run``) internally calls
``self.get_new_batch_prefill()``, Python finds it through
``OmniScheduler.__getattr__`` → looks it up on the upstream class → binds
it to this instance.  This gives us the full scheduling MRO without
inheriting from ``SGLangScheduler``.
"""
```

关键就是这个 `__getattr__`：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L887-918"
    def __getattr__(self, name: str):
        """Look up methods on the upstream SGLang Scheduler class.

        This gives us access to the full scheduling MRO (batch selection,
        result processing, memory checks, etc.) without inheriting.
        """
        if name == "grammar_queue":
            value = []
            self.__dict__[name] = value
            return value
        else:
            pass
        if name == "grammar_backend":
            self.__dict__[name] = None
            return None
        else:
            pass

        try:
            attr = getattr(_Upstream, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute {name!r}"
            ) from None

        # Bind unbound methods to this instance so they use our state
        if callable(attr):
            return types.MethodType(attr, self)
        else:
            pass
        return attr

```

Python 只在常规查找失败时才调 `__getattr__`。所以 `OmniScheduler` 自己定义的方法、自己 `__dict__` 里的属性优先；找不到的方法到上游的 `Scheduler` **类**上去拿，再用 `types.MethodType` 绑定到 `self`——绑定之后，上游方法里的 `self.xxx` 读写的全是 omni 自己的状态，上游方法里调用的 `self.get_new_batch_prefill()` 如果被 omni 覆盖了，调到的也是 omni 的版本。用一个最小的例子验证这个机制：

```python title="ch5_compose.py"
import types


class Upstream:
    """假装是 SGLang 的 Scheduler：get_next_batch_to_run 内部会调 self.get_new_batch_prefill()。"""

    def __init__(self):
        raise RuntimeError("上游的 __init__ 会连 ZMQ、起一堆组件——我们不想执行它")

    def get_next_batch_to_run(self):
        return f"plan({self.get_new_batch_prefill()}, running={self.running})"

    def get_new_batch_prefill(self):
        return "upstream-prefill"

    def log_stats(self):
        return f"stats of {type(self).__name__}"


class Composed:
    """不继承 Upstream，用 __getattr__ 借它的方法。"""

    def __init__(self):
        self.running = 3                    # 状态全在自己身上

    def get_new_batch_prefill(self):        # 想改的方法直接定义在自己身上
        return "omni-prefill"

    def __getattr__(self, name):
        attr = getattr(Upstream, name)
        return types.MethodType(attr, self) if callable(attr) else attr


c = Composed()
print(c.get_next_batch_to_run())   # 上游的方法，内部调到的是我们覆盖的版本
print(c.log_stats())               # 没覆盖的方法原样借用
print("isinstance:", isinstance(c, Upstream))
```

```text title="输出"
plan(omni-prefill, running=3)
stats of Composed
isinstance: False
```

为什么不直接继承？因为 SGLang 的 `Scheduler.__init__` 是为"独立进程里的一个完整引擎"写的：它按 `ServerArgs` 初始化一切，连接 TokenizerManager 和反分词进程的 ZMQ socket，起各种后台组件。omni 的自回归 stage 只需要其中的调度逻辑和 KV 管理，不需要那些进程间通信。组合让 omni 可以**完全不执行上游的构造函数**，只挑自己要的组件手工初始化。

在本书的 CPU 环境里，可以直接数一数两个类的方法：

```python title="ch5_methods.py"
import inspect
import warnings

warnings.filterwarnings("ignore")
from sglang.srt.managers.scheduler import Scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

upstream = {n for n, v in inspect.getmembers(Scheduler) if callable(v) and not n.startswith("__")}
own = {n for n, v in vars(OmniScheduler).items() if callable(v) and not n.startswith("__")}
print("SGLang Scheduler 的方法（含 mixin）：", len(upstream))
print("OmniScheduler 自己定义的方法：      ", len(own))
print("其中覆盖了上游同名方法的：")
for name in sorted(upstream & own):
    print("  ", name)
```

```text title="输出"
SGLang Scheduler 的方法（含 mixin）： 247
OmniScheduler 自己定义的方法：       112
其中覆盖了上游同名方法的：
   _add_request_to_queue
   event_loop_normal
   event_loop_overlap
   flush_cache
   get_new_batch_prefill
   get_next_batch_to_run
   get_num_allocatable_reqs
   process_batch_result
   process_input_requests
   run_batch
```

两百多个上游方法，omni 只覆盖了十个，剩下的全靠 `__getattr__` 借用。覆盖的这十个分成三类：

- **事件循环**：`event_loop_normal`、`event_loop_overlap`——下一节对照；
- **输入和执行**：`process_input_requests`（`StagePayload` → `Req`）、`run_batch`（交给 omni 的 ModelRunner）、`process_batch_result`、`_add_request_to_queue`、`flush_cache`；
- **选批**：`get_next_batch_to_run`、`get_new_batch_prefill`、`get_num_allocatable_reqs`。其中 `get_new_batch_prefill` 加了一个 omni 特有的策略"prefill 合并"（`prefill_coalesce_requests`）：decode 正在跑的时候，先压住零散到达的 prefill，攒够几个再一起准入，摊薄每一步的固定开销。覆盖的写法很统一——判断完自己的条件，最后还是 `_Upstream.get_new_batch_prefill(self, running_batch)`：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L1899-1911"
    def get_new_batch_prefill(self, running_batch):
        # Note: (maydomine) batch prefill admissions to amortize the fixed step
        # cost; the oldest-request deadline survives partial admission and aborts.
        #
        # Upstream passes running_batch in and expects a NextBatchPlan back,
        # so the coalesce hold-off returns an empty plan rather than None.
        if self.prefill_coalesce_requests <= 1 or self.chunked_req is not None:
            return _Upstream.get_new_batch_prefill(self, running_batch)
        else:
            pass
        decode_is_idle = running_batch is None or running_batch.is_empty()
        if not self.prefill_coalesce_when_idle and decode_is_idle:
            return _Upstream.get_new_batch_prefill(self, running_batch)
```

组合的代价是**上游 `__init__` 里设置的属性，omni 都要自己设**。代码里能看到很多这样的注释：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L709-720"
    def init_upstream_compat_flags(self, server_args: ServerArgs) -> None:
        self.enable_hisparse = bool(server_args.enable_hisparse)
        self.hisparse_coordinator = None
        self.enable_priority_preemption = bool(
            server_args.enable_priority_scheduling
            and not server_args.disable_priority_preemption
        )
        # High-water mark, not a cap. Mirrors upstream Scheduler.__init__ (sglang/srt/managers/scheduler.py).
        self.max_prefill_bs = 0
        self.use_ngram_embedding = False
        self.return_health_check_ipcs = []
        self.enable_overlap_mlx = False
```

上游哪天在 `__init__` 里新增一个属性、并在某个被借用的方法里读它，omni 这边就会在运行时抛 `AttributeError`——而且只在走到那条代码路径时才抛。这就是 omni 锁死 SGLang 版本、每次升级都单独开 PR 的原因：

```bash title="bumps.sh"
git log --date=short --format='%ad %h %s' "$REF" | grep -E 'Bump SGLang to' | while read -r d h rest; do
  printf '%s %s %-40s %3d 个文件\n' "$d" "$h" "$rest" "$(git show --name-only --format= "$h" | grep -c .)"
done
```

```text title="输出"
2026-10-04 c4670ad6 [Deps] Bump SGLang to 0.5.21 (#2505)      59 个文件
2026-09-22 bfa856d9 [Deps] Bump SGLang to 0.5.20 (#2275)      45 个文件
2026-09-09 76a2a48f [Deps] Bump SGLang to 0.5.19 (#2040)     124 个文件
2026-08-27 470965eb [Deps] Bump SGLang to 0.5.18 (#1719)     100 个文件
2026-08-01 a8d3dd14 Bump SGLang to 0.5.16 (#1183)            162 个文件
```

两个月五次，大约每两周一次。

## 两个事件循环

SGLang v0.5.21 的普通事件循环：

```python title="sglang:python/sglang/srt/managers/scheduler.py @ v0.5.21 L1906-1938"
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            self.ingest_requests()
            if self._engine_paused:
                self._record_scheduler_state_for_paused_engine()
                continue

            # Get the next batch to run
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self._sched_idled = True
                self.on_idle()

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()
```

omni 的：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L3176-3213"
    def event_loop_normal(self) -> None:
        # Note (Chenyang): yield the GIL when idle so co-located non-AR stages
        # (encoders, preprocessor) running in sibling threads aren't starved
        # of Python execution. Without this, in single-process mode the busy
        # AR scheduler loop pins the GIL and the audio_encoder forward pass
        # (which is mostly Python-side dispatch into many small CUDA kernels)
        # slows ~600x, dropping audio QPS from >10 to <0.5.
        while self.running:
            self.process_admin_requests()
            recv_reqs = self.recv_requests()
            recv_reqs.extend(self.take_deferred_request_payloads())
            self.process_input_requests(recv_reqs)
            if self._engine_paused:  # noqa: leading-underscore
                self.process_admin_requests()
                time.sleep(0.001)
                continue
            else:
                pass

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                if result is not _FAILED_BATCH_RESULT:
                    self.process_batch_result(batch, result)
                else:
                    pass
            else:
                self._sched_idled = True  # noqa: leading-underscore
                self.self_check_during_idle()
                self.sleep_during_idle()

            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.self_check_during_busy()
            else:
                pass
```

骨架一模一样——收请求、选批、执行、处理结果——差别都在"收"和"出"：

| 步骤 | SGLang | omni |
| --- | --- | --- |
| 收请求 | `ingest_requests()`：从 TokenizerManager 的 ZMQ socket 收 `TokenizedGenerateReqInput` | `recv_requests()`：从 inbox 取 `IncomingMessage`；`process_input_requests()` 用模型的 request builder 把 `StagePayload` 变成 `Req`（可以在线程池里并行构建） |
| 执行 | `run_batch` → `TpModelWorker` | `run_batch` → omni 的 `ModelRunner.execute()`，再把返回值转换成上游 `process_batch_result` 认识的 `GenerationBatchResult` |
| 出结果 | 发给反分词进程 | `stream_output` 截获完成的请求，放进 outbox；中间的流式输出在 `run_batch` 里就发了 |
| 空闲 | `on_idle()` | `sleep_during_idle()`：主动让出 GIL |

最后一行值得多说一句。注释里写着：单进程模式下，忙等的自回归循环会一直占着 GIL，同进程里另一个线程上的音频编码器（大量 Python 端的小 kernel 调度）会慢 600 倍，音频 QPS 从 10 以上掉到 0.5 以下。多个 stage 共享一个进程时（第七章的 colocated 部署），这类跨 stage 的干扰是真实存在的。

### 截获输出：换掉一个组件

SGLang 0.5.21 把调度器拆成了一组组件（`scheduler_components/`），批结果由 `SchedulerBatchResultProcessor` 处理，最后通过 `output_streamer.stream_output()` 发出去。omni 自己构造这些组件，并在构造时塞进一个假的 `output_streamer`：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L842-847,849-851,865-869"
        self.output_streamer = types.SimpleNamespace(
            stream_output=self.stream_output,
            _stream_output_generation=lambda reqs, return_logprob, **_kwargs: self.stream_output(
                reqs, return_logprob
            ),
        )
...
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
...
                model_config=self.model_config
            ),
            output_streamer=self.output_streamer,
            beam_coordinator=self.beam_coordinator,
            abort_request=lambda request: self.abort(request.rid),
```

于是上游的结果处理逻辑原封不动地跑，只是最后那一下"发出去"变成了调 omni 的 `stream_output`：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L2225-2231"
    def stream_output(self, reqs, return_logprob=False, skip_req=None):
        """Intercept finished requests and emit to outbox.

        Upstream calls this after process_batch_result to send results
        to the detokenizer via ZMQ.  We capture finished requests here
        and put them in the outbox so Stage can route them downstream.
        """
```

这是一个很典型的依赖注入用法：上游的组件只依赖"有个对象带 `stream_output` 方法"，omni 换一个实现就改变了输出的去向。

### 为什么禁掉重叠调度

SGLang 的零开销调度（重叠循环）让 CPU 提前一步准备下一批。omni 明确拒绝它：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L3215-3231"
    def event_loop_overlap(self) -> None:
        # Model runners read Req.inflight_middle_chunks at forward time under
        # a same-iteration process_batch_result contract. On this loop the
        # decrement lags one iteration, so a final prefill chunk still reads
        # as a middle chunk at forward time and the TTS model runners emit
        # wrong chunk boundaries — silently. No construction site enables
        # overlap today; refuse to run rather than corrupt chunked prefill if
        # one ever does.
        # The pre-guard loop body lives in git history ("Refuse the
        # OmniScheduler overlap event loop"); reviving it needs that drain,
        # not just deleting this raise.
        raise NotImplementedError(
            "OmniScheduler's overlap event loop is unsupported: "
            "Req.inflight_middle_chunks lags one iteration on this loop. "
            "Drain the result queue before the forward, then remove this "
            "guard."
        )
```

原因是 omni 的 TTS 模型 runner 在前向时要知道"这一块 prefill 是不是最后一块"（分块 prefill 时，最后一块要开始生成，中间块不生成）。这个信息来自 `Req.inflight_middle_chunks`，它在 `process_batch_result` 里更新；重叠循环把结果处理推迟了一轮，前向时读到的就是上一轮的旧值。最危险的是它**不报错**——只是悄悄切错 chunk 边界。注释里说得很直接：与其冒险，不如在入口就拒绝；要恢复它，得先在前向前把结果队列处理完。

omni 自己的重叠是另一个循环 `event_loop_async_decode`，只在 decode 步之间做"先发射、后处理"：

```python title="sglang_omni/scheduling/omni_scheduler.py @ 921ea2c8 L3486-3495"
    def event_loop_async_decode(self) -> None:
        """One-step-lookahead decode loop (single stream + CUDA event).

        Each iteration LAUNCHES the current decode step (GPU forward + on-GPU
        sample, then ``post_decode_launch`` publishes the resolve payload, no GPU
        wait) and THEN RESOLVES the previous step's host-side collect, so the
        resolve host work overlaps the current step's GPU forward (launch-first,
        D1 in design.md section 1.3). Prefill / empty batches flush any in-flight
        decode first and run synchronously (the in-flight step is never stranded).
        """
```

prefill 和空批次会先把在途的 decode 处理完再同步执行，所以分块 prefill 的边界问题不会出现。Qwen3-Omni 的 thinker 默认开启它（`enable_async_decode=True`，第九章）。

## 两个 ModelRunner

### omni 的 ModelRunner：前向路径上的钩子

`OmniScheduler.run_batch` 调的是 omni 自己的 `ModelRunner.execute`：

```python title="sglang_omni/model_runner/base.py @ 921ea2c8 L384-441"
    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Full synchronous pipeline: build → prepare → forward → post →
        sample → output.

        Used when async decode is disabled. Behavior is byte-identical to the
        pre-async implementation: it is a pure extraction over the same shared
        sub-steps (``_build_forward_batch`` / ``_prepare_and_forward`` /
        ``_finalize``) that ``execute_launch`` + ``execute_resolve`` also use,
        in the same order. Async decode splits this at the post-decode boundary.
        """
        schedule_batch = scheduler_output.batch_data
        if schedule_batch is None:
            return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})
        else:
            pass
        with self.execution_context(schedule_batch, isolate_sampling=True):
            built = self.build_forward_batch(scheduler_output)
            if built is None:
                return ModelRunnerOutput(outputs={}, req_ids=[], req_id_to_index={})
            else:
                pass
            forward_batch, schedule_batch, is_prefill = built
            batch_result = self.prepare_and_forward(
                forward_batch, schedule_batch, scheduler_output.requests, is_prefill
            )
            if is_prefill:
                self.post_prefill(
                    batch_result,
                    forward_batch,
                    schedule_batch,
                    scheduler_output.requests,
                )
            else:
                self.post_decode(
                    batch_result,
                    forward_batch,
                    schedule_batch,
                    scheduler_output.requests,
                )
            self.ensure_next_token_ids(
                batch_result,
                forward_batch,
                schedule_batch,
                scheduler_output,
            )
            self.publish_next_tokens(
                batch_result,
                forward_batch,
                schedule_batch,
                scheduler_output.requests,
            )
        return self.finalize(
            batch_result,
            forward_batch,
            schedule_batch,
            scheduler_output,
        )

```

`prepare_and_forward` 里是一串可以覆盖的钩子：

```python title="sglang_omni/model_runner/base.py @ 921ea2c8 L598-641"
    def prepare_and_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        is_prefill: bool,
        *,
        is_lookahead: bool = False,
    ) -> GenerationBatchResult:
        """Prepare hook → standard forward (if not custom) → sample-before-post
        block. Returns ``batch_result``."""
        try:
            if is_prefill:
                self.before_prefill(forward_batch, schedule_batch, requests)
                batch_result = self.custom_prefill_forward(
                    forward_batch, schedule_batch, requests
                )
            else:
                self.before_decode(
                    forward_batch,
                    schedule_batch,
                    requests,
                    is_lookahead=is_lookahead,
                )
                batch_result = self.custom_decode_forward(
                    forward_batch, schedule_batch, requests
                )
            if batch_result is None:
                batch_result = self.tp_worker.forward_batch_generation(forward_batch)
            else:
                pass

            if (
                not schedule_batch.is_prefill_only
                and batch_result.next_token_ids is None
                and (
                    self.sample_before_post_prefill(
                        forward_batch, schedule_batch, requests
                    )
                    if is_prefill
                    else self.sample_before_post_decode(
                        forward_batch, schedule_batch, requests
                    )
                )
```

`before_prefill` / `custom_prefill_forward` / `before_decode` / `custom_decode_forward` / `post_prefill` / `post_decode`……子类按需覆盖。两个典型：

- `ThinkerModelRunner` 覆盖 `custom_prefill_forward`：prefill 前把图像、视频、音频编码器算好的 embedding 按占位 token 的位置写进输入 embedding（`inject_multimodal_embeds`），再交给 SGLang 前向；
- talker 类的 runner（`QwenTalkerModelRunner`、`Qwen3TTSModelRunner`）是"反馈式自回归"：每一步主干生成一个码之后，模型内部的小头还要再预测其余几个码本，上一步的输出要写回 buffer 喂给下一步。官方文档把这种模式叫 `FeedbackARModelRunner`。

钩子返回 `None` 时走默认路径：`self.tp_worker.forward_batch_generation(forward_batch)`，也就是交给下一层。

### SGLang 的 ModelRunner：继承来的执行层

`ModelWorker` 是 omni 对 SGLang `TpModelWorker` 的替代，它在 `init_model_runner` 里创建 `SGLModelRunner`：

```python title="sglang_omni/model_runner/sglang_model_runner.py @ 921ea2c8 L252-278"
class SGLModelRunner(ModelRunner):
    """Thin wrapper to bootstrap SGLang ModelRunner from backend args."""

    def __init__(
        self,
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        model_arch_override: str | None = None,
        weight_prefix: str | None = None,
        total_gpu_memory_fraction: float | None = None,
        kv_cache_bytes: int | None = None,
    ) -> None:
        self.weight_prefix = weight_prefix
        self.total_gpu_memory_fraction = total_gpu_memory_fraction
        self.kv_cache_bytes = kv_cache_bytes
        self.model_arch_override = model_arch_override
        self.weight_share_config = None
        self.weight_share_record = None
        self.weight_ipc_leader_monitor = None
        self.register_omni_model()

        super().__init__(
            model_config=model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=gpu_id,
```

这一层是继承：SGLang 的 `ModelRunner` 负责的加载权重、KV 池、注意力后端、CUDA Graph，omni 全部直接用。构造之前的 `register_omni_model()` 把 omni 自己写的模型类登记进 SGLang 的 `ModelRegistry`，SGLang 按 HF 配置里的架构名加载模型时就能找到它们：

```python title="sglang_omni/model_runner/sglang_model_runner.py @ 921ea2c8 L596-610"
    def register_omni_model(self):
        # Register sglang_omni model classes directly in SGLang's model registry.
        import importlib

        from sglang.srt.models.registry import ModelRegistry

        sglang_omni_models = {
            "S2ProSGLangTextModel": "sglang_omni.models.fishaudio_s2_pro.sglang_model:S2ProSGLangTextModel",
            "Qwen3OmniTalker": "sglang_omni.models.qwen3_omni.components.talker:Qwen3OmniTalker",
            "Qwen3OmniThinkerForCausalLM": "sglang_omni.models.qwen3_omni.components.sglang_thinker:Qwen3OmniThinkerForCausalLM",
            "HiggsMultimodalQwen3ForConditionalGeneration": "sglang_omni.models.higgs_tts.model:HiggsTTSModel",
            "Qwen3TTSTalker": "sglang_omni.models.qwen3_tts.sglang_model:Qwen3TTSTalker",
            "MiniCPMOTalkerForCausalLM": "sglang_omni.models.minicpm_o.components.sglang_talker:MiniCPMOTalkerForCausalLM",
            "MingTTSSGLangModel": "sglang_omni.models.ming_tts.sglang_model:MingTTSSGLangModel",
            "MossTTSDelaySGLangModel": "sglang_omni.models.moss_tts.sglang_model:MossTTSDelaySGLangModel",
```

所以名字一样的两个类，一个是"omni 怎么用这次前向"，一个是"SGLang 怎么做这次前向"。读代码时看 import：`from sglang_omni.model_runner.base import ModelRunner` 是前者，`from sglang.srt.model_executor.model_runner import ModelRunner` 是后者。

## 引擎构建器：一个模板方法

每个自回归 stage 的工厂函数都长得差不多——构造一个模型专属的构建器，调 `build()`：

```python title="sglang_omni/models/qwen3_tts/stages.py @ 921ea2c8 L275-306"
def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    attn_implementation: str | None = None,
    prefill_coalesce_requests: int = 0,
    prefill_coalesce_wait_ms: float = 60.0,
    server_args_overrides: Mapping[str, object] | None = None,
    reference_encoder_cuda_graph_bucket_frames: Sequence[int] = (
        DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    ),
    leading_silence_mask_frames: int = DEFAULT_LEADING_SILENCE_MASK_FRAMES,
) -> OmniScheduler[Qwen3TTSSGLangRequestData]:
    from sglang_omni.models.qwen3_tts.engine_builder import Qwen3TtsEngineBuilder

    return Qwen3TtsEngineBuilder(
        attn_implementation=attn_implementation,
        prefill_coalesce_requests=prefill_coalesce_requests,
        prefill_coalesce_wait_ms=prefill_coalesce_wait_ms,
        reference_encoder_cuda_graph_bucket_frames=(
            reference_encoder_cuda_graph_bucket_frames
        ),
        leading_silence_mask_frames=leading_silence_mask_frames,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
```

`build()` 定义在基类 `SGLangGenerationEngineBuilder` 里，是一个典型的模板方法：

```python title="sglang_omni/scheduling/engine_factory.py @ 921ea2c8 L99-106"
class SGLangGenerationEngineBuilder(ABC, Generic[RequestDataT]):
    """Build the model-neutral parts of a SGLang AR engine stage.

    Model-specific builders provide checkpoint preprocessing, model setup,
    request/result adapters, validation policy, and any stage-owned resources.
    Family-specific builders such as :class:`AsrEngineBuilder` and
    :class:`TtsEngineBuilder` define the lifecycle policy for each modality.
    """
```

把 `build()` 里依次调用的钩子列出来：

```bash title="builder-hooks.sh"
git show "$REF:sglang_omni/scheduling/engine_factory.py" | awk 'NR>=129 && NR<=375' \
  | grep -oE 'self\.[a-z_]+\(' | awk '!seen[$0]++' | sed 's/($//' | paste -sd' ' | fold -s -w 96
```

```text title="输出"
self.resolve_checkpoint self.pre_infra_setup self.resolve_context_length 
self.generation_defaults self.adjust_overrides self.customize_server_args 
self.validate_before_infrastructure self.infra_kwargs self.before_memory_pool self.setup_model 
self.validate_after_model_setup self.compile_model self.post_cuda_graph_setup 
self.setup_model_resources self.setup_runtime_resources self.build_runtime 
self.post_scheduler_setup self.cleanup_build_failure
```

检查点 → 上下文长度 → 默认的批大小与显存参数 → ServerArgs 定制 → 校验 → 显存池之前 → 建模型 → 编译 → CUDA Graph 之后 → 模型资源 → 运行时资源 → 构造调度器 → 收尾。`AsrEngineBuilder`、`TtsEngineBuilder` 给一类模型定好生命周期，具体模型（`Qwen3TtsEngineBuilder`）只覆盖自己不同的那几步。新接一个自回归模型，主要的工作就是写这样一个子类。

## vendor 补丁：怎么打、怎么测

第一章看过 `vendor/sglang/` 的作用。补丁本身很克制，例如 `layers.py` 的文件头：

```python title="sglang_omni/vendor/sglang/layers.py @ 921ea2c8 L1-8"
"""Vendor wrapper for sglang.srt.layers.*

Centralize third-party imports and apply monkey patches here.

Patches applied to RMSNorm.forward_cuda:
  - Empty tensor early return (avoids CUDA kernel launch on zero-element tensors)
  - dtype mismatch fallback when residual or post_residual_addition differ from x.dtype
"""
```

每个补丁都有对应的单测（`tests/unit_test/vendor/test_sglang_layers_patch.py`），升级 SGLang 时先跑它们：上游如果已经修了，补丁就可以删；上游如果改了被补的函数，测试会第一时间告诉你。

## 练习

**1. 找一个"借来的"方法。** 在 `OmniScheduler` 里找一个调用了上游方法、但 omni 自己没有定义的名字（比如 `self.init_load_publisher()`），到 SGLang v0.5.21 里找到它的定义，列出它读写了 `self` 的哪些属性，再确认 omni 在调用它之前都设置了这些属性。

??? success "参考思路"
    `git show v0.5.21:python/sglang/srt/managers/scheduler.py | grep -n 'def init_load_publisher' -A30`，看它用到的 `self.xxx`；再到 `omni_scheduler.py` 里 grep 这些属性在 `__init__` 或 `init_upstream_*` 里有没有赋值。这正是升级 SGLang 时要做的检查——借来的方法读的每个属性，omni 都要负责初始化。

**2. 算一下 GIL 的影响。** 读 `sleep_during_idle` 的实现，说明它在什么条件下睡、睡多久，以及为什么在多 stage 共享进程时尤其重要。

??? success "参考思路"
    `grep -n 'def sleep_during_idle' -A20 sglang_omni/scheduling/omni_scheduler.py`。空闲（没有可运行的批）时短暂 sleep，把 GIL 让给同进程里的其他线程（其他 stage 的调度器线程、Stage 的事件循环）。忙等会让同进程的编码器线程拿不到 GIL，Python 端的 kernel 调度大幅变慢。

**3. 验证 omni 不执行上游的 `__init__`。** 用 `git grep` 证明 `omni_scheduler.py` 里没有调用 `Scheduler.__init__` 或 `super().__init__()`，然后找出 omni 自己初始化 KV 池（`req_to_token_pool`、`token_to_kv_pool_allocator`、`tree_cache`）的地方。

??? success "参考思路"
    `git grep -nE '_Upstream\.__init__|super\(\)\.__init__' 921ea2c8 -- sglang_omni/scheduling/omni_scheduler.py` 没有结果。KV 池在 `bind_model_runner` / `init_parallel_state` 一带从 `ModelWorker` 拿（`model_worker.get_memory_pool()`），`tree_cache` 由 `sglang_backend/cache.py` 按配置构造。

!!! interview "怎么讲清楚"
    讲 omni 怎么复用 SGLang，用三个词：组合、注入、继承。调度器用组合——`__getattr__` 从上游类借方法、绑到自己身上，自己只覆盖十个方法，上游 `__init__` 不执行；输出用注入——自己构造上游的结果处理组件，把 `output_streamer` 换成指向自己的实现；执行层用继承——`SGLModelRunner` 继承 SGLang 的 `ModelRunner`，权重、KV、CUDA Graph 全用上游的。代价也要讲：上游每次重构都要手工跟进，所以锁版本、两周一次升级 PR；重叠调度因为分块 prefill 的计数会晚一轮而被禁用，omni 用自己的 async decode 循环替代。

## 小结

- [x] `OmniScheduler` 组合 SGLang 的 `Scheduler`：`__getattr__` + `types.MethodType` 借方法，只覆盖 10 个，`__init__` 不执行，属性手工镜像。
- [x] 事件循环骨架相同，差在收（inbox + request builder）和出（替换 `output_streamer`，`stream_output` 放进 outbox）；空闲时让出 GIL。
- [x] 重叠调度被拒（`inflight_middle_chunks` 晚一轮会悄悄切错 chunk）；omni 用 `event_loop_async_decode` 只重叠 decode。
- [x] 两个 ModelRunner：omni 的管前向路径钩子（多模态注入、反馈式 AR），SGLang 的（经 `SGLModelRunner` 继承）管权重、KV、CUDA Graph。
- [x] 引擎构建器是模板方法：`build()` 固定流程，模型子类覆盖钩子；vendor 补丁配单测，升级时先跑。
