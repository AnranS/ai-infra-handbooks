# Stage 与调度器：IO 壳、inbox / outbox、扇入扇出与流式

<p class="lead">omni 里最重要的一条设计约束写在 Stage 的文档字符串里：Stage 是"IO 壳"，所有计算都经过调度器的 inbox / outbox，<b>Stage 不按调度器的类型分支</b>。于是同一个 Stage 类可以包住一个只会跑函数的 <code>SimpleScheduler</code>、一个嵌着 SGLang 的 <code>OmniScheduler</code>、一个按 chunk 解码的流式声码器——它们对 Stage 呈现完全相同的接口。这一章把这层接口拆开：Stage 的两个线程怎么分工，消息怎么进出调度器，结果怎么路由到下游，扇出时怎么投影、扇入时怎么汇合，流式 chunk 怎么边算边推。每一种机制都配一个能在 CPU 上跑的实验。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. Stage 的事件循环和调度器在同一个线程里吗？它们之间靠什么传消息？
    2. 调度器的 `outbox` 里能放哪几类消息？Stage 分别怎么处理？
    3. 一个 stage 要等三个上游都到齐才能开始算，配置里怎么写？到达顺序不固定，合并函数要注意什么？
    4. `SimpleScheduler` 的 `max_batch_wait_ms` 在空闲时为什么曾经是个性能问题？现在怎么解决的？
    5. talker 一边生成一边把码流推给声码器，配置上需要哪两个字段？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 不在。Stage 在进程的 asyncio 事件循环里做 IO（收 ZMQ、读写 relay、路由），调度器在一个专门的线程里跑 `scheduler.start()`。两者之间是两个线程安全的 `queue.Queue`：Stage 往 `inbox` 放 `IncomingMessage`，从 `outbox` 取 `OutgoingMessage`（用 `run_in_executor` 阻塞读，不卡事件循环）。
    2. `result`（完整结果，路由到下游或报给 Coordinator）、`stream`（流式 chunk，发给 `stream_to` 的下游或 Coordinator）、`error`（报失败）、`admitted`（准入回执）、`kv_transfer`（PD 分离时的 KV 页传输）。
    3. `wait_for=["a", "b", "c"]` + `merge_fn="模块.函数"`，Stage 会用 `AggregatedInput` 攒齐再调用合并函数；合并函数拿到的是 `{来源名: payload}` 字典，不能依赖字典的插入顺序（到达顺序不固定），要按名字取。如果"要等谁"取决于请求，再加 `wait_for_fn`。
    4. 原来第一个请求一到就开始计时，即使队列里没有别人也要干等满一个窗口；Qwen3-Omni 两个编码器的窗口是 50 ms，并发为 1 时每个请求白等 50 ms，约占语音首包时间的三分之一。#1628 改成"第二个请求加入之后才开始计时"（`batch_wait_when_idle=False` 对应这种行为），空闲时立刻执行。
    5. 上游写 `stream_to=["vocoder"]`（流式 chunk 发给谁），下游写 `can_accept_stream_before_payload=True`（允许 chunk 比完整 payload 先到）。下游的调度器要能处理 `stream_chunk` / `stream_done` 消息，比如继承 `StreamingSimpleScheduler`。

## IO 壳与两个线程

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L81-96"
class Stage:
    """IO shell for one pipeline stage.

    All stage compute is dispatched through the scheduler inbox/outbox
    contract, independent of scheduler implementation.

    Note on ``role``: ``role="single"`` means this stage owns its own ZMQ
    control plane and relay reader (i.e. it is NOT a TP follower). It does
    **not** imply this stage has its OS process to itself — since the
    declarative topology PR, multiple ``role="single"`` stages can share
    one OS process (and one asyncio event loop). When they do, they share
    a failure domain: see ``_run_process`` in ``stage_workers.py``.
    ``role="leader"`` / ``role="follower"`` continue to denote TP rank 0
    vs rank > 0 within a multi-rank TP stage; TP stages must own their OS
    process exclusively.
    """
```

![图：Stage 的两个线程和两个队列](../assets/figures/omni-stage-threads.svg){.aig-svg}

`Stage.start()` 里，调度器被放进一个专门的线程：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L241-247,265-268,281-287"
        # Start scheduler in dedicated thread
        if self.scheduler is not None:
            serving_thread_ready = threading.Event()

            def _run_scheduler() -> None:
                try:
                    try:
...
                        self.scheduler.warm_up_serving_thread()
                    finally:
                        serving_thread_ready.set()
                    self.scheduler.start()
...
            self.scheduler_thread = threading.Thread(
                target=_run_scheduler,
                name=f"scheduler-{self.name}",
                daemon=True,
            )
            self.scheduler_thread.start()
            await asyncio.to_thread(serving_thread_ready.wait)
```

为什么要分两个线程？Stage 这一侧全是 IO：ZMQ 收发、relay 读写、等下游的 ACK，天然适合 asyncio；调度器这一侧是阻塞的计算循环——`OmniScheduler` 的 `while self.running:` 里每一轮都要发射一次 GPU 前向。把阻塞循环放进事件循环会把所有 IO 卡死，反过来把 IO 放进计算线程又会拖慢每一步。两个 `queue.Queue` 是它们之间唯一的接触面。

主循环本身很短：收一条控制消息，按类型分发：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L384-410,438-463"
            while self.running:
                msg = await self.control_plane.recv()
                if (
                    self.role == "leader"
                    and self.tp_fanout is not None
                    and isinstance(
                        msg,
                        (
                            ShutdownMessage,
                            ProfilerStartMessage,
                            ProfilerStopMessage,
                            AdminMessage,
                        ),
                    )
                ):
                    await self.tp_fanout.fanout_control(msg)
                else:
                    pass
                if isinstance(msg, ShutdownMessage):
                    break
                else:
                    pass
                if isinstance(msg, TPWorkMessage):
                    await self.execute(msg.data)
                    continue
                else:
                    pass
...
    async def handle_message(
        self,
        msg: (
            SubmitMessage
            | DataAckMessage
            | DataReadyMessage
            | ProfilerStartMessage
            | ProfilerStopMessage
            | AdminMessage
        ),
    ) -> None:
        if isinstance(msg, SubmitMessage):
            await self.on_submit(msg)
        elif isinstance(msg, DataAckMessage):
            self.comm.ack_transfer(msg)
        elif isinstance(msg, DataReadyMessage):
            self.schedule_receive_task(msg)
        elif isinstance(msg, ProfilerStartMessage):
            self.on_profiler_start(msg)
        elif isinstance(msg, ProfilerStopMessage):
            self.on_profiler_stop(msg)
        elif isinstance(msg, AdminMessage):
            await self.on_admin(msg)
        else:
            pass

```

新请求（来自 Coordinator 的 `SubmitMessage` 或上游的 `DataReadyMessage`）最终都走 `execute`，变成一条 `IncomingMessage` 放进 inbox：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L1106-1131"
    async def execute(self, payload: StagePayload) -> None:
        request_id = payload.request_id
        if request_id in self.aborted:
            return
        else:
            pass
        _emit_event(
            request_id=request_id,
            stage=self.name,
            event_name="stage_dispatch",
        )
        if (
            self.role == "leader"
            and self.tp_fanout is not None
            and getattr(self.scheduler, "requires_tp_work_fanout", False)
        ):
            self.tp_fanout.fanout_work(payload)
        else:
            pass
        msg = IncomingMessage(request_id=request_id, type="new_request", data=payload)
        enqueue = getattr(self.scheduler, "enqueue", None)
        if enqueue is not None:
            enqueue(msg)
        else:
            self.scheduler.inbox.put(msg)

```

## 调度器的契约

所有调度器都满足同一个协议：

```python title="sglang_omni/scheduling/message.py @ 921ea2c8 L10-44"
IncomingMessageType = Literal["new_request", "stream_chunk", "stream_done", "abort"]


@dataclass
class IncomingMessage:
    request_id: str
    type: IncomingMessageType
    data: object = None


@dataclass
class OutgoingMessage:
    request_id: str
    type: Literal["result", "stream", "error", "kv_transfer", "admitted"]
    data: object = None
    target: str | None = None
    metadata: dict[str, object] | None = None


class StageScheduler(Protocol):
    """Scheduler lifecycle and message queues consumed by a pipeline stage."""

    @property
    def inbox(self) -> Queue[IncomingMessage]: ...

    @property
    def outbox(self) -> Queue[OutgoingMessage]: ...

    def warm_up_serving_thread(self) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def abort(self, request_id: str) -> None: ...
```

进来的消息只有四种（新请求、流式块、流结束、abort），出去的消息五种。Stage 一侧的 `drain_outbox_external` 在线程池里阻塞地读 outbox，按类型处理：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L1271-1305"
    async def drain_outbox_external(self) -> None:
        """Drain scheduler outbox and route results downstream."""
        loop = asyncio.get_running_loop()
        outbox = self.scheduler.outbox
        while self.running or not outbox.empty():
            try:
                out = await loop.run_in_executor(None, lambda: outbox.get(timeout=0.1))
            except _queue_mod.Empty:
                continue

            for batch_index in range(_OUTBOX_DRAIN_BATCH_SIZE):
                if out.type == "admitted":
                    if out.request_id not in self.aborted:
                        self.record_replica_bindings(
                            out.request_id, (out.metadata or {}).get("replica_bindings")
                        )
                        self.active_requests.add(out.request_id)
                    else:
                        pass
                elif out.type == "kv_transfer":
                    if out.request_id in self.active_requests:
                        self.launch_kv_transfer(out.data)
                    else:
                        self.discard_kv_transfer(out.data)
                elif out.request_id in self.active_requests:
                    if out.type == "result":
                        await self.route_result(out.request_id, out.data)
                    elif out.type == "stream":
                        if out.target is None:
                            if self.stream_targets:
                                await asyncio.gather(
                                    *(
                                        self.send_stream_to_target(
                                            out.request_id,
                                            out.data,
```

注意两点：只有 `active_requests` 里的请求的输出才会被路由——被 abort 掉的请求即使调度器晚一步吐出了结果，也会在这里被丢掉；一次最多连续处理 `_OUTBOX_DRAIN_BATCH_SIZE` 条就让出事件循环，避免一个吐得很快的调度器饿死同进程里的其他 stage。

基准提交里调度器的全部家族：

```bash title="schedulers.sh"
git grep -nE '^class \w*(Scheduler|VocoderBase)\b' "$REF" -- sglang_omni/scheduling/ | sed -E "s/^$REF:sglang_omni\/scheduling\///; s/\(.*//; s/:$//"
```

```text title="输出"
dllm_scheduler.py:46:class DllmScheduler
message.py:29:class StageScheduler
omni_scheduler.py:280:class OmniScheduler
pd_scheduler.py:117:class OmniPrefillScheduler
pd_scheduler.py:287:class OmniDecodeScheduler
session.py:169:class SessionScheduler
sglang_backend/ar_session.py:71:class BridgeScheduler
simple_scheduler.py:33:class SimpleScheduler
streaming_detokenizer.py:57:class StreamingDetokenizeScheduler
streaming_simple_scheduler.py:36:class StreamingSimpleScheduler
streaming_vocoder.py:103:class StreamingVocoderBase
threaded_simple_scheduler.py:75:class ThreadedSimpleScheduler
vocoder_base.py:22:class BatchVocoderBase
```

| 家族 | 典型用途 | 特点 |
| --- | --- | --- |
| `SimpleScheduler` | 预处理、编码器、聚合、decode | 一个函数；可选攒批（`batch_compute_fn`）或多线程并发（`max_concurrency`） |
| `ThreadedSimpleScheduler` | Qwen3-TTS 的预处理 | 多个工作线程并发跑同一个函数 |
| `StreamingSimpleScheduler` / `StreamingVocoderBase` | 声码器、流式反分词 | 处理 `stream_chunk` / `stream_done`，边收边算 |
| `SessionScheduler` | realtime、流式 ASR | 按会话保存状态，open / append / close |
| `OmniScheduler` | thinker、talker、TTS 引擎、ASR 引擎 | 组合 SGLang 的调度器（第五章） |
| `OmniPrefillScheduler` / `OmniDecodeScheduler` | PD 分离 | 在两个 stage 之间搬 KV 页 |
| `DllmScheduler` | LLaDA 这类扩散语言模型 | 非自回归的迭代去噪 |

## 路由：结果去哪

一个 stage 算完，结果去哪是在**进程启动时**就编译好的。`construct_stage` 根据配置生成一个 `get_next(request_id, output)` 函数：

```python title="sglang_omni/pipeline/stage_workers.py @ 921ea2c8 L803-827"
    # --- Build routing ---
    if spec.is_terminal:
        get_next = lambda request_id, output: None
    elif spec.route_fn:
        route_fn = import_string(spec.route_fn)
        allowed_route_targets = set(_target_list(spec.next_stages))

        def get_next(request_id, output, _fn=route_fn):
            return _target_result(
                _fn(request_id, output),
                allowed_targets=allowed_route_targets,
                allow_empty=False,
                hook_name="route_fn",
            )

    else:
        target = spec.next_stages
        if isinstance(target, str):
            get_next = lambda request_id, output, _t=target: _t
        elif isinstance(target, list):
            get_next = lambda request_id, output, _t=list(target): _t
        else:
            get_next = lambda request_id, output: None

    if spec.stream_done_to_fn:
```

终点返回 `None`；有 `route_fn` 就调它（返回值必须在静态 `next` 里）；否则就是静态的 `next`。`route_result` 用它决定把结果发给谁：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L1513-1576"
        # Send stream done to the active stream targets for this request.
        stream_targets = self.stream_targets
        if self.get_stream_done_targets is not None:
            resolved = self.get_stream_done_targets(request_id, result)
            if isinstance(resolved, str):
                stream_targets = [resolved]
            elif isinstance(resolved, list):
                stream_targets = resolved
            elif resolved is None:
                stream_targets = []
            else:
                pass
        else:
            pass
        stream_targets_for_request = set(stream_targets)
        for target in stream_targets:
            await self.send_stream_signal_to_target(
                request_id,
                target,
                is_done=True,
            )

        next_stages = (
            self.get_next(request_id, result) if session_operation is None else actual
        )
        if next_stages is None:
            # Terminal: notify coordinator
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": True},
            )
            await self.control_plane.send_complete(
                CompleteMessage(
                    request_id=request_id,
                    from_stage=self.name,
                    success=True,
                    result=result.data if isinstance(result, StagePayload) else result,
                )
            )
        else:
            if isinstance(next_stages, str):
                next_stages = [next_stages]
            else:
                pass
            is_single_target = len(next_stages) == 1
            _emit_event(
                request_id=request_id,
                stage=self.name,
                event_name="stage_complete",
                metadata={"terminal": False, "next": list(next_stages)},
            )
            for target in next_stages:
                await self.send_to_stage(
                    request_id,
                    target,
                    result,
                    allow_local_object=is_single_target,
                    allow_projected_local_object=not is_single_target,
                    stream_targets_for_request=stream_targets_for_request,
                )

        self.clear_request_state(request_id)
```

先给所有流式下游发"流结束"信号，再把完整结果发给下一个（或几个）stage；终点则把结果报给 Coordinator。多个下游时，结果会先经过各自的 **投影函数**（`project_payload`），每个下游只拿自己需要的那部分——Qwen3-Omni 的预处理把图像特征只发给图像编码器、把音频特征只发给音频编码器，就是靠它。

## 实验一：扇出、投影、扇入

```python title="ch4_fanin_stages.py"
"""扇出 + 扇入：split 把一份输出分给 upper 和 count，join 等两边都到齐再合并。"""
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_split():
    def fn(payload: StagePayload) -> StagePayload:
        text = payload.data["raw_inputs"]
        payload.data = {"text": text, "words": text.split()}
        return payload
    return SimpleScheduler(fn)


def project_to_upper(payload: StagePayload) -> StagePayload:
    """投影：upper 只需要原文。"""
    return StagePayload(payload.request_id, payload.request, {"text": payload.data["text"]})


def project_to_count(payload: StagePayload) -> StagePayload:
    """投影：count 只需要分好的词。"""
    return StagePayload(payload.request_id, payload.request, {"words": list(payload.data["words"])})


def create_upper():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"upper": payload.data["text"].upper(), "upper_saw": sorted(payload.data)}
        return payload
    return SimpleScheduler(fn)


def create_count():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"count": len(payload.data["words"]), "count_saw": sorted(payload.data)}
        return payload
    return SimpleScheduler(fn)


def merge(inputs: dict[str, StagePayload]) -> StagePayload:
    """inputs 是 {来源 stage 名: payload}；到达顺序不固定，所以按名字排序再合并。"""
    first = next(iter(inputs.values()))
    data = {"sources": sorted(inputs)}
    for name in sorted(inputs):
        data.update(inputs[name].data)
    return StagePayload(first.request_id, first.request, data)


def create_join():
    return SimpleScheduler(lambda payload: payload)
```

```python title="ch4_fanin.py"
import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

M = "ch4_fanin_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="split",
        stages=[
            StageConfig(name="split", process="p_split", factory_path=f"{M}.create_split",
                        next=["upper", "count"],
                        project_payload={"upper": f"{M}.project_to_upper", "count": f"{M}.project_to_count"}),
            StageConfig(name="upper", process="p_upper", factory_path=f"{M}.create_upper", next="join"),
            StageConfig(name="count", process="p_count", factory_path=f"{M}.create_count", next="join"),
            StageConfig(name="join", process="p_join", factory_path=f"{M}.create_join",
                        wait_for=["upper", "count"], merge_fn=f"{M}.merge", terminal=True),
        ],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    try:
        results = await asyncio.gather(*(runner.coordinator.submit(f"r{i}", t)
                                         for i, t in enumerate(["hello omni world", "stage by stage"])))
        for r in results:
            print(r)
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

```text title="输出"
{'sources': ['count', 'upper'], 'count': 3, 'count_saw': ['words'], 'upper': 'HELLO OMNI WORLD', 'upper_saw': ['text']}
{'sources': ['count', 'upper'], 'count': 3, 'count_saw': ['words'], 'upper': 'STAGE BY STAGE', 'upper_saw': ['text']}
```

`upper_saw`、`count_saw` 证明了投影生效：每个下游只看到了自己那一份。两个请求并发提交，join 同时为两个请求攒输入，互不干扰——因为攒的状态是按 `request_id` 分开的。这是 `AggregatedInput` 的全部逻辑：

```python title="sglang_omni/pipeline/stage/input.py @ 921ea2c8 L44-117"
class AggregatedInput(InputHandler):
    """Fan-in: wait for inputs from multiple sources then merge."""

    def __init__(
        self,
        sources: set[str],
        merge: Callable[[dict[str, StagePayload]], StagePayload],
        expected_sources_fn: ExpectedSourcesFn | None = None,
    ):
        self.sources = sources
        self.merge = merge
        self.expected_sources_fn = expected_sources_fn
        self.pending: dict[str, dict[str, StagePayload]] = {}
        self.expected_sources: dict[str, set[str]] = {}

    def receive(
        self, request_id: str, from_stage: str, data: "StagePayload"
    ) -> StagePayload | None:
        if from_stage not in self.sources:
            logger.warning(
                "AggregatedInput: unexpected source %s for request %s",
                from_stage,
                request_id,
            )
            return None
        else:
            pass

        if request_id not in self.pending:
            self.pending[request_id] = {}
        else:
            pass
        self.pending[request_id][from_stage] = data

        expected_sources = self.expected_sources.get(request_id)
        if expected_sources is None and self.expected_sources_fn is not None:
            resolved = self.expected_sources_fn(request_id, from_stage, data)
            if resolved is not None:
                expected_sources = self.normalize_expected_sources(
                    request_id,
                    resolved,
                )
                self.expected_sources[request_id] = expected_sources
            else:
                pass
        elif expected_sources is None:
            expected_sources = self.sources
        else:
            pass

        if expected_sources is None:
            return None
        else:
            pass

        pending_sources = set(self.pending[request_id])
        unexpected = pending_sources - expected_sources
        if unexpected:
            raise ValueError(
                "AggregatedInput: received sources outside expected fan-in for "
                f"request {request_id}: {sorted(unexpected)}. "
                f"Expected: {sorted(expected_sources)}"
            )
        else:
            pass

        if pending_sources == expected_sources:
            inputs = self.pending.pop(request_id)
            self.expected_sources.pop(request_id, None)
            return self.merge(inputs)
        else:
            pass

        return None
```

`expected_sources_fn`（配置里的 `wait_for_fn`）让"要等谁"可以按请求变化：Qwen3-Omni 的请求如果没有图片，就不等图像编码器。它的返回值同样只能是静态 `wait_for` 的子集。

## 实验二：攒批的窗口

编码器这类 stage 攒批能提高 GPU 利用率，但攒批要等。`SimpleScheduler.collect_batch` 的逻辑：

```python title="sglang_omni/scheduling/simple_scheduler.py @ 921ea2c8 L149-198"
    def collect_batch(self, first_msg: IncomingMessage) -> list[IncomingMessage]:
        batch = [first_msg]
        if self.batch_fn is None or self.max_batch_size <= 1:
            return batch
        else:
            pass

        batch_cost = self.message_cost(first_msg)
        deadline: float | None = (
            time.monotonic() + self.max_batch_wait_s
            if self.batch_wait_when_idle
            else None
        )
        while len(batch) < self.max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                if deadline is None:
                    break
                else:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                else:
                    pass
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.type == "new_request":
                if self.max_batch_cost is not None:
                    msg_cost = self.message_cost(msg)
                    if batch and batch_cost + msg_cost > self.max_batch_cost:
                        self.pending_messages.appendleft(msg)
                        break
                    else:
                        pass
                    batch_cost += msg_cost
                else:
                    pass
                batch.append(msg)
                if deadline is None:
                    deadline = time.monotonic() + self.max_batch_wait_s
                else:
                    pass
            else:
                self.pending_messages.append(msg)
        return batch
```

关键在 `deadline`：`batch_wait_when_idle=True` 时，第一个请求一到就开始倒计时；`False` 时 `deadline` 先是 `None`，队列空就立刻返回，直到第二个请求真的加入批次才开始计时。做个实验：

```python title="ch4_batch_stages.py"
"""一个会攒批的玩具编码器：batch_compute_fn 一次处理一批，并把批大小写进结果。"""
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_encoder(max_batch_wait_ms: float = 200, batch_wait_when_idle: bool = True):
    def one(payload: StagePayload) -> StagePayload:
        payload.data = {"batch_size": 1}
        return payload

    def batch(payloads: list[StagePayload]) -> list[StagePayload]:
        for p in payloads:
            p.data = {"batch_size": len(payloads)}
        return payloads

    return SimpleScheduler(one, batch_compute_fn=batch, max_batch_size=8,
                           max_batch_wait_ms=max_batch_wait_ms, batch_wait_when_idle=batch_wait_when_idle)
```

工厂函数的参数通过配置的 `factory` 组传进去（第七章会讲这个"按消费者分组"的配置设计）：

```python title="ch4_batch.py" ci="loose"
import asyncio
import tempfile
import time

from sglang_omni.config.schema import EndpointsConfig, FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner


def make_config(wait_when_idle: bool) -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="encoder",
        stages=[StageConfig(name="encoder", process="p_enc", terminal=True,
                            factory_path="ch4_batch_stages.create_encoder",
                            factory=FactoryArgs(max_batch_wait_ms=200, batch_wait_when_idle=wait_when_idle))],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def run(wait_when_idle: bool) -> None:
    runner = MultiProcessPipelineRunner(make_config(wait_when_idle))
    await runner.start(timeout=120)
    coord = runner.coordinator
    try:
        results = await asyncio.gather(*(coord.submit(f"burst-{i}", i) for i in range(4)))
        print(f"batch_wait_when_idle={wait_when_idle}")
        print("  同时到达的 4 个请求，各自所在批的大小：", [r["batch_size"] for r in results])
        t0 = time.perf_counter()
        lone = await coord.submit("lone", 0)
        waited = (time.perf_counter() - t0) * 1000
        print(f"  单独一个请求：批大小 {lone['batch_size']}，多等了 200 ms 吗：{waited >= 180}")
    finally:
        await runner.stop()


async def main() -> None:
    await run(True)
    await run(False)


if __name__ == "__main__":
    asyncio.run(main())
```

```text title="输出"
batch_wait_when_idle=True
  同时到达的 4 个请求，各自所在批的大小： [4, 4, 4, 4]
  单独一个请求：批大小 1，多等了 200 ms 吗：True
batch_wait_when_idle=False
  同时到达的 4 个请求，各自所在批的大小： [4, 4, 4, 4]
  单独一个请求：批大小 1，多等了 200 ms 吗：False
```

两种模式下，同时到达的请求都能攒成一批；区别只在空闲时：`True` 让孤零零的一个请求白等满 200 ms。这正是 omni 修过的一个真实问题，提交说明把来龙去脉写得很清楚：

```bash title="pr-1628.sh"
git log -S 'batch_wait_when_idle' --reverse --date=short --format='%ad %h %s' "$REF" -- sglang_omni/scheduling/simple_scheduler.py | head -1
git show -s --format=%B 7fa879d9 | sed -n '3,17p'
```

```text title="输出"
2026-08-20 7fa879d9 [Qwen3-Omni Perf] Encoder: conditional batching, layer CUDA graph, small-payload transport gate (#1628)
* perf(scheduler): arm the batch window only once a request has joined

SimpleScheduler._collect_batch armed the coalescing deadline as soon as it
took the first message, then blocked on the inbox until that deadline even
when nothing else was in flight. Every request arriving at an idle batching
stage paid the full window.

On the Qwen3-Omni speech path both encoders set the window to 50ms, so at
concurrency 1 each request burned a flat 50ms before the audio encoder ran
-- measured p95 50.26ms, invariant across requests, about a third of speech
TTFT.

Arm the deadline only after a second request actually joins the batch. An
empty inbox before that means the stage is idle and dispatches immediately;
once there is a backlog the window behaves exactly as before, so batching
```

一个 50 ms 的固定等待占了语音首包时间的三分之一——这种问题在压测里看不出来（高并发时窗口总能攒满），只有在低并发下量延迟才会暴露。issue #1147（2026-07-23）提出要扫这个参数，#1628（2026-08-20）用"第二个请求加入才开始计时"从根上解决；Qwen3-Omni 的编码器窗口现在默认是 0，可以用环境变量 `SGLANG_OMNI_ENCODER_BATCH_WAIT_MS` 打开。第十二章会把这个 PR 当成"性能类 PR 怎么写"的范例。

## 实验三：流式——边生成边推

全模态模型的首包延迟，取决于下游能不能在上游还没算完时就开工。omni 的流式通道是 `stream_to`：上游调度器往 outbox 放 `type="stream"` 的消息，Stage 把它们发给 `stream_to` 里的每个下游；上游算完时，`route_result` 先发"流结束"信号，再发完整结果。

下面用两个玩具 stage 模拟 talker → 声码器：talker 自己实现调度器协议，每"生成"一步就往外推一个码；声码器继承 `StreamingSimpleScheduler`，每收到一个码就"解码"成一段音频，流式发给客户端。

```python title="ch4_stream_stages.py"
"""一个"会说话"的玩具：talker 每步吐一个码（流式），vocoder 边收边"解码"。"""
import queue

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


class ToyTalkerScheduler:
    """自己实现 StageScheduler 协议：inbox / outbox / start / stop / abort。"""

    def __init__(self, steps: int = 3):
        self.inbox: queue.Queue[IncomingMessage] = queue.Queue()
        self.outbox: queue.Queue[OutgoingMessage] = queue.Queue()
        self.steps = steps
        self.running = False

    def warm_up_serving_thread(self) -> None:
        pass

    def start(self) -> None:
        self.running = True
        while self.running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            if msg.type != "new_request":
                continue
            payload: StagePayload = msg.data
            for step in range(self.steps):           # 自回归：一步一个码，立刻往下游推
                codes = torch.tensor([step * 10 + 1, step * 10 + 2])
                self.outbox.put(OutgoingMessage(msg.request_id, "stream", data=codes,
                                                metadata={"step": step}))
            payload.data = {"num_steps": self.steps}
            self.outbox.put(OutgoingMessage(msg.request_id, "result", data=payload))

    def stop(self) -> None:
        self.running = False

    def abort(self, request_id: str) -> None:
        pass


def create_talker():
    return ToyTalkerScheduler()


class ToyVocoderScheduler(StreamingSimpleScheduler):
    """收到一个码就"解码"成一段音频（这里是码乘 100），流式发给客户端。"""

    def __init__(self):
        super().__init__(compute_fn=None)
        self.decoded: dict[str, list[int]] = {}

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        return True

    def on_stream_chunk(self, request_id, item):
        samples = (item.data * 100).tolist()
        self.decoded.setdefault(request_id, []).extend(samples)
        return [OutgoingMessage(request_id, "stream", data={"samples": samples},
                                metadata={"modality": "audio"})]

    def on_stream_done(self, request_id):
        payload = self.stream_payloads[request_id]
        payload.data = {"total_samples": len(self.decoded.pop(request_id, []))}
        return [OutgoingMessage(request_id, "result", data=payload)]


def create_vocoder():
    return ToyVocoderScheduler()
```

```python title="ch4_stream.py"
import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import CompleteMessage, StreamMessage

M = "ch4_stream_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="talker",
        stages=[
            StageConfig(name="talker", process="p_talker", factory_path=f"{M}.create_talker",
                        next="vocoder", stream_to=["vocoder"]),
            StageConfig(name="vocoder", process="p_vocoder", factory_path=f"{M}.create_vocoder",
                        terminal=True, can_accept_stream_before_payload=True),
        ],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    try:
        async for msg in runner.coordinator.stream("r1", "say something"):
            if isinstance(msg, StreamMessage):
                print(f"流式块 #{msg.chunk_id} 来自 {msg.from_stage}：{msg.chunk}")
            elif isinstance(msg, CompleteMessage):
                print(f"完成 来自 {msg.from_stage}：{msg.result}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

```text title="输出"
流式块 #0 来自 vocoder：{'samples': [100, 200]}
流式块 #1 来自 vocoder：{'samples': [1100, 1200]}
流式块 #2 来自 vocoder：{'samples': [2100, 2200]}
完成 来自 vocoder：{'total_samples': 6}
```

这几十行就是 Qwen3-Omni 里 thinker → talker → code2wav 那条流式链路的骨架。要点：

- talker 的码是 `torch.Tensor`：跨进程的流式 chunk 必须是张量，通信层要把它放进 relay 的缓冲区（第六章）；
- 声码器的 `can_accept_stream_before_payload=True` 不能省：talker 的码流和"完整 payload"走的是两条路，码可能先到。`StreamingSimpleScheduler` 用 `pending_done` 和 `stream_payloads` 两张表处理"先收到流结束、后收到 payload"的情况；
- 终点 stage 的流式输出没有 `stream_to` 下游，于是被发给 Coordinator，变成 `coordinator.stream()` 里的 `StreamMessage`——HTTP 层的 SSE 就是从这里来的。

## 练习

**1. 加一条动态扇入。** 给实验一加一个 `wait_for_fn`：如果原文只有一个词，就只等 `upper`（同时让 `split` 的 `route_fn` 只发给 `upper`）。验证两种请求都能正确完成。

??? success "参考思路"
    `wait_for_fn` 的签名是 `(request_id, from_stage, payload) -> 来源列表`，它在第一份输入到达时被调用一次。可以根据 `payload.request.inputs` 判断单词数，返回 `["upper"]` 或 `["upper", "count"]`；`route_fn` 用同样的条件返回下游列表。两个函数的条件必须一致，否则 join 会永远等不齐。

**2. 让一个 stage 失败。** 在 `create_count` 里对某个输入抛异常，观察 Coordinator 那边 `submit` 抛出的错误，以及 join 里那个请求攒了一半的输入去了哪里。

??? success "参考思路"
    `SimpleScheduler` 捕获异常后放一条 `type="error"` 的消息，Stage 调 `send_failure` 报给 Coordinator；Coordinator 快速失败，广播 abort。join 收到 abort 后在 `on_abort` 里调用输入处理器的 `cancel(request_id)`，`AggregatedInput.cancel` 把 `pending` 里那一半删掉——否则每个失败请求都会泄漏一点内存。

**3. 读 `StreamingSimpleScheduler` 的乱序处理。** 找到"流结束信号比 payload 先到"时的代码路径，说明 `on_stream_done_before_payload` 这个钩子是为什么样的模型准备的。

??? success "参考思路"
    `handle_stream_done` 里如果 `request_id` 还不在 `stream_payloads`，就记进 `pending_done`，并立刻输出 `on_stream_done_before_payload` 返回的消息；等 payload 到了（`handle_streaming_new_request`）再补做完成。这个钩子给那些"已经算出来的东西不应该等 payload"的阶段用——比如声码器手里已经解码好的最后一段音频，应该马上发给客户端，而不是等完整 payload 走完另一条路。

!!! interview "怎么讲清楚"
    讲 omni 的 stage，先讲约束：Stage 是 IO 壳，所有调度器同一个接口，Stage 不按调度器类型分支。再讲机制：两个线程（asyncio 做 IO、调度器线程做计算）+ 两个队列；结果路由在启动时编译成 `get_next`，扇出时按目标投影，扇入时按 `request_id` 攒齐再合并；流式靠 `stream_to` 和"流结束"信号。最后用一个真实的权衡收尾：攒批窗口在低并发时会变成纯延迟，#1628 把计时改成"第二个请求加入才开始"。

## 小结

- [x] Stage = asyncio 事件循环（IO）+ 调度器线程（计算），之间只有 `inbox` / `outbox` 两个 `queue.Queue`。
- [x] 进 4 种消息（新请求、流式块、流结束、abort），出 5 种（result、stream、error、admitted、kv_transfer）；只路由仍活跃的请求。
- [x] 路由在启动时编译：终点 → Coordinator，`route_fn` 动态选择（限静态 `next` 的子集），`project_payload` 给每个下游投影。
- [x] 扇入用 `AggregatedInput`：按 `request_id` 攒、按来源名合并，`wait_for_fn` 动态决定要等谁。
- [x] 攒批窗口在空闲时是纯延迟：#1628 改成第二个请求加入才计时。
- [x] 流式：上游 `stream_to`，下游 `can_accept_stream_before_payload`，终点的流式输出变成 Coordinator 的 `StreamMessage`。
