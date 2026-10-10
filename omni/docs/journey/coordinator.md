# 从 HTTP 到 Coordinator：请求对象、生命周期与多终点

<p class="lead">一条请求进入 SGLang-Omni，先后经过三个对象：FastAPI 写的 HTTP 层、一个叫 <code>Client</code> 的内部适配器、一个全局唯一的 <code>Coordinator</code>。前两个只负责"翻译"——把 OpenAI 格式的请求变成内部的 <code>OmniRequest</code>，把结果拼回 OpenAI 格式；真正管请求生命周期的是 Coordinator：它把请求送进入口 stage，等终点 stage 报完成，必要时把多个终点的结果合并，出错或取消时向所有 stage 广播 abort。这一章用真实的 omni 运行时在 CPU 上搭玩具流水线，亲手验证这些行为。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. `sgl-omni serve` 启动时，流水线进程和 HTTP 服务谁先起来？
    2. 一条请求在 Coordinator 里有哪几种状态？Coordinator 什么时候不再跟踪它？
    3. 有两个终点 stage 时，`submit()` 返回的结果长什么样？只有一个终点时呢？
    4. 一个请求被 abort 之后，能不能用同一个 request id 再提交一次？为什么？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 流水线先起来：`run_server` 先 `await mp_runner.start()`（拉起所有 stage 进程、等它们就绪），拿到 Coordinator 后才构造 `Client` 和 FastAPI 应用、启动 uvicorn。
    2. `PENDING → RUNNING → COMPLETED / FAILED / ABORTED`。完成、失败（且 abort 广播成功）、取消之后都会从 `requests` 字典里删掉。
    3. 多终点时是一个字典 `{终点 stage 名: 该 stage 的结果}`；单终点时就是那个 stage 的结果本身，没有外面那层字典。
    4. 不能。stage 会把被 abort 或失败过的 id 记进 `aborted` 集合，之后用这个 id 提交，入口 stage 直接回一个失败："retired after abort or failure, use a fresh request ID"。

## 启动顺序：先流水线，后 HTTP

`sgl-omni serve` 最终调到 `serve/launcher.py` 的 `run_server`。把它的主干摘出来：

```python title="sglang_omni/serve/launcher.py @ 921ea2c8 L487-490,514-515,567-572"
    mp_runner = MultiProcessPipelineRunner(pipeline_config)
    startup_timeout = float(os.environ.get("SGLANG_OMNI_STARTUP_TIMEOUT", "600"))
    await mp_runner.start(timeout=startup_timeout)
    coordinator = mp_runner.coordinator
...
        cl_kwargs = client_kwargs or {}
        client = Client(coordinator, **cl_kwargs)
...
        )
        server = PipelineUvicornServer(config)
        await serve_with_failure_watch(server, [mp_runner.wait_failed()])
    finally:
        logger.info("Shutting down pipeline …")
        await mp_runner.stop()
```

三件事按顺序发生：

1. `MultiProcessPipelineRunner.start()` 解析配置、规划进程拓扑、创建 Coordinator，然后按"波次"拉起所有 stage 进程，等每个进程报就绪（第七章细讲）；
2. 用 Coordinator 构造 `Client`，再用 `Client` 构造 FastAPI 应用（`create_app` 注册了 `/v1/chat/completions`、`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/realtime` 等几十个路由）；
3. 启动 uvicorn，同时盯着 `mp_runner.wait_failed()`——任何一个 stage 进程挂掉，整个服务退出，而不是留一个半残的流水线继续接请求。

## Client：协议翻译

HTTP 层的每个端点最终都调 `Client` 的某个方法（`completion`、`completion_stream`、`speech`……），它们都建立在 `generate` 上：

```python title="sglang_omni/client/client.py @ 921ea2c8 L97-118"
    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        req_id = request_id or str(uuid.uuid4())
        omni_request = self.build_omni_request(request)
        if request.stream:
            coordinator_stream = self.coordinator.stream(req_id, omni_request)
            async with aclosing(coordinator_stream):
                async for msg in coordinator_stream:
                    if isinstance(msg, StreamMessage):
                        yield self.stream_builder(req_id, msg)
                    else:
                        yield self.result_builder(req_id, msg.result)
            return
        else:
            pass

        result = await self.coordinator.submit(req_id, omni_request)
        yield self.result_builder(req_id, result)

```

流式请求走 `coordinator.stream()`，一边收 `StreamMessage`（中间结果）一边往外 yield，最后收到 `CompleteMessage`；非流式请求走 `coordinator.submit()`，等一个最终结果。`build_omni_request` 把 OpenAI 格式的请求拆成三部分：

```python title="sglang_omni/proto/request.py @ 921ea2c8 L37-43"
@dataclass
class OmniRequest:
    """User-facing request with inputs and parameters."""

    inputs: object
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
```

`inputs` 是模型输入（文本、消息列表、音频……），`params` 是生成参数（采样、`max_tokens`、要不要音频输出……），`metadata` 是路由和观测用的附加信息。在 stage 之间流动的是包了一层的 `StagePayload`：

```python title="sglang_omni/proto/request.py @ 921ea2c8 L62-68"
@dataclass
class StagePayload:
    """Payload passed between stages with request context."""

    request_id: str
    request: OmniRequest
    data: object
```

`request` 始终是同一个 `OmniRequest`（下游 stage 随时能读到用户的原始参数），`data` 是上一个 stage 的输出，每个 stage 都会替换它。Coordinator 送进入口 stage 的第一个 payload，`data` 是 `{"raw_inputs": request.inputs}`。

## Coordinator：一张表、几个 future、两种 socket

Coordinator 的状态就是几张字典：

```python title="sglang_omni/pipeline/coordinator.py @ 921ea2c8 L122-136"
        # Control plane
        self.control_plane = CoordinatorControlPlane(
            completion_endpoint=completion_endpoint,
            abort_endpoint=abort_endpoint,
        )

        # Stage registry
        self.stages: dict[str, StageInfo] = {}

        # Request tracking
        self.requests: dict[str, RequestInfo] = {}
        self.completion_futures: dict[str, asyncio.Future] = {}
        self.stream_queues: dict[
            str, asyncio.Queue[CompleteMessage | StreamMessage]
        ] = {}
```

`requests` 记每个请求的 `RequestInfo`（状态、要等的终点、结果），`completion_futures` 给 `submit()` 的调用方挂一个 future，`stream_queues` 给流式调用方挂一个队列。和 stage 之间的通信全走 ZMQ：

```python title="sglang_omni/pipeline/control_plane.py @ 921ea2c8 L392-414"
class CoordinatorControlPlane:
    """Control plane interface for the Coordinator.

    Handles:
    - Submitting work to entry stage (PUSH)
    - Receiving completions from stages (PULL)
    - Broadcasting abort signals (PUB)
    """

    def __init__(self, completion_endpoint: str, abort_endpoint: str):
        self.completion_endpoint = completion_endpoint
        self.abort_endpoint = abort_endpoint
        self.completion_socket: PullSocket | None = None
        self.abort_socket: PubSocket | None = None
        self.stage_sockets: dict[str, PushSocket] = {}

    async def start(self) -> None:
        """Initialize all sockets."""
        self.completion_socket = PullSocket(self.completion_endpoint, bind=True)
        await self.completion_socket.start()
        self.abort_socket = PubSocket(self.abort_endpoint)
        await self.abort_socket.bind()
        logger.info("Coordinator control plane started")
```

![图：Coordinator 与 stage 之间的消息](../assets/figures/omni-control-plane.svg){.aig-svg}

- **PUSH → 入口 stage**：`SubmitMessage`，每个 stage 一个 socket，按需建立；
- **PULL ← 所有 stage**：`CompleteMessage`（终点完成或任何 stage 失败）和 `StreamMessage`（终点的流式中间结果）共用一个收件口；
- **PUB → 所有 stage**：`AbortMessage`，只带一个 `request_id`，每个 stage 用 SUB 订阅。

stage 和 stage 之间不经过 Coordinator：上游直接 PUSH `DataReadyMessage` 给下游（第六章）。所以 Coordinator 只看得到"进"和"出"，看不到中间——这是它能做到与 stage 实现无关的原因。

### 提交

`submit_request` 是入口。去掉检查和日志，核心是这几步：

```python title="sglang_omni/pipeline/coordinator.py @ 921ea2c8 L512-555"
        # Track request
        self.requests[request_id] = RequestInfo(
            request_id=request_id,
            state=RequestState.PENDING,
            current_stage=self.entry_stage,
            terminal_stages=(
                self.resolve_terminal_stages(request)
                if terminal_stages is None
                else terminal_stages
            ),
        )

        # Create future for completion
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self.completion_futures[request_id] = future
        if stream_queue is not None:
            self.stream_queues[request_id] = stream_queue
        else:
            pass

        payload = StagePayload(
            request_id=request_id,
            request=request,
            data={"raw_inputs": request.inputs},
        )

        _emit_event(
            request_id=request_id,
            stage="coordinator",
            event_name="request_admission",
            metadata={"entry_stage": self.entry_stage},
        )

        try:
            await self.control_plane.submit_to_stage(
                entry_instance,
                entry_info.control_endpoint,
                SubmitMessage(
                    request_id=request_id,
                    data=payload,
                    replica_bindings=replica_bindings,
                ),
            )
```

注意 `terminal_stages` 在提交时就算好了：`resolve_terminal_stages(request)` 可以按请求决定这次要等哪几个终点（下面的实验会用到）。提交前还有一道准入：如果配置了 `max_in_flight`（通常等于引擎的 `max_running_requests + max_queued_requests`），在途请求满了就直接抛 `QueueFullError`，不往流水线里塞。

### 完成与合并

所有 `CompleteMessage` 都进 `handle_completion`。失败的处理是"快速失败"：任何一个 stage 报失败，整个请求失败，并广播 abort 让其他 stage 清理。成功的处理分单终点和多终点：

```python title="sglang_omni/pipeline/coordinator.py @ 921ea2c8 L838-888"
        # Single active terminal (original behavior) or no terminal_stages configured
        if len(expected_terminal_stages) <= 1:
            info.state = RequestState.COMPLETED
            info.result = msg.result
            if request_id in self.completion_futures:
                future = self.completion_futures[request_id]
                if not future.done():
                    future.set_result(msg.result)
                else:
                    pass
            else:
                pass
            if request_id in self.stream_queues:
                await self.stream_queues[request_id].put(msg)
            else:
                pass
            self.requests.pop(request_id, None)
            return
        else:
            pass

        # Multi-terminal: collect partial results
        partials = self.partial_results.setdefault(request_id, {})
        partials[from_stage] = msg.result

        # Forward stream completion per-stage
        if request_id in self.stream_queues:
            await self.stream_queues[request_id].put(msg)
        else:
            pass

        if set(partials) < expected_terminal_stages:
            return  # still waiting
        else:
            pass

        # All terminal stages done -> merge and resolve
        merged = dict(partials)
        self.partial_results.pop(request_id)
        info.state = RequestState.COMPLETED
        info.result = merged

        if request_id in self.completion_futures:
            future = self.completion_futures[request_id]
            if not future.done():
                future.set_result(merged)
            else:
                pass
        else:
            pass
        self.requests.pop(request_id, None)
```

单终点时，future 的结果就是那个 stage 的 `result`；多终点时，先攒进 `partial_results`，等齐了再合并成 `{stage 名: 结果}`。Qwen3-Omni 带语音输出时就是两个终点：`decode` 出文本、`code2wav` 出音频。

## 动手：在 CPU 上跑真实的 Coordinator

下面的实验用的是 omni 真实的 `MultiProcessPipelineRunner`、`Coordinator`、`Stage` 和 ZMQ 控制面——只是每个 stage 的计算换成了几行 Python。环境的装法见[首页](../index.md#环境)，不需要 GPU。

先写几个玩具 stage。omni 里一个 stage 的"工厂"就是一个返回调度器的函数，这里全用最简单的 `SimpleScheduler`（下一章细讲），它包住一个 `StagePayload → StagePayload` 的函数：

```python title="ch3_stages.py"
"""第三章的玩具 stage：每个 stage 都是一个 SimpleScheduler 包着一个普通函数。"""
import time

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def create_split():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"text": payload.data["raw_inputs"]}
        return payload
    return SimpleScheduler(fn)


def create_text():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"upper": payload.data["text"].upper()}
        return payload
    return SimpleScheduler(fn)


def create_audio():
    def fn(payload: StagePayload) -> StagePayload:
        payload.data = {"samples": len(payload.data["text"]) * 100}
        return payload
    return SimpleScheduler(fn)


def route_split(request_id, output):
    """动态路由：只有请求要了音频才发给 audio。"""
    return ["text", "audio"] if output.request.params.get("want_audio") else ["text"]


def resolve_terminals(request):
    """按请求决定 Coordinator 要等哪几个终点。"""
    return ["text", "audio"] if request.params.get("want_audio") else ["text"]


def create_slow():
    def fn(payload: StagePayload) -> StagePayload:
        time.sleep(3)
        return payload
    return SimpleScheduler(fn)
```

### 实验一：单终点和多终点

一条三个 stage 的流水线：`split` 之后分叉到 `text` 和 `audio`，两个都是终点。`route_fn` 让 `split` 按请求决定发给谁，`terminal_stages_fn` 让 Coordinator 按请求决定等谁——这正是 Qwen3-Omni"要不要语音输出"的做法：

```python title="ch3_terminals.py"
import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest

M = "ch3_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="split",
        stages=[
            StageConfig(name="split", process="p_split", factory_path=f"{M}.create_split",
                        next=["text", "audio"], route_fn=f"{M}.route_split"),
            StageConfig(name="text", process="p_text", factory_path=f"{M}.create_text", terminal=True),
            StageConfig(name="audio", process="p_audio", factory_path=f"{M}.create_audio", terminal=True),
        ],
        terminal_stages_fn=f"{M}.resolve_terminals",
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    coord = runner.coordinator
    try:
        print("静态终点：", sorted(coord.terminal_stages))
        r1 = await coord.submit("r1", OmniRequest(inputs="hi omni", params={"want_audio": False}))
        print("只要文本：", r1)
        r2 = await coord.submit("r2", OmniRequest(inputs="hi omni", params={"want_audio": True}))
        print("文本+音频：", r2)
        print("还在跟踪的请求：", list(coord.requests))
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

```text title="输出"
静态终点： ['audio', 'text']
只要文本： {'upper': 'HI OMNI'}
文本+音频： {'text': {'upper': 'HI OMNI'}, 'audio': {'samples': 700}}
还在跟踪的请求： []
```

三个 stage 各在一个独立的 OS 进程里（`process` 各不相同），请求走了真实的 ZMQ 和共享内存。几个值得注意的地方：

- 单终点的结果没有外层字典，多终点的结果按 stage 名分开——客户端代码（`Client.default_result_builder`）要同时处理这两种形状；
- `resolve_terminals` 返回的终点必须是静态终点的子集，否则 `resolve_terminal_stages` 直接报错——动态逻辑只能在静态拓扑里做选择，不能凭空加边。`route_fn` 也有同样的约束（`construct_stage` 里的 `_target_result`）；
- 请求完成后 `requests` 就空了，Coordinator 不留历史。

`route_fn` 和 `terminal_stages_fn` 必须对得上：如果路由只发给了 `text`，Coordinator 却在等 `audio`，请求会永远挂着。omni 没有替你检查这件事，这是写新模型时容易出错的地方。

### 实验二：abort 与"退役"的 request id

```python title="ch3_abort.py"
import asyncio
import tempfile

from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

M = "ch3_stages"


def make_config() -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="slow",
        stages=[StageConfig(name="slow", process="p_slow", factory_path=f"{M}.create_slow", terminal=True)],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    coord = runner.coordinator
    try:
        task = asyncio.create_task(coord.submit("r1", "sleep please"))
        await asyncio.sleep(0.5)
        print("提交后的状态：", coord.get_request_info("r1").state.value)
        print("abort 返回：", await coord.abort("r1"))
        try:
            await task
        except asyncio.CancelledError as exc:
            print("submit 的结果：CancelledError", exc)
        print("abort 之后还在跟踪吗：", "r1" in coord.requests)
        try:
            await coord.submit("r1", "again")
        except Exception as exc:
            print("同一个 id 再提交：", type(exc).__name__, str(exc)[:70])
        print("换个 id：", await coord.submit("r2", "again"))
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

```text title="输出"
提交后的状态： running
abort 返回： True
submit 的结果：CancelledError Request r1 aborted
abort 之后还在跟踪吗： False
同一个 id 再提交： RuntimeError Request r1 (retired after abort or failure, use a fresh request ID) al
换个 id： {'raw_inputs': 'again'}
```

`slow` stage 正在 `time.sleep(3)` 的时候 abort 到了。Coordinator 这一侧的处理是 `run_abort`：

```python title="sglang_omni/pipeline/coordinator.py @ 921ea2c8 L677-711"
    async def run_abort(
        self,
        request_id: str,
    ) -> bool:
        await self.control_plane.broadcast_abort(AbortMessage(request_id=request_id))

        info = self.requests.get(request_id)
        if info is None:
            return False
        else:
            pass

        info.state = RequestState.ABORTED
        self.reject_completion_future(
            request_id, asyncio.CancelledError(f"Request {request_id} aborted")
        )
        stream_queue = self.stream_queues.get(request_id)
        if stream_queue is not None:
            await stream_queue.put(
                CompleteMessage(
                    request_id=request_id,
                    from_stage="coordinator",
                    success=False,
                    error="aborted",
                )
            )
        else:
            pass

        self.requests.pop(request_id, None)
        self.partial_results.pop(request_id, None)

        logger.info("Coordinator aborted req=%s", request_id)
        return True

```

先广播，再把状态改成 `ABORTED`、拒绝调用方的 future（所以 `submit` 抛 `CancelledError`）、给流式调用方塞一个失败的 `CompleteMessage`，最后删掉记录。stage 那一侧：收到 abort 的 stage 把 id 记进 `aborted` 集合，调度器算完这一步发现请求已被取消，结果直接丢掉（`SimpleScheduler.consume_if_aborted`）。

为什么同一个 id 不能再用？因为 abort 是**异步广播**：下游 stage 可能晚一点才收到 abort，也可能还有上游发出的数据在路上。如果允许复用 id，一条迟到的旧数据就可能被当成新请求的输入。stage 用"退役"集合把这扇门关死，入口 stage 收到退役 id 的 `SubmitMessage` 会明确回一个失败：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L527-538"
    async def on_submit(self, msg: SubmitMessage) -> None:
        request_id = msg.request_id
        if request_id in self.aborted:
            # A new coordinator admission needs an explicit answer. Late
            # downstream data still follows the silent stale-result path. The
            # message keeps the duplicate ID form so both map to one status.
            await self.send_failure(
                request_id,
                f"Request {request_id} (retired after abort or failure, use a "
                "fresh request ID) already exists",
            )
            return
```

HTTP 层每次都生成新的 UUID 作为 request id，所以正常使用碰不到这个问题；但写测试、做重试逻辑的时候要记住它。另外，这个退役集合是有上限的（`record_bounded_request_id`：超过一万个 id 时删掉五千个），所以"退役"只保证在一段时间内有效，不是永久的。

### 实验三：这些进程靠什么找到彼此

```python title="ch3_endpoints.py"
import asyncio
from pathlib import PurePosixPath

from ch3_terminals import make_config
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner


async def main() -> None:
    runner = MultiProcessPipelineRunner(make_config())
    await runner.start(timeout=120)
    try:
        for name, endpoint in sorted(runner.prep.endpoints.items()):
            scheme, _, path = endpoint.partition("://")
            print(f"{name:18s} {scheme}://<运行目录>/{PurePosixPath(path).name}")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

```text title="输出"
abort              ipc://<运行目录>/abort.sock
comm_audio_rank0   ipc://<运行目录>/comm_audio_rank0.sock
comm_split_rank0   ipc://<运行目录>/comm_split_rank0.sock
comm_text_rank0    ipc://<运行目录>/comm_text_rank0.sock
completion         ipc://<运行目录>/completion.sock
stage_audio        ipc://<运行目录>/stage_audio.sock
stage_split        ipc://<运行目录>/stage_split.sock
stage_text         ipc://<运行目录>/stage_text.sock
```

全是 `ipc://` 开头的 Unix 域套接字，放在一个临时运行目录里：`completion` 和 `abort` 属于 Coordinator，`stage_*` 是每个 stage 的收件口，`comm_*` 是数据面（第六章）用的。Unix 域套接字的路径有长度上限（Linux 上 108 字节），omni 在创建运行目录时会检查 `base_path + 最长的后缀` 是否超限——如果你的用户名或临时目录很长，跑 omni 自己的单测时可能遇到 "IPC endpoint path would exceed the Unix-domain socket path limit"，给 pytest 加 `--basetemp=/tmp/xx` 就好（第十一章）。

## 练习

**1. 读流式路径。** `Coordinator.stream()` 在什么条件下结束？如果调用方中途不再读（比如 HTTP 客户端断开），会发生什么？

??? success "参考思路"
    读 `coordinator.py` 的 `stream`：收齐 `expected_terminal_stages` 里每个终点的 `CompleteMessage` 就 `return`；任何一个失败消息会抛异常。它的 `finally` 里如果请求还在 `requests` 里，会调用 `self.abort(request_id)`——调用方停止迭代（异步生成器被关闭）就会触发 abort，所以客户端断开会把整条流水线上的这个请求取消掉。

**2. 失败也要广播。** `handle_completion` 收到失败消息时，如果广播 abort 本身失败了，为什么要"保留这个 id 和准入名额"而不是直接删掉？

??? success "参考思路"
    广播失败意味着别的 stage 可能还持有这个请求（占着 KV、占着队列）。如果 Coordinator 把它删掉，准入计数会以为有空位，再放新请求进来，实际资源却没释放。保留记录，等之后的 abort 或 stop 再释放，是"宁可少放、不可多放"。

**3. 自己写一个多终点的错误用法。** 把实验一里的 `route_split` 改成永远只返回 `["text"]`，但 `resolve_terminals` 不改，提交一个 `want_audio=True` 的请求会怎样？怎么用 `asyncio.wait_for` 让实验不至于卡死？

??? success "参考思路"
    请求会一直挂着：Coordinator 在等 `audio` 的完成，而没有任何数据发给 `audio`。用 `await asyncio.wait_for(coord.submit(...), timeout=5)` 观察 `TimeoutError`，超时后记得 `await coord.abort(...)`。这类"路由和终点对不上"的问题，在真实模型里通常表现为某类请求偶发超时。

!!! interview "怎么讲清楚"
    讲 omni 的请求生命周期，抓住"Coordinator 只看两头"：提交时 PUSH 一个 `SubmitMessage` 给入口 stage，并按请求算好要等的终点；中间 stage 之间直接传，Coordinator 看不到；终点通过 PULL 报完成，多终点时攒齐了合并成按 stage 名分开的字典；失败快速传播，abort 用 PUB/SUB 广播，id 一旦被取消就"退役"。启动顺序上先拉起流水线、后起 HTTP，任何 stage 进程挂掉整个服务退出。

## 小结

- [x] `run_server`：先 `mp_runner.start()`，再 `Client` → FastAPI → uvicorn，并监视 stage 进程。
- [x] `Client` 只做协议翻译：OpenAI 请求 → `OmniRequest(inputs, params, metadata)`；stage 之间传 `StagePayload(request_id, request, data)`。
- [x] Coordinator：`requests` / futures / stream 队列 + ZMQ（PUSH 提交、PULL 收完成和流、PUB 广播 abort）。
- [x] 单终点返回结果本身，多终点返回 `{stage: 结果}`；终点可以按请求动态决定，但必须是静态终点的子集。
- [x] abort 是异步广播，被取消的 id 会"退役"，不能复用。
