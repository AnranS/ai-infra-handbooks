# 进程间通信：控制面、数据面与按边选传输

<p class="lead">多阶段流水线的代价，是 stage 之间要搬数据：编码器输出的特征、thinker 的隐状态、talker 的码流，小的几十字节，大的几兆。omni 把通信分成两个面——ZMQ 上跑的控制面只传"消息和元数据"，张量走 relay 数据面——并且<b>每条边单独决定用什么方式传</b>：同进程直接传 Python 对象，同一张卡用 CUDA IPC 传句柄，同机跨卡用 CUDA IPC 显存池，CPU 张量走共享内存，跨机走 Mooncake。这一章读这套选择逻辑和打包格式，然后在 CPU 上打开通信追踪，亲眼看每条边选了什么，最后故意撞一次边界。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个 4 MB 的 CPU 张量从一个 stage 发到另一个进程里的 stage，走哪条路？如果两个 stage 在同一个进程里呢？
    2. relay 传的到底是什么？`StagePayload` 里的张量和别的字段是怎么分开的？
    3. 很小的流式 chunk（比如几十字节的码）走 relay 吗？
    4. README 说数据面支持 SHM、NCCL、NIXL、Mooncake 四种后端。在基准提交里，stage 之间的数据面实际会用到哪几种？
    5. relay 的 `credits` 是干什么的？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 跨进程的 CPU 张量走 SHM：发送方把张量拷进一块共享内存，控制面发一条带共享内存名字的 `DataReadyMessage`，接收方读出来后回一个 `DataAckMessage`。同进程（`process` 相同）时不经过通信层，直接把 Python 对象交给目标 stage（`LOCAL_OBJECT`）。
    2. `write_payload` 用 `extract_tensors` 把 `payload.data` 里的张量全部抽出来，换成占位符；张量拼成一个 `uint8` 大缓冲区交给 relay，剩下不含张量的 `StagePayload` 用 pickle + base64 编码成 header，和每个张量的形状、dtype、偏移一起放进 `DataRef`，随控制消息发出。
    3. 不走。不超过 16 KB 的 CPU 张量 chunk 直接 pickle 进控制消息（`inline`），省掉一次共享内存的分配和 ACK。这是 #1574 为了缩小 thinker → talker 的流式开销加的。
    4. 三种：`cuda_ipc`、`shm`、`mooncake`（`CommRouter.build_relay` 只会构造这三种）；另外还有不经过 relay 的同进程直传和"同一张卡的 PyTorch CUDA IPC 直传"。`relay/` 目录里有 NCCL、NIXL 的实现，但 stage 数据面的路由不会选它们。
    5. 背压：每个 relay 实例用一个信号量限制同时在途的传输数（默认 2），`put_async` 先拿信号量，接收方 ACK 之后才释放。下游读得慢，上游的发送就会在这里排队，而不是无限占内存。

## 两个面

![图：控制面与数据面](../assets/figures/omni-transport-ladder.svg){.aig-svg}

官方的通信文档有一张总表：

```text title="docs/developer_reference/communication.md @ 921ea2c8 L56-65"
| Path                     | Transport      | Carries                                                                                                      |
| ------------------------ | -------------- | ------------------------------------------------------------------------------------------------------------ |
| Coordination             | ZMQ `PUSH/PULL` | `SubmitMessage`, `DataReadyMessage`, `CompleteMessage`, `StreamMessage`, `ShutdownMessage`, profiler control |
| Broadcast coordination   | ZMQ `PUB/SUB`   | `AbortMessage`                                                                                               |
| Same-process movement    | LOCAL_OBJECT   | Full `StagePayload` objects and stream chunks passed by Python reference within one OS process                |
| Same-placement direct GPU movement | PyTorch CUDA IPC | CUDA storage handles plus ordinary payload or stream control metadata                              |
| Same-node pooled GPU movement | CUDA IPC relay | Packed payload tensor buffers, CUDA stream chunks, and stream metadata tensors                         |
| Local CPU relay movement | SHM relay      | Full payload tensor buffers and stream chunks that are not CUDA-local                                        |
| Cross-node movement      | Mooncake relay | Full payload tensor buffers and stream chunks over Mooncake-selected transport                               |

```

控制面（ZMQ）第三章已经看过：Coordinator 和 stage 之间、stage 和 stage 之间都用 PUSH/PULL 发消息，abort 用 PUB/SUB 广播。这一章的主角是数据面：张量怎么从一个进程到另一个进程。

## 按边选传输：CommRouter

每个 stage 有一个 `CommRouter`，它的职责写在类文档里：

```python title="sglang_omni/comm/router.py @ 921ea2c8 L23-28"
class CommRouter:
    """Maps an edge to the physical mover and owns relay instances.

    The router classifies locality. It does not define the stage protocol and it
    does not expose Mooncake-specific handles to stage code.
    """
```

"传输方式"一共四种：

```python title="sglang_omni/comm/data_ref.py @ 921ea2c8 L15-19"
class TransportKind(str, Enum):
    LOCAL_OBJECT = "local_object"
    CUDA_IPC = "cuda_ipc"
    SHM = "shm"
    MOONCAKE = "mooncake"
```

完整 payload 的选择逻辑：

```python title="sglang_omni/comm/router.py @ 921ea2c8 L301-321"
    def outbound_payload(self, target: str, payload: object) -> TransportKind:
        if target in self.remote_stage_names:
            kind = TransportKind.MOONCAKE
        else:
            devices = tensor_devices(getattr(payload, "data", payload))
            if not devices or devices == {"cpu"}:
                kind = TransportKind.SHM
            elif current_platform.device_type in devices and devices <= {
                "cpu",
                current_platform.device_type,
            }:
                if self.self_is_gpu and target in self.gpu_stage_names:
                    kind = self.intra_node_transport(target)
                else:
                    kind = TransportKind.SHM
            else:
                raise ValueError(
                    f"mixed or unsupported tensor devices in payload: {devices}"
                )
        return self.note_transport("payload", target, kind)

```

先看对方是不是远端（跨机 → Mooncake）；再看 payload 里张量在哪：没有张量或全在 CPU → SHM；在加速器上、且双方都是 GPU stage → 平台的机内传输（CUDA 平台上是 CUDA IPC，对方不可达时退回 SHM）。同进程的情况更早就被截走了：`Stage.send_to_stage` 发现目标在同一个进程里，直接调本进程的分发器把对象交过去，根本不进 `CommRouter`：

```python title="sglang_omni/pipeline/stage/runtime.py @ 921ea2c8 L1602-1612"
        projector = self.project_payload.get(target)
        projected_payload = projector(payload) if projector is not None else payload
        use_local_object = allow_local_object or (
            allow_projected_local_object
            and self.is_isolated_projected_payload(
                payload,
                projected_payload,
                projector_present=projector is not None,
            )
        )

```

这里有个细节：一份结果扇出给多个下游时（`allow_projected_local_object`），只有投影后的 payload 和原 payload **不共享可变容器**，才允许同进程直传——否则两个下游拿到同一个列表的引用，一个改了另一个也变。`is_isolated_projected_payload` 就是做这个检查的。

流式 chunk 的逻辑多一个分支：

```python title="sglang_omni/comm/router.py @ 921ea2c8 L233-253"
    def outbound_stream(self, target: str, data: torch.Tensor) -> TransportKind:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "relay-backed stream chunks must be torch.Tensor, got "
                f"{type(data).__name__}"
            )
        else:
            pass
        if target in self.remote_stage_names:
            kind = TransportKind.MOONCAKE
        elif data.device.type != current_platform.device_type:
            kind = TransportKind.SHM
        elif self.self_is_gpu and target in self.gpu_stage_names:
            kind = self.intra_node_transport(target)
        else:
            raise ValueError(
                f"{current_platform.device_type} stream chunk cannot be sent from "
                f"{self.stage_name!r} to non-GPU target {target!r}"
            )
        return self.note_transport("stream", target, kind)

```

注意最后的 `raise`：CPU 平台上，一个 CPU 张量 chunk 发给一个非 GPU 的目标，会直接报错。下面的实验会撞上它。

### 文档说的和代码做的

README 里写数据面"moves tensor payloads across shared-memory, NCCL, NIXL, and Mooncake backends"。实际构造 relay 的地方只有 `CommRouter.build_relay`：

```bash title="relay-kinds.sh"
echo "build_relay 构造的后端："
git grep -n -A1 'create_relay($' "$REF" -- sglang_omni/comm/router.py | grep -oE '"[a-z_]+"' | tr '\n' ' '; echo
echo "relay/ 目录里注册的后端："
git grep -hoE '^@register_relay\("[a-z_]+"\)' "$REF" -- sglang_omni/relay/ | grep -oE '"[a-z_]+"' | sort | tr '\n' ' '; echo
```

```text title="输出"
build_relay 构造的后端：
"cuda_ipc" "mooncake" "shm" 
relay/ 目录里注册的后端：
"cuda_ipc" "mooncake" "nccl" "nixl" "shm" 
```

NCCL 和 NIXL 的 relay 实现都在、也注册了，但 stage 之间的路由不会选到它们（`NCCL` 在仓库里另有用途：RL 训练时更新权重的进程组）。读一个快速演进的项目，**文档是线索，代码才是结论**；这种不一致本身也是一个可以提的小 PR（改文档，或者问维护者是否计划接回去）。

## 打包：张量和"其余部分"分开走

relay 只认一种东西：一个扁平的张量缓冲区。`write_payload` 负责把任意的 `StagePayload` 变成这个形状：

```python title="sglang_omni/comm/stage_io.py @ 921ea2c8 L508-542"
async def write_payload(
    relay: Relay,
    request_id: str,
    payload: StagePayload,
    *,
    transport: TransportKind,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> tuple[DataRef, RelayOperation]:
    data_without_tensors, tensors = extract_tensors(payload.data)
    packed, entries = pack_tensors(tensors, device=relay_device(relay))
    header = StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=data_without_tensors,
    )
    op = await relay.put_async(
        packed,
        request_id=request_id,
        receiver_id=to_stage,
    )
    data_ref = DataRef(
        version=1,
        object_id=f"{request_id}:payload:{from_stage or ''}:{to_stage or ''}",
        kind=DataKind.STAGE_PAYLOAD,
        transport=transport,
        layout=DataLayout.PACKED_TENSORS,
        buffer=BackendRef.from_relay_info(
            transport=transport,
            relay_info=op.metadata,
        ),
        header=base64.b64encode(pickle.dumps(header)).decode("ascii"),
        tensors=tuple(entries),
    )
    return data_ref, op
```

1. `extract_tensors` 递归遍历 `payload.data`，把每个张量换成一个占位符，记下路径；
2. `pack_tensors` 把所有张量按对齐要求拼进一个 `uint8` 缓冲区，记下每个张量的形状、dtype、偏移；
3. `relay.put_async` 把缓冲区放进 relay（SHM 是一块共享内存，CUDA IPC 是显存池里的一个槽位），返回接收方需要的元数据；
4. 不含张量的 `StagePayload` 被 pickle 后 base64 编码成 `header`，和张量清单、relay 元数据一起组成 `DataRef`，放进 `DataReadyMessage` 走控制面。

接收方反过来：读控制消息 → 从 relay 取缓冲区 → 按清单切出张量 → 填回占位符 → 回一个 ACK。发送方收到 ACK 才释放缓冲区。这个"发送方拥有缓冲区、接收方确认后释放"的约定，配合 relay 的信用额度，就是数据面的背压：

```python title="sglang_omni/relay/shm.py @ 921ea2c8 L173-198"
class ShmRelay(Relay):
    def __init__(
        self,
        engine_id: str,
        slot_size_mb: int = 64,
        credits: int = 2,
        device: str = "cpu",
    ):
        self.engine_id = engine_id
        self.device = device
        self.sem = asyncio.Semaphore(credits)
        self.slot_size_bytes = slot_size_mb * 1024 * 1024

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> ShmPutOperation:
        if request_id is None:
            request_id = str(uuid.uuid4())
        else:
            pass

        await self.sem.acquire()
```

`credits` 默认 2：同一个 relay 最多两笔在途，第三笔会在 `sem.acquire()` 上等，直到前面某一笔被 ACK、`release_cb` 释放信号量。

### 小 chunk 内联

流式 chunk 往往很小——talker 每步的码只有几十字节。为它们分配共享内存、等 ACK 不划算，于是有了内联：

```python title="sglang_omni/comm/stage_io.py @ 921ea2c8 L366-391"
_INLINE_STREAM_CHUNK_TYPE = "InlineStreamChunk"
_INLINE_STREAM_CHUNK_BYTES_LIMIT = 16 * 1024


def serialize_inline_stream_chunk(
    data: object, metadata: dict[str, object] | None
) -> InlineStreamChunkRef | None:
    if not isinstance(data, torch.Tensor) or data.device.type != "cpu":
        return None
    else:
        pass
    if contains_cuda_tensor(metadata) or contains_cpu_tensor(metadata):
        return None
    else:
        pass
    if data.element_size() * data.numel() > _INLINE_STREAM_CHUNK_BYTES_LIMIT:
        return None
    else:
        pass
    data = data.detach()
    if data.untyped_storage().nbytes() > _INLINE_STREAM_CHUNK_BYTES_LIMIT:
        data = data.clone(memory_format=torch.contiguous_format)
    else:
        pass
    payload = pickle.dumps((data, metadata))
    if len(payload) > _INLINE_STREAM_CHUNK_BYTES_LIMIT:
```

不超过 16 KB 的 CPU 张量直接 pickle 进控制消息。这个优化来自一个性能 PR：

```bash title="inline-origin.sh"
git log -S '_INLINE_STREAM_CHUNK_BYTES_LIMIT' --reverse --date=short --format='%ad %h %s' "$REF" -- sglang_omni/comm/stage_io.py | head -1
```

```text title="输出"
2026-08-24 1566230e [Qwen3-Omni Perf] Talker: shrink the thinker→talker stream payload (#1574)
```

## 动手：看每条边选了什么

一条小流水线：`src` 产生一个 4 MB 的"编码器输出"，分发给两个终点——同进程的 `near` 和另一个进程的 `far`；同时往 `far` 推一两段流。

??? example "ch6_stages.py（玩具 stage，展开看代码）"
    ```python title="ch6_stages.py"
    """src 产生一个 4 MB 的张量和一两段流，分别发给同进程的 near 和另一个进程的 far。"""
    import queue

    import torch

    from sglang_omni.proto import StagePayload
    from sglang_omni.scheduling.message import OutgoingMessage
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.scheduling.streaming_simple_scheduler import StreamingSimpleScheduler


    class SrcScheduler:
        def __init__(self, big_chunk: bool):
            self.inbox, self.outbox, self.running = queue.Queue(), queue.Queue(), False
            self.big_chunk = big_chunk

        def warm_up_serving_thread(self):
            pass

        def start(self):
            self.running = True
            while self.running:
                try:
                    msg = self.inbox.get(timeout=0.1)
                except queue.Empty:
                    continue
                if msg.type != "new_request":
                    continue
                chunks = [torch.arange(8)]                                   # 64 字节
                if self.big_chunk:
                    chunks.append(torch.zeros(16384, dtype=torch.float32))   # 64 KB
                for chunk in chunks:
                    self.outbox.put(OutgoingMessage(msg.request_id, "stream", data=chunk))
                payload = msg.data
                payload.data = {"features": torch.ones(1024, 1024)}   # 4 MB 的"编码器输出"
                self.outbox.put(OutgoingMessage(msg.request_id, "result", data=payload))

        def stop(self):
            self.running = False

        def abort(self, request_id):
            pass


    def create_src(big_chunk: bool = False):
        return SrcScheduler(big_chunk)


    def project(payload: StagePayload) -> StagePayload:
        return StagePayload(payload.request_id, payload.request, {"features": payload.data["features"]})


    def create_near():
        def fn(payload: StagePayload) -> StagePayload:
            payload.data = {"near_sum": int(payload.data["features"].sum())}
            return payload
        return SimpleScheduler(fn)


    class FarScheduler(StreamingSimpleScheduler):
        def __init__(self):
            super().__init__(compute_fn=None)
            self.sizes: dict[str, list[int]] = {}

        def is_streaming_payload(self, payload):
            return True

        def on_stream_chunk(self, request_id, item):
            self.sizes.setdefault(request_id, []).append(item.data.numel())
            return []

        def on_stream_done(self, request_id):
            payload = self.stream_payloads[request_id]
            payload.data = {"far_chunks": self.sizes.pop(request_id, []),
                            "far_sum": int(payload.data["features"].sum())}
            return [OutgoingMessage(request_id, "result", data=payload)]


    def create_far():
        return FarScheduler()
    ```

`src` 和 `near` 的 `process` 相同，`far` 单独一个进程：

```python title="ch6_pipeline.py" run="no"
"""python ch6_pipeline.py small|big：跑一次 src → (near, far)；日志打到标准输出，供 ch6_trace.py 解析。"""
import asyncio
import logging
import sys
import tempfile

from sglang_omni.config.schema import EndpointsConfig, FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

M = "ch6_stages"


def make_config(big_chunk: bool) -> PipelineConfig:
    return PipelineConfig(
        model_path="toy",
        entry_stage="src",
        stages=[
            StageConfig(name="src", process="p_main", factory_path=f"{M}.create_src", next=["near", "far"],
                        factory=FactoryArgs(big_chunk=big_chunk),
                        stream_to=["far"], project_payload={"near": f"{M}.project", "far": f"{M}.project"}),
            StageConfig(name="near", process="p_main", factory_path=f"{M}.create_near", terminal=True),
            StageConfig(name="far", process="p_far", factory_path=f"{M}.create_far", terminal=True,
                        can_accept_stream_before_payload=True),
        ],
        endpoints=EndpointsConfig(base_path=tempfile.mkdtemp(prefix="omni-")),
    )


async def main(big_chunk: bool) -> None:
    runner = MultiProcessPipelineRunner(make_config(big_chunk))
    await runner.start(timeout=120)
    try:
        print("RESULT", await runner.coordinator.submit("r0", "go"))
    finally:
        await runner.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)   # 子进程继承这个日志级别
    asyncio.run(main(sys.argv[1] == "big"))
```

`SGLANG_OMNI_COMM_TRACE=1` 打开通信追踪后，每条边第一次选传输方式、每次发送流式 chunk 都会打一行 JSON 日志。下面的脚本把流水线放在子进程里跑，解析这些日志：

```python title="ch6_trace.py"
"""在子进程里跑 ch6_pipeline.py，打开 SGLANG_OMNI_COMM_TRACE，整理出每条边用了什么传输方式。"""
import json
import os
import subprocess
import sys

EDGES = [("payload", "src", "near"), ("payload", "src", "far"), ("stream", "src", "far")]


def run(mode: str) -> None:
    env = dict(os.environ, SGLANG_OMNI_COMM_TRACE="1")
    out = subprocess.run([sys.executable, "ch6_pipeline.py", mode], env=env,
                         capture_output=True, text=True, timeout=600).stdout
    chosen, sizes = {}, []
    for line in out.splitlines():
        if "COMM_TRACE " in line:
            rec = json.loads(line.split("COMM_TRACE ", 1)[1])
            if rec["event"] == "comm_transport_selected":
                chosen[(rec["direction"], rec["stage"], rec["peer_stage"])] = rec["transport"]
            elif rec["event"] == "comm_stream_send":
                sizes.append((rec["bytes"], rec["transport"]))
    print(f"== {mode}")
    for edge in EDGES:
        print(f"  {edge[0]:7s} {edge[1]} → {edge[2]:5s} {chosen.get(edge, '（没经过通信层）')}")
    for nbytes, transport in sizes:
        print(f"  流式块 {nbytes} 字节 → {transport}")
    result = [l for l in out.splitlines() if l.startswith("RESULT")]
    errors = [l.strip() for l in out.splitlines() if l.startswith("ValueError")]
    print("  结果：", result[0][7:] if result else errors[0] if errors else "（无）")


run("small")
run("big")
```

```text title="输出"
== small
  payload src → near  （没经过通信层）
  payload src → far   shm
  stream  src → far   inline
  流式块 64 字节 → inline
  结果： {'near': {'near_sum': 1048576}, 'far': {'far_chunks': [8], 'far_sum': 1048576}}
== big
  payload src → near  （没经过通信层）
  payload src → far   （没经过通信层）
  stream  src → far   inline
  流式块 64 字节 → inline
  结果： ValueError: cpu stream chunk cannot be sent from 'src' to non-GPU target 'far'
```

`small` 这一轮：

- `src → near` 没有经过通信层：同进程直传，4 MB 的张量连拷贝都没有；
- `src → far` 的完整 payload 走 SHM：4 MB 拷进共享内存，`far` 读出后 ACK；
- 64 字节的流式 chunk 走 `inline`，直接塞进了控制消息；
- 两个终点的结果被 Coordinator 合并成按 stage 名分开的字典（第三章）。

`big` 这一轮多了一段 64 KB 的流式 chunk：超过内联上限，`outbound_stream` 要找 relay，而 `far` 不是 GPU stage，于是走到了上面看到的那个 `raise`，`src` 所在的进程直接退出，整条流水线挂掉。在真实的 GPU 部署里，产生大块流式数据的 stage（thinker、talker）都在 GPU 上，大 chunk 走 CUDA IPC，碰不到这条分支；但纯 CPU 的部署（仓库确实有 Intel CPU 平台的安装文档）理论上会撞上它。**这是我们写这一章时在 CPU 环境里自己撞到的边界**——它算不算 bug、该不该修，第十二章会把它当作一个"从自己的发现到一个 issue"的例子。

## 在 GPU 上会怎样

本书的环境没有 GPU，下面这部分只能读代码。CUDA 平台上 `get_intra_node_transport()` 返回 `CUDA_IPC`，于是有两条路：

- **同一张卡上的两个进程**（`direct_cuda_ipc_targets`）：用 PyTorch 自带的 CUDA IPC 规约把张量的存储句柄序列化进控制消息，接收方直接映射发送方的显存，没有拷贝、也没有 relay 的 ACK，生命周期交给 PyTorch 的引用计数；
- **同机的不同卡**：`CudaIpcRelay` 在发送方维护一个有界的显存池（默认按 64 KB 的槽位分配），把张量拷进槽位，把池的 IPC 句柄和槽位位置发给接收方，接收方拷走后 ACK 释放槽位：

```python title="sglang_omni/relay/cuda_ipc.py @ 921ea2c8 L1-2"
# SPDX-License-Identifier: Apache-2.0
"""CUDA IPC relay backed by a bounded sender-side GPU slot pool."""
```

跨机则是 Mooncake（RDMA）。三种在同一套 `DataRef` / ACK 协议下工作，Stage 的代码看不到它们的区别。

## 练习

**1. 让同进程直传失效。** 把 `project` 改成直接 `return payload`（不投影），再跑 `small`，`src → near` 还会同进程直传吗？为什么？

??? success "参考思路"
    扇出到两个下游时，同进程直传要求投影后的 payload 和原 payload 不共享可变容器（`is_isolated_projected_payload`）。直接返回原 payload 时 `projected_payload is original_payload`，检查返回 `False`，于是 `near` 也会走通信层（SHM）。这避免了"两个下游拿到同一个对象，一个修改影响另一个"。

**2. 量一量打包的开销。** 用 `sglang_omni.comm.stage_io.extract_tensors` 和 `pack_tensors` 对一个含三个张量的字典做一次打包，打印缓冲区大小和每个张量的偏移，验证对齐规则。

??? success "参考思路"
    `data, tensors = extract_tensors({"a": torch.zeros(3), "b": {"c": torch.zeros(2, dtype=torch.float16)}})`，再 `packed, entries = pack_tensors(tensors, device="cpu")`；`entries` 里有路径、形状、dtype、偏移和字节数。偏移按 dtype 的对齐（`dtype_alignment`）补齐，所以 float16 后面跟 float32 时会有填充。

**3. 写一个 issue 草稿。** 根据实验里 `big` 的结果，写一段英文 issue 草稿：现象、复现步骤、你认为的原因、你不确定的地方（例如纯 CPU 部署是否是受支持的场景），以及一个可能的修法。

??? success "参考思路"
    现象：CPU 平台上超过 16 KB 的流式 chunk 让发送方 stage 进程退出（`ValueError: cpu stream chunk cannot be sent ... to non-GPU target`）。原因：`CommRouter.outbound_stream` 在"张量在 CPU、平台设备也是 CPU、自己不是 GPU stage"时没有 SHM 分支。不确定：Intel CPU 平台上是否有模型会产生这么大的流式 chunk。可能的修法：非 GPU stage 之间的 CPU chunk 走 SHM（和 `outbound_payload` 对齐），并补一个单测。第十二章会把这个过程走完。

!!! interview "怎么讲清楚"
    讲多阶段推理服务的数据搬运，先分两个面：控制面（ZMQ）只传消息和元数据，数据面（relay）传张量。再讲"按边选传输"：同进程直传对象、同卡 CUDA IPC 传句柄、同机跨卡 CUDA IPC 显存池、CPU 张量走共享内存、跨机走 RDMA；小于 16 KB 的流式 chunk 直接内联进控制消息。然后讲协议：张量抽出来拼成一个缓冲区，其余部分 pickle 成 header；发送方拥有缓冲区，接收方 ACK 后释放，信用额度做背压。最后补一句"文档写四种后端、代码只用三种"，说明你是读了代码的。

## 小结

- [x] 控制面 ZMQ 传消息，数据面 relay 传张量；每条边由 `CommRouter` 单独选择传输方式。
- [x] 选择顺序：跨机 → Mooncake；同进程 → 直传对象（不进通信层）；GPU 张量且双方是 GPU stage → CUDA IPC；其余 → SHM。
- [x] 打包：`extract_tensors` + `pack_tensors` → 一个 `uint8` 缓冲区 + pickle 的 header + 张量清单（`DataRef`）。
- [x] 背压：发送方拥有缓冲区、接收方 ACK 后释放，`credits` 默认 2。
- [x] ≤16 KB 的 CPU 流式 chunk 内联（#1574）；纯 CPU 部署下更大的 chunk 会触发 `outbound_stream` 的报错。
- [x] 基准提交里数据面实际只构造 `cuda_ipc`、`shm`、`mooncake` 三种 relay，NCCL / NIXL 实现存在但不被路由选择。
