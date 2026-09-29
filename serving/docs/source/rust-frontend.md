# vLLM 的 Rust 前端：把服务层搬出 Python

<p class="lead"><a href="../vllm/">vLLM 源码导读</a>讲过，V1 把一个请求的处理分到几个进程里：前端进程负责 HTTP、对话模板、分词和反分词，EngineCore 进程负责调度和执行，两者之间只用 ZMQ + msgpack 通信。正因为这条边界和语言无关，vLLM 0.30 里出现了一个用 Rust 重写的前端（<code>rust/</code> 目录，约 12 万行），用 <code>VLLM_USE_RUST_FRONTEND=1</code> 打开，引擎核心仍然是原来的 Python 进程。这一章先量一量 Python 前端的开销，再看 Rust 前端的分层、它和引擎之间的协议、增量反分词和 gRPC 接口，最后说说 SGLang 用 Rust 写的网关。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 推理服务的前端要做哪些事？为什么 Python 前端要开多个进程（`--api-server-count`）？
    2. vLLM 的前端和 EngineCore 之间传的是什么？为什么说"字段顺序就是协议"？
    3. Rust 前端分成哪几层？哪一层负责对话模板和工具调用解析？
    4. `vllm-rs render` 模式有什么用？
    5. 它的 gRPC 接口除了生成，还提供了哪些控制功能？这些功能是给谁用的？

## Python 前端的开销

前端的活可以分成两类：**每个请求一次**的（解析 HTTP、校验参数、渲染对话模板、分词）和**每个输出 token 一次**的（增量反分词、检查停止字符串、解析工具调用和思考内容、拼成 SSE 事件写回）。后者和吞吐成正比。用真实的分词器和与 vLLM 相同结构的 pydantic 响应对象量一下：

```python
import asyncio
import time

from pydantic import BaseModel
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
ids = tok("推理服务的前端要做分词、渲染对话模板、增量反分词和流式输出。The frontend also parses tool calls. " * 60).input_ids[:1000]


# OpenAI 兼容接口的流式响应（字段与 vLLM 的 protocol 定义一致，做了精简）
class TopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int]


class LogprobContent(TopLogprob):
    top_logprobs: list[TopLogprob] = []


class Delta(BaseModel):
    content: str | None = None


class Choice(BaseModel):
    index: int
    delta: Delta
    logprobs: dict | None = None
    finish_reason: str | None = None


class Chunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[Choice]


def delta_text(prefix_ids):
    """增量反分词：只解码最后几个 token，取新增的部分（半个汉字时先不输出）"""
    window = prefix_ids[-6:]
    full, head = tok.decode(window), tok.decode(window[:-1])
    return "" if full.endswith("�") else full[len(head):]


def one_token(prefix_ids, top_k):
    logprobs = None
    if top_k:                                       # 请求了 logprobs：每个候选 token 都要解码成字符串
        alts = [TopLogprob(token=tok.decode([t]), logprob=-1.5, bytes=list(tok.decode([t]).encode())) for t in range(100, 100 + top_k)]
        logprobs = {"content": [LogprobContent(token=tok.decode(prefix_ids[-1:]), logprob=-0.1, bytes=[], top_logprobs=alts).model_dump()]}
    chunk = Chunk(id="chatcmpl-1", created=0, model="qwen", choices=[Choice(index=0, delta=Delta(content=delta_text(prefix_ids)), logprobs=logprobs)])
    return "data: " + chunk.model_dump_json(exclude_unset=True) + "\n\n"


for top_k in (0, 5):
    t0 = time.perf_counter()
    for i in range(1, len(ids) + 1):
        one_token(ids[:i], top_k)
    us = (time.perf_counter() - t0) / len(ids) * 1e6
    print(f"{'带 top-5 logprobs' if top_k else '只要文本'}：每个 token {us:.0f} 微秒，一个核最多 {1e6 / us:,.0f} token/s")


async def fanout(streams, steps):
    """每一步引擎输出一批 token：分发到每个请求的队列，每个请求一个协程取出来写回（这里只计数）"""
    queues = [asyncio.Queue() for _ in range(streams)]
    sent = 0

    async def client(q):
        nonlocal sent
        while (item := await q.get()) is not None:
            sent += 1

    tasks = [asyncio.create_task(client(q)) for q in queues]
    for _ in range(steps):
        for q in queues:
            q.put_nowait("data: {}\n\n")
        await asyncio.sleep(0)
    for q in queues:
        q.put_nowait(None)
    await asyncio.gather(*tasks)
    return sent


t0 = time.perf_counter()
n = asyncio.run(fanout(2000, 50))
print(f"2000 个并发流的协程调度：每个 token 额外 {(time.perf_counter() - t0) / n * 1e6:.1f} 微秒")
```

在一台 x86 开发机上的一次运行（耗时随机器变化，只看数量级）：

```text
只要文本：每个 token 17 微秒，一个核最多 59,330 token/s
带 top-5 logprobs：每个 token 63 微秒，一个核最多 15,912 token/s
2000 个并发流的协程调度：每个 token 额外 4.4 微秒
```

- **一个核能撑多少**：只要文本时每个 token 20 微秒左右，一个核每秒几万个 token，和一台 8 卡机的 decode 吞吐是同一个量级；请求了 logprobs（RL 的 rollout 几乎总要）就翻几倍，一个核只剩一万多；
- **GIL 让"加线程"不管用**：同一个进程里，流式输出、新请求的分词和模板渲染、HTTP 解析抢的是同一个核。一个几十万 token 的提示词分词要几十毫秒，这段时间里这个进程负责的所有流都停着——表现为 TPOT 的尖刺；
- **所以 Python 前端靠多进程**：vLLM 默认按数据并行数开多个 API server 进程（`--api-server-count`），每个进程各有一份分词器、模板和到 EngineCore 的连接。Rust 前端是多线程的，一个进程就够：`vllm/entrypoints/cli/serve.py` 里写明用 Rust 前端时 `api_server_count` 固定为 1。

## 分层

`rust/README.md` 画出了自下而上的几层（每层一个 crate）：

| crate | 做什么 |
| --- | --- |
| `vllm-engine-core-client` | 和 EngineCore 通信：ZMQ 传输（用纯 Rust 实现的 `zeromq` crate）+ MessagePack 协议，握手、请求簿记、多个引擎（数据并行）的路由 |
| `vllm-llm` | 一层很薄的"token 进、token 出"接口 |
| `vllm-text` / `vllm-tokenizer` | 分词与增量反分词；分词器支持 HF `tokenizers`、tiktoken 和 Mistral 的 tekken |
| `vllm-chat` | 对话补全：用 `minijinja` 渲染 HF 的对话模板，解析思考内容和工具调用（`vllm-parser` 里按模型各有一个解析器，例如 `kimi_k3.rs`、`gemma4.rs`、`minimax_m3.rs`） |
| `vllm-server` | 基于 axum 的 OpenAI 兼容 HTTP 接口，外加一套 gRPC 接口 |
| `vllm-cmd`（`vllm-rs`） | 命令行入口：由 Python 的 `vllm serve` 作为子进程拉起，或者自己管理引擎进程，或者只做预处理的 render 模式 |

另外还有 `vllm-bench`（Rust 写的压测工具）、`vllm-metrics`、`vllm-tracing` 和给测试用的 `mock-engine`。

两种启动方式：

- **Python 托管**：`VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B`。Python 负责启动，把监听好的 socket 和传输地址传给 `vllm-rs` 子进程，自己不再处理 HTTP；
- **前后端分开部署**：引擎用 `vllm serve ... --headless` 单独启动，`vllm-rs serve ... --data-parallel-size-local 0` 只起前端，按 `--data-parallel-size` 等待所有引擎完成握手。前端和引擎可以在不同的机器上，前端的数量也可以独立扩缩。

## 和引擎之间的协议

前端发给 EngineCore 的是两帧 ZMQ 消息：第一帧一个字节表示请求类型（`\x00` 添加、`\x01` 取消、`\x02` 数据并行的新一轮、`\x03` utility——加载 LoRA、collective_rpc 这类管理调用），第二帧是 msgpack 编码的 `EngineCoreRequest`。Python 那边它是 `msgspec.Struct(array_like=True, omit_defaults=True)`：按**字段顺序**编码成一个数组，末尾等于默认值的字段直接省略；Rust 那边用 `serde_tuple` 按同样的顺序解码，后面的字段都标了 `#[serde(default)]`：

```python
import msgpack

# EngineCoreRequest 在 Python 里是 msgspec.Struct(array_like=True, omit_defaults=True)：按字段顺序编码成一个数组，
# 末尾等于默认值的字段直接省略。Rust 那边用 serde_tuple 按同样的顺序解码（rust/src/engine-core-client/src/protocol/request.rs）
FIELDS = [("request_id", None), ("prompt_token_ids", None), ("mm_features", None), ("sampling_params", None),
          ("pooling_params", None), ("arrival_time", None), ("lora_request", None), ("cache_salt", None),
          ("data_parallel_rank", None), ("prompt_embeds", None), ("prompt_is_token_ids", None), ("client_index", 0),
          ("current_wave", 0), ("priority", 0)]
REQUIRED = 9                                    # 前 9 个字段没有默认值，必须出现


def encode(req, fields=FIELDS):
    values = [req.get(name, default) for name, default in fields]
    n = len(values)
    while n > REQUIRED and values[n - 1] == fields[n - 1][1]:     # omit_defaults：去掉末尾的默认值
        n -= 1
    return b"\x00" + msgpack.packb(values[:n])                    # 第一帧是请求类型：\x00 = ADD


def decode(frame, fields=FIELDS):
    kind, values = frame[:1], msgpack.unpackb(frame[1:])
    values += [default for _, default in fields[len(values):]]    # 省略的字段补默认值（Rust 端的 #[serde(default)]）
    return kind, dict(zip((name for name, _ in fields), values))


req = {"request_id": "req-7", "prompt_token_ids": list(range(151_000, 152_000)), "sampling_params": {"temperature": 0.7, "max_tokens": 256},
       "arrival_time": 1.0, "priority": 3}
frame = encode(req)
kind, back = decode(frame)
print(f"1000 个 token 的请求编码后 {len(frame)} 字节（token 编号每个 5 字节）；类型 {kind!r}；priority={back['priority']}，"
      f"client_index={back['client_index']}（被省略，补了默认值）")

# Python 那边在 priority 前面插了一个整数字段，Rust 没有同步：按位置解码，值就错位了，而且类型碰巧一致时不会报错
NEW = FIELDS[:-1] + [("tenant_weight", 1)] + FIELDS[-1:]
_, wrong = decode(encode({**req, "tenant_weight": 2}, NEW))
print(f"Python 加了字段、Rust 没改：Rust 读到 priority={wrong['priority']}（应该是 3）")
```

```text title="输出"
1000 个 token 的请求编码后 5066 字节（token 编号每个 5 字节）；类型 b'\x00'；priority=3，client_index=0（被省略，补了默认值）
Python 加了字段、Rust 没改：Rust 读到 priority=2（应该是 3）
```

- **数组比字典省**：不带字段名，一个 1000 token 的请求 5 KB，大部分是 token 编号；
- **字段顺序就是协议**：新字段只能加在末尾，并且要有默认值（老的一方解码时补默认值，新的一方编码时省略默认值）；两边的定义必须一起改。Rust 端的 `request.rs` 按同样的顺序定义了每个字段，并在注释里标出了暂不支持的（例如 `prompt_embeds`：Python 用单独的张量帧编码它）；
- 引擎的身份是 ZMQ ROUTER/DEALER 的路由标识，Python 的 `EngineCoreProc` 用两个字节的小端引擎编号，Rust 端（`transport.rs`）照此解析，数据并行时才能把"开始新一轮"这类控制消息发给正确的引擎。引擎死掉时会发一个单独的 `ENGINE_CORE_DEAD` 帧。

## 增量反分词

Rust 版（`rust/src/tokenizer/src/incremental.rs`）和 HF `tokenizers` 的 `DecodeStream`、vLLM Python 端用的是同一个思路：每来一个 token，只重新解码"已输出部分的末尾几个 token + 新 token"，和上次的解码结果做差得到新增的文字；结尾是 U+FFFD（半个 UTF-8 字符，比如一个汉字被切在两个 token 里）时先扣住不输出。一个细节：开始生成前，它从提示词末尾取 4～6 个 token 作为前缀，选能干净解码（不含 U+FFFD）的最短一段，这样既不用解码整个提示词，又不会在第一个输出 token 上出错。算法的推导和边界情况见[手写 mini-sglang 的反分词](minisgl://serve/tokenizer/)一章。

## gRPC 与 render 模式

- **gRPC**（`rust/proto/`）：`Inference` 服务提供 `Generate` 和 `GenerateStream`（token 进、token 出，不经过 OpenAI 的 JSON）；`Control` 服务提供服务器与模型信息、取消请求、LoRA 的加载与卸载、KV 事件源（给做缓存感知路由的网关订阅）、暂停 / 恢复生成、休眠 / 唤醒、以及 RL 训练用的权重更新（初始化传输、开始更新、给草稿模型更新权重）。这些正是网关和 RL 框架需要的控制面；schema 发布在 buf.build 上，Rust 的 `vllm-proto` crate 直接提供客户端；
- **render 模式**：`vllm-rs render Qwen/Qwen3-32B` 只加载分词器和配置，不启动也不连接引擎，提供 `/v1/chat/completions/render` 等接口，把 OpenAI 格式的请求变成"token 进"的 `GenerateRequest`。它把预处理从推理里拆了出来：网关可以先渲染和分词一次，按 token 前缀做缓存感知路由，再把 token 直接发给引擎；它不需要 GPU，也适合在 Mac 上编译运行来读代码（`./build_rust.sh` 需要 Rust 1.95 工具链）。

## SGLang 的 Rust 网关

SGLang 的跨实例路由由 Rust 写的 SGLang Model Gateway（原 sgl-router）负责，在独立的子项目里：它为每个 worker 维护一棵近似的基数树做缓存感知路由，负载差距过大时退回最短队列，也负责 PD 分离时 prefill / decode 实例的配对（见[全局调度](../frontier/disagg-sched.md#路由缓存与排队的权衡)）。vLLM 的 Rust 前端管"一个引擎（或一组数据并行引擎）前面的服务层"，SGLang 的网关管"很多实例之上的路由层"，两者在一个部署里是上下两层。[作品 B](../career/portfolio-guide.md#作品-bpd-感知的推理网关rust) 做的就是后一层，读这两份代码是最直接的参考。

!!! interview "面试怎么答"
    被问到"推理服务的前端为什么要用 Rust 重写"，先给数字：Python 前端每个输出 token 要十几到几十微秒（反分词、拼 JSON、协程调度），带 logprobs 时翻几倍，一个核每秒一万到几万个 token，和一台 8 卡机的吞吐同一量级；再讲 GIL：长提示词的分词、模板渲染和流式输出抢同一个核，只能开多个进程（`--api-server-count`），尾延迟也受影响。然后讲 vLLM 的做法：引擎核心不动，只换前端，靠的是一条和语言无关的边界（ZMQ + 按字段顺序编码的 msgpack 数组，所以新字段只能加在末尾、两边一起改）；Rust 前端分成传输、token 接口、分词与反分词、对话与解析、HTTP / gRPC 几层，多线程、一个进程；gRPC 的控制面（KV 事件、暂停、权重更新）面向网关和 RL 框架；render 模式把预处理拆出来给网关用。

## 练习

**1. 为什么引擎核心没有一起用 Rust 重写？**

??? success "参考答案"
    引擎核心里最重的是模型执行，已经在 GPU kernel 里（CUDA / Triton / CUTLASS），Python 只负责调度和组织，而调度的开销已经用重叠调度、CUDA Graph、把工作挪到独立线程等方法压下去了；更重要的是模型代码、量化、注意力后端这些都和 PyTorch 生态绑在一起，改语言的代价极大、收益有限。前端正好相反：全是 CPU 上的字符串和 JSON 处理，和 PyTorch 无关，边界清楚（一条消息协议），用 Rust 重写收益大、风险小。

**2. 一个服务同时要支持 Python 前端和 Rust 前端，最容易出问题的是什么？怎么防？**

??? success "参考答案"
    行为的一致性：对话模板（Jinja2 与 minijinja 的细微差别）、工具调用和思考内容的解析、停止字符串、logprobs 的格式、各种参数的默认值和校验，都可能两边不一样；消息协议的字段顺序也必须同步。防的办法是用同一组用例测两边（vLLM 的 Rust 代码里有大量对照测试，例如 `chat/tests/roundtrip.rs`、`server/src/routes/tests.rs`），协议的定义在一处改、另一处在注释里逐字段对应，并在 CI 里用 mock 引擎跑端到端的对照。

**3. render 模式对缓存感知路由有什么帮助？**

??? success "参考答案"
    缓存感知路由比较的是请求的 token 前缀和各实例缓存里的前缀。如果网关只看到 OpenAI 格式的 JSON，就得自己渲染模板、分词，才能知道前缀是什么，而且要和引擎的渲染结果完全一致。render 模式让网关用和引擎同一套代码得到 token 序列，再按 token 前缀选实例，最后把 token 直接发过去（gRPC 的 `Generate` 或 `/inference/v1/generate`），引擎不用再做一遍预处理。

## 小结

- [x] Python 前端每个输出 token 十几到几十微秒，带 logprobs 更贵；受 GIL 限制只能多进程扩展，长提示词的分词会拖慢同一进程里的所有流。
- [x] vLLM 0.30 的 Rust 前端只替换服务层，引擎核心不变：分成传输、token 接口、分词与反分词、对话与解析、HTTP / gRPC 几层，多线程、一个进程。
- [x] 前端和引擎之间是 ZMQ + 按字段顺序编码的 msgpack 数组：新字段只能加在末尾并带默认值，两边一起改。
- [x] gRPC 的控制面（KV 事件、暂停与恢复、休眠、权重更新）面向网关和 RL 框架；render 模式把预处理拆出来，方便按 token 前缀路由。
- [x] SGLang 的 Rust 网关在更上一层做多实例路由；两者合起来就是一个推理平台的服务层与路由层。
