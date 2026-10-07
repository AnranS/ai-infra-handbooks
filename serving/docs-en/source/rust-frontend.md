# vLLM's Rust frontend: moving the serving layer out of Python

<p class="lead">As <a href="../vllm/">the vLLM walkthrough</a> explained, V1 splits a request's handling across several processes: the frontend process handles HTTP, chat templates, tokenization and detokenization, the EngineCore process handles scheduling and execution, and the two talk only through ZMQ + msgpack. Precisely because this boundary is language-independent, vLLM 0.30 gained a frontend rewritten in Rust (the <code>rust/</code> directory, about 120,000 lines), switched on with <code>VLLM_USE_RUST_FRONTEND=1</code>, while the engine core remains the same Python process. This chapter first measures the Python frontend's overhead, then looks at the Rust frontend's layers, its protocol with the engine, incremental detokenization and the gRPC interface, and finally SGLang's gateway written in Rust.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What does the frontend of an inference service do? Why does the Python frontend run several processes (`--api-server-count`)?
    2. What travels between vLLM's frontend and EngineCore? Why do we say "the field order is the protocol"?
    3. What layers is the Rust frontend split into? Which layer handles chat templates and tool-call parsing?
    4. What is the `vllm-rs render` mode for?
    5. Besides generation, what control functions does its gRPC interface provide? Who are they for?

??? success "Answers (try first, then expand to compare)"
    1. HTTP ingress, rendering chat templates, tokenization, sending requests to the engine, detokenization, assembling streaming responses (JSON), and parsing tool calls and reasoning content. The Python frontend spends a dozen to several dozen microseconds per output token, the GIL limits a process to one core, and tokenizing a long prompt slows every stream in the same process, so the only way to scale is more processes.
    2. Messages for requests and outputs: msgpack-encoded over ZMQ, and encoded as arrays in field order (without field names). The receiver decodes by position, so the order of fields is itself the protocol: new fields can only be appended at the end with defaults, and both sides must change together, or values shift (with no error when the types happen to match).
    3. Transport (talking to EngineCore), the token interface, tokenization and detokenization, chat and parsing, and the HTTP / gRPC server. Rendering chat templates and parsing tool calls and reasoning content are in the "chat and parsing" layer.
    4. It loads only the tokenizer and config, neither starting nor connecting to an engine, and renders and tokenizes OpenAI-format requests into "tokens in" requests. A gateway can render once, route cache-aware by token prefix, then send the tokens straight to an engine; it needs no GPU, which also makes it handy for reading the code.
    5. Control functions such as KV events (which blocks are cached), pause and resume, sleep, and weight updates, aimed at gateways above it (cache-aware routing) and RL frameworks (pausing generation, updating weights).

## The Python frontend's overhead {#python-前端的开销}

The frontend's work falls into two kinds: **once per request** (parsing HTTP, validating parameters, rendering the chat template, tokenizing) and **once per output token** (incremental detokenization, checking stop strings, parsing tool calls and reasoning content, assembling SSE events and writing them back). The latter scales with throughput. Measure it with a real tokenizer and pydantic response objects with the same structure as vLLM's:

```python
import asyncio
import time

from pydantic import BaseModel
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
ids = tok("推理服务的前端要做分词、渲染对话模板、增量反分词和流式输出。The frontend also parses tool calls. " * 60).input_ids[:1000]


# streaming responses of the OpenAI-compatible API (fields match vLLM's protocol definitions, trimmed down)
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
    if top_k:                                       # logprobs requested: every candidate token must be decoded into a string
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

One run on an x86 development machine (timings vary by machine; look only at the order of magnitude):

```text
只要文本：每个 token 17 微秒，一个核最多 59,330 token/s
带 top-5 logprobs：每个 token 63 微秒，一个核最多 15,912 token/s
2000 个并发流的协程调度：每个 token 额外 4.4 微秒
```

- **How much one core can handle**: with text only, about 20 microseconds per token, so one core handles tens of thousands of tokens per second, the same order of magnitude as the decode throughput of an 8-GPU machine; request logprobs (RL rollouts almost always do) and it grows several-fold, leaving one core at just over ten thousand;
- **The GIL makes "add threads" useless**: in one process, streaming output, tokenizing and rendering templates for new requests, and HTTP parsing all compete for the same core. Tokenizing a prompt of several hundred thousand tokens takes tens of milliseconds, during which every stream the process handles stalls, showing up as TPOT spikes;
- **So the Python frontend relies on multiple processes**: vLLM by default starts several API server processes according to the data-parallel size (`--api-server-count`), each with its own tokenizer, templates and connection to EngineCore. The Rust frontend is multithreaded, so one process suffices: `vllm/entrypoints/cli/serve.py` states that with the Rust frontend, `api_server_count` is fixed at 1.

## Layers {#分层}

`rust/README.md` draws the layers from the bottom up (one crate each):

| Crate | What it does |
| --- | --- |
| `vllm-engine-core-client` | talks to EngineCore: ZMQ transport (with the pure-Rust `zeromq` crate) + the MessagePack protocol, handshakes, request bookkeeping, routing across several engines (data parallelism) |
| `vllm-llm` | a very thin "tokens in, tokens out" interface |
| `vllm-text` / `vllm-tokenizer` | tokenization and incremental detokenization; tokenizers include HF `tokenizers`, tiktoken and Mistral's tekken |
| `vllm-chat` | chat completions: renders HF chat templates with `minijinja` and parses reasoning content and tool calls (`vllm-parser` has one parser per model, such as `kimi_k3.rs`, `gemma4.rs` and `minimax_m3.rs`) |
| `vllm-server` | an OpenAI-compatible HTTP interface based on axum, plus a gRPC interface |
| `vllm-cmd` (`vllm-rs`) | the command-line entry: launched as a subprocess by Python's `vllm serve`, managing engine processes itself, or the preprocessing-only render mode |

There are also `vllm-bench` (a load-testing tool in Rust), `vllm-metrics`, `vllm-tracing` and a `mock-engine` for tests.

Two ways to start it:

- **Managed by Python**: `VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-0.6B`. Python does the launching, passes the already-listening socket and the transport addresses to the `vllm-rs` subprocess, and no longer handles HTTP itself;
- **Frontend and backend deployed separately**: start the engines alone with `vllm serve ... --headless`, and start only the frontend with `vllm-rs serve ... --data-parallel-size-local 0`, which waits for all engines to complete the handshake according to `--data-parallel-size`. The frontend and engines can be on different machines, and the number of frontends can scale independently.

## The protocol with the engine {#和引擎之间的协议}

The frontend sends EngineCore two ZMQ frames: the first is one byte for the request type (`\x00` add, `\x01` abort, `\x02` a new data-parallel wave, `\x03` utility, i.e. management calls such as loading a LoRA or collective_rpc), and the second is the msgpack-encoded `EngineCoreRequest`. On the Python side it is a `msgspec.Struct(array_like=True, omit_defaults=True)`: encoded as an array **in field order**, with trailing fields equal to their defaults simply omitted; the Rust side decodes in the same order with `serde_tuple`, and all later fields are marked `#[serde(default)]`:

```python
import msgpack

# in Python, EngineCoreRequest is a msgspec.Struct(array_like=True, omit_defaults=True): encoded as an array in field order,
# with trailing fields equal to their defaults omitted. The Rust side decodes in the same order with serde_tuple (rust/src/engine-core-client/src/protocol/request.rs)
FIELDS = [("request_id", None), ("prompt_token_ids", None), ("mm_features", None), ("sampling_params", None),
          ("pooling_params", None), ("arrival_time", None), ("lora_request", None), ("cache_salt", None),
          ("data_parallel_rank", None), ("prompt_embeds", None), ("prompt_is_token_ids", None), ("client_index", 0),
          ("current_wave", 0), ("priority", 0)]
REQUIRED = 9                                    # the first 9 fields have no defaults and must be present


def encode(req, fields=FIELDS):
    values = [req.get(name, default) for name, default in fields]
    n = len(values)
    while n > REQUIRED and values[n - 1] == fields[n - 1][1]:     # omit_defaults: drop trailing defaults
        n -= 1
    return b"\x00" + msgpack.packb(values[:n])                    # the first frame is the request type: \x00 = ADD


def decode(frame, fields=FIELDS):
    kind, values = frame[:1], msgpack.unpackb(frame[1:])
    values += [default for _, default in fields[len(values):]]    # fill omitted fields with defaults (#[serde(default)] on the Rust side)
    return kind, dict(zip((name for name, _ in fields), values))


req = {"request_id": "req-7", "prompt_token_ids": list(range(151_000, 152_000)), "sampling_params": {"temperature": 0.7, "max_tokens": 256},
       "arrival_time": 1.0, "priority": 3}
frame = encode(req)
kind, back = decode(frame)
print(f"1000 个 token 的请求编码后 {len(frame)} 字节（token 编号每个 5 字节）；类型 {kind!r}；priority={back['priority']}，"
      f"client_index={back['client_index']}（被省略，补了默认值）")

# Python inserted an integer field before priority and Rust was not updated: decoding by position shifts the values, with no error when the types happen to match
NEW = FIELDS[:-1] + [("tenant_weight", 1)] + FIELDS[-1:]
_, wrong = decode(encode({**req, "tenant_weight": 2}, NEW))
print(f"Python 加了字段、Rust 没改：Rust 读到 priority={wrong['priority']}（应该是 3）")
```

```text title="output"
1000 个 token 的请求编码后 5066 字节（token 编号每个 5 字节）；类型 b'\x00'；priority=3，client_index=0（被省略，补了默认值）
Python 加了字段、Rust 没改：Rust 读到 priority=2（应该是 3）
```

- **Arrays are cheaper than dicts**: no field names, so a 1000-token request is 5 KB, mostly token IDs;
- **The field order is the protocol**: new fields can only be appended at the end, and must have defaults (an older side fills in defaults when decoding, a newer side omits defaults when encoding); both definitions must change together. The Rust side's `request.rs` defines every field in the same order and notes in comments which are not supported yet (for example `prompt_embeds`, which Python encodes in a separate tensor frame);
- An engine's identity is the ZMQ ROUTER/DEALER routing identity: Python's `EngineCoreProc` uses its engine number as two little-endian bytes, and the Rust side (`transport.rs`) parses it the same way, so that under data parallelism control messages such as "start a new wave" reach the right engine. When an engine dies, it sends a separate `ENGINE_CORE_DEAD` frame.

## Incremental detokenization {#增量反分词}

The Rust version (`rust/src/tokenizer/src/incremental.rs`) follows the same idea as HF `tokenizers`' `DecodeStream` and vLLM's Python side: for each new token, re-decode only "the last few tokens of what has been output + the new token" and diff against the previous decode to get the new text; if it ends with U+FFFD (half a UTF-8 character, such as a Chinese character cut across two tokens), hold it back. One detail: before generation starts, it takes 4–6 tokens from the end of the prompt as the prefix, choosing the shortest run that decodes cleanly (with no U+FFFD), so it neither decodes the whole prompt nor gets the first output token wrong. The derivation and edge cases of the algorithm are in the [detokenization chapter of mini-sglang from Scratch](minisgl://serve/tokenizer/).

## gRPC and the render mode {#grpc-与-render-模式}

- **gRPC** (`rust/proto/`): the `Inference` service provides `Generate` and `GenerateStream` (tokens in, tokens out, bypassing OpenAI's JSON); the `Control` service provides server and model information, aborting requests, loading and unloading LoRAs, a KV event source (for gateways doing cache-aware routing to subscribe to), pausing / resuming generation, sleeping / waking, and weight updates for RL training (initializing the transfer, starting an update, updating the draft model's weights). These are exactly the control plane gateways and RL frameworks need; the schema is published on buf.build, and the Rust `vllm-proto` crate provides a client directly;
- **The render mode**: `vllm-rs render Qwen/Qwen3-32B` loads only the tokenizer and config, neither starting nor connecting to an engine, and provides endpoints such as `/v1/chat/completions/render` that turn OpenAI-format requests into "tokens in" `GenerateRequest`s. It splits preprocessing out of inference: a gateway can render and tokenize once, route cache-aware by token prefix, then send the tokens straight to an engine; it needs no GPU, which also makes it suitable for compiling and running on a Mac to read the code (`./build_rust.sh` needs the Rust 1.95 toolchain).

## SGLang's Rust gateway {#sglang-的-rust-网关}

SGLang's routing across instances is handled by SGLang Model Gateway (formerly sgl-router), written in Rust, in a separate subproject: it keeps an approximate radix tree per worker for cache-aware routing, falls back to the shortest queue when loads diverge too far, and also pairs prefill and decode instances under PD disaggregation (see [global scheduling](../frontier/disagg-sched.md#路由缓存与排队的权衡)). vLLM's Rust frontend handles "the serving layer in front of one engine (or a group of data-parallel engines)", while SGLang's gateway handles "the routing layer above many instances"; in a deployment, they are two layers, one above the other. [Project B](../career/portfolio-guide.md#作品-bpd-感知的推理网关rust) builds the latter, and reading these two codebases is the most direct reference.

!!! interview "In an interview"
    When asked "why rewrite an inference service's frontend in Rust", start with numbers: the Python frontend spends a dozen to several dozen microseconds per output token (detokenization, building JSON, coroutine scheduling), several times more with logprobs, so one core handles ten to several tens of thousands of tokens per second, the same order of magnitude as an 8-GPU machine's throughput; then the GIL: long-prompt tokenization, template rendering and streaming output compete for the same core, so the only option is more processes (`--api-server-count`), and tail latency suffers too. Then describe vLLM's approach: leave the engine core alone and swap only the frontend, made possible by a language-independent boundary (ZMQ + msgpack arrays encoded in field order, so new fields can only be appended and both sides must change together); the Rust frontend has layers for transport, the token interface, tokenization and detokenization, chat and parsing, and HTTP / gRPC, multithreaded in one process; the gRPC control plane (KV events, pausing, weight updates) serves gateways and RL frameworks; and the render mode splits out preprocessing for gateways.

## Exercises {#练习}

**1. Why wasn't the engine core rewritten in Rust as well?**

??? success "Answer"
    The heaviest part of the engine core, model execution, is already in GPU kernels (CUDA / Triton / CUTLASS), with Python only scheduling and organizing, and scheduling overhead has already been pushed down with overlap scheduling, CUDA Graphs and moving work to separate threads; more importantly, the model code, quantization and attention backends are all tied to the PyTorch ecosystem, so changing languages would cost enormously for limited gain. The frontend is the opposite: all CPU string and JSON processing, unrelated to PyTorch, with a clear boundary (one message protocol), so rewriting it in Rust has a large payoff and small risk.

**2. If a service must support both the Python and the Rust frontend, what is most likely to go wrong? How do you guard against it?**

??? success "Answer"
    Behavioral consistency: chat templates (subtle differences between Jinja2 and minijinja), parsing of tool calls and reasoning content, stop strings, the logprobs format, and the defaults and validation of all kinds of parameters may differ between the two; the field order of the message protocol must also stay in sync. The guard is to test both sides with the same set of cases (vLLM's Rust code has many comparison tests, such as `chat/tests/roundtrip.rs` and `server/src/routes/tests.rs`), change the protocol definition in one place with the other annotated field by field in comments, and run end-to-end comparisons against a mock engine in CI.

**3. How does the render mode help cache-aware routing?**

??? success "Answer"
    Cache-aware routing compares a request's token prefix with the prefixes in each instance's cache. If the gateway sees only OpenAI-format JSON, it must render the template and tokenize by itself to know the prefix, and match the engine's rendering exactly. The render mode lets the gateway get the token sequence with the same code as the engine, choose the instance by token prefix, and send the tokens straight over (gRPC's `Generate` or `/inference/v1/generate`), so the engine need not preprocess again.

## Summary {#小结}

- [x] The Python frontend spends a dozen to several dozen microseconds per output token, more with logprobs; the GIL means it can only scale with more processes, and tokenizing a long prompt slows every stream in the same process.
- [x] vLLM 0.30's Rust frontend replaces only the serving layer and leaves the engine core unchanged: layers for transport, the token interface, tokenization and detokenization, chat and parsing, and HTTP / gRPC, multithreaded in one process.
- [x] Between the frontend and the engine is ZMQ + msgpack arrays encoded in field order: new fields can only be appended at the end with defaults, and both sides change together.
- [x] The gRPC control plane (KV events, pause and resume, sleep, weight updates) serves gateways and RL frameworks; the render mode splits out preprocessing, making routing by token prefix easy.
- [x] SGLang's Rust gateway does multi-instance routing one layer up; together they are the serving layer and routing layer of an inference platform.
