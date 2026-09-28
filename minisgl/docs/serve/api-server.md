# API Server 与启动器

<p class="lead">最后一块拼图：一个 FastAPI 应用接收 OpenAI 格式的请求，给每个请求分配 uid，发给 tokenizer，再把陆续回来的增量文本以 SSE 流式写回客户端；一个启动器按正确的顺序拉起所有进程。完成这一章，<code>python -m minisgl --model Qwen/Qwen3-0.6B</code> 就是一个可以用任何 OpenAI 客户端访问的推理服务。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 成百上千个并发的 HTTP 请求共用一个 ZMQ 接收队列，每个请求怎样只拿到属于自己的回复？
    2. SSE（Server-Sent Events）的格式是什么？OpenAI 流式接口的最后一条消息是什么？
    3. 客户端中途断开连接，服务端要做什么？
    4. 启动器为什么用 `spawn` 而不是 `fork` 创建子进程？

**本章要写的文件**：`server/args.py`、`server/api_server.py`、`server/launch.py`、`__main__.py`、`shell.py`。

## 一个共享队列，多个等待者

API Server 是一个 asyncio 程序。所有请求的回复都从同一个 ZMQ 队列进来，需要按 uid 分发给各自的处理协程：

@@code python/minisgl/server/api_server.py:FrontendManager@@

- `new_user()` 为新请求分配 uid，并创建一个回复列表和一个 `asyncio.Event`；
- 后台协程 `listen()` 不断从 ZMQ 收 `UserReply`，追加到对应 uid 的列表里，并 `set()` 它的事件；
- 每个请求的处理协程在 `wait_for_ack()` 里 `await event.wait()`，醒来后取走列表里的所有回复逐个 yield，直到遇到 `finished`；
- `listen()` 在第一次发送时才创建：必须在事件循环已经运行之后，而 FastAPI 应用启动时还没有事件循环。

这是 asyncio 里"一个生产者、多个消费者按键分发"的标准写法。事件被 `clear()` 之后可能有新回复到达并再次 `set()`，所以取走回复时是把整个列表换掉，不会漏掉。

## OpenAI 兼容接口

@@code python/minisgl/server/api_server.py:v1_chat_completions@@

- `messages`（对话）和 `prompt`（纯文本）二选一，前者由 tokenizer 套用对话模板；
- 采样参数原样转给调度器，`ignore_eos` 是一个扩展字段（压测时常用）；
- 流式时返回 `text/event-stream`：每条回复变成一个 `data: {...}\n\n`，其中 `choices[0].delta.content` 是增量文本；最后发一条 `finish_reason: "stop"` 和 `data: [DONE]`；
- 非流式时收集全部回复拼成一个完整的 JSON。

@@code python/minisgl/server/api_server.py:FrontendManager.stream_chat_completions@@

## 断开连接时中止请求

客户端断开后，如果服务端继续生成，这个请求会一直占用一个请求槽和它的 KV 缓存，直到达到 `max_tokens`。所以每发一段数据前检查连接是否还在：

@@code python/minisgl/server/api_server.py:FrontendManager.stream_with_cancellation@@

检测到断开后，删除这个 uid 的等待状态，发送 `AbortMsg`。它经过 tokenizer 变成 `AbortBackendMsg`，调度器从等待队列或 decode 集合里找到这个请求，释放资源（第 11 章讨论了它与重叠调度交织时的细节）。

## 启动器

@@code python/minisgl/server/launch.py:start_backend@@

启动顺序：

1. 主进程解析参数，创建 `FrontendManager`（此时绑定前端的 ZMQ 地址）；
2. 用 `spawn` 方式启动每个 TP rank 的调度器进程、一个 detokenizer 进程、N 个 tokenizer 进程。CUDA 不能在 `fork` 出来的子进程里重新初始化，必须用 `spawn`；
3. 每个进程准备好后往一个 `multiprocessing.Queue` 里放一条"就绪"消息（调度器只有 rank 0 发，且在所有 rank 完成 `sync_all_ranks` 之后）；主进程收齐 `num_tokenizer + 2` 条后才继续；
4. 启动 uvicorn（或者交互式 shell）。

@@code python/minisgl/server/launch.py:_run_scheduler@@

交互模式（`--shell`）不起 HTTP 服务，直接在终端里对话；它把 `max_running_req` 设为 1、只录制批大小为 1 的 CUDA Graph，因为只有一个用户。

## 运行

@@code examples/ch15_server.py@@

@@output ch15_server@@

- 进程：API Server 主进程，加上一个调度器进程和一个 tokenizer 进程（默认分词、反分词共用）。
- 非流式：Qwen3 的回答先是一段空的 `<think></think>`（`/no_think` 关闭了思考），然后是答案。
- 流式：每个 token 一个片段，特殊 token `<think>` 也作为一个完整片段发出。
- 8 个并发请求被调度器合成一个 batch，总用时与单个请求几乎相同——这就是连续批处理带来的吞吐。

!!! upstream "官方实现"
    - @@upstream server/api_server.py:FrontendManager@@
    - @@upstream server/launch.py:launch_server@@
    - @@upstream server/args.py:parse_args@@

    官方的 `/v1/chat/completions` 非流式响应里 `usage` 字段恒为 0；我们直接省略了它。官方交互模式在退出时用 `psutil` 杀掉所有子进程，我们沿用了这个做法。

## 测试

@@code tests/test_ch15_server.py:test_client_disconnect_aborts_request@@

`tests/test_ch15_server.py` 另外两个测试：非流式输出与 Hugging Face 的贪心结果逐字相同；流式与并发请求都能正确返回。

## 练习

1. 在非流式响应里加上正确的 `usage`（提示词 token 数、生成 token 数）。这些数字在哪个进程里知道？需要在哪些消息里加字段？
2. 实现 `/v1/completions`（非对话的补全接口），它和 `/v1/chat/completions` 的请求、响应格式有什么不同？
3. 现在 uid 由 API Server 单调递增分配。如果将来要起多个 API Server 进程共同对外服务，uid 怎么分配才不会冲突？

??? success "参考答案"
    1. 提示词长度在 tokenizer 里知道，生成数量在调度器里知道。可以在 `UserReply` 里加 `prompt_tokens`、`completion_tokens`（由 detokenizer 在结束时填入：它能数出收到了几个 token，提示词长度需要 tokenizer 在分词时记下、或者由调度器随最后一条 `DetokenizeMsg` 带回）。
    2. 请求用 `prompt` 字段而不是 `messages`，不套对话模板；响应里是 `choices[0].text` 而不是 `message.content`，流式时每个片段是 `choices[0].text`，`object` 是 `text_completion`。
    3. 让每个 API Server 使用不同的前缀或步长，例如 uid = 进程编号 + 进程数 × 本地计数；或者使用 UUID。detokenizer 的回复也要能路由回对应的 API Server（每个 API Server 一个接收地址，消息里带上来源）。

## 小结

- [x] API Server 用"按 uid 分发的回复列表 + `asyncio.Event`"让所有请求共用一个 ZMQ 接收队列。
- [x] OpenAI 流式接口：SSE，每个片段一个 `delta.content`，最后是 `finish_reason` 和 `[DONE]`。
- [x] 客户端断开时发送 `AbortMsg`，释放请求占用的请求槽和 KV。
- [x] 启动器用 `spawn` 拉起调度器和 tokenizer 进程，等所有进程就绪再启动 HTTP 服务。
