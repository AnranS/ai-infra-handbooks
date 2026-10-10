# 阶段三的里程碑：变成一个服务

<p class="lead">阶段二结束时，引擎还是一个 Python 对象：同一个进程里调 <code>generate</code>，请求一次给齐，结果一次拿回。阶段三的四章把它变成一个多进程的在线服务：API Server 收 HTTP、tokenizer 进程分词、调度器进程计算、消息用 ZMQ 在进程之间传，对外是 OpenAI 兼容的接口，支持流式输出和断连中止。这一页先把服务起起来，用标准的 OpenAI 客户端访问一遍。</p>

**这一阶段结束时你手里有什么**

- `python -m minisgl --model ...` 一条命令起服务，任何 OpenAI 客户端（或一条 `curl`）都能连；
- 请求是陆续到的、互不相识的，它们在调度器里自然地进同一个 batch；
- 流式输出：第一个 token 出来就发出去，不等整句；
- 客户端断开，请求被中止，KV 页被回收。

## 先跑起来

```bash
cd minisgl && python examples/stage3_milestone.py
```

@@code examples/stage3_milestone.py@@

@@output stage3_milestone@@

## 读这几个数字

**1 个 API Server 进程 + 3 个工作进程。** 阶段二的一切都在一个进程里；现在分词、调度、HTTP 各在自己的进程里，用 ZMQ 传消息。多进程不是为了并行计算——计算全在调度器进程——而是为了让 HTTP 的事件循环和 Python 的分词永远不卡住正在算的那一步，反过来也一样。

**流式：首个片段 157 ms，全部 1086 ms。** 非流式要等 32 个 token 全部生成才返回；流式让用户 157 ms 就看到第一个字。首 token 延迟（TTFT）和每 token 延迟（TPOT）从这里开始成为两个独立的指标，推理服务的 SLA 都是用它们定义的。

**8 个并发请求 1.41 秒，每个 1.40～1.40 秒。** 8 个请求从 8 个线程同时发出，经过 HTTP、分词、ZMQ 进入调度器，被放进**同一个 batch**，所以每个请求的耗时几乎一样——没有谁在排队。这和阶段二 `generate` 一次给 8 个提示词的效果相同，区别是现在它们来自网络、互不相识。

## 一个进程变成了哪几个

| 阶段二 | 阶段三 | 在哪一章 |
| --- | --- | --- |
| `generate` 的参数就是请求 | 消息对象 + 通用序列化 + ZMQ 队列（PUSH/PULL、PUB/SUB） | [消息、序列化与 ZMQ](message.md) |
| `self.tokenizer(prompt)` 在同一个进程里 | 独立的 tokenizer 进程；反分词是增量的（流式输出的每个片段） | [Tokenizer 与增量反分词](tokenizer.md) |
| 调度器直接拿到 Python 列表 | 调度器从 ZMQ 收请求、往 ZMQ 发结果；多 rank 时 rank 0 广播给其他 rank | [调度器的收发与多 rank 同步](scheduler-io.md) |
| 没有 HTTP | FastAPI 的 OpenAI 兼容接口、流式 SSE、断连中止；启动器拉起所有进程 | [API Server 与启动器](api-server.md) |

## 怎么读这四章

先读[消息](message.md)：这一阶段所有进程之间传的东西都在那一个文件里定义，后面三章只是收发它们。读到 [API Server](api-server.md) 时，把本页脚本最后打印的那条 `curl` 真的敲一遍，再用 `stream=True` 看 SSE 的每一帧长什么样。四章读完，对照本书第 11 章和第 12 章里修正官方实现的那几个问题（EOS 之后多发一条过期消息、PUB/SUB 没有等订阅者）——它们都出在这一阶段的边界上。

!!! abstract "验收：做到这些再进阶段四"
    - [ ] 画出一个请求经过的进程和消息：HTTP → tokenizer → 调度器 → detokenizer → HTTP；
    - [ ] 解释为什么分词要单独一个进程，而不是在 API Server 里做；
    - [ ] 流式输出时，一个 token 变成一个片段要经过几次序列化；
    - [ ] 用 `Ctrl-C` 中断一个流式请求，确认调度器里对应的 KV 页被回收（第 15 章的测试）。

## 小结

- [x] 阶段三把一个进程里的 `generate` 变成多进程的 OpenAI 兼容服务，消息走 ZMQ。
- [x] 流式输出让首 token 延迟和每 token 延迟成为两个独立的指标。
- [x] 来自网络的并发请求自然进同一个 batch：8 个请求各 1.40 秒，没有排队。
- [x] 多进程是为了让 HTTP 和分词永远不卡住计算，不是为了并行算。
