# 消息、序列化与 ZMQ

<p class="lead">前两部分的所有代码都在一个进程里运行。要变成在线服务，API Server、tokenizer、各个调度器进程之间需要传递消息。mini-sglang 的做法很朴素：每种消息是一个 dataclass，用一个 70 行的通用序列化函数变成由基本类型组成的字典，再用 msgpack 变成字节，通过 ZMQ 的几种 socket 发送。这一章把这套机制写出来。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 系统里一共有哪几类消息？分别从哪个进程发往哪个进程？
    2. 张量（提示词的 token）怎样放进 msgpack？
    3. ZMQ 的 PUSH/PULL 和 PUB/SUB 有什么区别？调度器分别在哪里用它们？
    4. 为什么 rank 0 转发给其他 rank 的是原始字节，而不是解码后的消息？

**本章要写的文件**：`message/utils.py`、`message/backend.py`、`message/tokenizer.py`、`message/frontend.py`、`message/__init__.py`、`utils/mp.py`。

## 三组消息

按接收方分成三组，每组有一个基类：

| 基类 | 接收方 | 消息 |
| --- | --- | --- |
| `BaseTokenizerMsg` | tokenizer / detokenizer | `TokenizeMsg`（文本或对话 + 采样参数）、`DetokenizeMsg`（一个新 token + 是否结束）、`AbortMsg` |
| `BaseBackendMsg` | 调度器 | `UserMsg`（token 张量 + 采样参数）、`AbortBackendMsg`、`ExitMsg` |
| `BaseFrontendMsg` | API Server | `UserReply`（增量文本 + 是否结束） |

每组还有一个 `Batch...Msg`，把多条消息打包成一条发送，减少消息数。

@@code python/minisgl/message/backend.py@@

注意 `decoder` 里的 `globals()`：反序列化时按类名在**定义消息的那个模块**的全局命名空间里找类。这就是为什么三组消息分在三个文件里、各自有自己的 `decoder`——每个接收方只认识发给自己的那组消息。

## 通用序列化

@@code python/minisgl/message/utils.py@@

规则很简单：任何对象变成 `{"__type__": 类名, 字段名: 值...}`，递归处理字段；基本类型原样保留；1 维张量变成原始字节加 dtype 字符串。反序列化反过来：看到 `__type__` 就按类名构造对象。

@@code examples/ch12_message.py@@

@@output ch12_message@@

一条带 3 个 token 的请求消息大约 200 多字节，其中大部分是字段名。字段名在每条消息里重复出现，这是通用序列化的代价；换来的是新增一种消息只需要写一个 dataclass，不用写任何编解码代码。提示词的 token 以原始字节存放（`int32` 每个 4 字节），一个 8000 token 的提示词也只有 32 KB。

## ZMQ 队列

@@code python/minisgl/utils/mp.py:ZmqPushQueue@@

`utils/mp.py` 把 ZMQ socket 包成五种队列，都是"编码器 + msgpack + socket"的薄封装：

- **PUSH / PULL**：一对一（或多对一）的管道，消息按顺序到达、不会丢失。tokenizer → 调度器、调度器 → detokenizer、detokenizer → API Server 都是这种；
- **异步版本**：API Server 运行在 asyncio 事件循环里，用 `zmq.asyncio` 的 socket，`await` 收发不阻塞事件循环；
- **PUB / SUB**：一对多的广播，rank 0 调度器用它把消息转发给其他 rank。

`create=True` 的一方 `bind`（占住地址），另一方 `connect`。地址是 `ipc:///tmp/minisgl_N.pid=启动进程的 PID`，同一台机器上起多个服务也不会冲突。

PUB/SUB 有一个经典陷阱——"慢订阅者"（slow joiner）：**订阅在连接建立之后才异步生效，在此之前发布的消息会被直接丢弃**。如果 rank 0 的第一条广播赶在 rank 1 的订阅生效之前，rank 1 就会永远等下去。官方靠启动时序躲开了它（调度器初始化完还要等 tokenizer、API Server 启动，时间很充裕），但这只是时间差，不是保证。我们把发布端换成 **XPUB**：它能收到"有订阅者订阅了"的通知，`wait_for_subscribers(n)` 等 n 个订阅者都到齐再返回：

@@code python/minisgl/utils/mp.py:ZmqPubQueue@@

`XPUB_VERBOSE` 让每个订阅者的订阅通知都送达（默认会把相同主题的重复订阅合并成一条）。

## 为什么转发原始字节

rank 0 从 tokenizer 收到消息后，要原样转发给其他 rank。`ZmqPullQueue` 提供了 `get_raw()` 和 `decode()` 两个方法：rank 0 先取原始字节、原样 `put_raw` 给其他 rank，再自己解码。这样省掉了一次"解码再编码"，更重要的是保证所有 rank 收到的是**逐字节相同**的消息——各 rank 必须做出完全相同的调度决策（第 14 章）。

!!! upstream "官方实现"
    - 序列化：@@upstream message/utils.py:serialize_type@@
    - 队列：@@upstream utils/mp.py:ZmqPushQueue@@
    - 官方的 `tests/misc/test_serialize.py` 测试了序列化的往返

    我们的队列在 `stop()` 时使用 `close(linger=0)`：官方用默认的 `close()`，如果还有未送出的消息，进程退出时可能会卡住等待。

!!! diff "与官方的差异：XPUB"
    官方的 `ZmqPubQueue` 用普通的 PUB socket，没有等待订阅者。本书第 14 章的演示程序（主进程启动两个 rank 后立即发消息）用官方的写法，在本书的机器上每次都会卡死：rank 0 收到第一条消息后广播、然后在"广播条数"的集合通信里等 rank 1，而 rank 1 永远收不到那条被丢弃的广播。换成 XPUB 并等待订阅者后就不会了。`tests/test_ch12_message.py` 里有对应的测试。

## 测试

@@code tests/test_ch12_message.py:test_zmq_push_pull_over_ipc@@

## 练习

1. 序列化只支持 1 维张量。如果要传一个 `[batch, seq]` 的二维张量，需要改哪几行？
2. 通用序列化每条消息都带着字段名。设计一种更紧凑的格式（例如按字段顺序存成列表），它的代价是什么？
3. 为什么 API Server 用异步版本的队列，而调度器和 tokenizer 用同步版本？

??? success "参考答案"
    1. 序列化时额外存 `shape`（`list(obj.shape)`），反序列化时 `np.frombuffer(...).reshape(shape)`；去掉 `dim() == 1` 的断言。
    2. 收发双方必须对字段顺序达成一致，加字段、删字段都要同步修改两边，且版本不同的进程无法通信；可读性和调试也变差。msgpack 已经很紧凑，瓶颈通常不在这里。
    3. API Server 在一个事件循环里同时处理成百上千个 HTTP 连接，任何阻塞调用都会卡住所有连接；调度器和 tokenizer 是单线程的循环，本来就在"收消息 → 处理"之间轮转，同步调用更简单。

## 小结

- [x] 三组消息分别发往 tokenizer、调度器、API Server，每组按类名在自己的模块里反序列化。
- [x] 通用序列化：对象 → `{"__type__": 类名, ...}` → msgpack 字节；1 维张量存原始字节。
- [x] PUSH/PULL 做点对点管道，PUB/SUB 做 rank 0 到其他 rank 的广播；订阅者必须先连上。
- [x] rank 0 原样转发原始字节，保证所有 rank 看到相同的消息。
