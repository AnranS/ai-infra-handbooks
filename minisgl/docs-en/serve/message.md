# Messages, serialization and ZMQ

<p class="lead">Everything in the first two parts ran in one process. To become an online service, the API server, the tokenizer and the scheduler processes have to pass messages. mini-sglang's approach is plain: each kind of message is a dataclass, a 70-line generic serializer turns it into a dictionary of primitives, msgpack turns that into bytes, and a few kinds of ZMQ socket carry them. This chapter writes all of it.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. What kinds of messages are there, and which process sends each to which?
    2. How does a tensor (the prompt's tokens) go into msgpack?
    3. What is the difference between ZMQ's PUSH/PULL and PUB/SUB? Where does the scheduler use each?
    4. Why does rank 0 forward the raw bytes to the other ranks rather than the decoded message?

??? success "Answers (try it yourself first, then expand)"
    1. Three groups, by receiver: to the tokenizer / detokenizer (`TokenizeMsg` from the API server, `DetokenizeMsg` carrying a new token from the scheduler, and `AbortMsg`); to the scheduler (`UserMsg` from the tokenizer, `AbortBackendMsg`, `ExitMsg`); and to the API server (`UserReply` from the detokenizer). Each group also has a `Batch...Msg` that packs several together, and each is deserialized by class name inside its own module.
    2. A one-dimensional tensor is stored as raw bytes (along with its dtype and so on) in a msgpack bytes field, and the receiver rebuilds the tensor from them.
    3. PUSH / PULL is a point-to-point pipe (one receiver takes each message); PUB / SUB is a broadcast (every subscriber gets a copy). The scheduler PULLs requests from upstream and PUSHes results out, and rank 0 PUBs what it receives to the other ranks.
    4. Forwarding the raw bytes guarantees every rank sees exactly the same message, saves a decode-then-encode round trip, and avoids any difference re-encoding might introduce.

**Files you will write**: `message/utils.py`, `message/backend.py`, `message/tokenizer.py`, `message/frontend.py`, `message/__init__.py`, `utils/mp.py`.

## Three groups of messages {#三组消息}

They split into three groups by receiver, each with a base class:

| Base class | Receiver | Messages |
| --- | --- | --- |
| `BaseTokenizerMsg` | tokenizer / detokenizer | `TokenizeMsg` (text or a conversation plus sampling parameters), `DetokenizeMsg` (one new token plus whether it finished), `AbortMsg` |
| `BaseBackendMsg` | the scheduler | `UserMsg` (a token tensor plus sampling parameters), `AbortBackendMsg`, `ExitMsg` |
| `BaseFrontendMsg` | the API server | `UserReply` (the delta text plus whether it finished) |

Each group also has a `Batch...Msg` that packs several messages into one and cuts the message count.

@@code python/minisgl/message/backend.py@@

Note the `globals()` in `decoder`: deserialization looks the class up by name in the global namespace of **the module where the message is defined**. That is why the three groups live in three files with their own `decoder`, since each receiver only knows the group addressed to it.

## Generic serialization {#通用序列化}

@@code python/minisgl/message/utils.py@@

The rules are simple: any object becomes `{"__type__": class name, field: value...}` with its fields handled recursively; primitives stay as they are; and a one-dimensional tensor becomes raw bytes plus a dtype string. Deserialization goes the other way: a `__type__` means construct an object of that class.

@@code examples/ch12_message.py@@

@@output ch12_message@@

A request message carrying 3 tokens is a little over 200 bytes, most of it field names. Repeating the field names in every message is what generic serialization costs; what it buys is that a new kind of message needs only a dataclass and no encoding or decoding code at all. The prompt's tokens are stored as raw bytes (4 bytes each as `int32`), so even an 8000-token prompt is only 32 KB.

## ZMQ queues {#zmq-队列}

@@code python/minisgl/utils/mp.py:ZmqPushQueue@@

`utils/mp.py` wraps ZMQ sockets into five kinds of queue, all thin layers over "encoder plus msgpack plus socket":

- **PUSH / PULL**: a one-to-one (or many-to-one) pipe where messages arrive in order and are not dropped. tokenizer to scheduler, scheduler to detokenizer and detokenizer to API server all use it;
- **the async versions**: the API server runs in an asyncio event loop and uses `zmq.asyncio` sockets, so `await` on send and receive does not block the loop;
- **PUB / SUB**: a one-to-many broadcast, used by scheduler rank 0 to forward messages to the other ranks.

Whichever side passes `create=True` does the `bind` (claiming the address) and the other does the `connect`. The address is `ipc:///tmp/minisgl_N.pid=the launching process's PID`, so several services on one machine do not collide.

PUB/SUB has a classic trap, the slow joiner: **a subscription only takes effect asynchronously after the connection is established, and anything published before that is dropped outright**. If rank 0's first broadcast beats rank 1's subscription, rank 1 waits forever. Upstream dodges it by startup timing (after the scheduler initializes it still waits for the tokenizer and the API server, which leaves plenty of room), but that is a timing gap, not a guarantee. We switch the publishing side to **XPUB**, which is told when a subscriber subscribes, so `wait_for_subscribers(n)` returns only once all n have arrived:

@@code python/minisgl/utils/mp.py:ZmqPubQueue@@

`XPUB_VERBOSE` makes every subscriber's notification arrive (by default repeated subscriptions to the same topic are collapsed into one).

## Why the raw bytes are forwarded {#为什么转发原始字节}

After rank 0 receives a message from the tokenizer it forwards it verbatim to the other ranks. `ZmqPullQueue` offers both `get_raw()` and `decode()`: rank 0 takes the raw bytes, `put_raw`s them to the other ranks, and only then decodes them itself. That saves a decode-then-encode round trip and, more importantly, guarantees that every rank receives a **byte-for-byte identical** message, since all ranks must make exactly the same scheduling decisions (chapter 14).

!!! upstream "The official implementation"
    - serialization: @@upstream message/utils.py:serialize_type@@
    - the queues: @@upstream utils/mp.py:ZmqPushQueue@@
    - the official `tests/misc/test_serialize.py` tests the serialization round trip

    Our queues use `close(linger=0)` in `stop()`; upstream uses the default `close()`, which can hang on exit waiting for messages that have not been sent.

!!! diff "Difference from upstream: XPUB"
    The official `ZmqPubQueue` uses a plain PUB socket and does not wait for subscribers. The demo program of chapter 14 (the main process starts two ranks and sends a message immediately) hangs every single time on this book's machine with the official version: rank 0 receives the first message, broadcasts it, then waits for rank 1 in the collective that counts the broadcasts, while rank 1 never receives the broadcast that was dropped. With XPUB and waiting for subscribers it does not. `tests/test_ch12_message.py` has a test for it.

## Tests {#测试}

@@code tests/test_ch12_message.py:test_zmq_push_pull_over_ipc@@

!!! interview "How to explain it"
    On inter-process communication: three groups of messages, from the API server to the tokenizer (the request), from the tokenizer to the scheduler (the tokenized request), and from the scheduler through the detokenizer back to the API server (the results). Generic serialization turns an object into `{"__type__": class name, ...}` and encodes it with msgpack, with one-dimensional tensors stored as raw bytes, which is faster and safer than pickle. ZMQ's PUSH/PULL is a point-to-point pipe and PUB/SUB is a broadcast (subscribers have to connect first or early messages are lost); rank 0 broadcasts the raw bytes of a request verbatim to the other ranks so that every rank sees exactly the same message. To explain why NCCL is not used for control messages: they are small and irregular, and NCCL only suits tensors.

## Exercises {#练习}

1. Serialization only supports one-dimensional tensors. Which lines change to send a two-dimensional `[batch, seq]` tensor?
2. Generic serialization carries the field names in every message. Design a more compact format (storing the fields as a list in order, say). What does it cost?
3. Why does the API server use the async queues while the scheduler and the tokenizer use the synchronous ones?

??? success "Answers"
    1. Store `shape` as well when serializing (`list(obj.shape)`) and `np.frombuffer(...).reshape(shape)` when deserializing; drop the `dim() == 1` assertion.
    2. Both sides have to agree on the field order, so adding or removing a field means changing both at once and processes at different versions cannot talk; readability and debugging suffer too. msgpack is already compact and this is rarely the bottleneck.
    3. The API server handles hundreds or thousands of HTTP connections in one event loop, where any blocking call stalls all of them; the scheduler and the tokenizer are single-threaded loops that already alternate between receiving and processing, so synchronous calls are simpler.

## Summary {#小结}

- [x] Three groups of messages go to the tokenizer, the scheduler and the API server, each deserialized by class name inside its own module.
- [x] Generic serialization: object to `{"__type__": class name, ...}` to msgpack bytes, with one-dimensional tensors stored as raw bytes.
- [x] PUSH/PULL makes the point-to-point pipes and PUB/SUB the broadcast from rank 0 to the other ranks; subscribers must connect first.
- [x] rank 0 forwards the raw bytes verbatim so that every rank sees the same message.
