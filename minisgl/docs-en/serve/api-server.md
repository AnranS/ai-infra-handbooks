# The API server and the launcher

<p class="lead">The last piece: a FastAPI application takes OpenAI-format requests, gives each a uid, sends it to the tokenizer, and streams the delta text that comes back to the client over SSE; a launcher starts every process in the right order. With this chapter done, <code>python -m minisgl --model Qwen/Qwen3-0.6B</code> is an inference service any OpenAI client can talk to.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Hundreds or thousands of concurrent HTTP requests share one ZMQ receive queue. How does each one get only its own replies?
    2. What does SSE (Server-Sent Events) look like? What is the last message of OpenAI's streaming interface?
    3. The client disconnects mid-stream. What does the server have to do?
    4. Why does the launcher use `spawn` rather than `fork` for its subprocesses?

??? success "Answers (try it yourself first, then expand)"
    1. Each request has a unique `uid`: the receiving coroutine puts a reply into the list for that uid and sets that request's `asyncio.Event`, and each request's handler coroutine waits only on its own event and reads only its own list.
    2. Each message is `data: <JSON>` followed by a blank line; OpenAI's streaming interface gives each chunk a `delta.content`, the last chunk a `finish_reason`, and then sends `data: [DONE]`.
    3. Send an `AbortMsg` to the scheduler, free the request slot and KV it holds, and stop generating for it.
    4. `fork` copies the parent's state, including an already initialized CUDA context, threads and the ZMQ context, none of which can be used safely in the child; `spawn` starts a clean interpreter that initializes its own.

**Files you will write**: `server/args.py`, `server/api_server.py`, `server/launch.py`, `__main__.py`, `shell.py`.

@@tree@@

**This step's main**: `examples/ch15_server.py` — it uses only the files above; `python tools/steps.py check` rebuilds this tree chapter by chapter and runs it.

## One shared queue, many waiters {#一个共享队列多个等待者}

The API server is an asyncio program. Every request's replies come in through the same ZMQ queue and have to be dispatched to the right handler coroutine by uid:

@@code python/minisgl/server/api_server.py:FrontendManager@@

- `new_user()` assigns a uid to a new request and creates a reply list and an `asyncio.Event` for it;
- the background coroutine `listen()` keeps taking `UserReply`s from ZMQ, appends each to its uid's list, and `set()`s its event;
- each request's handler coroutine `await`s `event.wait()` inside `wait_for_ack()`, then takes everything in the list and yields it one by one until it sees `finished`;
- `listen()` is created on the first send, because it needs the event loop to be running already, and there is no loop yet when the FastAPI application starts.

This is the standard asyncio shape for "one producer, many consumers dispatched by key". New replies can arrive and `set()` the event again after it is `clear()`ed, so taking the replies swaps the whole list out, and nothing is missed.

## The OpenAI-compatible endpoints {#openai-兼容接口}

@@code python/minisgl/server/api_server.py:v1_chat_completions@@

- either `messages` (a conversation) or `prompt` (plain text), with the former going through the chat template in the tokenizer;
- the sampling parameters pass straight to the scheduler, and `ignore_eos` is an extension field, often used for load testing;
- a streaming response returns `text/event-stream`: each reply becomes a `data: {...}\n\n` whose `choices[0].delta.content` is the delta text, followed by a `finish_reason: "stop"` and `data: [DONE]`;
- a non-streaming response collects everything into one complete JSON.

@@code python/minisgl/server/api_server.py:FrontendManager.stream_chat_completions@@

## Aborting on disconnect {#断开连接时中止请求}

If the server keeps generating after a client disconnects, that request holds a request slot and its KV cache until `max_tokens`. So the connection is checked before every chunk:

@@code python/minisgl/server/api_server.py:FrontendManager.stream_with_cancellation@@

On a disconnect it deletes that uid's waiting state and sends an `AbortMsg`. That becomes an `AbortBackendMsg` through the tokenizer, and the scheduler finds the request in the waiting queue or the decode set and frees its resources (chapter 11 covered the details of how this interleaves with overlap scheduling).

## The launcher {#启动器}

@@code python/minisgl/server/launch.py:start_backend@@

The startup order:

1. the main process parses the arguments and creates the `FrontendManager` (which binds the frontend's ZMQ address);
2. `spawn` starts a scheduler process per TP rank, one detokenizer process and N tokenizer processes. CUDA cannot be reinitialized in a `fork`ed child, so `spawn` is required;
3. each process puts a "ready" message into a `multiprocessing.Queue` once it is set up (for the schedulers only rank 0 does, and only after every rank has completed `sync_all_ranks`); the main process waits for `num_tokenizer + 2` of them before continuing;
4. uvicorn starts (or the interactive shell).

@@code python/minisgl/server/launch.py:_run_scheduler@@

Interactive mode (`--shell`) runs no HTTP service and chats right in the terminal; it sets `max_running_req` to 1 and captures only a batch-size-1 CUDA Graph, since there is only one user.

## Running it {#运行}

@@code examples/ch15_server.py@@

@@output ch15_server@@

- Processes: the API server's main process, plus one scheduler process and one tokenizer process (tokenizing and detokenizing share it by default).
- Non-streaming: Qwen3's answer starts with an empty `<think></think>` (`/no_think` turned thinking off) and then the answer.
- Streaming: one chunk per token, with the special token `<think>` emitted as a complete chunk of its own.
- The 8 concurrent requests are merged into one batch by the scheduler and take almost as long in total as a single request, which is the throughput continuous batching buys.

!!! upstream "The official implementation"
    - @@upstream server/api_server.py:FrontendManager@@
    - @@upstream server/launch.py:launch_server@@
    - @@upstream server/args.py:parse_args@@

    The `usage` field in upstream's non-streaming `/v1/chat/completions` response is always 0; we leave it out. Upstream's interactive mode kills every child process with `psutil` on exit, which we kept.

## Tests {#测试}

@@code tests/test_ch15_server.py:test_client_disconnect_aborts_request@@

The other two tests in `tests/test_ch15_server.py`: a non-streaming response matches Hugging Face's greedy output word for word, and both streaming and concurrent requests return correctly.

!!! interview "How to explain it"
    On the API server: hundreds or thousands of HTTP requests share one ZMQ receive queue through "a reply list dispatched by uid plus one `asyncio.Event` per request", where a background task puts each reply into its request's list and wakes it. OpenAI's streaming interface is SSE: one `data: {...}` line per chunk with the content in `delta.content`, a `finish_reason` on the last one, and then `data: [DONE]`. A mid-stream disconnect has to send an abort so the request slot and KV are freed, or the engine keeps generating for nobody. The launcher uses `spawn` rather than `fork` because forking copies the parent's already-initialized CUDA, threads and locks, which is not safe.

## Exercises {#练习}

1. Add a correct `usage` (prompt tokens, completion tokens) to the non-streaming response. Which process knows those numbers? Which messages need new fields?
2. Implement `/v1/completions` (the non-chat completion endpoint). How do its request and response formats differ from `/v1/chat/completions`?
3. uids are currently assigned by a counter in the API server. If several API server processes served together later, how would you assign uids without collisions?

??? success "Answers"
    1. The prompt length is known in the tokenizer and the generated count in the scheduler. Add `prompt_tokens` and `completion_tokens` to `UserReply`, filled by the detokenizer at the end (it can count the tokens it received; the prompt length has to be recorded by the tokenizer at encoding time, or carried back by the scheduler on the last `DetokenizeMsg`).
    2. The request uses `prompt` rather than `messages` and skips the chat template; the response has `choices[0].text` rather than `message.content`, each streaming chunk is a `choices[0].text`, and `object` is `text_completion`.
    3. Give each API server a different prefix or stride, for instance uid = process index + process count × local counter, or use UUIDs. The detokenizer's replies also have to route back to the right API server (one receive address per server, with the origin carried in the message).

## Summary {#小结}

- [x] The API server shares one ZMQ receive queue across all requests through "a reply list dispatched by uid plus an `asyncio.Event`".
- [x] OpenAI's streaming interface is SSE: one `delta.content` per chunk, then a `finish_reason` and `[DONE]`.
- [x] A client disconnect sends an `AbortMsg` so the request's slot and KV are freed.
- [x] The launcher `spawn`s the scheduler and tokenizer processes and waits for all of them before starting the HTTP service.
