# Stage 3 milestone: make it a service

<p class="lead">At the end of stage 2 the engine is still a Python object: you call <code>generate</code> in the same process, hand it every request at once and collect every result at once. The four chapters of stage 3 turn it into a multi-process online service: an API server takes HTTP, a tokenizer process tokenizes, a scheduler process computes, messages travel between them over ZMQ, and the outside sees an OpenAI-compatible interface with streaming and abort on disconnect. This page starts the service first and talks to it with the standard OpenAI client.</p>

**What you hold at the end of this stage**

- One command, `python -m minisgl --model ...`, starts the service; any OpenAI client (or one `curl`) can connect;
- Requests arrive one by one, knowing nothing of each other, and still land in the same batch inside the scheduler;
- Streaming: the first token is sent as soon as it exists, without waiting for the whole answer;
- When a client disconnects, its request is aborted and its KV pages reclaimed.

## Run it first {#先跑起来}

```bash
cd minisgl && python examples/stage3_milestone.py
```

@@code examples/stage3_milestone.py@@

@@output stage3_milestone@@

## Reading the numbers {#读这几个数字}

**1 API server process + 3 worker processes.** Everything in stage 2 lived in one process; now tokenizing, scheduling and HTTP each have their own, passing messages over ZMQ. The point of multiple processes is not parallel computation — all the computing happens in the scheduler process — but to make sure the HTTP event loop and Python tokenization never stall the step being computed, and vice versa.

**Streaming: first chunk after 118 ms, all of it after 1027 ms.** Non-streaming returns only once all 32 tokens exist; streaming shows the user the first word after 118 ms. From here on, time to first token (TTFT) and time per output token (TPOT) are two separate metrics, and inference SLAs are written in terms of them.

**8 concurrent requests in 1.35 seconds, each taking 1.34–1.35 seconds.** Eight requests leave eight threads at once, pass through HTTP, tokenization and ZMQ into the scheduler, and land in **the same batch**, so each takes almost exactly as long as the others — nobody queues. The effect is the same as handing `generate` eight prompts in stage 2; the difference is that these came over the network and know nothing of each other.

## What one process became {#一个进程变成了哪几个}

| Stage 2 | Stage 3 | Where |
| --- | --- | --- |
| the arguments of `generate` are the requests | message objects + generic serialization + ZMQ queues (PUSH/PULL, PUB/SUB) | [Messages, serialization and ZMQ](message.md) |
| `self.tokenizer(prompt)` in the same process | a separate tokenizer process; detokenization is incremental (one chunk per streamed token) | [The tokenizer and incremental detokenization](tokenizer.md) |
| the scheduler receives a Python list | the scheduler reads requests from ZMQ and sends results back; with several ranks, rank 0 broadcasts to the rest | [Scheduler IO and multi-rank sync](scheduler-io.md) |
| no HTTP | a FastAPI OpenAI-compatible API, SSE streaming, abort on disconnect; a launcher that starts every process | [The API server and the launcher](api-server.md) |

## How to read these four chapters {#怎么读这四章}

Read [messages](message.md) first: everything the processes of this stage pass to each other is defined in that one file, and the next three chapters only send and receive it. At [the API server](api-server.md), really type the `curl` line this page's script prints last, then set `stream=True` and look at what each SSE frame contains. After the four chapters, revisit the fixes to the official implementation in chapters 11 and 12 (one stale message sent after EOS, a PUB/SUB that does not wait for subscribers) — all of them sit on this stage's boundaries.

!!! abstract "Checkpoint: do these before stage 4"
    - [ ] Draw the processes and messages a request passes through: HTTP → tokenizer → scheduler → detokenizer → HTTP;
    - [ ] Explain why tokenization gets its own process instead of running inside the API server;
    - [ ] Count how many serializations one streamed token goes through on its way to a chunk;
    - [ ] Interrupt a streaming request with `Ctrl-C` and confirm its KV pages are reclaimed in the scheduler (chapter 15's test).

## Summary {#小结}

- [x] Stage 3 turns a single-process `generate` into a multi-process OpenAI-compatible service with ZMQ messages.
- [x] Streaming makes time to first token and time per token two separate metrics.
- [x] Concurrent requests from the network land in one batch naturally: 8 requests at 1.35 seconds each, no queueing.
- [x] Multiple processes exist so that HTTP and tokenization never stall the computation, not to compute in parallel.
