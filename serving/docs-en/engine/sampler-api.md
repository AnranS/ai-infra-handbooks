# The sampler, streaming output and the OpenAI API

<p class="lead">After the model outputs logits, there is still a stretch of road that is easy to overlook but that every inference engine must get right: each request in a batch has different sampling parameters, and sampling must be done in one go; generated tokens must be turned into text incrementally, handling half characters and stop strings; results must be streamed back in an OpenAI-compatible format; and HTTP and tokenization must not slow down the engine's main loop. This chapter fills these in, turning the mini engine into a service you can call with curl.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. A batch has some requests that are greedy, some with top-p=0.9 and some with top-k=20. How is sampling done in one go?
    2. Why doesn't vLLM use `torch.multinomial`? What does it use instead?
    3. In streaming output, why "hold back" the last few characters?
    4. Why do inference engines put the HTTP server, tokenization and model execution in different processes?

??? success "Answers (try first, then expand to compare)"
    1. Lay each request's sampling parameters (temperature, top-k, top-p) out as tensors and process the whole batch at once: temperature scales each row; top-k and top-p share one sort, and each row is masked with its own thresholds; greedy requests take the argmax directly (or use a tiny temperature).
    2. `torch.multinomial` introduces a CPU-GPU synchronization and does not support an independent random seed per request. vLLM uses an exponential race (an equivalent form of the Gumbel trick): `argmax(p / q)` with q exponentially distributed, done in one vectorized operation, with per-request seeds.
    3. A stop string may span several tokens: the last few characters generated may be the first half of a stop string, and once sent they cannot be taken back if what follows completes the stop string. So hold back the last `max(len(stop)) − 1` characters and send them once it is clear they are not the start of a stop string.
    4. HTTP parsing, tokenization and detokenization are CPU work; in the same Python process as scheduling and model execution, they compete through the GIL, and the GPU idles while waiting for the CPU; split into different processes, they work in parallel.

## Batched sampling {#批量采样}

Every request has its own temperature, top-k, top-p and random seed. Sampling request by request would make the GPU run hundreds of small kernels, so the engine turns the parameters into tensors too and does the whole batch at once:

```python title="sampler.py"
"""sampler.py —— 批量采样：一个批次里每个请求可以有不同的温度、top-k、top-p 和随机种子。

做法与 vLLM 相同：top-k/top-p 用一次排序对整批完成；采样用"指数竞赛"代替 torch.multinomial：
argmax(p_i / E_i)（E_i 独立服从指数分布）恰好以概率 p_i 选中 i，而且每个请求可以用自己的随机数生成器。
"""

import torch


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """logits: [B, V]；k: [B]（0 表示不限制）；p: [B]（1.0 表示不限制）。先 top-k，再在剩下的 token 上做 top-p。"""
    sorted_logits, idx = logits.sort(dim=-1, descending=True)
    rank = torch.arange(logits.shape[-1]).expand_as(sorted_logits)
    k = torch.where(k > 0, k, logits.shape[-1])
    mask = rank >= k[:, None]                                            # top-k: drop the lower ranks
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    mask |= (probs.cumsum(dim=-1) - probs) >= p[:, None]                 # top-p: drop tokens whose predecessors already reach p
    return torch.empty_like(logits).scatter_(1, idx, sorted_logits.masked_fill(mask, float("-inf")))


class Sampler:
    def __call__(self, logits: torch.Tensor, reqs) -> list[int]:
        params = [r.params for r in reqs]
        out = logits.argmax(dim=-1)                                          # requests with temperature 0: greedy
        rows = [i for i, sp in enumerate(params) if sp.temperature > 0]
        if rows:
            temps = torch.tensor([params[i].temperature for i in rows], dtype=logits.dtype)
            k = torch.tensor([params[i].top_k for i in rows])
            p = torch.tensor([params[i].top_p for i in rows], dtype=logits.dtype)
            probs = apply_top_k_top_p(logits[rows] / temps[:, None], k, p).softmax(dim=-1)
            q = torch.empty_like(probs).exponential_()
            for j, i in enumerate(rows):
                if reqs[i].generator is not None:                            # requests with a seed use their own generator
                    q[j].exponential_(generator=reqs[i].generator)
            out[rows] = (probs / q).argmax(dim=-1)
        return out.tolist()
```

Two key points:

1. **top-k and top-p are done with one sort**. Each row has different k and p, but all can be masked with different thresholds on the same sorted matrix. The result is exactly the same as the LLM book's [row-by-row filtering](llm://inference/decoding/#实现).
2. **An exponential race instead of `torch.multinomial`**. If the $E_i$ are independent exponentials with rate 1, then $\arg\max_i p_i / E_i$ picks $i$ with probability exactly $p_i$ (because $E_i / p_i$ is exponential with rate $p_i$, and the probability that the minimum of several exponentials is $i$ is proportional to $p_i$). It needs only generating noise, one division and an argmax, all batchable tensor operations; `torch.multinomial` on a GPU introduces a CPU-GPU synchronization. Another benefit: each request can use its own random generator, so a seeded request gives **the same result no matter whom it is batched with**.

```python
import torch
from sampling import top_k_filter, top_p_filter
from sampler import apply_top_k_top_p

torch.manual_seed(0)
logits = torch.randn(4, 1000) * 3
k, p = torch.tensor([0, 50, 5, 20]), torch.tensor([0.9, 1.0, 0.5, 0.7])
batched = apply_top_k_top_p(logits, k, p)
one_by_one = torch.cat([top_p_filter(top_k_filter(logits[i:i + 1], int(k[i])), float(p[i])) for i in range(4)])
print("与逐行过滤一致：", torch.equal(batched, one_by_one), " 每行保留的 token 数：", (batched > -1e9).sum(1).tolist())

probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
q = torch.empty(200_000, 4).exponential_()
freq = torch.bincount((probs / q).argmax(-1), minlength=4) / 200_000
print("指数竞赛的采样频率：", [round(f, 3) for f in freq.tolist()])
assert torch.equal(batched, one_by_one) and (freq - probs).abs().max() < 0.01
```

```text
与逐行过滤一致： True  每行保留的 token 数： [6, 50, 1, 8]
指数竞赛的采样频率： [0.502, 0.298, 0.151, 0.05]
```

Check in the engine that "the seed is independent of the batch composition": the same seeded request gives the same result run alone and run together with other randomly sampled requests:

```python
from transformers import AutoTokenizer
from mini_llm import Transformer
from nano_engine import LLMEngine, SamplingParams
from sampler import Sampler

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)

def chat(q):
    return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)).input_ids

seeded = SamplingParams(max_tokens=30, temperature=0.9, top_p=0.95, seed=1234)
alone = LLMEngine(model, eos_token_id=tok.eos_token_id, sample_fn=Sampler()).generate([chat("给猫起三个名字。")], seeded)
engine = LLMEngine(model, eos_token_id=tok.eos_token_id, sample_fn=Sampler())
mixed = engine.generate([chat("给狗起三个名字。"), chat("给猫起三个名字。"), chat("你好")],
                        [SamplingParams(max_tokens=30, temperature=1.0), seeded, SamplingParams(max_tokens=30)])
print(tok.decode(alone[0], skip_special_tokens=True))
print("单独运行与混在批次中结果相同：", alone[0] == mixed[1])
assert alone[0] == mixed[1]
```

```text
当然可以！以下是三个适合给猫咪的名字，简洁又可爱：

1. **喵星**  
2. **咪咪**  
3.
单独运行与混在批次中结果相同： True
```

(`generate` also accepts a list of parameters, giving each request different sampling parameters.)

vLLM's sampler has more steps on top of this, in order: compute the raw logprobs (if requested) → convert to FP32 → allowed/banned tokens → processors that affect the greedy result (`min_tokens`, `logit_bias`) → repetition/frequency/presence penalties → greedy or temperature → `min_p` → top-k/top-p → sample → gather logprobs. Every step is an operation on tensors of the whole batch.

## Stop conditions and incremental detokenization {#停止条件与增量反分词}

Inside the engine everything is tokens; users want text. Converting tokens to text has two complications:

- **Half characters**: byte-level BPE may split a Chinese character or an emoji into several tokens, and decoding the first alone gives only the garbage "�" (see the LLM book's [streaming output and incremental detokenization](llm://basics/tokenization/#流式输出与增量反分词));
- **Stop strings**: if the user specifies `stop=["\n解释"]`, the text must be cut and the request finished as soon as it appears. But a stop string may span several tokens, and text already sent cannot be taken back, so the last few characters that "might be the start of a stop string" are **held back** first and sent once it is clear they are not one.

```python title="detokenizer.py"
"""detokenizer.py —— 增量反分词与停止字符串：流式输出时每次只吐出新的、完整的文本。"""


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop: list[str] = ()):
        self.tok, self.stop = tokenizer, list(stop)
        self.ids: list[int] = []
        self.prefix_offset = self.read_offset = 0
        self.text = ""              # the complete output settled so far
        self.num_sent = 0           # characters already sent to the client
        self.stopped = False
        # hold back a tail that might start a stop string; send it once it is clearly not one
        self.holdback = max((len(s) for s in self.stop), default=1) - 1

    def update(self, token_id: int) -> str:
        """加入一个新 token，返回可以发送给客户端的新文本。"""
        self.ids.append(token_id)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        if len(full) > len(prefix) and not full.endswith("�"):   # does not end with half a UTF-8 character
            self.text += full[len(prefix):]
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        for s in self.stop:
            pos = self.text.find(s)
            if pos != -1:
                self.text, self.stopped = self.text[:pos], True
        return self._emit(final=self.stopped)

    def flush(self) -> str:
        return self._emit(final=True)

    def _emit(self, final: bool) -> str:
        end = len(self.text) if final else max(self.num_sent, len(self.text) - self.holdback)
        delta, self.num_sent = self.text[self.num_sent:end], end
        return delta
```

Feed in the tokens one by one and see what can be sent each time (the text says "the parrot 🦜 says: the answer is 42.", then a newline and "解释如下", "explanation follows"):

```python
from detokenizer import IncrementalDetokenizer

text = "鹦鹉🦜说：答案是42。\n解释如下"
ids = tok(text).input_ids
detok = IncrementalDetokenizer(tok, stop=["\n解释"])
for i in ids:
    delta = detok.update(i)
    print(f"{tok.decode([i])!r:10} 发出 {delta!r}")
    if detok.stopped:
        break
print("最终文本：", repr(detok.text))
```

```text title="output"
'鹦'        发出 ''
'鹉'        发出 ''
'�'        发出 ''
'�'        发出 '鹦'
'说'        发出 '鹉'
'：'        发出 '🦜'
'答案'       发出 '说：'
'是'        发出 '答'
'4'        发出 '案'
'2'        发出 '是'
'。\n'      发出 '42'
'解释'       发出 '。'
最终文本： '鹦鹉🦜说：答案是42。'
```

The stop string `\n解释` is 3 characters, so the last 2 characters are always held back: each send lags the decoded text by two characters. 🦜 is split into two byte tokens (each shown as � when decoded alone); when the first arrives the text has no new complete character, and only after the second does 🦜 enter the text. When `\n解释` appears, the text is cut just before it, the held-back part is sent in one go, and the request finishes.

## An OpenAI-compatible streaming service {#openai-兼容的流式服务}

The last step: wrap the engine in an HTTP service. The structure is the same as vLLM and SGLang:

- HTTP threads only parse the request, apply the chat template and tokenize, hand the request to the engine, then take results from the request's own output queue and send them chunk by chunk in Server-Sent Events (SSE) format;
- One background thread owns the engine and keeps calling `step()`, detokenizing each step's new tokens into each request's queue.

```python title="api_server.py"
"""api_server.py —— 给 nano_engine 套上 OpenAI 兼容的 HTTP 接口（/v1/completions、/v1/chat/completions，支持流式）。

结构与 vLLM、SGLang 相同：HTTP 线程只负责收发，一个后台线程独占引擎、不停地 step()；
两者之间用队列通信。每个请求有自己的输出队列，流式响应以 Server-Sent Events 的格式逐块发送。
"""

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from detokenizer import IncrementalDetokenizer
from nano_engine import LLMEngine, SamplingParams


class AsyncEngine:
    """在后台线程里运行引擎主循环。submit() 返回一个队列，依次收到 (增量文本, finish_reason)。"""

    def __init__(self, engine: LLMEngine, tokenizer):
        self.engine, self.tok = engine, tokenizer
        self.inbox: queue.Queue = queue.Queue()
        self.streams: dict[str, tuple[queue.Queue, IncrementalDetokenizer]] = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt_ids: list[int], params: SamplingParams, stop: list[str]) -> queue.Queue:
        out: queue.Queue = queue.Queue()
        self.inbox.put((prompt_ids, params, stop, out))
        return out

    def _loop(self):
        while True:
            # block for new requests when idle; when busy, only take what has arrived, without delaying the next step
            block = not self.engine.scheduler.has_unfinished()
            while True:
                try:
                    prompt_ids, params, stop, out = self.inbox.get(block=block)
                except queue.Empty:
                    break
                req = self.engine.add_request(prompt_ids, params)
                self.streams[req.request_id] = (out, IncrementalDetokenizer(self.tok, stop))
                block = False
            for req in self.engine.step():
                out, detok = self.streams[req.request_id]
                delta = detok.update(req.output_ids[-1])
                if detok.stopped and req.finish_reason is None:       # hit a stop string: the frontend finishes the request
                    self.engine.scheduler.finish(req, "stop")
                if req.finish_reason is not None:
                    out.put((delta + detok.flush(), req.finish_reason))
                    del self.streams[req.request_id]
                elif delta:
                    out.put((delta, None))


def make_handler(async_engine: AsyncEngine, model_name: str):
    tok = async_engine.tok

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            chat = self.path == "/v1/chat/completions"
            if chat:
                prompt = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                prompt = body["prompt"]
            prompt_ids = tok(prompt).input_ids
            stop = body.get("stop") or []
            params = SamplingParams(max_tokens=body.get("max_tokens", 64), temperature=body.get("temperature", 1.0),
                                    top_p=body.get("top_p", 1.0), top_k=body.get("top_k", 0), seed=body.get("seed"))
            events = async_engine.submit(prompt_ids, params, [stop] if isinstance(stop, str) else stop)
            rid, created = f"cmpl-{uuid.uuid4().hex[:12]}", int(time.time())
            kind = "chat.completion" if chat else "text_completion"

            def choice(text, finish, delta):
                if not chat:
                    return {"index": 0, "text": text, "finish_reason": finish}
                key = "delta" if delta else "message"
                return {"index": 0, key: {"role": "assistant", "content": text}, "finish_reason": finish}

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    text, finish = events.get()
                    chunk = {"id": rid, "object": kind + (".chunk" if chat else ""), "created": created,
                             "model": model_name, "choices": [choice(text, finish, True)]}
                    self._write_chunk(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n")
                    if finish is not None:
                        break
                self._write_chunk("data: [DONE]\n\n")
                self._write_chunk("")
            else:
                parts, finish, n = [], None, 0
                while finish is None:
                    text, finish = events.get()
                    parts.append(text)
                resp = {"id": rid, "object": kind, "created": created, "model": model_name,
                        "choices": [choice("".join(parts), finish, False)],
                        "usage": {"prompt_tokens": len(prompt_ids)}}
                data = json.dumps(resp, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def _write_chunk(self, s: str):
            data = s.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    return Handler


def serve(engine: LLMEngine, tokenizer, model_name: str, host="127.0.0.1", port=0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(AsyncEngine(engine, tokenizer), model_name))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
```

Start the service, with three clients requesting **concurrently**: two streaming chat requests (one greedy, one sampling with a seed) and one non-streaming completion request (with a stop string):

```python
import http.client
import json
import threading
import time
from api_server import serve

server = serve(LLMEngine(model, eos_token_id=tok.eos_token_id, sample_fn=Sampler()), tok, "qwen2.5-0.5b-instruct")
port = server.server_address[1]

def call(body):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if not body.get("stream"):
        return json.loads(resp.read())
    pieces, t0, ttft = [], time.perf_counter(), None
    for line in resp:                                     # SSE: each event is a line "data: {...}"
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            ttft = ttft or time.perf_counter() - t0
            pieces.append(json.loads(line[6:])["choices"][0]["delta"]["content"])
    return pieces, ttft

bodies = [
    {"messages": [{"role": "user", "content": "用一句话介绍北京。"}], "max_tokens": 40, "temperature": 0, "stream": True},
    {"messages": [{"role": "user", "content": "列出三种水果，用顿号分隔。"}], "max_tokens": 40,
     "temperature": 0.8, "seed": 42, "stream": True},
    {"prompt": "The three primary colors are", "max_tokens": 30, "temperature": 0, "stop": ["."]},
]
results = {}
threads = [threading.Thread(target=lambda i=i, b=b: results.__setitem__(i, call(b))) for i, b in enumerate(bodies)]
t0 = time.perf_counter()
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"3 个并发请求共用时 {time.perf_counter() - t0:.1f} s")
for i in (0, 1):
    pieces, ttft = results[i]
    print(f"请求 {i}：{len(pieces)} 个流式分块，首块 {ttft:.2f} s 到达：{''.join(pieces)}")
print("请求 2：", json.dumps(results[2]["choices"][0], ensure_ascii=False))
print("带种子的请求再发一次，结果相同：", "".join(call(bodies[1])[0]) == "".join(results[1][0]))
```

```text
3 个并发请求共用时 3.5 s
请求 0：18 个流式分块，首块 0.20 s 到达：北京是中国的首都，位于中国北方，是世界著名的历史文化名城。
请求 1：8 个流式分块，首块 0.38 s 到达：苹果、香蕉、橙汁。
请求 2： {"index": 0, "text": " red, blue, and yellow", "finish_reason": "stop"}
带种子的请求再发一次，结果相同： True
```

curl works too (replace the port with the actual one):

```bash
curl -N http://127.0.0.1:PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "你好"}], "stream": true, "max_tokens": 32}'
```

### Reasoning models: splitting out the thinking {#思考模型把推理过程拆出来}

Reasoning models such as Qwen3 and DeepSeek-R1 first write their reasoning between `<think>` and `</think>`, then give the answer. An OpenAI-compatible API returns the two parts separately: the reasoning goes in its own field (`reasoning_content` in SGLang, `reasoning` in vLLM 0.30, where the old name is deprecated), and `content` holds only the answer. In streaming output this must be done on incremental text, and like stop strings it runs into "a tag split across two chunks":

```python
def split_reasoning(chunks, start="<think>", end="</think>"):
    """流式地把 <think>…</think> 之间的文本标成 reasoning，其余标成 content；末尾可能是半个标签，先扣住"""
    buf, in_think, out = "", False, []
    for chunk in chunks:
        buf += chunk
        while buf:
            tag = end if in_think else start
            i = buf.find(tag)
            if i >= 0:                                            # found a complete tag: switch state
                if i:
                    out.append(("reasoning" if in_think else "content", buf[:i]))
                buf, in_think = buf[i + len(tag):], not in_think
                continue
            keep = next((k for k in range(len(tag) - 1, 0, -1) if buf.endswith(tag[:k])), 0)
            if len(buf) > keep:
                out.append(("reasoning" if in_think else "content", buf[:len(buf) - keep]))
            buf = buf[len(buf) - keep:]
            break
    if buf:
        out.append(("reasoning" if in_think else "content", buf))
    return out

stream = ["<thi", "nk>\n用户问 1+1", "，答案是 2。\n</th", "ink>\n\n1+1 等于 ", "2。"]
for kind, text in split_reasoning(stream):
    print(f"{kind:9s} {text!r}")
```

```text title="output"
reasoning '\n用户问 1+1'
reasoning '，答案是 2。\n'
content   '\n\n1+1 等于 '
content   '2。'
```

Real parsers (`--reasoning-parser qwen3` in both vLLM and SGLang) must also handle a few variants: some templates already put `<think>` in the prompt, so the model's output starts directly with the reasoning; some models do not write the opening tag; blank lines at the start of the answer are stripped. Two more things that concern the engine:

- **The switch travels with the request**: `enable_thinking` is a chat template parameter that clients pass to the server through `chat_template_kwargs`, and it can differ per request;
- **The template deletes reasoning from the history**: in multi-turn chat, the previous turn's reasoning no longer appears in the next turn's prompt, so the KV computed in the previous turn no longer lines up, and the prefix cache can only hit up to the previous turn's prompt (see the experiment in [hierarchical KV caching](../distributed/kv-offload.md)).

## Why multiple processes {#为什么要多进程}

![Figure: the process structure of a streaming service: the HTTP process, the engine process, incremental detokenization](../assets/figures/api-processes.svg){.aig-svg}

Our mini service uses threads: the HTTP threads and the engine thread share one Python interpreter. Under real load this does not work: Python's GIL lets only one thread execute Python code at a time, and tokenization, chat templates, JSON serialization and detokenization are all CPU-intensive Python work. With many concurrent requests they compete with the engine's main loop for the GIL, leaving the GPU idle. So both mainstream engines split them into different processes:

<!-- i18n:diagram 2acc84f4d9 -->
```text
vLLM V1                                 SGLang
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│ API server process (one or more) │    │ main process: HTTP +             │
│ HTTP, chat template, tokenize    │    │ TokenizerManager                 │
│ AsyncLLM → OutputProcessor       │    │ HTTP, chat template, tokenize    │
│ (detokenize, stop strings,       │    └─────────────────┬────────────────┘
│  streaming)                      │                      │ ZMQ
└─────────────────┬────────────────┘    ┌─────────────────▼────────────────┐
                  │ ZMQ + msgpack       │ Scheduler process (one per TP    │
┌─────────────────▼────────────────┐    │ rank): scheduling, prefix        │
│ EngineCore process               │    │ cache, model execution           │
│ scheduler + KV cache manager     │    └─────────────────┬────────────────┘
│ → executor → GPU worker          │                      │ ZMQ
│   processes                      │    ┌─────────────────▼────────────────┐
└──────────────────────────────────┘    │ DetokenizerManager process       │
                                        │ detokenize, send results back    │
                                        │ to the main process              │
                                        └──────────────────────────────────┘
```

The idea is the same in both: the engine core process (scheduling + model execution) does only what directly affects GPU throughput, and everything else is moved out. vLLM even allows several API server processes (`--api-server-count`), and is developing a Rust frontend (`VLLM_USE_RUST_FRONTEND=1`) to cut frontend overhead further.

!!! source "Source code"
    - **vLLM**: the sampler is in `vllm/v1/sample/sampler.py`, and the processing order listed above comes from its docstring; `random_sample` in `ops/topk_topp_sampler.py` implements the exponential race with `q.exponential_()`, with a comment explaining that it avoids `torch.multinomial` because that causes a CPU-GPU synchronization, and it overwrites the noise for seeded requests one by one with their own generators. On GPU, top-k/top-p default to a Triton implementation (`topk_topp_triton.py`), and FlashInfer's sampling kernels can be used too.
    - Detokenization happens in the frontend process: `IncrementalDetokenizer` in `vllm/v1/engine/detokenizer.py`, where `stop_buffer_length = max(len(s) for s in stop) - 1` is this chapter's "number of characters held back"; fast tokenizers use the `DecodeStream` of the `tokenizers` library. `OutputProcessor` (`output_processor.py`) turns the tokens EngineCore sends back into `RequestOutput`s. The frontend and backend communicate over ZMQ, with messages encoded in msgpack (`vllm/v1/serial_utils.py`).
    - **SGLang**: `TokenizerManager` (`srt/managers/tokenizer_manager.py`), `Scheduler` (`scheduler.py`, started by `run_scheduler_process`) and `DetokenizerManager` (`detokenizer_manager.py`), launched in `_launch_subprocesses` in `srt/entrypoints/engine.py`. Sampling is in `srt/layers/sampler.py`, and the batched tensors of sampling parameters are in `srt/sampling/`.

!!! interview "How to explain it"
    To explain "why does vLLM put EngineCore in a separate process", the keywords are **the GIL** and **overlapping CPU overhead with GPU computation**: tokenization, detokenization and HTTP handling are CPU-intensive Python code, and in the engine process they make the GPU wait for the CPU; split apart, the frontend's work truly runs in parallel with GPU computation. Add one sentence on the cost: data must be serialized and sent between processes, so vLLM uses msgpack and minimizes what is sent (for example, only token IDs, not text).

## Exercises {#练习}

**1. Add repetition penalty to `Sampler`.** What extra data does each request need? How do you keep it a whole-batch computation?

??? success "Approach"
    Repetition penalty needs to know which tokens each request has produced. Build a `[B, V]` boolean mask for the whole batch (or a count tensor, since frequency penalty needs counts), then compute `torch.where(mask, where(logits > 0, logits / penalty, logits * penalty), logits)` over the whole batch, with `penalty` a `[B, 1]` tensor. vLLM implements the three penalties in `ops/penalties.py`, maintaining the count tensors incrementally per request rather than recounting every step.

**2. logprobs.** A client requests `logprobs=5` (return the 5 most likely tokens at each position with their log-probabilities). Should these logprobs be computed before or after temperature and top-p?

??? success "Answer"
    It is a design choice. vLLM V1 returns the **raw** logprobs by default (`logprobs_mode="raw_logprobs"`), i.e. the model's distribution before any penalty, temperature or truncation, since that better reflects the model's own confidence and is convenient for evaluation. V0 returned the processed distribution. If processed values are needed, set `logprobs_mode="processed_logprobs"`. Reinforcement learning cares about this in particular: when computing importance-sampling ratios, the logprobs the inference side returns must be computed the same way as on the training side.

## Summary {#小结}

- [x] Sampling parameters become tensors and the whole batch is done at once; top-k/top-p share one sort; an exponential race replaces `multinomial` and supports per-request seeds.
- [x] Incremental detokenization handles split multi-byte characters; stop strings require holding back the last `max(len(stop)) - 1` characters before sending.
- [x] The streaming responses of an OpenAI-compatible API are SSE: one `data: {...}` per line, ending with `data: [DONE]`.
- [x] For reasoning models, a reasoning parser splits the reasoning into its own field on the streamed text; templates delete reasoning from the history, so prefix caching in multi-turn chat can only hit up to the previous turn's prompt.
- [x] Real engines split HTTP, tokenization and detokenization from the engine core into different processes, so the GIL does not leave the GPU waiting.
