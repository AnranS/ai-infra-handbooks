# 采样器、流式输出与 OpenAI 接口

<p class="lead">模型输出 logits 之后，还有一段容易被忽视、但每个推理引擎都必须做对的路：一个批次里的每个请求有不同的采样参数，要一次完成采样；生成的 token 要增量地变成文本，处理半个汉字和停止字符串；结果要以 OpenAI 兼容的格式流式返回；HTTP 与分词还不能拖慢引擎主循环。这一章把这些补齐，让迷你引擎成为一个能用 curl 调用的服务。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 一个批次里有的请求贪心、有的 top-p=0.9、有的 top-k=20，怎样一次完成采样？
    2. 为什么 vLLM 不用 `torch.multinomial`？它用什么替代？
    3. 流式输出时，为什么要"扣住"末尾的几个字符不发？
    4. 为什么推理引擎要把 HTTP 服务、分词和模型执行放在不同的进程里？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 把每个请求的采样参数（温度、top-k、top-p）排成张量，整批一次处理：温度按行缩放；top-k、top-p 共用一次排序，每行用自己的阈值做掩码；贪心的请求直接取 argmax（或者用极小的温度）。
    2. `torch.multinomial` 会引入 CPU-GPU 同步，而且不支持每个请求独立的随机种子。vLLM 用指数竞赛（Gumbel 技巧的等价形式）：`argmax(p / q)`，其中 q 服从指数分布，一次向量化操作完成，还能按请求设置种子。
    3. 停止字符串可能跨越多个 token：已经生成的末尾几个字符可能是停止字符串的前半截，发出去之后发现后面凑成了停止字符串，就收不回来了。所以扣住末尾 `max(len(stop)) − 1` 个字符，确认不是停止字符串的开头再发。
    4. HTTP 解析、分词、反分词都是 CPU 工作，和调度、模型执行放在一个 Python 进程里，会因为 GIL 互相抢占，GPU 在等 CPU 的时候空转；拆成不同的进程并行工作。

## 批量采样

每个请求都有自己的温度、top-k、top-p、随机种子。逐个请求采样会让 GPU 执行几百个小 kernel，所以引擎把参数也做成张量，对整批一次完成：

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
    mask = rank >= k[:, None]                                            # top-k：排名靠后的去掉
    probs = sorted_logits.masked_fill(mask, float("-inf")).softmax(dim=-1)
    mask |= (probs.cumsum(dim=-1) - probs) >= p[:, None]                 # top-p：前面的概率已经够 p 的去掉
    return torch.empty_like(logits).scatter_(1, idx, sorted_logits.masked_fill(mask, float("-inf")))


class Sampler:
    def __call__(self, logits: torch.Tensor, reqs) -> list[int]:
        params = [r.params for r in reqs]
        out = logits.argmax(dim=-1)                                          # 温度为 0 的请求：贪心
        rows = [i for i, sp in enumerate(params) if sp.temperature > 0]
        if rows:
            temps = torch.tensor([params[i].temperature for i in rows], dtype=logits.dtype)
            k = torch.tensor([params[i].top_k for i in rows])
            p = torch.tensor([params[i].top_p for i in rows], dtype=logits.dtype)
            probs = apply_top_k_top_p(logits[rows] / temps[:, None], k, p).softmax(dim=-1)
            q = torch.empty_like(probs).exponential_()
            for j, i in enumerate(rows):
                if reqs[i].generator is not None:                            # 指定了种子的请求用自己的生成器
                    q[j].exponential_(generator=reqs[i].generator)
            out[rows] = (probs / q).argmax(dim=-1)
        return out.tolist()
```

两个要点：

1. **top-k、top-p 一次排序完成**。每行的 k、p 不同，但都可以在同一个排好序的矩阵上用不同的阈值做掩码。它与大模型手册中[逐行实现的过滤](llm://inference/decoding/#实现)结果完全相同。
2. **指数竞赛代替 `torch.multinomial`**。如果 $E_i$ 独立服从参数为 1 的指数分布，那么 $\arg\max_i p_i / E_i$ 恰好以概率 $p_i$ 选中 $i$（因为 $E_i / p_i$ 服从参数为 $p_i$ 的指数分布，而若干个指数分布中最小的那个是 $i$ 的概率正比于 $p_i$）。它只需要生成噪声、做一次除法和 argmax，全都是可以批量执行的张量操作；`torch.multinomial` 在 GPU 上会引入 CPU-GPU 同步。另一个好处是：每个请求可以用自己的随机数生成器，有种子的请求**无论和谁在同一批，结果都一样**。

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

在引擎里验证"种子与批次组成无关"：同一个带种子的请求，单独运行和与其他随机采样的请求一起运行，结果相同：

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

（`generate` 也接受一个参数列表，给每个请求不同的采样参数。）

vLLM 的采样器在此之外还有更多步骤，按顺序是：计算原始 logprobs（如果请求了）→ 转成 FP32 → 允许/禁止的 token → 与贪心结果相关的处理器（`min_tokens`、`logit_bias`）→ 重复/频率/存在惩罚 → 贪心或温度 → `min_p` → top-k/top-p → 采样 → 收集 logprobs。每一步都是对整批张量的操作。

## 停止条件与增量反分词

引擎内部处理的是 token，用户要的是文本。token 到文本的转换有两个麻烦：

- **半个字符**：字节级 BPE 可能把一个汉字或 emoji 拆成多个 token，单独解码第一个只能得到乱码"�"（见大模型手册[流式输出与增量反分词](llm://basics/tokenization/#流式输出与增量反分词)）；
- **停止字符串**：用户指定 `stop=["\n解释"]`，文本里一旦出现就要截断并结束请求。可停止字符串可能跨越多个 token，已经发出去的文本就收不回来了，所以末尾"可能是停止字符串开头"的几个字符要先**扣住不发**，确认不是停止字符串之后再发。

```python title="detokenizer.py"
"""detokenizer.py —— 增量反分词与停止字符串：流式输出时每次只吐出新的、完整的文本。"""


class IncrementalDetokenizer:
    def __init__(self, tokenizer, stop: list[str] = ()):
        self.tok, self.stop = tokenizer, list(stop)
        self.ids: list[int] = []
        self.prefix_offset = self.read_offset = 0
        self.text = ""              # 已经确定的完整输出
        self.num_sent = 0           # 已经发给客户端的字符数
        self.stopped = False
        # 可能是某个停止字符串开头的尾巴先扣住不发，确认不是停止字符串后再发
        self.holdback = max((len(s) for s in self.stop), default=1) - 1

    def update(self, token_id: int) -> str:
        """加入一个新 token，返回可以发送给客户端的新文本。"""
        self.ids.append(token_id)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        if len(full) > len(prefix) and not full.endswith("�"):   # 末尾不是半个 UTF-8 字符
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

逐个 token 喂进去，看每次能发出什么：

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

```text title="输出"
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

停止字符串 `\n解释` 有 3 个字符，所以末尾始终扣住 2 个字符不发：每次发出的内容都比已解码的文本滞后两个字符。🦜 被拆成了两个字节 token（单独解码显示为 �），第一个到来时文本没有任何新的完整字符；第二个到来后，🦜 才进入文本。遇到 `\n解释` 后，文本被截断在它之前，扣住的部分一次发完，请求结束。

## OpenAI 兼容的流式服务

最后一步：把引擎包装成 HTTP 服务。结构与 vLLM、SGLang 相同：

- HTTP 线程只负责解析请求、应用对话模板、分词，然后把请求交给引擎，再从请求自己的输出队列里取结果、按 Server-Sent Events（SSE）格式逐块发送；
- 一个后台线程独占引擎，不停地 `step()`，把每步的新 token 反分词后放进各请求的队列。

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
            # 空闲时阻塞等待新请求；忙时只把已经到达的请求取走，不耽误下一步计算
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
                if detok.stopped and req.finish_reason is None:       # 命中停止字符串：由前端结束请求
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

启动服务，三个客户端**并发**请求：两个流式的对话请求（一个贪心、一个带种子采样），一个非流式的补全请求（带停止字符串）：

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
    for line in resp:                                     # SSE：每个事件是一行 "data: {...}"
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

用 curl 也能直接调用（把端口换成实际端口）：

```bash
curl -N http://127.0.0.1:PORT/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "你好"}], "stream": true, "max_tokens": 32}'
```

### 思考模型：把推理过程拆出来

Qwen3、DeepSeek-R1 这类思考模型先在 `<think>` 和 `</think>` 之间写推理过程，再给出回答。OpenAI 兼容接口要把两部分分开返回：推理过程放进单独的字段（SGLang 叫 `reasoning_content`，vLLM 0.30 叫 `reasoning`，旧名字已弃用），`content` 里只有回答。流式输出时，这件事要在增量文本上做，和停止字符串一样会遇到"标签被拆在两个分块里"的问题：

```python
def split_reasoning(chunks, start="<think>", end="</think>"):
    """流式地把 <think>…</think> 之间的文本标成 reasoning，其余标成 content；末尾可能是半个标签，先扣住"""
    buf, in_think, out = "", False, []
    for chunk in chunks:
        buf += chunk
        while buf:
            tag = end if in_think else start
            i = buf.find(tag)
            if i >= 0:                                            # 找到完整的标签：切换状态
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

```text title="输出"
reasoning '\n用户问 1+1'
reasoning '，答案是 2。\n'
content   '\n\n1+1 等于 '
content   '2。'
```

真实的解析器（vLLM、SGLang 都是 `--reasoning-parser qwen3`）还要处理几种变体：有的模板在提示词里已经放了 `<think>`，模型的输出从推理过程直接开始；有的模型不写开始标签；回答开头的空行要去掉。另外两件和引擎有关的事：

- **开关随请求走**：`enable_thinking` 是对话模板的参数，客户端通过 `chat_template_kwargs` 传给服务端，每个请求可以不同；
- **历史里的推理过程会被模板删掉**：多轮对话时，上一轮的推理过程不再出现在下一轮的提示词里，上一轮算过的 KV 就对不上了，前缀缓存只能命中到上一轮的提示词为止（见 [KV 分层缓存](../distributed/kv-offload.md)的实验）。

## 为什么要多进程

我们的迷你服务用的是线程：HTTP 线程和引擎线程共享一个 Python 解释器。在真实负载下这是不行的：Python 的 GIL 让同一时刻只有一个线程在执行 Python 代码，而分词、对话模板、JSON 序列化、反分词都是 CPU 密集的 Python 工作。并发请求一多，它们就会和引擎主循环抢 GIL，导致 GPU 空闲等待。所以两个主流引擎都把它们拆到不同的进程里：

```text
vLLM V1                                         SGLang
┌──────────────────────────────┐                ┌──────────────────────────────┐
│ API server 进程（可以多个）     │                │ 主进程：HTTP + TokenizerManager │
│ HTTP、对话模板、分词             │                │ HTTP、对话模板、分词             │
│ AsyncLLM → OutputProcessor    │                └──────────────┬───────────────┘
│ （反分词、停止字符串、流式）        │                               │ ZMQ
└──────────────┬───────────────┘                ┌──────────────▼───────────────┐
               │ ZMQ + msgpack                  │ Scheduler 进程（每个 TP rank 一个）│
┌──────────────▼───────────────┐                │ 调度、前缀缓存、模型执行           │
│ EngineCore 进程                │                └──────────────┬───────────────┘
│ 调度器 + KV Cache 管理          │                               │ ZMQ
│ → 执行器 → GPU worker 进程       │                ┌──────────────▼───────────────┐
└──────────────────────────────┘                │ DetokenizerManager 进程         │
                                                │ 反分词，结果发回主进程             │
                                                └──────────────────────────────┘
```

两者的思路一致：引擎核心（调度 + 模型执行）的进程里只做对 GPU 吞吐有直接影响的事，其余全部挪走。vLLM 甚至允许起多个 API server 进程（`--api-server-count`），并且正在开发 Rust 实现的前端（`VLLM_USE_RUST_FRONTEND=1`），进一步降低前端开销。

!!! source "源码对照"
    - **vLLM**：采样器在 `vllm/v1/sample/sampler.py`，上面列出的处理顺序就来自它的文档字符串；`ops/topk_topp_sampler.py` 中的 `random_sample` 用 `q.exponential_()` 实现指数竞赛，注释写明"不用 `torch.multinomial` 是因为它会导致 CPU-GPU 同步"，并对有种子的请求逐个用各自的生成器覆盖噪声。GPU 上默认用 Triton 实现的 top-k/top-p（`topk_topp_triton.py`），也可以用 FlashInfer 的采样 kernel。
    - 反分词在前端进程：`vllm/v1/engine/detokenizer.py` 的 `IncrementalDetokenizer`，其中 `stop_buffer_length = max(len(s) for s in stop) - 1` 就是本章的"扣住的字符数"；快速分词器用 `tokenizers` 库的 `DecodeStream`。`OutputProcessor`（`output_processor.py`）把 EngineCore 发回的 token 转成 `RequestOutput`。前后端通信用 ZMQ，消息用 msgpack 编码（`vllm/v1/serial_utils.py`）。
    - **SGLang**：`TokenizerManager`（`srt/managers/tokenizer_manager.py`）、`Scheduler`（`scheduler.py`，由 `run_scheduler_process` 启动）、`DetokenizerManager`（`detokenizer_manager.py`），在 `srt/entrypoints/engine.py` 的 `_launch_subprocesses` 中启动。采样在 `srt/layers/sampler.py`，采样参数的批量张量在 `srt/sampling/`。

!!! interview "面试怎么答"
    被问到"为什么 vLLM 要把 EngineCore 放到单独的进程"时，关键词是 **GIL** 和 **CPU 开销与 GPU 计算的重叠**：分词、反分词、HTTP 处理是 CPU 密集的 Python 代码，放在引擎进程里会让 GPU 等 CPU；拆开之后，前端的工作与 GPU 计算真正并行。再补充一句代价：进程间要序列化和传输数据，所以 vLLM 用 msgpack 并尽量减少传输的内容（例如只传 token ID，不传文本）。

## 练习

**1. 为 `Sampler` 加上重复惩罚。** 需要为每个请求准备什么额外的数据？怎样保持整批计算？

??? success "参考思路"
    重复惩罚需要知道每个请求出现过哪些 token。可以为整批构造一个 `[B, V]` 的布尔掩码（或计数张量，频率惩罚需要计数），然后对整批做 `torch.where(mask, where(logits > 0, logits / penalty, logits * penalty), logits)`，`penalty` 是 `[B, 1]` 的张量。vLLM 在 `ops/penalties.py` 中实现了三种惩罚，计数张量随请求增量维护，而不是每步重新统计。

**2. logprobs。** 客户端请求 `logprobs=5`（返回每个位置概率最高的 5 个 token 及其对数概率）。这些 logprobs 应该在温度、top-p 之前还是之后计算？

??? success "参考答案"
    这是一个设计选择。vLLM V1 默认返回**原始**的 logprobs（`logprobs_mode="raw_logprobs"`），即在任何惩罚、温度、截断之前的模型分布，因为这更能反映模型本身的置信度，也便于做评测。V0 返回的是处理之后的分布。如果需要处理后的值，可以设置 `logprobs_mode="processed_logprobs"`。强化学习的场景特别在意这一点：计算重要性采样比时，推理端返回的 logprobs 要与训练端的计算方式一致。

## 小结

- [x] 采样参数做成张量，整批一次完成；top-k/top-p 共用一次排序；指数竞赛代替 `multinomial`，并支持每个请求独立的种子。
- [x] 增量反分词处理被拆开的多字节字符；停止字符串要扣住末尾 `max(len(stop)) - 1` 个字符再发送。
- [x] OpenAI 兼容接口的流式响应是 SSE：一行一个 `data: {...}`，以 `data: [DONE]` 结束。
- [x] 思考模型的推理过程由 reasoning parser 在流式文本上拆成单独的字段；模板会删掉历史里的推理过程，多轮对话的前缀缓存因此只能命中到上一轮的提示词。
- [x] 真实引擎把 HTTP、分词、反分词与引擎核心拆到不同进程，避免 GIL 让 GPU 空等。
