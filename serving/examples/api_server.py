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
