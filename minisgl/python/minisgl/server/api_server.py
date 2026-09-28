"""前端：FastAPI + uvicorn 提供 OpenAI 兼容接口，通过 ZMQ 与 tokenizer 进程通信。

每个 HTTP 请求分配一个 uid。后台有一个协程 listen() 不断从 ZMQ 收 UserReply，按 uid 放进
ack_map 并唤醒对应请求的 asyncio.Event；每个请求的处理协程在 wait_for_ack() 里等待自己的事件，
把收到的增量文本流式地写回客户端。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from pydantic import BaseModel, Field

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: List[Message] | None = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


def _unwrap(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        return [m for m in msg.data if isinstance(m, UserReply)]
    assert isinstance(msg, UserReply)
    return [msg]


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self) -> None:
        while True:
            for reply in _unwrap(await self.recv_tokenizer.get()):
                if reply.uid not in self.ack_map:  # 已经结束或已取消的请求
                    continue
                self.ack_map[reply.uid].append(reply)
                self.event_map[reply.uid].set()

    async def send_one(self, msg: BaseTokenizerMsg) -> None:
        if not self.initialized:  # 第一次调用时（此时事件循环已经在运行）才能创建后台任务
            asyncio.create_task(self.listen())
            self.initialized = True
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int) -> AsyncIterator[UserReply]:
        event = self.event_map[uid]
        try:
            while True:
                await event.wait()
                event.clear()
                pending, self.ack_map[uid] = self.ack_map[uid], []
                for ack in pending:
                    yield ack
                    if ack.finished:
                        return
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def stream_generate(self, uid: int) -> AsyncIterator[bytes]:
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
        yield b"data: [DONE]\n"

    async def stream_chat_completions(self, uid: int) -> AsyncIterator[bytes]:
        first = True
        async for ack in self.wait_for_ack(uid):
            delta: Dict[str, str] = {}
            if first:
                delta["role"] = "assistant"
                first = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output
            chunk = {"id": f"chatcmpl-{uid}", "object": "chat.completion.chunk",
                     "choices": [{"delta": delta, "index": 0, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        end = {"id": f"chatcmpl-{uid}", "object": "chat.completion.chunk",
               "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
        yield f"data: {json.dumps(end)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def stream_with_cancellation(self, gen: AsyncIterator[bytes], request: Request,
                                       uid: int) -> AsyncIterator[bytes]:
        """客户端断开连接时，通知后端中止这个请求，释放它占用的 KV 缓存。"""
        try:
            async for chunk in gen:
                if await request.is_disconnected():
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int) -> None:
        await asyncio.sleep(0.1)
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
        logger.warning(f"Aborting request {uid}")
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self) -> None:
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


_STATE: FrontendManager | None = None


def get_state() -> FrontendManager:
    assert _STATE is not None, "Frontend is not initialized"
    return _STATE


app = FastAPI(title="mini-sglang API server")


def _sampling_params(req: OpenAICompletionRequest) -> SamplingParams:
    return SamplingParams(temperature=req.temperature, top_k=req.top_k, top_p=req.top_p,
                          ignore_eos=req.ignore_eos, max_tokens=req.max_tokens)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    state = get_state()
    uid = state.new_user()
    await state.send_one(TokenizeMsg(uid=uid, text=req.prompt, sampling_params=SamplingParams(
        ignore_eos=req.ignore_eos, max_tokens=req.max_tokens)))
    return StreamingResponse(state.stream_with_cancellation(state.stream_generate(uid), request, uid),
                             media_type="text/event-stream")


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OpenAICompletionRequest, request: Request):
    state = get_state()
    if req.messages:
        text: str | List[Dict[str, str]] = [m.model_dump() for m in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        text = req.prompt
    uid = state.new_user()
    await state.send_one(TokenizeMsg(uid=uid, text=text, sampling_params=_sampling_params(req)))
    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream")
    content = ""
    async for ack in state.wait_for_ack(uid):
        content += ack.incremental_output
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


@app.get("/v1/models")
async def available_models():
    path = get_state().config.model_path
    return ModelList(data=[ModelCard(id=path, root=path)])


async def shell() -> None:
    """交互式对话：直接调用上面的流式生成，不经过 HTTP。/reset 清空历史，/exit 退出。"""
    from prompt_toolkit import PromptSession

    session: PromptSession = PromptSession("$ ")
    history: List[Tuple[str, str]] = []
    state = get_state()
    try:
        while True:
            cmd = (await session.prompt_async()).strip()
            if not cmd:
                continue
            if cmd == "/exit":
                return
            if cmd == "/reset":
                history = []
                continue
            messages = []
            for user, assistant in history:
                messages += [{"role": "user", "content": user},
                             {"role": "assistant", "content": assistant}]
            messages.append({"role": "user", "content": cmd})
            uid = state.new_user()
            await state.send_one(TokenizeMsg(uid=uid, text=messages, sampling_params=SamplingParams(
                temperature=ENV.SHELL_TEMPERATURE.value, max_tokens=ENV.SHELL_MAX_TOKENS.value)))
            answer = ""
            async for ack in state.wait_for_ack(uid):
                answer += ack.incremental_output
                print(ack.incremental_output, end="", flush=True)
            print()
            history.append((cmd, answer))
    except EOFError:
        pass
    finally:
        state.shutdown()
        import psutil

        for child in psutil.Process().children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    global _STATE
    assert _STATE is None
    _STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(config.zmq_frontend_addr, create=True,
                                         decoder=BaseFrontendMsg.decoder),
        send_tokenizer=ZmqAsyncPushQueue(config.zmq_tokenizer_addr,
                                         create=config.frontend_create_tokenizer_link,
                                         encoder=BaseTokenizerMsg.encoder),
    )
    start_backend()
    if run_shell:
        asyncio.run(shell())
    else:
        logger.info(f"API server is ready on http://{config.server_host}:{config.server_port}")
        uvicorn.run(app, host=config.server_host, port=config.server_port, log_level="warning")
