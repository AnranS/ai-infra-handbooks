---
title: OpenAI 兼容的流式响应（SSE）
chapter: serve/api-server.md
difficulty: 简单
tags: [SSE, OpenAI API, 流式输出]
---
`/v1/chat/completions` 在 `stream=true` 时用 Server-Sent Events 返回：每条事件是一行 `data: <JSON>` 加一个空行，最后是 `data: [DONE]`。实现服务端和客户端两个函数：

**`chat_stream(req_id, model, created, deltas, finish_reason, usage=None)`**：返回字符串列表，每个元素是一条完整的事件（`"data: ...\n\n"`）：

1. 第一条：`choices[0].delta` 是 `{"role": "assistant", "content": ""}`；
2. `deltas` 里每个**非空**字符串一条：`delta` 是 `{"content": 文本}`；
3. 结束的一条：`delta` 是 `{}`，`finish_reason` 是给定的值（`"stop"` 或 `"length"`），其他条的 `finish_reason` 是 `null`；
4. 如果给了 `usage`（字典），再发一条 `choices` 为空列表、带 `usage` 字段的事件；
5. 最后 `"data: [DONE]\n\n"`。

每条 JSON（除 `[DONE]`）都包含 `"id": req_id`、`"object": "chat.completion.chunk"`、`"created": created`、`"model": model`；`choices` 里的元素是 `{"index": 0, "delta": ..., "finish_reason": ...}`。中文不要转义成 `\uXXXX`。

**`parse_stream(chunks)`**：客户端：输入一串**任意切分**的字节块（网络上收到的数据不一定按事件边界切开，一个 UTF-8 字符也可能被切断），返回 `(text, finish_reason, usage)`。

<!-- 题解 -->
服务端就是拼 JSON：`"data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"`。

客户端要缓冲：把字节块累积起来，按 `b"\n\n"` 切出完整的事件，最后一个不完整的部分留到下一次；在**字节**层面切分再解码，避免多字节字符被切断。
遇到 `data: [DONE]` 结束。书中的 API server 用 FastAPI 的 `StreamingResponse` 包一个异步生成器，每从 detokenizer 收到一段增量文本就 `yield` 一条事件。
