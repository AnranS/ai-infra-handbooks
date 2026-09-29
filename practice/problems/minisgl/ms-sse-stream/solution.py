import json


def _event(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def chat_stream(req_id, model, created, deltas, finish_reason, usage=None):
    base = {"id": req_id, "object": "chat.completion.chunk", "created": created, "model": model}

    def chunk(delta, finish=None):
        return _event({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    out = [chunk({"role": "assistant", "content": ""})]
    out += [chunk({"content": d}) for d in deltas if d]
    out.append(chunk({}, finish_reason))
    if usage is not None:
        out.append(_event({**base, "choices": [], "usage": usage}))
    out.append("data: [DONE]\n\n")
    return out


def parse_stream(chunks):
    buf = b""
    text, finish, usage = [], None, None
    for c in chunks:
        buf += c
        while b"\n\n" in buf:
            event, buf = buf.split(b"\n\n", 1)
            line = event.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                return "".join(text), finish, usage
            obj = json.loads(payload)
            if obj.get("usage") is not None:
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                text.append(ch["delta"].get("content", ""))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return "".join(text), finish, usage
