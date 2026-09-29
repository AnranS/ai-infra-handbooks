import json


def chat_stream(req_id, model, created, deltas, finish_reason, usage=None):
    return ["data: " + json.dumps({"text": d}) + "\n\n" for d in deltas]


def parse_stream(chunks):
    text = ""
    for c in chunks:
        text += c.decode()          # 没有按事件切分，也会切断多字节字符
    return text, None, None
