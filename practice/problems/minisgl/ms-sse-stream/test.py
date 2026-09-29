import json
import random

from checker import check
from solution import chat_stream, parse_stream


def events(lines):
    out = []
    for l in lines:
        assert l.startswith("data: ") and l.endswith("\n\n"), f"事件格式不对：{l!r}"
        out.append(l[6:-2])
    return out


def test_example():
    lines = chat_stream("chatcmpl-1", "qwen3", 1700000000, ["你好", "", "，世界"], "stop",
                        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8})
    ev = events(lines)
    check(ev[-1], "[DONE]", "最后一条")
    objs = [json.loads(e) for e in ev[:-1]]
    check(len(objs), 5, "角色 + 两段文本（空串跳过）+ 结束 + usage")
    check(objs[0]["choices"][0]["delta"], {"role": "assistant", "content": ""}, "第一条的 delta")
    check([o["choices"][0]["delta"].get("content") for o in objs[1:3]], ["你好", "，世界"], "文本")
    check((objs[3]["choices"][0]["delta"], objs[3]["choices"][0]["finish_reason"]), ({}, "stop"), "结束的一条")
    check((objs[4]["choices"], objs[4]["usage"]["total_tokens"]), ([], 8), "usage")
    for o in objs:
        check((o["id"], o["object"], o["created"], o["model"]), ("chatcmpl-1", "chat.completion.chunk", 1700000000, "qwen3"),
              "公共字段")
    assert "你好" in lines[1], "中文不要转义"


def test_roundtrip_random_splits():
    rng = random.Random(0)
    for trial in range(50):
        deltas = ["".join(rng.choice("ab 中文😀\n\"") for _ in range(rng.randint(0, 4))) for _ in range(rng.randint(0, 8))]
        usage = {"completion_tokens": len(deltas)} if rng.random() < 0.5 else None
        lines = chat_stream("x", "m", 1, deltas, rng.choice(["stop", "length"]), usage)
        data = "".join(lines).encode()
        cuts = sorted(rng.sample(range(len(data)), min(len(data), rng.randint(0, 20))))
        chunks = [data[a:b] for a, b in zip([0] + cuts, cuts + [len(data)])]
        text, fin, u = parse_stream(chunks)
        want_fin = json.loads(events(lines)[-2 if usage is None else -3])["choices"][0]["finish_reason"]
        check((text, fin, u), ("".join(deltas), want_fin, usage), f"随机用例 {trial}")
