from checker import check, raises
from solution import REQUIRED, compatible, decode, encode

S = [("request_id", REQUIRED), ("prompt", REQUIRED), ("priority", 0), ("client_index", 0)]


def test_example():
    check(encode({"request_id": "r1", "prompt": [1, 2]}, S), ["r1", [1, 2]], "末尾的默认值都省略")
    check(encode({"request_id": "r1", "prompt": [1, 2], "client_index": 3}, S), ["r1", [1, 2], 0, 3], "中间的默认值要占位")
    check(decode(["r1", [1, 2]], S)["priority"], 0, "省略的字段补默认值")


def test_roundtrip_and_errors():
    obj = {"request_id": "r2", "prompt": [7], "priority": 5}
    check(encode(obj, S), ["r2", [7], 5], "只省略 client_index")
    check(decode(encode(obj, S), S), {**obj, "client_index": 0}, "编码再解码")
    check(encode({"request_id": "r", "prompt": [], "priority": 0, "client_index": 0}, S), ["r", []], "显式给出的默认值也省略")
    with raises(ValueError, "缺少必填字段"):
        encode({"request_id": "r"}, S)
    with raises(ValueError, "未知字段"):
        encode({"request_id": "r", "prompt": [], "tenant": 1}, S)
    with raises(ValueError, "值比字段多"):
        decode(["r", [], 0, 0, 9], S)
    with raises(ValueError, "少于必填字段"):
        decode(["r"], S)
    check(encode({"request_id": "", "prompt": 0}, [("request_id", REQUIRED), ("prompt", REQUIRED)]), ["", 0], "必填字段即使是假值也不能省")


def test_compatibility():
    check(compatible(S, S + [("cache_salt", None)]), True, "末尾追加带默认值的字段")
    check(compatible(S, S + [("tenant", REQUIRED)]), False, "追加必填字段：老版本发来的消息缺这个字段")
    check(compatible(S, S[:2] + [("tenant", 1)] + S[2:]), False, "在中间插字段：后面的全部错位")
    check(compatible(S, S[:2] + [("priority", 1)] + S[3:]), False, "改默认值：省略时两边理解不同")
    check(compatible(S, [S[1], S[0]] + S[2:]), False, "改顺序")
    check(compatible(S + [("x", 0)], S), False, "删字段")
    old_msg = encode({"request_id": "r", "prompt": [1], "priority": 2}, S)
    new = S + [("cache_salt", None)]
    check(decode(old_msg, new)["cache_salt"], None, "新版本读老消息：补默认值")
    check(decode(encode({"request_id": "r", "prompt": [1]}, new), S)["priority"], 0, "新字段为默认值时，老版本也能读")
