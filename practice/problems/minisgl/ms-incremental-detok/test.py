import random

from checker import check
from solution import Detokenizer

# 模拟分词器：前 100 个 id 是文字片段，100 + b 是原始字节 b
PIECES = ["▁Hello", "▁world", "!", "▁", "▁你", "好", "ing", "▁the", "\n", "▁a", "b", "▁😀"]
EOS = 99


def decode(ids):
    data = b"".join(PIECES[i].encode() if i < 100 else bytes([i - 100]) for i in ids)
    text = data.decode("utf-8", errors="replace").replace("▁", " ")
    return text[1:] if text.startswith(" ") else text


def byte_tokens(s):
    return [100 + b for b in s.encode()]


def stream(ids):
    d = Detokenizer(decode, EOS)
    outs = [d.step(t, finished=(i == len(ids) - 1)) for i, t in enumerate(ids)]
    return outs


def test_example():
    ids = [0, 1, 2]
    check(stream(ids), ["Hello", " world", "!"], "每一步的新增文本（第二个 token 的空格要保留）")


def test_multibyte_split():
    ids = [0] + byte_tokens("中") + [1]
    outs = stream(ids)
    check(outs, ["Hello", "", "", "中", " world"], "汉字的 3 个字节：前两个字节时不输出")
    assert all("�" not in o for o in outs), "不能输出替换字符"


def test_eos_not_emitted():
    ids = [7, 9, EOS]
    outs = stream(ids)
    check("".join(outs), "the a", "EOS 不出现在文本里")
    check(outs[-1], "", "最后一步（EOS）没有新增文本")


def test_random_streams():
    rng = random.Random(0)
    alphabet = "a 中文😀é\n"
    for trial in range(300):
        ids = []
        for _ in range(rng.randint(1, 12)):
            if rng.random() < 0.5:
                ids.append(rng.randrange(len(PIECES)))
            else:
                ids += byte_tokens(rng.choice(alphabet))
        outs = stream(ids)
        full = decode(ids)
        check("".join(outs), full, f"随机用例 {trial}：拼接结果 = 完整解码（ids={ids}）")
        if "�" not in full:
            assert all("�" not in o for o in outs), f"随机用例 {trial}：输出了替换字符 {outs}"
