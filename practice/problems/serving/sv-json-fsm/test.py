import random
import re

from checker import check
from solution import IntArrayFSM, accepts, allowed_tokens, constrained_greedy

PATTERN = re.compile(r"\[(-?(0|[1-9][0-9]*)(, -?(0|[1-9][0-9]*))*)?\]\Z")
VOCAB = ["", "[", "]", "[]", "0", "1", "12", "-", "-5", ",", ", ", " ", "3]", "7, ", "01", "a", "]]"]


def test_example():
    f = IntArrayFSM()
    for good in ["[]", "[0]", "[12, -3, 0]", "[-10, 999]"]:
        check(accepts(f, good), True, f"{good!r} 合法")
    for bad in ["[01]", "[1,2]", "[-]", "[1, ]", "[", "[1 ]", "[, 1]", "[-0]x", "", "[1, -02]"]:
        check(accepts(f, bad), bool(PATTERN.match(bad)), f"{bad!r}")


def test_random_strings_match_regex():
    f = IntArrayFSM()
    rng = random.Random(0)
    alphabet = "[]-0123, "
    for _ in range(3000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 10)))
        check(accepts(f, s), bool(PATTERN.match(s)), f"accepts({s!r})")


def run_prefix(f, text):
    s = f.start()
    for ch in text:
        s = f.step(s, ch)
    return s


def test_allowed_tokens():
    f = IntArrayFSM()
    check(allowed_tokens(f, f.start(), VOCAB, 0), {1, 3}, "开头只能是 [ 或 []")
    check(allowed_tokens(f, run_prefix(f, "[1"), VOCAB, 0), {2, 4, 5, 6, 9, 10, 12, 13, 14}, "[1 之后（\"01\" 接在 1 后面是 101，合法）")
    check(allowed_tokens(f, run_prefix(f, "[0"), VOCAB, 0), {2, 9, 10}, "[0 之后不能再接数字")
    check(allowed_tokens(f, run_prefix(f, "[]"), VOCAB, 0), {0}, "读完之后只能结束")


def test_allowed_tokens_random():
    f = IntArrayFSM()
    rng = random.Random(1)
    prefixes = ["[", "[3", "[3, ", "[-", "[12, 0", "[7, -1"]
    for p in prefixes:
        st = run_prefix(f, p)
        want = {i for i, t in enumerate(VOCAB) if i != 0 and t and _prefix_ok(p + t)}
        if f.is_final(st):
            want.add(0)
        check(allowed_tokens(f, st, VOCAB, 0), want, f"前缀 {p!r} 之后允许的 token")


def _prefix_ok(s):
    for tail in ["", "]", "0]", "1]", " 1]", ", 1]"]:          # 能补成一个合法数组，就是合法的前缀
        if PATTERN.match(s + tail):
            return True
    return False


def test_constrained_greedy():
    f = IntArrayFSM()
    # 打分函数偏爱不合法的 token（"a"、"01"），约束之后仍然得到合法的输出
    prefs = {"": 1, "[": 5, "7, ": 9, "a": 100, "01": 50, "3]": 8, "12": 7}

    def score(text):
        s = [prefs.get(t, 0) for t in VOCAB]
        if text.count("7") >= 2:
            s[VOCAB.index("7, ")] = -1
        return s

    out = constrained_greedy(f, VOCAB, 0, score, 20)
    check(out, "[7, 7, 3]", "约束贪心解码的结果")
    check(accepts(f, out), True, "结果合法")
