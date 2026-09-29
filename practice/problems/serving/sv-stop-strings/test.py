import random

from checker import check
from solution import StopChecker


def oneshot(text, stops, include):
    best = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (best is None or i < best[0] or (i == best[0] and len(s) > len(best[1]))):
            best = (i, s)
    if best is None:
        return text, False
    return (text[: best[0] + len(best[1])] if include else text[: best[0]]), True


def stream(text, stops, cuts, include=False):
    sc = StopChecker(stops, include)
    maxlen = max(len(s) for s in stops)
    out, fin, prev = [], False, 0
    for c in cuts + [len(text)]:
        e, fin = sc.feed(text[prev:c])
        out.append(e)
        prev = c
        held = c - len("".join(out))
        assert fin or held <= maxlen - 1, f"扣住了 {held} 个字符，最多只能扣住 {maxlen - 1} 个"
        if fin:
            break
    if not fin:
        out.append(sc.flush())
    return "".join(out), fin


def test_example():
    sc = StopChecker(["\nUser:"])
    check(sc.feed("Hello!\nUs"), ("Hello!", False), "第一段：扣住可能是停止字符串开头的 \\nUs")
    check(sc.feed("er: hi"), ("", True), "第二段：拼成了停止字符串")
    check(sc.feed("more"), ("", True), "结束之后")


def test_false_alarm_released():
    sc = StopChecker(["###"])
    check(sc.feed("a##"), ("a", False), "扣住 ##")
    check(sc.feed("b"), ("##b", False), "后面不是 #，放出来")
    check(sc.flush(), "", "flush")


def test_include_stop_and_priority():
    sc = StopChecker(["ab", "abc", "b"], include_stop=True)
    check(sc.feed("xxabc"), ("xxabc", True), "同一位置取更长的停止字符串")
    sc = StopChecker(["cd", "b"])
    check(sc.feed("abcd"), ("a", True), "取最早出现的")


def test_flush_holds():
    sc = StopChecker(["STOP"])
    check(sc.feed("the end ST"), ("the end ", False), "扣住 ST")
    check(sc.flush(), "ST", "生成结束，放出扣住的文本")


def test_random_splits():
    rng = random.Random(0)
    alphabet = "ab#\nUs:er"
    for trial in range(300):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        stops = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4))) for _ in range(rng.randint(1, 3))]
        cuts = sorted(rng.sample(range(len(text) + 1), rng.randint(0, len(text) + 1) if text else 0))
        include = rng.random() < 0.5
        check(stream(text, stops, cuts, include), oneshot(text, stops, include),
              f"随机用例 {trial}：text={text!r}, stops={stops}, 切分点={cuts}")
