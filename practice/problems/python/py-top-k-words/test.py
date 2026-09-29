import random

from checker import check, time_limit
from solution import top_k_words


def test_example():
    check(top_k_words(["i", "love", "llm", "i", "love", "gpu"], 2), ["i", "love"])


def test_example_tie():
    """次数相同按字典序"""
    check(top_k_words(["b", "a", "c", "a", "b", "c"], 2), ["a", "b"])


def test_most_common_order_is_not_enough():
    """most_common 在并列时按出现顺序，这里要按字典序"""
    check(top_k_words(["z", "y", "x", "x", "y", "z", "w"], 3), ["x", "y", "z"])


def test_k_equals_all():
    check(top_k_words(["a"], 1), ["a"])
    check(top_k_words(["x", "y", "y"], 2), ["y", "x"])


def test_random_against_sort():
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(50)]
    for _ in range(20):
        words = [rng.choice(vocab) for _ in range(rng.randint(1, 300))]
        k = rng.randint(1, len(set(words)))
        cnt = {w: words.count(w) for w in set(words)}
        want = sorted(cnt, key=lambda w: (-cnt[w], w))[:k]
        check(top_k_words(words, k), want, f"k={k} 的结果")


def test_large():
    rng = random.Random(2)
    words = [f"t{rng.randint(0, 5000)}" for _ in range(100_000)]
    with time_limit(1.0, "10 万个单词"):
        got = top_k_words(words, 10)
    check(len(got), 10, "返回的个数")
