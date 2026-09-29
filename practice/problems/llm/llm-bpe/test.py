import random
from collections import Counter

from checker import check
from solution import encode, train_bpe

EOW = "</w>"
CORPUS = ["low " * 5 + "lower " * 2 + "newest " * 6 + "widest " * 3]


def ref_merge(sym, pair):
    out, i = [], 0
    while i < len(sym):
        if i + 1 < len(sym) and (sym[i], sym[i + 1]) == pair:
            out.append(sym[i] + sym[i + 1])
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return tuple(out)


def ref_train(corpus, n):
    vocab = Counter()
    for line in corpus:
        for w in line.split():
            vocab[tuple(w) + (EOW,)] += 1
    merges = []
    for _ in range(n):
        pairs = Counter()
        for s, c in vocab.items():
            for i in range(len(s) - 1):
                pairs[s[i], s[i + 1]] += c
        if not pairs:
            break
        top = max(pairs.values())
        best = sorted(p for p, c in pairs.items() if c == top)[0]
        merges.append(best)
        nv = Counter()
        for s, c in vocab.items():
            nv[ref_merge(s, best)] += c
        vocab = nv
    return merges


def ref_encode(word, merges):
    rank = {p: i for i, p in enumerate(merges)}
    s = tuple(word) + (EOW,)
    while True:
        cands = [(rank[(a, b)], (a, b)) for a, b in zip(s, s[1:]) if (a, b) in rank]
        if not cands:
            return list(s)
        s = ref_merge(s, min(cands)[1])


def test_example():
    merges = train_bpe(CORPUS, 10)
    check(merges[:3], [("e", "s"), ("es", "t"), ("est", "</w>")], "前 3 条合并规则")
    check(encode("lowest", merges), ["low", "est</w>"], 'encode("lowest")')


def test_matches_reference_on_example():
    check(train_bpe(CORPUS, 10), ref_train(CORPUS, 10), "10 条合并规则")


def test_stops_early():
    merges = train_bpe(["ab ab"], 100)
    check(merges, [("a", "b"), ("ab", "</w>")], "没有相邻对时提前停止")
    check(encode("ab", merges), ["ab</w>"], 'encode("ab")')


def test_overlapping_pairs():
    """aaaa：合并 (a, a) 要从左到右、不重叠"""
    merges = train_bpe(["aaaa"], 1)
    check(merges, [("a", "a")], "第一条规则")
    check(encode("aaaaa", merges), ["aa", "aa", "a", "</w>"], 'encode("aaaaa")')


def test_unknown_chars_and_order():
    merges = train_bpe(CORPUS, 30)
    check(encode("xyz", merges), ["x", "y", "z", "</w>"], "没见过的字符原样保留")
    for w in ["lower", "newer", "widest", "slow", "low"]:
        check(encode(w, merges), ref_encode(w, merges), f'encode("{w}")')


def test_random_corpora():
    rng = random.Random(0)
    for trial in range(15):
        words = ["".join(rng.choice("abcde") for _ in range(rng.randint(1, 7))) for _ in range(rng.randint(3, 12))]
        corpus = [" ".join(rng.choice(words) for _ in range(rng.randint(5, 40)))]
        n = rng.randint(1, 25)
        want = ref_train(corpus, n)
        check(train_bpe(corpus, n), want, f"随机语料 {trial} 的合并规则")
        for w in words[:4] + ["abcabc"]:
            check(encode(w, want), ref_encode(w, want), f'随机语料 {trial}：encode("{w}")')
