import math

from checker import check, check_close
from solution import BigramLM

DATA = [["我", "爱", "GPU"], ["我", "爱", "CUDA"]]


def test_example():
    lm = BigramLM(DATA, k=1)
    check(sorted(lm.vocab), sorted(["我", "爱", "GPU", "CUDA", "</s>", "<unk>"]), "词表")
    check_close(lm.prob("我", "爱"), 0.375, what='prob("我", "爱")')
    p = [lm.prob("<s>", "我"), lm.prob("我", "爱"), lm.prob("爱", "GPU"), lm.prob("GPU", "</s>")]
    want = math.exp(-sum(map(math.log, p)) / 4)
    check_close(lm.perplexity(["我", "爱", "GPU"]), want, what="perplexity")


def test_probabilities_sum_to_one():
    lm = BigramLM(DATA + [["GPU", "很", "快"]], k=0.5)
    for prev in ["<s>", "我", "爱", "GPU", "很", "快", "没见过"]:
        total = sum(lm.prob(prev, w) for w in lm.vocab)
        check_close(total, 1.0, what=f"P(· | {prev}) 之和")


def test_unknown_words():
    lm = BigramLM(DATA, k=1)
    check_close(lm.prob("我", "TPU"), lm.prob("我", "<unk>"), what="未登录词按 <unk> 计算")
    check_close(lm.prob("TPU", "爱"), 1 / 6, what="未见过的上下文：(0 + 1) / (0 + 6)")
    assert math.isfinite(lm.perplexity(["TPU", "很", "贵"])), "有未登录词时困惑度也应该是有限值"


def test_k_zero_and_smoothing_effect():
    lm0 = BigramLM(DATA, k=0)
    check_close(lm0.prob("爱", "GPU"), 0.5, what="k=0 时 prob('爱', 'GPU')")
    lm1 = BigramLM(DATA, k=1)
    seen, unseen = ["我", "爱", "GPU"], ["GPU", "我", "爱"]
    assert lm1.perplexity(seen) < lm1.perplexity(unseen), "训练语料里的句子困惑度应该更低"


def test_counts_from_larger_corpus():
    corpus = [["a", "b"]] * 3 + [["a", "c"]]
    lm = BigramLM(corpus, k=1)
    check_close(lm.prob("a", "b"), (3 + 1) / (4 + 5), what='prob("a", "b")')
    check_close(lm.prob("<s>", "a"), (4 + 1) / (4 + 5), what='prob("<s>", "a")')
