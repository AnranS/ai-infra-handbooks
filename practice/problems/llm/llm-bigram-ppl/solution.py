import math
from collections import Counter

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class BigramLM:
    def __init__(self, sentences, k=1.0):
        self.k = k
        self.vocab = {w for s in sentences for w in s} | {EOS, UNK}
        self.pairs = Counter()
        self.ctx = Counter()
        for s in sentences:
            words = [BOS] + [self._norm(w) for w in s] + [EOS]
            for a, b in zip(words, words[1:]):
                self.pairs[a, b] += 1
                self.ctx[a] += 1

    def _norm(self, w):
        return w if w in self.vocab or w == BOS else UNK

    def prob(self, prev, word):
        prev, word = self._norm(prev), self._norm(word)
        return (self.pairs[prev, word] + self.k) / (self.ctx[prev] + self.k * len(self.vocab))

    def perplexity(self, sentence):
        words = [BOS] + list(sentence) + [EOS]
        logp = sum(math.log(self.prob(a, b)) for a, b in zip(words, words[1:]))
        return math.exp(-logp / (len(words) - 1))
