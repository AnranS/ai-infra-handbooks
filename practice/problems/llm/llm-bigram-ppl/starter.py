import math
from collections import Counter


class BigramLM:
    def __init__(self, sentences, k=1.0):
        self.k = k
        self.vocab = set()

    def prob(self, prev, word):
        pass

    def perplexity(self, sentence):
        pass
