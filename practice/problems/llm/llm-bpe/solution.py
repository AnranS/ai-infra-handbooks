from collections import Counter

EOW = "</w>"


def _merge(symbols, pair):
    a, b = pair
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    words = Counter(w for line in corpus for w in line.split())
    vocab = {tuple(w) + (EOW,): c for w, c in words.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, c in vocab.items():
            for p in zip(symbols, symbols[1:]):
                pairs[p] += c
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        new_vocab = Counter()
        for symbols, c in vocab.items():
            new_vocab[_merge(symbols, best)] += c
        vocab = new_vocab
    return merges


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    rank = {p: i for i, p in enumerate(merges)}
    symbols = tuple(word) + (EOW,)
    while len(symbols) > 1:
        cands = [p for p in zip(symbols, symbols[1:]) if p in rank]
        if not cands:
            break
        symbols = _merge(symbols, min(cands, key=rank.__getitem__))
    return list(symbols)
