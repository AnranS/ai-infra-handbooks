"""bpe.py —— 从零实现的 byte-level BPE 分词器（教学用）。"""

import re
from collections import Counter

# 预切分：按"可选的前导空格 + 连续的字母/数字"、"标点"、"空白"切开，合并不会跨越这些边界
PATTERN = re.compile(r" ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+")


def merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class ByteBPE:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}          # (左, 右) -> 新 token 的编号；编号越小越先合并
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        words = Counter(PATTERN.findall(text))                 # 相同的片段只统计一次，乘以频次
        seqs = {w: list(w.encode("utf-8")) for w in words}
        for new_id in range(256, vocab_size):
            counts = Counter()
            for w, freq in words.items():
                s = seqs[w]
                for pair in zip(s, s[1:]):
                    counts[pair] += freq
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))   # 次数最多，平局取编号小的，保证确定性
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            for w in words:
                seqs[w] = merge(seqs[w], best, new_id)

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in PATTERN.findall(text):
            seq = list(piece.encode("utf-8"))
            while len(seq) >= 2:
                # 在当前所有相邻对中，找最早学到的那条合并规则
                pair = min(zip(seq, seq[1:]), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                seq = merge(seq, pair, self.merges[pair])
            ids.extend(seq)
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
