"""字节级 BPE。规则（测试按这些规则逐字检查）：

1. 预切分：用正则 PRETOKENIZE 把文本切成"片段"，合并只在片段内部进行，不跨片段；
2. 初始词表是 256 个单字节，编号 0～255；第 i 次合并产生的新 token 编号为 256 + i；
3. 每一轮统计所有片段中相邻 token 对的出现次数（按片段出现的次数加权），合并次数最多的一对；
   次数相同时，选 (左边的字节串, 右边的字节串) 字典序最大的一对；没有可合并的对时提前结束；
4. 编码：对每个片段，反复找出"合并顺序最靠前"的相邻对进行合并，直到没有可合并的对。
"""

import re  # noqa: F401

PRETOKENIZE = r" ?[^\s]+|\s+"


def train_bpe(text: str, vocab_size: int) -> list[tuple[bytes, bytes]]:
    """在 text 上训练 BPE，返回按顺序排列的合并规则 [(左, 右), ...]，长度不超过 vocab_size - 256。"""
    raise NotImplementedError


class Tokenizer:
    def __init__(self, merges: list[tuple[bytes, bytes]]):
        """merges 是 train_bpe 的返回值。"""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """把 token 编号还原成字节串，再按 UTF-8 解码（无法解码的字节用 errors="replace" 处理）。"""
        raise NotImplementedError
