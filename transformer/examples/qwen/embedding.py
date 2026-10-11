"""嵌入层与输出层：进门查表把 token id 变成向量，出门用同一张表把向量变回词表上的分数。"""
import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))   # 一行一个 token

    def forward(self, ids):
        return self.weight[ids]                                     # 查表就是取行：[B, T] -> [B, T, dim]


class LMHead(nn.Module):
    """tie_word_embeddings=True：输出层直接用嵌入层那张表，不另存一份（省下 254M 参数）"""

    def __init__(self, weight):
        super().__init__()
        self.weight = weight                                        # 同一个 Parameter 对象

    def forward(self, x):
        return x @ self.weight.T                                    # [B, T, dim] @ [dim, vocab] -> 每个位置在词表上的分数
