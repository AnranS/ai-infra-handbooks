from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        for msg in msgs:
            if isinstance(msg.text, list):  # chat 消息列表：先套对话模板
                prompt = self.tokenizer.apply_chat_template(
                    msg.text, tokenize=False, add_generation_prompt=True)
            else:
                prompt = msg.text
            ids = self.tokenizer.encode(prompt, return_tensors="pt")
            results.append(ids.view(-1).to(torch.int32))
        return results
