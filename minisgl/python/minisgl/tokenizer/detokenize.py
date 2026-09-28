"""增量反分词：每来一个新 token，就算出"新增的那段文本"发给前端。

难点有两个：
1. 一个字符可能被拆到几个 token 里（UTF-8 多字节、字节级 BPE），单独解码会得到 "�"，要等后续 token；
2. 反分词不是逐 token 可加的：decode(a + b) 不一定等于 decode(a) + decode(b)（空格、合并规则）。

做法（与 SGLang、vLLM 相同）：只解码最近的一个窗口。surr_offset 到 read_offset 之间是"已经输出过、
用来提供上下文"的 token，read_offset 之后是新 token。解码 [surr, 末尾] 与 [surr, read]，
两段文本之差就是新增文本；如果它以 "�" 结尾，说明字符还不完整，先不推进偏移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.message import DetokenizeMsg
from transformers import PreTrainedTokenizerBase


def _is_chinese_char(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F
    )


def find_printable_text(text: str) -> str:
    """文本还不完整时，只输出能确定的部分：到换行、到中文字符，或到最后一个空格为止。"""
    if text.endswith("\n"):
        return text
    if text and _is_chinese_char(ord(text[-1])):
        return text
    if len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int  # 已经确认解码的 token 数
    surr_offset: int  # 上下文窗口的起点
    sent_offset: int  # 已经发给前端的字符数


class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []
        for msg in msgs:
            s = self.decode_map.setdefault(msg.uid, DecodeStatus([], "", 0, 0, 0))
            if not (msg.finished and msg.next_token == self.eos_token_id):  # EOS 不输出
                s.decoded_ids.append(msg.next_token)
            read_ids.append(s.decoded_ids[s.surr_offset :])
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])
        # 一批请求一起解码，比逐个调用快
        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        outputs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str) :]
            if new_text and not new_text.endswith("�"):  # 新文本完整：推进窗口
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:  # 不完整：只输出确定的部分，窗口不动
                output_str = s.decoded_str + find_printable_text(new_text)
            outputs.append(output_str[s.sent_offset :])
            s.sent_offset = len(output_str)
            if msg.finished:
                del self.decode_map[msg.uid]
        return outputs
