from minisgl.message import DetokenizeMsg
from minisgl.tokenizer import DetokenizeManager, find_printable_text
from transformers import AutoTokenizer

from conftest import QWEN3


def test_incremental_detokenize_equals_full_decode():
    tok = AutoTokenizer.from_pretrained(QWEN3)
    text = "推理引擎 😀 streams text: naïve café, 1+1=2.\n第二行结束。"
    ids = tok(text).input_ids
    dm = DetokenizeManager(tok)
    pieces = []
    for i, t in enumerate(ids):
        pieces += dm.detokenize([DetokenizeMsg(uid=0, next_token=t, finished=i == len(ids) - 1)])
    assert "".join(pieces) == tok.decode(ids)
    assert all("�" not in p for p in pieces)  # 不完整的字符从不发给前端
    assert 0 not in dm.decode_map  # 结束后清理状态


def test_find_printable_text():
    assert find_printable_text("hello wor") == "hello "
    assert find_printable_text("你好") == "你好"
    assert find_printable_text("line\n") == "line\n"
