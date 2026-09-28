"""第 13 章：对话模板，以及增量反分词如何避免输出半个字符。"""

from minisgl.core import SamplingParams
from minisgl.message import DetokenizeMsg, TokenizeMsg
from minisgl.tokenizer import DetokenizeManager, TokenizeManager
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
ids = TokenizeManager(tok).tokenize([TokenizeMsg(
    uid=0, text=[{"role": "user", "content": "你好"}], sampling_params=SamplingParams())])[0]
print("对话模板展开后:", repr(tok.decode(ids)))

text = "鱻䨻🦩 is rare, 推理 naïve"  # 前三个字符各被拆成两个 token
out_ids = tok(text).input_ids
dm = DetokenizeManager(tok)
print(f"{'token':>7}  {'单独解码':<12} 增量输出")
pieces = []
for i, t in enumerate(out_ids):
    piece = dm.detokenize([DetokenizeMsg(uid=1, next_token=t, finished=i == len(out_ids) - 1)])[0]
    pieces.append(piece)
    print(f"{t:>7}  {tok.decode([t])!r:<12} {piece!r}")
print("拼起来:", repr("".join(pieces)), " 与整体解码相同:", "".join(pieces) == tok.decode(out_ids))
