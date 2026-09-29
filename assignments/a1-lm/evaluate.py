"""评分脚本：读取你的检查点，在验证集上计算每字节的交叉熵（nats/byte），与达标线比较。

检查点格式（torch.save 的字典）：{"config": Config 的字段组成的 dict, "model": state_dict, "merges": train_bpe 的返回值}
用法：python evaluate.py out/model.pt
"""

import math
import sys
from pathlib import Path

import torch

from lm.bpe import Tokenizer
from lm.model import Config, TransformerLM

TARGET = 0.10          # 达标线：验证集每字节交叉熵（nats/byte）；参考实现约 0.089


@torch.no_grad()
def nats_per_byte(ckpt_path: str) -> float:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    text = (Path(__file__).parent / "data" / "val.txt").read_text(encoding="utf-8")
    tok = Tokenizer(ckpt["merges"])
    ids = torch.tensor(tok.encode(text))
    assert tok.decode(ids.tolist()) == text, "分词器必须能无损还原验证集"
    n = cfg.context_length
    probe = ids[:n][None].clone()                             # 防作弊：改动后半段不能影响前半段的输出（因果性）
    changed = probe.clone()
    changed[0, n // 2:] = torch.randint(0, cfg.vocab_size, (n - n // 2,))
    assert torch.allclose(model(probe)[0, :n // 2], model(changed)[0, :n // 2], atol=1e-4), "模型不是因果的"
    total = 0.0
    for s in range(0, len(ids) - 1, n):                      # 不重叠的窗口：每个 token 只算一次
        x, y = ids[s:s + n], ids[s + 1:s + n + 1]
        x = x[:len(y)]
        logits = model(x[None])[0]
        total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
    return total / len(text.encode("utf-8"))


if __name__ == "__main__":
    score = nats_per_byte(sys.argv[1] if len(sys.argv) > 1 else "out/model.pt")
    print(f"验证集：{score:.4f} nats/byte（{score / math.log(2):.4f} bits/byte），达标线 {TARGET}："
          f"{'通过' if score <= TARGET else '未通过'}")
