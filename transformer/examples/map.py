"""全书地图：权重文件里有什么。只读目录和形状，不加载数据。"""
import json
import re
from collections import OrderedDict
from pathlib import Path

from safetensors import safe_open


def find_model(name="Qwen3.5-0.8B"):
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "models" / name).is_dir():
            return d / "models" / name
    raise FileNotFoundError(f"先下载权重：hf download Qwen/{name} --local-dir models/{name}")


path = find_model()
cfg = json.loads((path / "config.json").read_text())["text_config"]
print(f"{cfg['num_hidden_layers']} 层：{cfg['layer_types'].count('linear_attention')} 层线性注意力 + "
      f"{cfg['layer_types'].count('full_attention')} 层全注意力，排法是", " ".join("L" if t.startswith("linear") else "A" for t in cfg["layer_types"]))
print(f"hidden {cfg['hidden_size']}，前馈 {cfg['intermediate_size']}，词表 {cfg['vocab_size']}，"
      f"注意力 {cfg['num_attention_heads']} 个 query 头 / {cfg['num_key_value_heads']} 组 KV × {cfg['head_dim']} 维，"
      f"线性注意力 {cfg['linear_num_value_heads']} 个头 × {cfg['linear_value_head_dim']} 维")

groups = OrderedDict()                                          # 名字里的层号换成 N，同类张量归成一组
with safe_open(next(path.glob("*.safetensors")), "pt") as f:
    for name in f.keys():
        key = re.sub(r"\.\d+\.", ".N.", name)
        shape = tuple(f.get_slice(name).get_shape())
        g = groups.setdefault(key, {"shape": shape, "n": 0, "params": 0})
        g["n"] += 1
        g["params"] += __import__("math").prod(shape)

parts = {"model.language_model.": "语言模型（这本书）", "model.visual.": "视觉编码器（不碰）", "mtp.": "MTP 草稿头（第 14 章）"}
for prefix, title in parts.items():
    rows = [(k, g) for k, g in groups.items() if k.startswith(prefix)]
    total = sum(g["params"] for _, g in rows)
    print(f"\n== {title}：{sum(g['n'] for _, g in rows)} 个张量，{total / 1e6:.1f}M 参数")
    if prefix != "model.language_model.":
        continue
    for k, g in rows:
        print(f"  {k[len(prefix):]:<48} {str(g['shape']):<14} ×{g['n']:<3} {g['params'] / 1e6:7.1f}M")
