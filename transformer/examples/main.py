"""第 0 步：先把终点跑起来。模型本身先当黑盒（Hugging Face 的实现），解码循环自己写。"""
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(max(1, os.cpu_count() // 2))          # CPU 上线程不是越多越好（第 15 章量过）


def find_model(name="Qwen3.5-0.8B"):
    """从当前目录往上找 models/<name>"""
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "models" / name).is_dir():
            return d / "models" / name
    raise FileNotFoundError(f"先下载权重：hf download Qwen/{name} --local-dir models/{name}")


path = find_model()
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
print(type(model).__name__, f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M 参数")

# 1. 文字 -> token：套上聊天模板（第 12 章手写它），再切成 token id
messages = [{"role": "user", "content": "用一句话介绍一下 Transformer。"}]
ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                    return_tensors="pt", return_dict=True)["input_ids"]
print(f"提示词 {ids.shape[1]} 个 token：", tokenizer.decode(ids[0]).replace("\n", "\\n"))
STOP = {tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}   # 两个"说完了"的 token

# 2. 解码循环：整段序列喂进去，取最后一个位置的分数，分最高的 token 接在后面，再来一遍
t0 = time.time()
out = ids
with torch.no_grad():
    for _ in range(40):
        logits = model(out).logits[0, -1]                    # [词表]：下一个 token 的分数
        nxt = logits.argmax()                                 # 贪心：永远取最高分
        out = torch.cat([out, nxt.view(1, 1)], dim=1)
        if nxt.item() in STOP:
            break
new = out[0, ids.shape[1]:]
print(f"生成 {len(new)} 个 token，{time.time() - t0:.1f} 秒")
print(tokenizer.decode(new, skip_special_tokens=True))

# 3. 和 HF 自己的 generate 对一下：同样的贪心，应该一个 token 都不差
ref = model.generate(ids, max_new_tokens=40, do_sample=False, eos_token_id=list(STOP))
print("和 model.generate 一样：", ref[0].tolist() == out[0].tolist())
