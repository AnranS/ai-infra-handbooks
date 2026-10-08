import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

ckpt = torch.load("ckpt.pt")
model = GPT(GPTConfig(**ckpt["model_config"]))
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer.json")
prompt = torch.tensor([tok.encode("孔明曰：「").ids])

with torch.no_grad():
    probs = model(prompt)[0, -1].softmax(-1)
top = probs.topk(5)
print("下一个 token 最可能是：", "  ".join(f"{tok.decode([int(i)])} {p:.1%}" for p, i in zip(top.values, top.indices)))
for temperature, top_k in [(0.3, None), (1.0, None), (1.0, 20), (1.5, None)]:
    gen = torch.Generator().manual_seed(7)
    out = model.generate(prompt, 40, temperature=temperature, top_k=top_k, generator=gen)
    text = tok.decode(out[0, prompt.shape[1]:].tolist()).replace("\n", " ")
    print(f"温度 {temperature}{'，top-k ' + str(top_k) if top_k else ''}：{text}")
