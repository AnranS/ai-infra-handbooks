"""跟训好的模型聊天：python talk.py（Ctrl-C 退出）"""
import torch

from chat import IM_END, chat_ids, load_tokenizer
from model import GPT, GPTConfig

tok = load_tokenizer("tokenizer_chat.json")
sft = torch.load("sft.pt", weights_only=False)
cfg = GPTConfig(**sft["model_config"])
model = GPT(cfg)
model.load_state_dict(sft["model"])
model.eval()
print("它只读过《三国演义》，会做三件事：接下来写 X、「X」的下一句是什么、这句话是谁说的。Ctrl-C 退出。")
history = []
while True:
    try:
        question = input("\n你：").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not question:
        continue
    history.append({"role": "user", "content": question})
    idx, out = torch.tensor([chat_ids(tok, history, add_generation_prompt=True)[0]]), []
    with torch.no_grad():
        for _ in range(60):
            logits = model(idx[:, -cfg.seq_len:])[:, -1] / 0.8          # 温度 0.8，比贪心活一点
            nxt = torch.multinomial(logits.softmax(-1), 1)
            if nxt.item() == tok.token_to_id(IM_END):
                break
            out.append(nxt.item())
            idx = torch.cat([idx, nxt], dim=1)
    reply = tok.decode(out, skip_special_tokens=False)
    print("模型：" + reply)
    history = (history + [{"role": "assistant", "content": reply}])[-4:]  # 上下文只有 128 个 token，只留最近两轮
