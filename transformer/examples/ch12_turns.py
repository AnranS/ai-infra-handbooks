"""第 12 章（二）：多轮对话里，第二轮的提示词能接上第一轮已经算过的多少 token？（不用模型，只比两段 token）"""
from tokenizers import Tokenizer

from qwen.chat import IM_END, build_prompt
from qwen.weights import find_model

tk = Tokenizer.from_file(str(find_model() / "tokenizer.json"))
enc = lambda s: tk.encode(s).ids
system = {"role": "system", "content": "你是一个乐于助人的助手。"}
q1 = {"role": "user", "content": "北京在哪个国家？"}
answer = "北京是中国的首都，位于华北平原北部。"                       # 假设模型第一轮这样回答（这里不跑模型，只看 token）
p1 = enc(build_prompt([system, q1]))                                 # 关思考
done = p1 + enc(answer) + [tk.token_to_id(IM_END)]                   # 第一轮结束时，算过 KV 的就是这些 token
p2 = enc(build_prompt([system, q1, {"role": "assistant", "content": answer}, {"role": "user", "content": "它的面积大概多少？"}]))
hit = next((i for i, (a, b) in enumerate(zip(p2, done)) if a != b), min(len(p2), len(done)))   # 从头数，两段一样的有多长
print(f"第一轮：提示词 {len(p1)} 个 token，生成 {len(done) - len(p1)} 个，算过 KV 的一共 {len(done)} 个")
print(f"第二轮：提示词 {len(p2)} 个 token，开头和第一轮一样的 {hit} 个，要重新 prefill {len(p2) - hit} 个")
print(f"从这里开始不一样，第一轮是 {tk.decode(done[hit - 3:hit + 4], skip_special_tokens=False)!r}")
print(f"                   第二轮是 {tk.decode(p2[hit - 3:hit + 4], skip_special_tokens=False)!r}")
