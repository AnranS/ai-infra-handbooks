"""第 12 章：手写模板长什么样、流式输出、思考模式、多轮"""
import os

import torch

from qwen.chat import Chat, build_prompt
from qwen.model import Model
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)

messages = [{"role": "system", "content": "你是一个诗人。"}, {"role": "user", "content": "写一句诗"}]   # 1. 模板长什么样
for thinking in (False, True):
    text = build_prompt(messages, thinking)
    ids = chat.tk.encode(text).ids
    print(f"思考={thinking!s:5}：{len(ids)} 个 token，末尾 {ids[-6:]} = {[chat.tk.decode([i]) for i in ids[-6:]]!r}")
print(repr(build_prompt([{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "<think>\n想一下\n</think>\n\n2"}, {"role": "user", "content": "再加 1 呢"}])))
print("特殊 token 的 id：", {t: chat.tk.token_to_id(t) for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")})

print("\n--- 流式（关思考）")                                        # 2. 流式输出
chat.ask([{"role": "user", "content": "用一句话介绍一下 Transformer。"}], temperature=0)

print("\n--- 开思考")                                                # 3. 思考模式：模型先在 <think> 里推理
answer = chat.ask([{"role": "user", "content": "9.11 和 9.8 哪个大？"}], thinking=True, temperature=0, max_new_tokens=160)

print("\n--- 多轮：第二轮带着第一轮的问答")                            # 4. 多轮对话
history = [{"role": "user", "content": "北京在哪个国家？"}]
history.append({"role": "assistant", "content": chat.ask(history, temperature=0)})
history.append({"role": "user", "content": "它的面积大概多少？"})
chat.ask(history, temperature=0)
