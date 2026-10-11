"""第 12 章：手写模板和官方 Jinja 模板逐 token 对；流式输出；思考模式；多轮"""
import os

import torch
from transformers import AutoTokenizer

from qwen.chat import Chat, build_prompt
from qwen.model import Model
from qwen.weights import find_model

torch.set_num_threads(max(1, os.cpu_count() // 2))
path = find_model()
model = Model.from_pretrained(path)
chat = Chat(model, path)

hf_tok = AutoTokenizer.from_pretrained(path)                       # 1. 和官方模板对：transformers 最后一次出场
cases = [([{"role": "user", "content": "你好"}], False),
         ([{"role": "user", "content": "你好"}], True),
         ([{"role": "system", "content": "你是一个诗人。"}, {"role": "user", "content": "写一句诗"}], False),
         ([{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "<think>\n想一下\n</think>\n\n2"}, {"role": "user", "content": "再加 1 呢"}], False)]
for messages, thinking in cases:
    mine = chat.tk.encode(build_prompt(messages, thinking)).ids
    ref = hf_tok(hf_tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=False))["input_ids"]
    print(f"{len(messages)} 条消息，思考={thinking!s:5}：{len(mine)} 个 token，和官方模板一致 = {mine == ref}")
print(repr(build_prompt(cases[3][0])))

print("\n--- 流式（关思考）")                                        # 2. 流式输出
chat.ask([{"role": "user", "content": "用一句话介绍一下 Transformer。"}], temperature=0)

print("\n--- 开思考")                                                # 3. 思考模式：模型先在 <think> 里推理
answer = chat.ask([{"role": "user", "content": "9.11 和 9.8 哪个大？"}], thinking=True, temperature=0, max_new_tokens=160)

print("\n--- 多轮：第二轮带着第一轮的问答")                            # 4. 多轮对话
history = [{"role": "user", "content": "北京在哪个国家？"}]
history.append({"role": "assistant", "content": chat.ask(history, temperature=0)})
history.append({"role": "user", "content": "它的面积大概多少？"})
chat.ask(history, temperature=0)
