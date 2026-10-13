# 12. The chat template and thinking mode

<p class="lead">The model only knows tokens. "The user said this", "this turn is over", "think before answering" — all of it is marked by a few special tokens in the template. This chapter writes Qwen3.5's chat template by hand (we checked it: it matches the official chat_template.jinja token by token), builds a streaming <code>Chat</code>, and turns thinking mode on to watch the model "think".</p>

!!! note "What this chapter runs"
    `ch12_chat.py`, about 1 minute (four generations; the thinking-mode one is the longest).

@@files@@

## The template {#模板}

The official template is a 300-line Jinja file (`chat_template.jinja`), most of it about tool calls and images. The conversation itself has only these rules:

- every message: `<|im_start|>role\ncontent<|im_end|>\n`;
- assistant turns in the history **do not keep their thinking** (`<think>…</think>` is removed);
- finally `<|im_start|>assistant\n`, and then — with thinking on, write `<think>\n` so the model continues thinking; with thinking off, write an empty thinking block `<think>\n\n</think>\n\n` on its behalf.

```python title="qwen/chat.py"
"""聊天模板与思考模式。模型只认 token："谁在说""说完了没有""要不要先想一想"全靠模板里的特殊 token 标出来。"""
import torch
from tokenizers import Tokenizer

from qwen.sample import generate
from qwen.weights import find_model

IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"


def build_prompt(messages, thinking=False):
    """messages: [{"role": "system" | "user" | "assistant", "content": ...}] -> 套好模板的字符串，末尾是 assistant 回合的开头"""
    text = ""
    for m in messages:
        content = m["content"].strip()
        if m["role"] == "assistant" and "</think>" in content:
            content = content.split("</think>")[-1].strip()                 # thinking in earlier turns is dropped (the official template does the same)
        text += f"{IM_START}{m['role']}\n{content}{IM_END}\n"
    text += f"{IM_START}assistant\n"
    text += "<think>\n" if thinking else "<think>\n\n</think>\n\n"          # thinking on: let the model continue thinking; off: write an empty thinking block for it
    return text


class Chat:
    def __init__(self, model, path=None):
        path = path or find_model()
        self.model = model
        self.tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # only the tokenizers library, no transformers
        self.stop = (self.tk.token_to_id(IM_END), self.tk.token_to_id(EOT))

    def encode(self, text):
        return torch.tensor([self.tk.encode(text).ids], device=next(self.model.parameters()).device)

    def ask(self, messages, thinking=False, stream=True, **kw):
        """messages 同 build_prompt；kw 是采样参数（temperature、top_k、top_p ...）。流式打印，返回整段回答。"""
        ids = self.encode(build_prompt(messages, thinking))
        out, printed = [], ""

        def on_token(t):
            nonlocal printed
            if t in self.stop:
                return
            out.append(t)
            text = self.tk.decode(out, skip_special_tokens=False)         # re-decode the whole text every time: one character may be split across two tokens
            if stream and not text.endswith("�"):
                print(text[len(printed):], end="", flush=True)
                printed = text

        generate(self.model, ids, stop=self.stop, on_token=on_token, **kw)
        if stream:
            print()
        return self.tk.decode(out, skip_special_tokens=False)
```

The streaming in `Chat.ask` has one subtlety: a Chinese character may be split across two tokens (its three UTF-8 bytes in two tokens), and decoding one of them alone gives `�`. So every new token triggers a re-decode of the whole text, only the new part is printed, and a trailing `�` is held back.

## The template, streaming, thinking, multi-turn {#模板流式思考多轮}

```python title="ch12_chat.py" ci="loose"
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

messages = [{"role": "system", "content": "你是一个诗人。"}, {"role": "user", "content": "写一句诗"}]   # 1. what the template looks like
for thinking in (False, True):
    text = build_prompt(messages, thinking)
    ids = chat.tk.encode(text).ids
    print(f"思考={thinking!s:5}：{len(ids)} 个 token，末尾 {ids[-6:]} = {[chat.tk.decode([i]) for i in ids[-6:]]!r}")
print(repr(build_prompt([{"role": "user", "content": "1+1=?"}, {"role": "assistant", "content": "<think>\n想一下\n</think>\n\n2"}, {"role": "user", "content": "再加 1 呢"}])))
print("特殊 token 的 id：", {t: chat.tk.token_to_id(t) for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")})

print("\n--- 流式（关思考）")                                        # 2. streaming output
chat.ask([{"role": "user", "content": "用一句话介绍一下 Transformer。"}], temperature=0)

print("\n--- 开思考")                                                # 3. thinking mode: the model reasons inside <think> first
answer = chat.ask([{"role": "user", "content": "9.11 和 9.8 哪个大？"}], thinking=True, temperature=0, max_new_tokens=160)

print("\n--- 多轮：第二轮带着第一轮的问答")                            # 4. multi-turn chat
history = [{"role": "user", "content": "北京在哪个国家？"}]
history.append({"role": "assistant", "content": chat.ask(history, temperature=0)})
history.append({"role": "user", "content": "它的面积大概多少？"})
chat.ask(history, temperature=0)
```

```text title="输出"
思考=False：23 个 token，末尾 [74455, 198, 248068, 271, 248069, 271] = ['assistant', '\n', '<think>', '\n\n', '</think>', '\n\n']
思考=True ：21 个 token，末尾 [198, 248045, 74455, 198, 248068, 198] = ['\n', '', 'assistant', '\n', '<think>', '\n']
'<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n<|im_start|>user\n再加 1 呢<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
特殊 token 的 id： {'<|im_start|>': 248045, '<|im_end|>': 248046, '<|endoftext|>': 248044, '<think>': 248068, '</think>': 248069}

--- 流式（关思考）
Transformer 是一种基于深度学习的神经网络架构，通过自注意力机制解决了传统卷积神经网络在处理长距离依赖和复杂语义关系时的局限性，从而实现了强大的语言理解与生成能力。

--- 开思考
Thinking Process:

1.  **Analyze the Request:** The user is asking which number is larger between 9.11 and 9.8.

2.  **Identify the Numbers:**
    *   Number A: 9.11
    *   Number B: 9.8

3.  **Compare the Numbers:**
    *   Both numbers have the same integer part (9).
    *   Compare the decimal parts:
        *   9.11 has a tenths digit of 1.
        *   9.8 has a tenths digit of 8.
    *   Since 8 > 1, 9.8 is greater than 9.11.

4.

--- 多轮：第二轮带着第一轮的问答
北京是中国的首都，位于**中国**境内。
北京是中国面积最大的直辖市，其总面积约为**1.66 万平方公里**。

这个数据包含了北京城（包括市区、郊区和公园）以及周边的郊区、自然保护区和未开发区域。如果仅计算北京城区的面积，约为 1.3 万平方公里。

此外，北京是中国重要的交通枢纽和战略地区，也是世界著名的历史文化名城，拥有众多世界文化遗产和著名地标。
```

The first lines lay the template open: thinking off is 2 tokens longer than thinking on — the `\n\n</think>\n\n` stretch closes the thinking block on the model's behalf; the assistant turn in the history keeps only the answer "2", its thinking removed; `<|im_start|>` and the other special tokens are single tokens whose ids sit at the very end of the vocabulary. In thinking mode the model's output starts with a `<think>` block holding its reasoning, and the answer comes after `</think>`; the 160-token limit cuts it off here, in real use let it finish. Multi-turn chat is simply the history re-assembled with the template and prefilled once more — chapter 10 mentioned that inference engines use prefix caching to skip the repeated computation.

!!! interview "How to explain it"
    The core sentence about chat templates: the structure of the conversation is encoded into the sequence with special tokens, and after pretraining the model is trained to "start answering when it sees `<|im_start|>assistant` and emit `<|im_end|>` when done". Thinking mode is just part of the template: a leading `<think>` lets the model continue thinking; to turn it off, write an empty thinking block for it. Then the engineering trap: with a wrong template (an extra newline, a missing space) the model still answers, but worse — so compare with the official template token by token.

## Summary {#小结}

- [x] The template = `<|im_start|>role\ncontent<|im_end|>\n` per message + the start of the assistant turn; thinking on/off differs only by an empty `<think>` block.
- [x] The hand-written template was checked against the official Jinja template token by token; assistant turns in the history keep only the answer, the thinking never enters the next turn's context.
- [x] `Chat.ask`: streaming (re-decode the whole text to avoid half a character), thinking mode, multi-turn.

**File created in this chapter**: `qwen/chat.py`.
