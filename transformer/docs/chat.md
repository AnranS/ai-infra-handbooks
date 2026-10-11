# （十二）聊天模板与思考模式

<p class="lead">模型只认 token。"这句是用户说的""这句说完了""先想一想再回答"，全靠模板里的几个特殊 token 标出来。这一章手写 Qwen3.5 的聊天模板（和官方的 Jinja 模板逐 token 对齐），做一个流式打印的 <code>Chat</code>，再打开思考模式看看模型怎么"想"。从此不再需要 transformers——它在这一章只出场一次，用来核对模板。</p>

!!! note "这一章跑什么"
    `ch12_chat.py`，约 1 分钟（三次生成，其中思考模式那次最长）。

## 模板

官方模板是一个 300 行的 Jinja 文件（`chat_template.jinja`），大部分在处理工具调用和图片。对话本身只有这几条规则：

- 每条消息：`<|im_start|>角色\n内容<|im_end|>\n`；
- 历史里的 assistant 回合**不保留思考过程**（`<think>…</think>` 被去掉）；
- 末尾加上 `<|im_start|>assistant\n`，然后——开思考就写 `<think>\n` 让模型接着想；关思考就替它写一个空的思考块 `<think>\n\n</think>\n\n`。

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
            content = content.split("</think>")[-1].strip()                 # 历史回合里的思考过程不保留（官方模板也这样）
        text += f"{IM_START}{m['role']}\n{content}{IM_END}\n"
    text += f"{IM_START}assistant\n"
    text += "<think>\n" if thinking else "<think>\n\n</think>\n\n"          # 开思考：让模型接着想；关思考：替它写一个空的思考块
    return text


class Chat:
    def __init__(self, model, path=None):
        path = path or find_model()
        self.model = model
        self.tk = Tokenizer.from_file(str(path / "tokenizer.json"))         # 只用 tokenizers 库，不用 transformers
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
            text = self.tk.decode(out, skip_special_tokens=False)         # 每次重新解码整段：一个汉字可能拆在两个 token 里
            if stream and not text.endswith("�"):
                print(text[len(printed):], end="", flush=True)
                printed = text

        generate(self.model, ids, stop=self.stop, on_token=on_token, **kw)
        if stream:
            print()
        return self.tk.decode(out, skip_special_tokens=False)
```

`Chat.ask` 的流式打印有一个细节：一个汉字可能被拆成两个 token（UTF-8 的 3 个字节分在两个 token 里），单独解码其中一个会得到 `�`。所以每来一个 token 就把整段重新解码，只打印新增的部分，末尾是 `�` 就先不打。

## 对齐、流式、思考

```python title="ch12_chat.py" ci="loose"
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
```

```text title="输出"
1 条消息，思考=False：13 个 token，和官方模板一致 = True
1 条消息，思考=True ：11 个 token，和官方模板一致 = True
2 条消息，思考=False：23 个 token，和官方模板一致 = True
3 条消息，思考=False：32 个 token，和官方模板一致 = True
'<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n<|im_start|>user\n再加 1 呢<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

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

思考模式下模型的输出以 `<think>` 块开头，里面是它的推理过程，`</think>` 之后才是答案；160 个 token 的上限截断了它，真实使用时让它想完。多轮对话就是把历史按模板拼起来再 prefill 一次——第 10 章说过，推理引擎用前缀缓存把这一步的重复计算省掉。

!!! interview "怎么讲清楚"
    讲聊天模板，核心一句话：对话结构是用特殊 token 编码进序列的，模型是在预训练之后被训练成"看到 `<|im_start|>assistant` 就开始回答、答完吐 `<|im_end|>`"的。思考模式也只是模板的一部分：`<think>` 开头让模型接着想；要关掉，就替它写一个空的思考块。再提一句工程上的坑：模板不对（多一个换行、少一个空格）模型照样能答，但质量会掉，所以要逐 token 和官方模板对。

## 小结

- [x] 模板 = 每条消息 `<|im_start|>角色\n内容<|im_end|>\n` + assistant 开头；开/关思考只差一个空的 `<think>` 块。
- [x] 手写模板和官方 Jinja 模板在四种情况下逐 token 一致。
- [x] `Chat.ask`：流式打印（整段重解码避免半个汉字）、思考模式、多轮。

**这一章新建的文件**：`qwen/chat.py`。
