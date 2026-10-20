# （十二）聊天模板与思考模式

<p class="lead">模型只认 token。"这句是用户说的""这句说完了""先想一想再回答"，全靠模板里的几个特殊 token 标出来。这一章手写 Qwen3.5 的聊天模板（我们核对过，和官方的 chat_template.jinja 逐 token 一致），做一个流式打印的 <code>Chat</code>，再打开思考模式看看模型怎么"想"。</p>

!!! abstract "学完这章你能"
    1. 手写 Qwen3.5 的聊天模板，说清每个特殊 token 的作用，以及开、关思考差在哪。
    2. 写出流式打印，知道为什么要整段重新解码。
    3. 说清多轮对话里每一轮送进模型的是什么，以及第二轮为什么接不上第一轮算过的全部 token。

!!! note "这一章跑什么"
    `ch12_chat.py`，约 1 分钟（四次生成，其中思考模式那次最长）。另有一个不用模型的 `ch12_turns.py`（1 秒），看多轮对话能复用多少。

@@files@@

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

## 模板、流式、思考、多轮

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

前几行把模板摊开看：关思考比开思考多 2 个 token——`\n\n</think>\n\n` 那一截替模型把思考块关上了；历史里的 assistant 回合只剩答案"2"，思考过程被去掉了；`<|im_start|>` 这些都是单个 token，id 在词表最末尾。思考模式下模型的输出以 `<think>` 块开头，里面是它的推理过程，`</think>` 之后才是答案；160 个 token 的上限截断了它，真实使用时让它想完。多轮对话就是把历史按模板拼起来再 prefill 一次——第 10 章说过，推理引擎用前缀缓存把这一步的重复计算省掉。

## 多轮对话：第二轮能接上多少

`Chat.ask` 每一轮都把整段对话套上模板、从头 prefill 一遍：模型本身不记得上一轮，"记得"靠的是每次把历史一起送进去。真实的推理引擎会把上一轮算过的 KV 留下（前缀缓存），第二轮只要提示词的开头和上一轮一模一样，那一段就不用重算。那能接上多少？

```python title="ch12_turns.py"
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
```

```text title="输出"
第一轮：提示词 28 个 token，生成 11 个，算过 KV 的一共 39 个
第二轮：提示词 53 个 token，开头和第一轮一样的 24 个，要重新 prefill 29 个
从这里开始不一样，第一轮是 '<|im_start|>assistant\n<think>\n\n</think>\n\n'
                   第二轮是 '<|im_start|>assistant\n北京是中国的首都'
```

第一轮算过 KV 的有 39 个 token，第二轮只接上了前 24 个：断在第一轮的回答开头。第一轮的提示词以 `<|im_start|>assistant\n` 加一个空的思考块结尾，模型在它后面写出回答；到了第二轮，这一回合已经是历史，模板只写 `<|im_start|>assistant\n` 加回答，空的思考块没有了。开头差了这几个 token，后面的回答虽然一字不差，位置也全错开了，只能重算。开思考时也一样：历史回合里的思考过程会被去掉（上面讲模板时说过），缓存同样断在 assistant 回合开头。

对一问一答的聊天，重算一段回答不算什么；对调工具的 agent，每次工具返回都要接着问一轮，提示词里还有很长的工具定义，能不能接上就很要紧。推理系统手册的 [Agent 负载下的推理](serving://topics/agent/) 一章接着讲：真实的引擎怎么复用、哪里会断、怎么调度。

## 常见疑问

**追问时，是把新问题拼到之前的内容后面，重新 prefill，再 decode 吗？**

是的，模型每一轮看到的都是"整段历史 + 新问题"。区别只在引擎：没有前缀缓存时整段重新 prefill（就是这一章的 `Chat`）；有前缀缓存时，开头和上一轮一样的部分直接用缓存，只 prefill 后面不一样的部分，然后照常 decode。

**为什么模板要把历史里的思考过程删掉？**

思考过程往往很长，留在历史里会挤占上下文，而且官方训练时历史回合里就没有思考。删掉的代价是：上一轮的回答在第二轮里换了位置，要重新算一遍。

!!! interview "怎么讲清楚"
    讲聊天模板，核心一句话：对话结构是用特殊 token 编码进序列的，模型是在预训练之后被训练成"看到 `<|im_start|>assistant` 就开始回答、答完吐 `<|im_end|>`"的。思考模式也只是模板的一部分：`<think>` 开头让模型接着想；要关掉，就替它写一个空的思考块。再提一句工程上的坑：模板不对（多一个换行、少一个空格）模型照样能答，但质量会掉，所以要逐 token 和官方模板对。

## 小结

- [x] 模板 = 每条消息 `<|im_start|>角色\n内容<|im_end|>\n` + assistant 开头；开/关思考只差一个空的 `<think>` 块。
- [x] 手写模板核对过和官方 Jinja 模板逐 token 一致；历史里的 assistant 回合只保留答案，思考过程不进下一轮的上下文。
- [x] `Chat.ask`：流式打印（整段重解码避免半个汉字）、思考模式、多轮。
- [x] 多轮对话每轮都送整段历史；模板改写了历史回合（去掉思考块），第二轮只能接上第一轮算过的一部分，断在上一轮回答的开头。

## 思考题

**1. 第二轮的提示词 53 个 token，开头 24 个能接上。如果引擎有前缀缓存，第二轮要 prefill 多少？没有缓存呢？**

??? success "参考答案"
    有前缀缓存：53 − 24 = 29 个（上一轮的回答 11 个要重算，再加上新的追问）。没有缓存：53 个全部重算。轮数越多、历史越长，两者差得越多。

**2. 如果模板在历史回合里也保留那个空的思考块，第二轮能接上多少？**

??? success "参考答案"
    第一轮算过的 39 个 token（提示词 28 + 回答 11，回答以 `<|im_end|>` 结尾）就都和第二轮的开头一样了，只有 `<|im_end|>` 之后的换行和新的追问要 prefill。代价是历史里多了几个没有意义的 token，而且和模型训练时见过的格式不一样。

**这一章新建的文件**：`qwen/chat.py`。
