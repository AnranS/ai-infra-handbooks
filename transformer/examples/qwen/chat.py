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
