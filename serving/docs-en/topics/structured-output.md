# Structured output and tool calling

<p class="lead">Agents and tool calling need models to output JSON that strictly follows a format: one extra character or one missing quote and downstream parsing fails. Prompting "please output JSON" is not reliable; inference engines use <b>constrained decoding</b>: at each step, allow only the tokens that keep the output conforming to the grammar. This chapter implements a minimal constrained decoder (a template matcher + a vocabulary prefix tree), compares a real model's output with and without constraints, then discusses performance problems and implementations in inference engines, and the full path of a tool call.</p>

!!! question "Self-test: if you can answer these, skip this chapter"
    1. What does constrained decoding do at each step? Why does it change only sampling and not the model?
    2. With a vocabulary of 150,000 tokens, what goes wrong if every step checks them one by one? How does xgrammar speed this up?
    3. What is jump-forward? What is its hidden risk?
    4. For one tool-calling request, what steps lie between the `tools` parameter and the returned `tool_calls`?

??? success "Answers (try first, then expand to compare)"
    1. Compile the grammar (JSON Schema, regex, EBNF) into an automaton; at each step, compute which tokens are legal from the automaton's current state, set the logits of illegal tokens to negative infinity, sample, then advance the automaton with the chosen token. Only sampling changes, the model is untouched, and the output is guaranteed to follow the grammar.
    2. Checking 150,000 tokens one by one is too slow, and doing it every step slows decode badly. xgrammar precomputes masks for "context-independent" tokens (whose legality depends only on the automaton's state), so at runtime only a few context-dependent tokens need checking, and it overlaps mask computation with the GPU's forward pass.
    3. When the grammar allows only one determined piece of text at some position (such as a JSON key name or fixed punctuation), append that text directly instead of generating it token by token, saving many steps. The hidden risk: the appended text may be tokenized differently from how the model would generate it, producing token sequences the model has never seen, which hurts the quality of later output.
    4. The chat template renders the `tools` definitions into the prompt → the model generates a tool call in the agreed format (its arguments can be constrained with JSON Schema) → a tool parser extracts the tool name and arguments from the generated text → `tool_calls` is returned in OpenAI's format (with incremental parsing when streaming).

<!-- comic ../assets/comics/structured-output.webp is in Chinese; put it back once the English version exists -->

## How it works {#原理}

Represent the target format as an automaton (a regular expression corresponds to a finite automaton; JSON Schema or any context-free grammar to a pushdown automaton). During decoding, maintain the automaton's current state, and at each step:

1. Compute the **set of allowed tokens**: a token is allowed if and only if every character it decodes to is accepted by the automaton in turn;
2. Set the logits of disallowed tokens to −∞ (a bitmask the size of the vocabulary), then sample as usual;
3. Advance the automaton's state with the characters of the chosen token.

The model itself is completely unchanged; there is just a mask before sampling. Because only illegal tokens are blocked, the relative probabilities among legal tokens stay the same, and within what the format allows, the model still chooses by its own preferences.

![Figure: grammar-constrained decoding: the automaton's state determines the allowed token set, and the mask is added to the logits before sampling](../assets/figures/fsm-mask.svg){.aig-svg}

## Implementation {#实现}

A "JSON template" matcher (literals + three kinds of fields: string, integer, enum), plus a character prefix tree built from the whole vocabulary. When computing the allowed set, **walk down the prefix tree in step with the matcher**: as soon as a character is not accepted, the whole subtree can be skipped, which is far faster than checking 150,000 tokens one by one.

```python title="constrained.py"
"""constrained.py —— 语法约束解码的最小实现：一个"JSON 模板"匹配器 + 词表前缀树，每步算出允许的 token 集合。

模板由字面量和带类型的字段组成，例如：
    ['{"city": "', STR, '", "temp_c": ', INT, ', "weather": "', ENUM("晴", "多云", "雨"), '"}']
匹配器的状态是 (第几段, 这一段已经读了什么)。一个 token 允许出现，当且仅当它的每个字符都能被匹配器接受。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    kind: str                      # "str" | "int" | "enum"
    options: tuple = ()
    max_len: int = 16


STR, INT = Field("str"), Field("int", max_len=3)


def ENUM(*options):
    return Field("enum", tuple(options))


class TemplateMatcher:
    def __init__(self, template):
        self.template = template

    def step(self, state, ch):
        """在状态 state 下读入字符 ch，返回新状态；不合法返回 None。"""
        seg, buf = state
        while seg < len(self.template):
            part = self.template[seg]
            if isinstance(part, str):                                  # literal: must match character by character
                if part[len(buf)] == ch:
                    buf += ch
                    return (seg + 1, "") if len(buf) == len(part) else (seg, buf)
                return None
            if part.kind == "str" and ch not in '"\\\n' and len(buf) < part.max_len:
                return seg, buf + ch
            if part.kind == "int" and ch.isdigit() and len(buf) < part.max_len and not (buf == "0"):
                return seg, buf + ch
            if part.kind == "enum" and any(o.startswith(buf + ch) for o in part.options):
                return seg, buf + ch
            if self._field_complete(part, buf):                        # the field may end here: hand over to the next segment
                seg, buf = seg + 1, ""
                continue
            return None
        return None

    @staticmethod
    def _field_complete(field, buf):
        if field.kind == "enum":
            return buf in field.options
        return len(buf) > 0

    def is_final(self, state):
        seg, buf = state
        return seg == len(self.template)

    def forced_text(self, state):
        """当前状态下，接下来"唯一可能"的字面量（可以直接写出，不必让模型生成）。"""
        seg, buf = state
        if seg < len(self.template) and isinstance(self.template[seg], str):
            return self.template[seg][len(buf):]
        return ""


class VocabTrie:
    """把词表中每个 token 解码后的字符串插入一棵字符前缀树，节点上记录"在这里结束的 token"。"""

    def __init__(self, tokenizer, special_ids=()):
        self.root = {}
        for tid in range(len(tokenizer)):
            if tid in special_ids:
                continue
            s = tokenizer.decode([tid])
            if not s or "�" in s:                                 # an incomplete UTF-8 fragment: not allowed in this implementation
                continue
            node = self.root
            for ch in s:
                node = node.setdefault(ch, {})
            node.setdefault(None, []).append(tid)

    def allowed(self, matcher: TemplateMatcher, state):
        """与匹配器同步遍历前缀树：只沿合法的字符往下走，返回 (允许的 token 列表, 访问的节点数)。"""
        result, visited, stack = [], 0, [(self.root, state)]
        while stack:
            node, st = stack.pop()
            visited += 1
            for ch, child in node.items():
                if ch is None:
                    result.extend(child)
                    continue
                nxt = matcher.step(st, ch)
                if nxt is not None:
                    stack.append((child, nxt))
        return result, visited


def advance(matcher, state, text):
    for ch in text:
        state = matcher.step(state, ch)
    return state
```

Ask the model to describe the weather in JSON, first without constraints, then with them:

```python
import json
import time
import torch
from transformers import AutoTokenizer
from mini_llm import KVCache, Transformer, generate
from constrained import ENUM, INT, STR, TemplateMatcher, VocabTrie, advance

torch.set_num_threads(16)
path = "models/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(path)
model = Transformer.from_pretrained(path)
trie = VocabTrie(tok, special_ids=set(tok.all_special_ids))
matcher = TemplateMatcher(['{"city": "', STR, '", "temp_c": ', INT, ', "weather": "', ENUM("晴", "多云", "雨"), '"}'])

messages = [{"role": "user", "content": "用 JSON 描述今天北京的天气，包含城市、摄氏温度和天气。"}]
prompt = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)).input_ids
free = generate(model, torch.tensor([prompt]), 60, eos_token_id=tok.eos_token_id)[0].tolist()
print("不加约束：", repr(tok.decode(free, skip_special_tokens=True)))

cache, state, output, log = KVCache(model.cfg.num_hidden_layers), (0, ""), [], []
with torch.no_grad():
    logits = model(torch.tensor([prompt]), cache)[0, -1]
    while not matcher.is_final(state):
        t0 = time.perf_counter()
        allowed, visited = trie.allowed(matcher, state)
        mask_ms = (time.perf_counter() - t0) * 1e3
        forced = matcher.forced_text(state)
        bias = torch.full_like(logits, float("-inf"))
        bias[allowed] = 0.0
        token = (logits + bias).argmax().item()               # greedy among the allowed tokens
        log.append((len(allowed), visited, mask_ms, bool(forced)))
        output.append(token)
        state = advance(matcher, state, tok.decode([token]))
        logits = model(torch.tensor([[token]]), cache)[0, -1]
text = tok.decode(output)
print("加约束：  ", text, "→ 解析结果", json.loads(text))
print(f"共 {len(output)} 步；允许的 token 数：{[n for n, *_ in log]}")
in_field = [(v, ms) for n, v, ms, forced in log if not forced]
print(f"字段内的步：平均访问前缀树 {sum(v for v, _ in in_field) / len(in_field):.0f} 个节点，"
      f"平均耗时 {sum(ms for _, ms in in_field) / len(in_field):.0f} ms；"
      f"{sum(forced for *_, forced in log)} 步处在字面量中（输出其实是确定的）")
```

```text
不加约束： '```json\n{\n  "city": "北京",\n  "temperature": 25,\n  "weather": "晴"\n}\n```'
加约束：   {"city": "北京", "temp_c": 22, "weather": "晴"} → 解析结果 {'city': '北京', 'temp_c': 22, 'weather': '晴'}
共 20 步；允许的 token 数：[2, 4, 2, 110, 146283, 145716, 2, 4, 2, 2, 1, 28, 29, 27, 2, 3, 2, 2, 3, 2]
字段内的步：平均访问前缀树 76867 个节点，平均耗时 77 ms；13 步处在字面量中（输出其实是确定的）
```

Without constraints, the model wrapped its output in a Markdown code block and changed a field name (writing `temperature` where `temp_c` was required), so a downstream program fails outright; with constraints, the output follows the template exactly and parses directly.

## Performance problems and optimizations {#性能问题与优化}

The statistics above expose two performance problems of constrained decoding:

1. **Computing masks is expensive**: inside a string field almost every token is legal (over 140,000), and the prefix tree's two hundred thousand-odd nodes must all be walked; field steps average seventy to eighty milliseconds, about one forward pass of this small model on CPU. Even in C++, computing over a 150,000-token vocabulary at every step slows each step down.
2. **Many steps are actually determined**: inside a literal (key names and punctuation like `", "temp_c": `) there is only one possible output, yet the forward pass still runs token by token.

The corresponding optimizations:

- **Precomputation and caching** (the core idea of xgrammar): whether most tokens are legal depends only on the automaton's "local state", not on the contents of the pushdown stack (context-independent tokens), so they can be precomputed and cached per state; only the few stack-dependent tokens (for example those that might close a bracket) are checked at runtime. This brings per-step mask computation down to microseconds.
- **Overlapping with GPU compute**: the mask depends only on tokens already generated, so it can be computed on the CPU while the GPU runs the forward pass. vLLM's EngineCore computes the grammar mask (`get_grammar_bitmask`) after calling `execute_model` and before waiting for the result, then hands it to sampling; SGLang's overlap scheduling also has dedicated handling for this.
- **jump-forward**: when the upcoming text is determined (a literal), append it to the output directly and process several tokens in one forward pass instead of generating them one by one. SGLang implemented this optimization first. Its hidden risk is **tokenization mismatch**: the tokens the tokenizer splits the appended string into may differ from the tokens the model would choose "generating step by step on its own", and the model has never seen that split, so later quality may suffer. Implementations usually back off a few characters and re-tokenize so the split at the junction matches normal tokenization.

## The full path of a tool call {#工具调用的完整链路}

Tool calling in the OpenAI interface (the `tools` parameter) is implemented in inference engines like this:

1. **Rendering**: write the `tools` JSON Schema into the prompt through the model's chat template (each model family has its own format; Qwen uses the Hermes style, putting function signatures in the system prompt);
2. **Generation**: the model outputs the call in the format it learned in training; for example, Qwen outputs `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`;
3. **Parsing**: the engine's **tool parser** extracts the calls from the output text and converts them into the OpenAI-format `tool_calls` field (incrementally when streaming); reasoning models also need a **reasoning parser** to separate the thinking content (such as `<think>...</think>`) into `reasoning_content`;
4. **Constraints (optional)**: when `tool_choice` requires calling a specific function, or requires calling some tool, constrained decoding with the function arguments' JSON Schema can guarantee legal arguments; otherwise it relies only on the model's own formatting ability.

So the reliability of tool calling depends on three things: the model's own capability, whether the chat template is correct, and whether the parser matches the model's output format. When launching a new model, all three must be verified.

!!! source "Source code"
    - **vLLM**: `vllm/v1/structured_output/`, where `StructuredOutputManager` manages each request's grammar state, with backends `backend_xgrammar.py` (the default), `backend_guidance.py` (llguidance), `backend_outlines.py` and `backend_lm_format_enforcer.py`; when `get_grammar_bitmask` in EngineCore works together with speculative decoding, draft tokens must also pass grammar validation. Tool parsers are in `vllm/tool_parsers/` (`--enable-auto-tool-choice --tool-call-parser hermes`), and reasoning parsers in `vllm/reasoning/` (`--reasoning-parser`).
    - **SGLang**: `srt/constrained/` has grammar backends such as xgrammar, outlines and llguidance (`--grammar-backend`); tool-call parsing is in `srt/function_call/` (`--tool-call-parser`), and the reasoning parser is set with `--reasoning-parser`.

!!! interview "How to explain it"
    "How is structured output implemented?": **principle** (grammar → automaton; compute the allowed-token mask each step and sample after masking, without changing the relative probabilities of legal tokens) → **difficulties** (the cost of computing masks with a large vocabulary; context-free grammars need pushdown automata) → **optimizations** (xgrammar's precomputation and caching of context-independent tokens, overlapping with the GPU forward pass, jump-forward and its tokenization consistency problem) → **interaction with other features** (speculative drafts must also pass validation; state synchronization under overlap scheduling). Saying "constraints only mask, they don't reweight" and what that means for the output distribution earns bonus points.

## Exercises {#练习}

**1. Do constraints change quality?** Constrained decoding guarantees the format, but can it make the content worse? Give an example.

??? success "Answer"
    Yes. Constraints force the model to choose among legal tokens, but its "best path" may lie outside the legal range in the first place. For example, the model meant to output `"temperature": "25℃"`; constrained to an integer field it can only output a number, but its earlier "plan" was a string with a unit, and once forced to change, what follows may be incoherent. More extreme examples: when the field order differs from what the model is used to, or the Schema demands a format the model is unfamiliar with, quality drops noticeably. The mitigation is to explain the format and give examples in the prompt so the model "already wants" to output this format, with constraints only as a safety net.

**2. Why does this chapter's implementation forbid tokens that decode to "�"?** How must real systems handle them?

??? success "Answer"
    These tokens are only part of a multi-byte UTF-8 character (for example, some rare Chinese characters or emoji split into two byte tokens), so decoding one alone does not yield a complete character, and a "character-level" automaton cannot judge its legality. For simplicity this chapter forbids them outright, which means those characters can never appear in the output. Real systems (xgrammar and others) run the automaton at the **byte level**: each token is treated as a string of bytes and the automaton advances on bytes, which both handles split multi-byte characters and matches byte-level BPE vocabularies naturally.

## Summary {#小结}

- [x] Constrained decoding: grammar → automaton; compute the allowed tokens each step, mask the rest and sample, then advance the automaton with the chosen token.
- [x] Walking the vocabulary prefix tree in step with the automaton visits only legal prefixes; but almost every token is legal inside a field, so per-step computation is still expensive.
- [x] xgrammar precomputes masks for context-independent tokens, and engines overlap mask computation with the GPU forward pass; jump-forward skips determined text, minding tokenization consistency.
- [x] Tool calling = chat template rendering + the model generating in its format + a tool parser extracting the calls, with Schema constraints on arguments when needed.
