---
title: 增量反分词
chapter: serve/tokenizer.md
difficulty: 困难
tags: [反分词, UTF-8, 流式输出]
---
流式输出时，每来一个 token 就要算出"新增的那段文本"。两个难点：

1. **一个字符可能被拆到几个 token 里**（字节级 BPE 把一个汉字拆成 3 个字节 token），只解码前一个 token 会得到替换字符 `"�"`（`�`），必须等后面的 token；
2. **反分词不是逐 token 可加的**：`decode(a + b) != decode(a) + decode(b)`（例如 SentencePiece 的 `▁` 表示空格，但解码时会去掉开头的空格）。

书中（以及 SGLang、vLLM）的做法：维护 `read_offset`（已经确认解码的 token 数）和 `surr_offset`（上下文窗口起点），每次解码 `ids[surr:]` 和 `ids[surr:read]`，两段文本之差就是新增文本；
新增文本以 `"�"` 结尾（或为空）时说明字符还不完整：这次什么也不输出、偏移不动；否则输出它，并推进窗口：`surr = read`、`read = len(ids)`。

实现 `Detokenizer(decode, eos_id)`：

- `step(token, finished=False) -> str`：收到一个新 token，返回这次可以发给前端的新增文本。`finished=True` 且 `token == eos_id` 时，EOS 本身不加入文本；
- 所有 `step` 的返回值拼起来，在最后一个 token（`finished=True`）之后，应该等于 `decode(全部非 EOS 的 token)`；
- 返回的文本里**永远不能**出现 `"�"`（前提是完整解码的结果里没有它）。

测试会用一个模拟的分词器：token 可以是一段文字（`"▁"` 表示空格）或者一个原始字节；`decode` 把它们拼成字节串、按 UTF-8 解码（不完整的字节变成 `"�"`），再把 `"▁"` 换成空格、去掉开头的一个空格。

<!-- 题解 -->
```python
self.ids.append(token)            # EOS 不加
read_str = decode(self.ids[self.surr:])
surr_str = decode(self.ids[self.surr:self.read])
new = read_str[len(surr_str):]
if new and not new.endswith("�"):
    self.surr, self.read = self.read, len(self.ids)
    return new
return ""
```

为什么要保留一小段"上下文"（`surr` 到 `read`）而不是只解码新 token：新 token 单独解码时会丢掉开头的空格（`▁world` → `world`），带着前一个 token 一起解码再相减，空格就保留下来了。
书中的实现还会在不完整时用 `find_printable_text` 先输出能确定的部分（到最后一个空格、换行或汉字为止），进一步降低延迟。
