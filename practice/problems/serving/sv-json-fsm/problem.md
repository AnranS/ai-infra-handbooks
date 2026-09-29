---
title: 结构化输出：用有限状态机约束解码
chapter: topics/structured-output.md
difficulty: 困难
tags: [结构化输出, 有限状态机, token 掩码]
---
约束解码的核心：把输出格式写成有限状态机，每一步只允许那些"读完之后状态机仍然合法"的 token。这里的格式是**整数数组**（JSON 的一个子集）：

```text
array   := "[" ( int ( ", " int )* )? "]"
int     := "-"? ( "0" | [1-9][0-9]* )
```

逗号后面**恰好一个空格**，其他地方没有空白。例如 `[]`、`[0]`、`[12, -3, 0]` 合法，`[01]`、`[1,2]`、`[-]`、`[1, ]` 不合法。实现：

1. 状态机 `IntArrayFSM`：`start()` 返回初始状态；`step(state, ch)` 返回读入字符 `ch` 后的新状态，不合法时返回 `None`；`is_final(state)`：是否已经读完一个完整的数组。状态用什么表示由你决定（可哈希即可）；
2. `accepts(fsm, text)`：整个字符串是否合法；
3. `allowed_tokens(fsm, state, vocab, eos_id)`：`vocab` 是 token 字符串列表（一个 token 可能包含多个字符），`eos_id` 是结束符的下标（它的字符串是空串）。
   返回允许的 token 下标集合：非空 token 要求依次读入它的每个字符都合法；结束符只在 `is_final(state)` 时允许；
4. `constrained_greedy(fsm, vocab, eos_id, score_fn, max_steps)`：每一步调用 `score_fn(text_so_far)` 得到每个 token 的分数（列表），只在允许的 token 里取分数最高的（同分取下标小的），直到选中结束符或达到 `max_steps`。返回生成的文本。

<!-- 题解 -->
状态可以用字符串标签：`"start"`（等 `[`）、`"open"`（刚读完 `[`，可以是 `]`、`-`、数字）、`"neg"`（读了 `-`）、`"zero"`（数字是单独的 0）、`"digits"`（非零开头的数字中）、`"comma"`（读了 `,`，必须接空格）、`"space"`（读了 `, `，必须接数字或 `-`）、`"done"`（读完 `]`）。

`allowed_tokens` 对每个 token 模拟一遍，复杂度是词表总字符数；真实系统（Outlines、XGrammar）会预先为每个状态算好允许的 token 集合，或者用字节级的下推自动机处理完整的 JSON 文法。
