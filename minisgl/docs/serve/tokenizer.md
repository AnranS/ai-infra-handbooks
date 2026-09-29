# Tokenizer 与增量反分词

<p class="lead">tokenizer 进程做两件方向相反的事：把 API Server 发来的文本（或对话）变成 token 送给调度器，把调度器发来的一个个新 token 变回文本送给 API Server。前者一行 <code>apply_chat_template</code> 加一行 <code>encode</code>；后者要难得多——一个汉字或 emoji 可能被拆到两三个 token 里，单独解码只会得到"�"；而且解码不是逐 token 可加的。这一章实现 tokenizer 进程，重点是增量反分词。</p>

!!! question "自测：能答上来就可以跳过本章"
    1. 为什么不能每来一个 token 就 `tokenizer.decode([token])` 发给前端？
    2. 反分词为什么"不可加"？举一个 `decode(a + b) != decode(a) + decode(b)` 的例子。
    3. 为什么 tokenizer 和 detokenizer 可以放在同一个进程里？什么时候应该分开？
    4. EOS token 要不要发给前端？

??? success "自测参考答案（先自己答，再展开对照）"
    1. 一个 token 可能只是一个字符的一部分（中文、emoji 的 UTF-8 字节被拆开），单独解码会得到乱码；而且前后 token 之间的空格、合并规则会让单独解码的结果和整体解码不同。
    2. 字节级 BPE 下，一个汉字的 3 个 UTF-8 字节可能被分到两个 token 里：分别解码得到两段乱码（替换字符），合起来解码才是正确的汉字；类似地，SentencePiece 里的前导空格标记单独解码时会被去掉。
    3. 分词和反分词都是轻量的 CPU 工作，一个进程就够，还省掉一次进程间通信；请求很多、分词或反分词成为瓶颈时（长提示词、高并发），应该拆成多个进程（但反分词的状态在进程内，拆分要按请求路由）。
    4. 不要：EOS 只表示生成结束，不是要显示的文本；在服务端判断停止，并作为 `finish_reason` 返回。

**本章要写的文件**：`tokenizer/tokenize.py`、`tokenizer/detokenize.py`、`tokenizer/server.py`。

## 分词

@@code python/minisgl/tokenizer/tokenize.py:TokenizeManager@@

`TokenizeMsg.text` 可以是纯文本（`/generate` 接口、或者 OpenAI 请求里的 `prompt`），也可以是对话消息列表（OpenAI 的 `messages`）。后者先用模型自带的对话模板展开成一段文本，并加上"助手开始回答"的提示（`add_generation_prompt=True`）。模板用错是推理服务最常见的效果问题之一：模型会续写用户的话、或者停不下来（见[大模型原理手册的分词一章](llm://basics/tokenization/)）。

## 增量反分词

字节级 BPE 的词表建立在 UTF-8 字节上：一个不常见的汉字（3 个字节）或 emoji（4 个字节）很可能被拆到两个 token 里。单独解码前一个 token，只能得到不完整的字节，显示为替换字符"�"。另外，SentencePiece 类的分词器会根据上下文决定是否在开头加空格，所以逐个解码再拼接，与整体解码的结果可能不同。

通用的做法（SGLang、vLLM、HF 的 `TextStreamer` 都类似）是**只解码最近的一个窗口，用两次解码之差得到新增文本**：

@@code python/minisgl/tokenizer/detokenize.py:DetokenizeManager.detokenize@@

每个请求维护几个偏移：

- `surr_offset` 到 `read_offset` 之间是"已经确认输出、用来提供上下文"的 token；
- `read_offset` 之后是还没确认的新 token。

每次解码 `[surr_offset, 末尾]` 和 `[surr_offset, read_offset]` 两段，前者减去后者就是新增的文本。如果新文本完整（不以"�"结尾），就确认它、把窗口往前推；否则暂时不推进，只输出能确定的部分（`find_printable_text`：到换行、到中文字符、或到最后一个空格为止），等后续 token 补全字符。`sent_offset` 记录已经发给前端的字符数，保证每个字符只发一次。

@@code examples/ch13_tokenizer.py@@

@@output ch13_tokenizer@@

"鱻"被拆成两个 token，第一个到达时增量输出为空，第二个到达时才输出完整的"鱻"；"推"与前面的空格合在一个 token 里，第一个 token 只输出了能确定的空格。拼起来与整体解码完全相同，前端从来不会收到"�"。

其他细节：

- 一批消息一起 `batch_decode`，比逐条调用快；
- 请求结束（`finished=True`）且最后一个 token 是 EOS 时，EOS 不加入解码（它会被解码成 `<|im_end|>` 之类的特殊字符串）；
- 请求结束后删除它的状态，否则长时间运行会泄漏内存（这就是上一章"问题二"里过期消息的危害）。

## tokenizer 进程

@@code python/minisgl/tokenizer/server.py:tokenize_worker@@

主循环：阻塞地收一条消息，再把已经到达的消息尽量多收一些（最多 `local_bs` 条），然后按类型分三组处理：`DetokenizeMsg` 反分词后发给 API Server，`TokenizeMsg` 分词后发给调度器，`AbortMsg` 转换成 `AbortBackendMsg` 转给调度器。多条结果打包成一个 `Batch...Msg` 发送。

默认（`--num-tokenizer 0`）只起一个进程，同时负责分词和反分词：API Server 发来的 `TokenizeMsg` 和调度器发来的 `DetokenizeMsg` 进入同一个队列。分词和反分词都很快，一个进程足以支撑一张卡的吞吐。当请求很多、提示词很长时，可以用 `--num-tokenizer N` 起 N 个专门分词的进程，与反分词进程分开。

!!! upstream "官方实现"
    - @@upstream tokenizer/detokenize.py:DetokenizeManager.detokenize@@（注释写明借鉴自 SGLang）
    - @@upstream tokenizer/server.py:tokenize_worker@@
    - 官方的 `load_tokenizer` 还处理了 Mistral 把对话模板放在单独 JSON 文件里的情况

## 测试

@@code tests/test_ch13_tokenizer.py:test_incremental_detokenize_equals_full_decode@@

## 练习

1. `find_printable_text` 在文本以非中文字符结尾时，只输出到最后一个空格。为什么？对日文、韩文有什么影响？
2. 如果两个 detokenizer 进程同时存在（每个请求的消息可能被分到不同的进程），会出什么问题？
3. 给 `DetokenizeMsg` 加上"停止字符串"（stop strings）支持：输出文本中出现指定字符串时结束。应该在 detokenizer 里检查，还是在调度器里检查？

??? success "参考答案"
    1. 一个英文单词可能被拆成几个 token，而且下一个 token 可能改变前面部分的解码结果（比如 SentencePiece 的空格处理），在空格处截断最保险。日文假名、韩文谚文不在 `_is_chinese_char` 的范围内，它们会被当作"单词"一直等到空格或换行才输出，流式体验变差。
    2. 增量反分词的状态（`decode_map`）在进程内。同一个请求的 token 被分到两个进程，每个进程都只看到一部分，输出错乱。所以反分词只能有一个进程（或者按 uid 固定路由）。
    3. 停止字符串是文本层面的，必须在反分词之后检查，所以在 detokenizer 里检查最自然；但检查到之后还要通知调度器中止这个请求、释放资源——需要一条从 detokenizer 到调度器的中止消息。SGLang 0.5.20 就是在调度器进程里解码输出末尾的若干 token 来检查停止字符串的（`schedule_batch.py` 中 `Req` 的 `check_match_stop_str_prefix` 等方法）。

## 小结

- [x] 分词：对话先套模板再编码；模板必须用模型自带的。
- [x] 增量反分词只解码最近的窗口，用两次解码之差得到新文本；字符不完整时等待，只输出确定的部分。
- [x] tokenizer 进程按消息类型分组处理，默认分词和反分词共用一个进程；反分词状态在进程内，只能有一个 detokenizer。
