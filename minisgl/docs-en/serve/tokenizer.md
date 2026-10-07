# The tokenizer and incremental detokenization

<p class="lead">The tokenizer process does two opposite things: turn the text (or conversation) the API server sends into tokens for the scheduler, and turn the new tokens the scheduler sends back into text for the API server. The first is one <code>apply_chat_template</code> and one <code>encode</code>; the second is much harder, because a Chinese character or an emoji can be split across two or three tokens, decoding one alone gives nothing but "&#xfffd;", and decoding is not additive. This chapter implements the tokenizer process, with the focus on incremental detokenization.</p>

!!! question "Self-test: if you can answer these, skip the chapter"
    1. Why not call `tokenizer.decode([token])` on each new token and send that to the frontend?
    2. Why is detokenization "not additive"? Give an example where `decode(a + b) != decode(a) + decode(b)`.
    3. Why can the tokenizer and the detokenizer share one process? When should they be separated?
    4. Should the EOS token be sent to the frontend?

??? success "Answers (try it yourself first, then expand)"
    1. One token may be only part of a character (the UTF-8 bytes of Chinese text or an emoji get split), and decoding it alone gives mojibake; the spaces and merge rules between neighbouring tokens also make piecewise decoding differ from decoding the whole.
    2. With byte-level BPE the 3 UTF-8 bytes of a Chinese character can land in two tokens: decoded separately they give two pieces of mojibake (replacement characters), and only together do they give the right character. Likewise SentencePiece's leading-space marker gets stripped when a token is decoded alone.
    3. Tokenizing and detokenizing are both light CPU work, so one process is enough and it saves a hop between processes; when there are many requests and tokenizing or detokenizing becomes the bottleneck (long prompts, high concurrency), split it into several processes (but the detokenizer's state is per-process, so a split has to route by request).
    4. No: EOS only marks the end of generation, it is not text to display. Stop on it server-side and return it as the `finish_reason`.

**Files you will write**: `tokenizer/tokenize.py`, `tokenizer/detokenize.py`, `tokenizer/server.py`.

## Tokenizing {#分词}

@@code python/minisgl/tokenizer/tokenize.py:TokenizeManager@@

`TokenizeMsg.text` can be plain text (the `/generate` endpoint, or `prompt` in an OpenAI request) or a list of conversation messages (OpenAI's `messages`). The latter is expanded into text by the model's own chat template, with the "the assistant starts answering" prompt added (`add_generation_prompt=True`). Getting the template wrong is one of the most common quality problems in an inference service: the model continues the user's sentence, or never stops (see the [tokenization chapter of the LLM Internals handbook](llm://basics/tokenization/)).

## Incremental detokenization {#增量反分词}

A byte-level BPE vocabulary is built over UTF-8 bytes: an uncommon Chinese character (3 bytes) or an emoji (4 bytes) is very likely split across two tokens. Decoding the first one alone yields incomplete bytes, shown as the replacement character "&#xfffd;". SentencePiece-style tokenizers also decide from context whether to add a leading space, so decoding one by one and concatenating can differ from decoding the whole.

The usual approach, shared by SGLang, vLLM and HF's `TextStreamer`, is to **decode only a recent window and take the difference between two decodes as the new text**:

@@code python/minisgl/tokenizer/detokenize.py:DetokenizeManager.detokenize@@

Each request keeps a few offsets:

- the tokens between `surr_offset` and `read_offset` are already confirmed output, kept to provide context;
- everything after `read_offset` is new and unconfirmed.

Each time it decodes both `[surr_offset, end]` and `[surr_offset, read_offset]`, and the first minus the second is the new text. If the new text is complete (it does not end in "&#xfffd;"), it is confirmed and the window moves forward; otherwise the window stays put and only the certain part is emitted (`find_printable_text`: up to a newline, up to a Chinese character, or up to the last space), waiting for later tokens to complete the character. `sent_offset` records how many characters have gone to the frontend, so each character is sent exactly once.

@@code examples/ch13_tokenizer.py@@

@@output ch13_tokenizer@@

The character 鱻 is split across two tokens, so the first produces no delta and only the second emits the complete 鱻; 推 shares a token with the space before it, and the first token emits only the space it is sure of. Concatenated, the result is identical to decoding the whole, and the frontend never receives a "&#xfffd;".

Other details:

- a group of messages is decoded together with `batch_decode`, which beats one call each;
- when a request has finished (`finished=True`) and its last token is EOS, the EOS is left out of the decode (it would come back as a special string like `<|im_end|>`);
- a finished request's state is deleted, or a long-running service leaks memory (which is exactly what made the stale messages of "problem two" in the last chapter harmful).

## The tokenizer process {#tokenizer-进程}

@@code python/minisgl/tokenizer/server.py:tokenize_worker@@

The main loop: block for one message, then take as many already-arrived messages as it can (up to `local_bs`), and handle them in three groups by type. A `DetokenizeMsg` is detokenized and sent to the API server, a `TokenizeMsg` is tokenized and sent to the scheduler, and an `AbortMsg` becomes an `AbortBackendMsg` forwarded to the scheduler. Several results are packed into one `Batch...Msg` to send.

By default (`--num-tokenizer 0`) there is one process doing both directions: the `TokenizeMsg`s from the API server and the `DetokenizeMsg`s from the scheduler land in one queue. Both are fast, and one process is enough to sustain one GPU's throughput. With many requests and long prompts, `--num-tokenizer N` starts N dedicated tokenizing processes, separate from the detokenizer.

!!! upstream "The official implementation"
    - @@upstream tokenizer/detokenize.py:DetokenizeManager.detokenize@@ (the comments credit SGLang)
    - @@upstream tokenizer/server.py:tokenize_worker@@
    - the official `load_tokenizer` also handles Mistral keeping its chat template in a separate JSON file

## Tests {#测试}

@@code tests/test_ch13_tokenizer.py:test_incremental_detokenize_equals_full_decode@@

!!! interview "Answering in an interview"
    On detokenization: you cannot `decode([token])` each token as it arrives, because detokenization is not additive. A Chinese character or an emoji may be several byte-level tokens (decoded alone they are mojibake), and spacing and merge rules depend on context. Incremental detokenization decodes only a recent window and takes the difference between two decodes as the new text, holding back when the end is an incomplete character (`�`) and emitting only what is certain. A conversation goes through the model's own chat template before encoding, and EOS is never sent to the frontend. The tokenizer and the detokenizer share one process by default (both are light); under load tokenizing can be parallelized across processes, but the detokenizer's state is per-process, so there can be only one.

## Exercises {#练习}

1. When the text does not end in a Chinese character, `find_printable_text` emits only up to the last space. Why? What does that mean for Japanese and Korean?
2. What goes wrong if two detokenizer processes exist and a request's messages can land in either?
3. Add stop-string support to `DetokenizeMsg`: finish when a given string appears in the output. Should the check live in the detokenizer or in the scheduler?

??? success "Answers"
    1. An English word may be split across several tokens, and the next token can change how the earlier part decodes (SentencePiece's space handling, for instance), so cutting at a space is the safest. Japanese kana and Korean hangul fall outside `_is_chinese_char`, so they are treated as "words" and held back until a space or newline, which makes streaming feel worse.
    2. Incremental detokenization's state (`decode_map`) is per-process. Split one request's tokens across two processes and each sees only part of them, so the output is garbled. There can therefore be only one detokenizer process (or routing fixed by uid).
    3. A stop string lives at the text level and has to be checked after detokenization, so the detokenizer is the natural place; but once it matches, the scheduler still has to be told to abort the request and free its resources, which needs an abort message from the detokenizer to the scheduler. SGLang 0.5.20 instead decodes the last few output tokens inside the scheduler process to check stop strings (`Req.check_match_stop_str_prefix` and friends in `schedule_batch.py`).

## Summary {#小结}

- [x] Tokenizing: a conversation goes through the template before encoding, and the template must be the model's own.
- [x] Incremental detokenization decodes only a recent window and takes the difference between two decodes as the new text, waiting when a character is incomplete and emitting only what is certain.
- [x] The tokenizer process handles messages in groups by type, with tokenizing and detokenizing sharing one process by default; the detokenizer's state is per-process, so there can be only one.
