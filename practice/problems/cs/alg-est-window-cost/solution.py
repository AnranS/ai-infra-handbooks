def chunks_of(prompt_len, chunk):
    return (prompt_len + chunk - 1) // chunk


def attention_tokens(prompt_len, chunk):
    total = 0
    done = 0                                   # 前面已经处理的 token 数
    while done < prompt_len:
        c = min(chunk, prompt_len - done)
        total += c * done + c * (c + 1) // 2   # 看前面所有块 + 块内的因果注意力
        done += c
    return total


def overhead(prompt_len, chunk):
    return attention_tokens(prompt_len, chunk) / attention_tokens(prompt_len, prompt_len)
