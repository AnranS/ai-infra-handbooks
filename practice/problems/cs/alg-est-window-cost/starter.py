def chunks_of(prompt_len, chunk):
    return prompt_len // chunk                 # 整除：最后不满一块的被漏掉了


def attention_tokens(prompt_len, chunk):
    total = 0
    done = 0
    while done < prompt_len:
        c = min(chunk, prompt_len - done)
        total += c * (done + c)                # 块内部按满矩阵算，没有考虑因果掩码
        done += c
    return total


def overhead(prompt_len, chunk):
    return attention_tokens(prompt_len, chunk) / attention_tokens(prompt_len, prompt_len)
