from checker import check, check_close
from solution import attention_tokens, chunks_of, overhead


def test_example_chunks():
    check(chunks_of(5000, 2048), 3, "不满的最后一块也算")
    check(chunks_of(4096, 2048), 2, "正好整除")
    check(chunks_of(1, 2048), 1, "比一块还短")
    check(chunks_of(0, 2048), 0, "空 prompt")


def test_example_attention():
    check(attention_tokens(4, 4), 10, "不分块：因果注意力是 4*5/2")
    check(attention_tokens(4, 2), 10, "分两块，总量不变")
    check(attention_tokens(4, 1), 10, "每块一个 token，总量还是不变")


def test_overhead_is_exactly_one():
    for chunk in (128, 512, 2048, 8192):
        check_close(overhead(8192, chunk), 1.0, rtol=1e-9, what=f"块大小 {chunk}：分块不改变注意力总量")


def test_partial_last_chunk():
    check(attention_tokens(5, 2), 15, "5*6/2")
    check(chunks_of(5, 2), 3, "三块")


def test_large():
    n = 131072
    check(chunks_of(n, 2048), 64, "128K 上下文切成 64 块")
    check(attention_tokens(n, 2048), n * (n + 1) // 2, "总量正好是 n(n+1)/2")
