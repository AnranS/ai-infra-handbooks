from checker import check
from solution import kv_tokens_per_group, max_concurrent_groups, prefill_tokens


def test_example():
    check(prefill_tokens([1000, 500], 8, share=True), 1500, "共享：每组提示词只算一次")
    check(prefill_tokens([1000, 500], 8, share=False), 12000, "不共享")
    check(kv_tokens_per_group(1000, 8, 512, share=False, block_size=16), 8 * 1520, "不共享的 KV")
    check(kv_tokens_per_group(1000, 8, 512, share=True, block_size=16), 992 + 8 * 528, "共享的 KV")


def test_block_boundaries():
    check(kv_tokens_per_group(32, 4, 16, True, 16), 32 + 4 * 16, "提示词正好装满块")
    check(kv_tokens_per_group(33, 4, 16, True, 16), 32 + 4 * 32, "多出 1 个 token")
    check(kv_tokens_per_group(10, 1, 5, True, 16), kv_tokens_per_group(10, 1, 5, False, 16), "n=1 时共享不省")


def test_concurrency():
    prompts = [2000] * 100
    a = max_concurrent_groups(prompts, 16, 1024, 1_000_000, False, 16)
    b = max_concurrent_groups(prompts, 16, 1024, 1_000_000, True, 16)
    check((a, b), (20, 54), "100 万 token 的 KV 预算")
    check(max_concurrent_groups([10, 10**6, 10], 2, 10, 1000, True, 16), 1, "放不下的组会挡住后面的组")
