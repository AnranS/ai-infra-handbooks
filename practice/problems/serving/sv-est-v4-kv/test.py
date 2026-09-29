from checker import check, check_close
from solution import attended, fixed_bytes, per_token_bytes, request_bytes

FLASH = [0, 0] + [4, 128] * 20 + [4]
PRO = [128, 128] + [4, 128] * 29 + [4]


def test_example():
    check(len(FLASH), 43, "V4-Flash 的层数")
    check_close(per_token_bytes(FLASH), 21 * (584 + 132) / 4 + 20 * 584 / 128, what="V4-Flash 每 token 的字节数")
    check(fixed_bytes(FLASH), 43 * 128 * 584, "V4-Flash 每请求的滑窗部分")
    check(attended(4, 1_048_575), 640, "C4 层在 1M 上下文时看的条数")
    check(attended(128, 1_048_575), 8320, "C128 层在 1M 上下文时看的条数")


def test_layer_types():
    check_close(per_token_bytes([0]), 0.0, what="只有滑窗的层不随上下文增长")
    check_close(per_token_bytes([4]), (584 + 132) / 4, what="一层 C4")
    check_close(per_token_bytes([128]), 584 / 128, what="一层 C128")
    check_close(per_token_bytes([4], entry_bytes=1000, index_bytes=68), 1068 / 4, what="换字节数（例如 MXFP4 的索引器 key）")
    check(attended(0, 10), 11, "短上下文、只有滑窗")
    check(attended(0, 5000), 128, "长上下文、只有滑窗")
    check(attended(4, 10), 11 + 2, "11 个 token：滑窗 11 条 + 压好的 2 条")
    check(attended(4, 100_000, topk=1024), 128 + 1024, "C4 层最多看 topk 条")
    check(attended(128, 1000, window=64), 64 + 7, "C128 层全看")


def test_request():
    L = 131_072
    check_close(request_bytes(PRO, L), per_token_bytes(PRO) * L + len(PRO) * 128 * 584, what="V4-Pro 的 128K 请求")
    check_close(request_bytes(FLASH, 50), per_token_bytes(FLASH) * 50 + 43 * 50 * 584, what="短于窗口：滑窗只存实际的 token")
    v32 = 61 * (656 + 132) * L
    ratio = v32 / request_bytes(FLASH, L)
    check(ratio > 10, True, f"128K 时 V3.2 的 KV 是 V4-Flash 的十倍以上（实际 {ratio:.1f} 倍）")
