from checker import check, raises
from solution import num_image_tokens, prefill_tokens, smart_resize


def test_example():
    check(smart_resize(1080, 1920), (728, 1316), "1080p：超过 max_pixels，按比例缩小")
    check(num_image_tokens(1080, 1920), 1222, "1080p 的 token 数")
    check(smart_resize(1080, 1920, max_pixels=28 * 28 * 16384), (1092, 1932), "像素上限足够大时只做取整")


def test_limits():
    check(smart_resize(4000, 3000), (1148, 840), "超过 max_pixels 时按比例缩小")
    check(smart_resize(20, 30), (56, 84), "低于 min_pixels 时放大")
    check(smart_resize(224, 224), (224, 224), "正好是 28 的倍数")
    with raises(ValueError, "宽高比 300"):
        smart_resize(10, 3000)


def test_custom_limits_and_prefill():
    check(num_image_tokens(1080, 1920, max_pixels=28 * 28 * 256), 252, "max_pixels = 256 个 token 的面积")
    check(prefill_tokens(100, [(224, 224), (1080, 1920)]), 100 + 64 + 2 + 1222 + 2, "一段文字加两张图片")
    check(prefill_tokens(5, []), 5, "纯文本")
