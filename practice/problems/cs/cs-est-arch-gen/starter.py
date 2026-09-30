def ridge(tflops, gbs):
    return tflops * 1e12 / (gbs * 1e9)


def breakeven_batch(tflops, gbs, weight_bytes=2):
    return ridge(tflops, gbs) / weight_bytes          # 方向搞反了：权重越小，门槛应该越低


def attn_intensity(group_size, kv_bytes=2):
    return 2 * group_size / kv_bytes


def decode_floor_ms(weight_gb, gbs):
    return weight_gb / gbs * 1e3                      # 少了 GB 与 GB/s 的单位换算
