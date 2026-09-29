def kv_transfer_ms(prompt_len, kv_bytes_per_token, link_gbps, n_links=1, efficiency=0.8):
    bytes_per_s = link_gbps / 8 * 1e9 * n_links * efficiency
    return prompt_len * kv_bytes_per_token / bytes_per_s * 1e3


def exposed_ms(prefill_ms, transfer_ms, n_layers):
    c, t = prefill_ms / n_layers, transfer_ms / n_layers
    finish = n_layers * c + t if t <= c else c + n_layers * t
    return finish - prefill_ms


def ttft_ms(prefill_ms, transfer_ms, n_layers, layerwise=True):
    extra = exposed_ms(prefill_ms, transfer_ms, n_layers) if layerwise else transfer_ms
    return prefill_ms + extra
