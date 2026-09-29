from dataclasses import dataclass


@dataclass
class PendingReq:
    uid: int
    input_len: int
    output_len: int
    cached_len: int = 0
    chunked: int | None = None


def schedule_prefill(pending, token_budget, reserved_size, available_kv, free_rows):
    batch = []
    for req in pending:                       # 没有准入控制，也没有分块
        batch.append((req.uid, req.cached_len, req.input_len))
    return batch, []
