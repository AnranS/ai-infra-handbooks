from dataclasses import dataclass


@dataclass
class PendingReq:
    uid: int
    input_len: int
    output_len: int
    cached_len: int = 0
    chunked: int | None = None


def schedule_prefill(pending, token_budget, reserved_size, available_kv, free_rows):
    batch, chunked_list, taken = [], [], 0
    budget = token_budget
    for req in pending:
        if budget <= 0:
            break
        if req.chunked is not None:
            start = req.chunked
        else:
            need = req.input_len - req.cached_len + req.output_len
            if free_rows == 0 or need + reserved_size > available_kv:
                break
            free_rows -= 1
            reserved_size += need
            start = req.cached_len
        n = min(budget, req.input_len - start)
        budget -= n
        batch.append((req.uid, start, start + n))
        taken += 1
        if start + n < req.input_len:
            req.chunked = start + n
            chunked_list.append(req)
        else:
            req.chunked = None
    return batch, chunked_list + list(pending[taken:])
