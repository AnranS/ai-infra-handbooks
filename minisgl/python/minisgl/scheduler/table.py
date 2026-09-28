import torch


class TableManager:
    """管理 page table / token pool 的行：每个运行中的请求占一行。"""

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # token_pool[i, j]：第 i 行请求第 j 个位置的 token id。它和 page_table 形状相同、放在设备上，
        # 这样模型输入可以直接在设备上用下标取出，采样结果也可以直接写回（重叠调度的关键，见第 11 章）。
        # dummy 请求也从这里取输入，所以要初始化成合法的 token id（0）。
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
