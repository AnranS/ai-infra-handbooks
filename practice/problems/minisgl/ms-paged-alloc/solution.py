import contextlib
import math

import numpy as np


class PageAllocator:
    def __init__(self, num_pages, page_size, max_rows, max_len):
        self.ps = page_size
        self.free_slots = [p * page_size for p in range(num_pages)]
        self.page_table = np.full((max_rows, max_len), -1, dtype=np.int32)
        self._lazy = None

    def allocate_paged(self, reqs):
        ps = self.ps
        info, need = [], 0
        for row, cached, device in reqs:
            f, l = math.ceil(cached / ps), math.ceil(device / ps)
            if l > f:
                info.append((row, f, l))
                need += l - f
        if need > len(self.free_slots):
            raise MemoryError(f"需要 {need} 页，只有 {len(self.free_slots)} 页空闲")
        pages = self.free_slots[:need]
        del self.free_slots[:need]
        tokens = (np.array(pages, dtype=np.int32)[:, None] + np.arange(ps, dtype=np.int32)).ravel()
        off = 0
        for row, f, l in info:
            n = (l - f) * ps
            self.page_table[row, f * ps:l * ps] = tokens[off:off + n]
            off += n

    def out_loc(self, reqs):
        parts = [self.page_table[row, c:d] for row, c, d in reqs]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)

    def free(self, indices):
        heads = [int(x) for x in np.asarray(indices)[::self.ps]]
        if self._lazy is not None:
            self._lazy.append(heads)
        else:
            self.free_slots.extend(heads)

    @contextlib.contextmanager
    def lazy_free(self):
        self._lazy = []
        try:
            yield
        finally:
            for heads in self._lazy:
                self.free_slots.extend(heads)
            self._lazy = None
