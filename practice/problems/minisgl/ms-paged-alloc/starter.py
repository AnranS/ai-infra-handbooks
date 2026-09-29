import contextlib
import math

import numpy as np


class PageAllocator:
    def __init__(self, num_pages, page_size, max_rows, max_len):
        self.ps = page_size
        self.free_slots = [p * page_size for p in range(num_pages)]
        self.page_table = np.full((max_rows, max_len), -1, dtype=np.int32)

    def allocate_paged(self, reqs):
        pass

    def out_loc(self, reqs):
        pass

    def free(self, indices):
        pass

    @contextlib.contextmanager
    def lazy_free(self):
        yield
