import math
from collections.abc import Sequence


class ArithSeq(Sequence):
    def __init__(self, start, stop, step=1):
        self.start, self.stop, self.step = start, stop, step

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError
