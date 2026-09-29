from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16
    stop: tuple = ()

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(f"temperature 必须 >= 0，收到 {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p 必须在 (0, 1] 内，收到 {self.top_p}")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f"top_k 必须是 -1 或 >= 1，收到 {self.top_k}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {self.max_tokens}")
        stop = (self.stop,) if isinstance(self.stop, str) else tuple(self.stop)
        object.__setattr__(self, "stop", stop)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    def with_(self, **changes) -> "SamplingParams":
        return replace(self, **changes)
