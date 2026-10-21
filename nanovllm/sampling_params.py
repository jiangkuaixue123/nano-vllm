from dataclasses import dataclass
from math import isfinite


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos must be boolean")
