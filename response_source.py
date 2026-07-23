from enum import IntEnum
from typing import NamedTuple


class ResponseSource(IntEnum):
    Intent = 0
    Code = 1
    Knowledge = 2
    Generation = 3
    Fallback = 4


class ResponseCandidate(NamedTuple):
    text: str
    confidence: float
    source: ResponseSource