from collections import deque
from typing import List, Optional
from response_source import ResponseSource, ResponseCandidate


SOURCE_WEIGHT = {
    ResponseSource.Intent: 1.15,
    ResponseSource.Code: 1.12,
    ResponseSource.Knowledge: 1.02,
    ResponseSource.Generation: 0.70,
    ResponseSource.Fallback: 0.10,
}

ACCEPT_THRESHOLD = 0.30
MAX_RECENT = 4


class FrontalLobe:
    def __init__(self):
        self._recent: deque = deque(maxlen=MAX_RECENT)
        self.last_source: ResponseSource = ResponseSource.Fallback

    @property
    def last_source_label(self) -> str:
        return {
            ResponseSource.Intent: "reacción",
            ResponseSource.Code: "código (NeuralCode)",
            ResponseSource.Knowledge: "conocimiento (corpus)",
            ResponseSource.Generation: "generación (red)",
            ResponseSource.Fallback: "reserva",
        }.get(self.last_source, "reserva")

    def decide(self, candidates: List[ResponseCandidate]) -> str:
        best: Optional[ResponseCandidate] = None
        best_score = float("-inf")
        fallback: Optional[ResponseCandidate] = None

        for c in candidates:
            if not c.text or not c.text.strip():
                continue
            if c.source == ResponseSource.Fallback:
                fallback = c
            weight = SOURCE_WEIGHT.get(c.source, 1.0)
            score = c.confidence * weight
            if self._recently_said(c.text):
                score *= 0.4
            if score > best_score:
                best_score = score
                best = c

        accept = best_score >= ACCEPT_THRESHOLD
        chosen = best if (accept or fallback is None) else fallback

        if chosen is not None:
            self.last_source = chosen.source
            self._remember(chosen.text)
            return chosen.text
        return ""

    def _recently_said(self, text: str) -> bool:
        return any(r.lower() == text.lower() for r in self._recent)

    def _remember(self, text: str):
        self._recent.append(text)