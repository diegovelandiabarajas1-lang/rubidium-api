from typing import Dict, List, Optional, Set
from .base import ITokenizer, TokenizerConfig, TokenizerType
from .bpe import BPETokenizer


class DynamicVocabularyTokenizer(ITokenizer):
    def __init__(self, inner: ITokenizer, config: TokenizerConfig):
        super().__init__(config)
        self._inner = inner
        self._growth_interval = config.dynamic_growth_interval
        self._growth_threshold = config.dynamic_growth_threshold
        self._max_vocab_size = config.vocab_size * 2
        self._steps_since_last_growth = 0
        self._oov_tracker: Dict[str, int] = {}

    @property
    def type(self) -> TokenizerType:
        return TokenizerType.DynamicVocabulary

    @property
    def name(self) -> str:
        return f"Dynamic Vocabulary ({self._inner.name})"

    @property
    def vocab_size(self) -> int:
        return self._inner.vocab_size

    def add_special_token(self, token: str):
        self._inner.add_special_token(token)

    def train(self, corpus: List[str], vocab_size: int):
        self._inner.train(corpus, vocab_size)

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for word in text.split():
            word_tokens = self._inner.encode(word)
            has_oov = any(self._inner.is_special_token(t) for t in word_tokens)
            if has_oov:
                lower = word.lower()
                self._oov_tracker[lower] = self._oov_tracker.get(lower, 0) + 1
            tokens.extend(word_tokens)

            if self._inner.vocab_size < self._max_vocab_size:
                self._steps_since_last_growth += 1
                if self._steps_since_last_growth >= self._growth_interval:
                    self._try_grow_vocabulary()
        return tokens

    def _try_grow_vocabulary(self):
        self._steps_since_last_growth = 0
        total = sum(self._oov_tracker.values())
        if total == 0:
            return

        candidates = sorted(
            [(k, v) for k, v in self._oov_tracker.items() if v / total >= self._growth_threshold],
            key=lambda x: -x[1]
        )

        for token, _ in candidates:
            if self._inner.vocab_size >= self._max_vocab_size:
                break
            if self._inner.token_to_id(token) is not None:
                continue
            if len(token) <= 1:
                continue
            self._inner.add_special_token(token)
            del self._oov_tracker[token]

        if len(self._oov_tracker) > 10000:
            to_remove = sorted(self._oov_tracker.items(), key=lambda x: x[1])[:len(self._oov_tracker) - 5000]
            for key, _ in to_remove:
                del self._oov_tracker[key]

    def decode(self, tokens: List[int]) -> str:
        return self._inner.decode(tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        return self._inner.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        return self._inner.id_to_token(id)

    def get_vocab(self) -> Dict[str, int]:
        return self._inner.get_vocab()

    def set_vocab(self, vocab: Dict[str, int]):
        self._inner.set_vocab(vocab)

    def is_special_token(self, id: int) -> bool:
        return self._inner.is_special_token(id)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for token, count in self._oov_tracker.items():
                f.write(f"oov\t{token}\t{count}\n")
            f.write(f"growth_steps={self._steps_since_last_growth}\n")
        self._inner.save(path + ".inner")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split("\t")
                if parts[0] == "oov":
                    self._oov_tracker[parts[1]] = int(parts[2])
                elif parts[0].startswith("growth_steps="):
                    self._steps_since_last_growth = int(parts[0].split("=")[1])
        import os
        if os.path.exists(path + ".inner"):
            self._inner.load(path + ".inner")