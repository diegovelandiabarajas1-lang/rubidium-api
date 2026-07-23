from enum import IntEnum
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field


class TokenizerType(IntEnum):
    BPE = 0
    Unigram = 1
    WordPiece = 2
    SentencePieceBPE = 3
    SentencePieceUnigram = 4
    DynamicVocabulary = 5


@dataclass
class TokenizerConfig:
    type: TokenizerType = TokenizerType.BPE
    vocab_size: int = 1024
    add_special_tokens: bool = True
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    min_frequency: int = 2
    pre_tokenize: bool = True
    pre_tokenize_regex: str = r"\w+|[^\w\s]"
    max_unigram_seed_vocab: int = 30000
    dynamic_growth_interval: int = 1000
    dynamic_growth_threshold: float = 0.05
    rust_acceleration: bool = True
    python_automation: bool = True


class ITokenizer:
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self._vocab: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}
        self._special_tokens: Set[int] = set()
        self._next_id: int = 0
        self._config = config or TokenizerConfig()

    @property
    def type(self) -> TokenizerType:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def add_special_token(self, token: str):
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._special_tokens.add(self._next_id)
            self._next_id += 1

    def train(self, corpus: List[str], vocab_size: int):
        raise NotImplementedError

    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    def token_to_id(self, token: str) -> Optional[int]:
        return self._vocab.get(token)

    def id_to_token(self, id: int) -> Optional[str]:
        return self._id_to_token.get(id)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._vocab)

    def set_vocab(self, vocab: Dict[str, int]):
        self._vocab = dict(vocab)
        self._id_to_token = {v: k for k, v in vocab.items()}
        self._next_id = max(self._vocab.values()) + 1 if self._vocab else 0

    def is_special_token(self, id: int) -> bool:
        return id in self._special_tokens

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def _add_special_if_needed(self):
        config = self._config
        if config.add_special_tokens:
            self.add_special_token(config.pad_token)
            self.add_special_token(config.unk_token)
            self.add_special_token(config.bos_token)
            self.add_special_token(config.eos_token)