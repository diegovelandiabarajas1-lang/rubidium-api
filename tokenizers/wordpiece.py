import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from .base import ITokenizer, TokenizerConfig, TokenizerType


WORD_PREFIX = "##"


class WordPieceTokenizer(ITokenizer):
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self._add_special_if_needed()

    @property
    def type(self) -> TokenizerType:
        return TokenizerType.WordPiece

    @property
    def name(self) -> str:
        return "WordPiece Tokenizer"

    def train(self, corpus: List[str], vocab_size: int):
        word_counts: Dict[str, int] = {}
        for text in corpus:
            for word in text.split():
                lower = word.lower()
                word_counts[lower] = word_counts.get(lower, 0) + 1

        for word in word_counts:
            for c in word:
                if c not in self._vocab:
                    self._vocab[c] = self._next_id
                    self._id_to_token[self._next_id] = c
                    self._next_id += 1

        total_count = sum(word_counts.values())

        while len(self._vocab) < vocab_size:
            pair_scores: Dict[Tuple[str, str], float] = defaultdict(float)

            for word, freq in word_counts.items():
                tokens = self._tokenize_word(word)
                if len(tokens) < 2:
                    continue
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair[0] in self._vocab and pair[1] in self._vocab:
                        pXY = freq / total_count if total_count > 0 else 1.0
                        score = pXY / (1.0 * 1.0 + 1e-10)
                        pair_scores[pair] += math.log(score + 1e-10) * freq

            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            token_a, token_b = best_pair

            if not token_a.startswith(WORD_PREFIX):
                merged = token_a + WORD_PREFIX + token_b
            else:
                merged = token_a + token_b[len(WORD_PREFIX):] if token_b.startswith(WORD_PREFIX) else token_a + token_b

            if merged not in self._vocab:
                self._vocab[merged] = self._next_id
                self._id_to_token[self._next_id] = merged
                self._next_id += 1

    def _tokenize_word(self, word: str) -> List[str]:
        tokens: List[str] = []
        pos = 0
        unk = self._config.unk_token
        while pos < len(word):
            longest_match = 0
            best_token = ""
            if pos == 0:
                for length in range(1, len(word) - pos + 1):
                    sub = word[pos:pos + length]
                    if sub in self._vocab and length > longest_match:
                        longest_match = length
                        best_token = sub
            else:
                for length in range(1, len(word) - pos + 1):
                    sub = WORD_PREFIX + word[pos:pos + length]
                    if sub in self._vocab and length > longest_match:
                        longest_match = length
                        best_token = sub
            if longest_match > 0:
                tokens.append(best_token)
                pos += longest_match
            else:
                tokens.append(unk)
                pos += 1
        return tokens

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        unk = self._config.unk_token
        unk_id = self._vocab.get(unk, 0)
        for word in text.split():
            word_tokens = self._tokenize_word(word.lower())
            for token in word_tokens:
                tokens.append(self._vocab.get(token, unk_id))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        result: List[str] = []
        unk = self._config.unk_token
        for tid in tokens:
            if tid in self._special_tokens:
                continue
            token = self._id_to_token.get(tid, unk)
            if token.startswith(WORD_PREFIX):
                result.append(token[2:])
            else:
                if result:
                    result.append(" ")
                result.append(token)
        return "".join(result).strip()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self._next_id}\n")
            for token, tid in self._vocab.items():
                f.write(f"{token}\t{tid}\n")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self._next_id = int(lines[0])
        self._vocab.clear()
        self._id_to_token.clear()
        for line in lines[1:]:
            parts = line.split("\t")
            tid = int(parts[1])
            self._vocab[parts[0]] = tid
            self._id_to_token[tid] = parts[0]