import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from .base import ITokenizer, TokenizerConfig, TokenizerType


class UnigramTokenizer(ITokenizer):
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self._token_log_probs: Dict[int, double] = {}
        self._max_seed_vocab = self._config.max_unigram_seed_vocab
        self._add_special_if_needed()

    @property
    def type(self) -> TokenizerType:
        return TokenizerType.Unigram

    @property
    def name(self) -> str:
        return "Unigram Tokenizer"

    def train(self, corpus: List[str], vocab_size: int):
        raw_text = " ".join(corpus)
        self._build_seed_vocab(raw_text, max(self._max_seed_vocab, vocab_size * 10))
        corpus_list = list(corpus)

        while len(self._vocab) > vocab_size:
            self._e_step(corpus_list)
            removed = self._m_step(vocab_size)
            if removed <= 0:
                break

        self._e_step(corpus_list)
        total_prob = sum(math.exp(lp) for lp in self._token_log_probs.values())
        if total_prob > 0:
            for tid in list(self._token_log_probs.keys()):
                self._token_log_probs[tid] = math.log(math.exp(self._token_log_probs[tid]) / total_prob)

    def _build_seed_vocab(self, text: str, max_tokens: int):
        freq: Dict[str, int] = {}
        max_len = min(15, len(text))
        for length in range(1, max_len + 1):
            for i in range(len(text) - length + 1):
                sub = text[i:i + length]
                freq[sub] = freq.get(sub, 0) + 1

        sorted_items = sorted(freq.items(), key=lambda x: -x[1])[:max_tokens]
        total = sum(v for _, v in sorted_items)

        for token, count in sorted_items:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._token_log_probs[self._next_id] = math.log(count / total) if total > 0 else 0.0
                self._next_id += 1

    def _e_step(self, corpus: List[str]):
        expected: Dict[int, float] = {tid: 0.0 for tid in self._vocab.values()}
        for text in corpus:
            path, _ = self._viterbi(text)
            for tid in path:
                if tid in expected:
                    expected[tid] += 1.0

        total = sum(expected.values())
        if total > 0:
            for tid in list(expected.keys()):
                self._token_log_probs[tid] = math.log(expected[tid] / total)

    def _m_step(self, target_vocab_size: int) -> int:
        sorted_ids = sorted(self._token_log_probs.keys(), key=lambda tid: -self._token_log_probs[tid])
        to_remove = len(self._vocab) - target_vocab_size
        if to_remove <= 0:
            return 0

        removed = 0
        for tid in reversed(sorted_ids):
            if removed >= to_remove:
                break
            if tid not in self._special_tokens:
                token = self._id_to_token[tid]
                del self._vocab[token]
                del self._id_to_token[tid]
                del self._token_log_probs[tid]
                removed += 1
        return removed

    def encode(self, text: str) -> List[int]:
        path, _ = self._viterbi(text)
        return path

    def _viterbi(self, text: str) -> Tuple[List[int], float]:
        n = len(text)
        dp = [float("-inf")] * (n + 1)
        back = [0] * (n + 1)
        dp[0] = 0.0
        unk = self._config.unk_token
        unk_id = self._vocab.get(unk, 0)
        unk_lp = self._token_log_probs.get(unk_id, -10.0)

        for i in range(n):
            if dp[i] == float("-inf"):
                continue
            for j in range(i + 1, min(n, i + 20) + 1):
                sub = text[i:j]
                if sub in self._vocab:
                    tid = self._vocab[sub]
                    lp = dp[i] + self._token_log_probs.get(tid, float("-inf"))
                    if lp > dp[j]:
                        dp[j] = lp
                        back[j] = i
            if dp[i + 1] == float("-inf"):
                dp[i + 1] = dp[i] + unk_lp
                back[i + 1] = i

        path: List[int] = []
        pos = n
        while pos > 0:
            start = back[pos]
            sub = text[start:pos]
            tid = self._vocab.get(sub, unk_id)
            path.append(tid)
            pos = start
        path.reverse()
        return path, dp[n]

    def decode(self, tokens: List[int]) -> str:
        return "".join(self._id_to_token.get(tid, self._config.unk_token) for tid in tokens)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self._next_id}\n")
            for token, tid in self._vocab.items():
                lp = self._token_log_probs.get(tid, 0.0)
                f.write(f"{token}\t{tid}\t{lp:.10f}\n")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self._next_id = int(lines[0])
        self._vocab.clear()
        self._id_to_token.clear()
        self._token_log_probs.clear()
        for line in lines[1:]:
            parts = line.split("\t")
            tid = int(parts[1])
            self._vocab[parts[0]] = tid
            self._id_to_token[tid] = parts[0]
            self._token_log_probs[tid] = float(parts[2])