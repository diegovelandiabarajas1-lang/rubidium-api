import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from .base import ITokenizer, TokenizerConfig, TokenizerType


class SentencePieceTokenizer(ITokenizer):
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self._use_unigram = self._config.type == TokenizerType.SentencePieceUnigram
        self._token_log_probs: Dict[int, double] = {}
        self._merge_priority: Dict[Tuple[int, int], int] = {}
        self._add_special_if_needed()

    @property
    def type(self) -> TokenizerType:
        return self._config.type

    @property
    def name(self) -> str:
        return f"SentencePiece ({'Unigram' if self._use_unigram else 'BPE'})"

    @staticmethod
    def _normalize_to_bytes(text: str) -> str:
        return "".join(chr(b) for b in text.encode("utf-8"))

    def train(self, corpus: List[str], vocab_size: int):
        byte_corpus = [self._normalize_to_bytes(t) for t in corpus]
        if self._use_unigram:
            self._train_unigram(byte_corpus, vocab_size)
        else:
            self._train_bpe(byte_corpus, vocab_size)

    def _train_bpe(self, byte_corpus: List[str], vocab_size: int):
        word_freqs: Dict[Tuple[int, ...], int] = {}
        for text in byte_corpus:
            for word in text.split():
                chars = tuple([ord(c) for c in word] + [256])
                word_freqs[chars] = word_freqs.get(chars, 0) + 1

        seen = set()
        for ids in word_freqs:
            for c in ids:
                seen.add(c)

        for c in sorted(seen):
            token = chr(c) if c <= 255 else "</s>"
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._next_id += 1

        while len(self._vocab) < vocab_size:
            pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for ids, freq in word_freqs.items():
                for i in range(len(ids) - 1):
                    pair_counts[(ids[i], ids[i + 1])] += freq

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            a, b = best_pair
            merged = self._id_to_token.get(a, "?") + self._id_to_token.get(b, "?")
            merged_id = self._next_id
            self._next_id += 1
            self._vocab[merged] = merged_id
            self._id_to_token[merged_id] = merged
            self._merge_priority[(a, b)] = merged_id

            new_word_freqs: Dict[Tuple[int, ...], int] = {}
            for ids, freq in word_freqs.items():
                new_ids: List[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        new_ids.append(merged_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                new_word_freqs[tuple(new_ids)] = freq
            word_freqs = new_word_freqs

    def _train_unigram(self, byte_corpus: List[str], vocab_size: int):
        text = " ".join(byte_corpus)
        max_len = min(15, len(text))
        freq: Dict[str, int] = {}
        for length in range(1, max_len + 1):
            for i in range(len(text) - length + 1):
                sub = text[i:i + length]
                freq[sub] = freq.get(sub, 0) + 1

        sorted_items = sorted(freq.items(), key=lambda x: -x[1])[:max(vocab_size * 10, 10000)]
        total = sum(v for _, v in sorted_items)
        for token, count in sorted_items:
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._token_log_probs[self._next_id] = math.log(count / total) if total > 0 else 0.0
                self._next_id += 1

        corpus_list = byte_corpus
        while len(self._vocab) > vocab_size:
            expected: Dict[int, float] = {tid: 0.0 for tid in self._vocab.values()}
            for t in corpus_list:
                path, _ = self._viterbi(t)
                for tid in path:
                    if tid in expected:
                        expected[tid] += 1.0

            e_total = sum(expected.values())
            if e_total > 0:
                for tid in list(expected.keys()):
                    self._token_log_probs[tid] = math.log(expected[tid] / e_total)

            sorted_ids = sorted(self._token_log_probs.keys(), key=lambda x: -self._token_log_probs[x])
            to_remove = len(self._vocab) - vocab_size
            for tid in reversed(sorted_ids):
                if to_remove <= 0:
                    break
                if tid not in self._special_tokens:
                    del self._vocab[self._id_to_token[tid]]
                    del self._id_to_token[tid]
                    del self._token_log_probs[tid]
                    to_remove -= 1

    def _viterbi(self, text: str) -> Tuple[List[int], float]:
        n = len(text)
        dp = [float("-inf")] * (n + 1)
        back = [0] * (n + 1)
        dp[0] = 0.0
        unk_tok = self._config.unk_token
        unk_id = self._vocab.get(unk_tok, 0)
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
            path.append(self._vocab.get(sub, unk_id))
            pos = start
        path.reverse()
        return path, dp[n]

    def encode(self, text: str) -> List[int]:
        normalized = self._normalize_to_bytes(text)
        if self._use_unigram:
            path, _ = self._viterbi(normalized)
            return path
        else:
            return self._encode_bpe(normalized)

    def _encode_bpe(self, text: str) -> List[int]:
        chars = [ord(c) for c in text] + [256]
        changed = True
        while changed:
            changed = False
            best_priority = float("inf")
            best_pos = -1
            best_pair = None
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                if pair in self._merge_priority:
                    prio = self._merge_priority[pair]
                    if prio < best_priority:
                        best_priority = prio
                        best_pos = i
                        best_pair = pair
            if best_pair is not None:
                merged_id = self._merge_priority[best_pair]
                new_chars: List[int] = []
                i = 0
                while i < len(chars):
                    if i == best_pos:
                        new_chars.append(merged_id)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
                changed = True

        unk_id = self._vocab.get(self._config.unk_token, 0)
        return [c if c in self._id_to_token else unk_id for c in chars]

    def decode(self, tokens: List[int]) -> str:
        raw: List[int] = []
        unk = self._config.unk_token
        for tid in tokens:
            if tid in self._special_tokens:
                continue
            token = self._id_to_token.get(tid, unk)
            for c in token:
                raw.append(ord(c))
        try:
            return bytes(raw).decode("utf-8")
        except Exception:
            return "".join(chr(b) for b in raw)

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