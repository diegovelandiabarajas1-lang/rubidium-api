import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from .base import ITokenizer, TokenizerConfig, TokenizerType


WORD_END_ID = 256


class BPETokenizer(ITokenizer):
    def __init__(self, config: Optional[TokenizerConfig] = None):
        super().__init__(config)
        self._merge_priority: Dict[Tuple[int, int], int] = {}
        self._pre_tokenize = self._config.pre_tokenize
        self._pre_tokenize_re = re.compile(self._config.pre_tokenize_regex)
        self._add_special_if_needed()

    @property
    def type(self) -> TokenizerType:
        return TokenizerType.BPE

    @property
    def name(self) -> str:
        return f"BPE Tokenizer{' (pretok)' if self._pre_tokenize else ''}"

    def train(self, corpus: List[str], vocab_size: int):
        word_freqs: Dict[str, int] = {}
        byte_freqs: Dict[int, int] = {}

        for text in corpus:
            if self._pre_tokenize:
                pretokens = [m.group(0) for m in self._pre_tokenize_re.finditer(text)]
            else:
                pretokens = text.split()

            for word in pretokens:
                word_freqs[word] = word_freqs.get(word, 0) + 1
                for c in word:
                    b = ord(c)
                    byte_freqs[b] = byte_freqs.get(b, 0) + 1

        for b in sorted(byte_freqs.keys()):
            token = chr(b)
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._id_to_token[self._next_id] = token
                self._next_id += 1

        if "</w>" not in self._vocab:
            self._vocab["</w>"] = self._next_id
            self._id_to_token[self._next_id] = "</w>"
            self._next_id += 1

        word_ids: Dict[Tuple[int, ...], int] = {}
        for word, freq in word_freqs.items():
            ids = tuple([ord(c) for c in word] + [WORD_END_ID])
            word_ids[ids] = freq

        target_vocab = max(vocab_size, self._next_id)

        while self._next_id < target_vocab:
            pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for ids, freq in word_ids.items():
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    pair_counts[pair] += freq

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

            new_word_ids: Dict[Tuple[int, ...], int] = {}
            for ids, freq in word_ids.items():
                new_ids: List[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        new_ids.append(merged_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                new_word_ids[tuple(new_ids)] = freq
            word_ids = new_word_ids

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        if self._pre_tokenize:
            words = [m.group(0) for m in self._pre_tokenize_re.finditer(text)]
        else:
            words = text.split()

        for word in words:
            ids = [ord(c) for c in word] + [WORD_END_ID]
            changed = True
            while changed:
                changed = False
                best_priority = float("inf")
                best_pos = -1
                best_pair = None
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self._merge_priority:
                        prio = self._merge_priority[pair]
                        if prio < best_priority:
                            best_priority = prio
                            best_pos = i
                            best_pair = pair
                if best_pair is not None:
                    merged_id = self._merge_priority[best_pair]
                    new_ids: List[int] = []
                    i = 0
                    while i < len(ids):
                        if i == best_pos:
                            new_ids.append(merged_id)
                            i += 2
                        else:
                            new_ids.append(ids[i])
                            i += 1
                    ids = new_ids
                    changed = True

            unk_token = self._config.unk_token
            unk_id = self._vocab.get(unk_token, 0)
            for tid in ids:
                if tid in self._id_to_token:
                    tokens.append(tid)
                elif tid <= 255:
                    ch = chr(tid)
                    tokens.append(self._vocab.get(ch, unk_id))
                else:
                    tokens.append(unk_id)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        result: List[str] = []
        space = True
        unk = self._config.unk_token
        for tid in tokens:
            if tid in self._special_tokens:
                continue
            token = self._id_to_token.get(tid, unk)
            if token == "</w>":
                space = True
                continue
            if len(token) == 1 and not token[0].isalnum():
                if not space:
                    result.append(" ")
                result.append(token)
                space = False
            else:
                if not space and token[0].isalnum():
                    result.append(" ")
                result.append(token)
                space = False
        return "".join(result).strip()

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self._next_id}\n")
            for token, tid in self._vocab.items():
                f.write(f"{token}\t{tid}\n")
            for (a, b), mid in self._merge_priority.items():
                f.write(f"m\t{a}\t{b}\t{mid}\n")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        self._next_id = int(lines[0])
        self._vocab.clear()
        self._id_to_token.clear()
        self._merge_priority.clear()
        for line in lines[1:]:
            parts = line.split("\t")
            if parts[0] == "m":
                self._merge_priority[(int(parts[1]), int(parts[2]))] = int(parts[3])
            else:
                tid = int(parts[1])
                self._vocab[parts[0]] = tid
                self._id_to_token[tid] = parts[0]