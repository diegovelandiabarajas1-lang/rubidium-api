import re
from typing import List, Dict, Set, Optional
import numpy as np


STOP_WORDS: Set[str] = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "a", "y", "o", "u",
    "que", "qué", "como", "cómo", "en", "con", "por", "para", "es", "son", "soy", "eres", "ser",
    "estar", "esta", "este", "esto", "yo", "tu", "tú", "te", "se", "lo", "le", "les", "mi", "muy",
    "su", "sus", "hay", "más", "pero", "si", "sí", "no", "ya", "the", "of", "to", "and", "or", "in",
    "on", "is", "are", "it", "this", "that", "me", "uno", "dos", "tres", "tras", "ante", "bajo"
}

_TOKEN_RE = re.compile(r"[\p{L}\p{N}]+(?:['\u2019][\p{L}\p{N}]+)*|[^\s\p{L}\p{N}]", re.UNICODE)


class WordEmbeddings:
    def __init__(self):
        self.dimension: int = 0
        self._vectors: Dict[str, np.ndarray] = {}

    def try_embed(self, word: str) -> Optional[np.ndarray]:
        return self._vectors.get(word.lower())

    def add_embedding(self, word: str, vector: np.ndarray):
        self._vectors[word.lower()] = vector
        self.dimension = len(vector)


def content_tokens(text: str) -> Set[str]:
    tokens = set()
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0).lower()
        if len(t) > 2 and t not in STOP_WORDS and t[0].isalpha():
            tokens.add(t)
    return tokens


def content_sequence(text: str) -> List[str]:
    result = []
    for m in _TOKEN_RE.finditer(text):
        t = m.group(0).lower()
        if len(t) > 2 and t not in STOP_WORDS and t[0].isalpha():
            result.append(t)
    return result


def sentence_vector(tokens: Set[str], embeddings: Optional[WordEmbeddings]) -> Optional[np.ndarray]:
    if embeddings is None or embeddings.dimension == 0:
        return None
    acc = np.zeros(embeddings.dimension, dtype=np.float64)
    n = 0
    for t in tokens:
        v = embeddings.try_embed(t)
        if v is not None:
            acc += v
            n += 1
    if n == 0:
        return None
    return acc / n


def combined(a_tokens: Set[str], a_vec: Optional[np.ndarray],
             b_tokens: Set[str], b_vec: Optional[np.ndarray]) -> float:
    jac = jaccard(a_tokens, b_tokens)
    if a_vec is None or b_vec is None:
        return jac
    cos = max(0.0, cosine(a_vec, b_vec))
    return 0.55 * jac + 0.45 * cos


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = sum(1 for x in a if x in b)
    union = len(a) + len(b) - inter
    return inter / union if union > 0 else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    denom = na * nb
    return dot / denom if denom > 1e-12 else 0.0