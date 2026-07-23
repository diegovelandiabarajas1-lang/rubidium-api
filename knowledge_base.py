from typing import List, Optional, Set, Tuple
import numpy as np
from text_similarity import content_tokens, sentence_vector, combined, WordEmbeddings


MAX_PASSAGES = 600
MIN_PASSAGE_WORDS = 8


class _Passage:
    def __init__(self, text: str, tokens: Set[str], vector: Optional[np.ndarray], sentences: List[str]):
        self.text = text
        self.tokens = tokens
        self.vector = vector
        self.sentences = sentences


class KnowledgeBase:
    def __init__(self):
        self._passages: List[_Passage] = []
        self._corpus: str = ""
        self._embeddings: Optional[WordEmbeddings] = None
        self._built: bool = False

    @property
    def passage_count(self) -> int:
        return len(self._passages)

    def load(self, corpus_text: str):
        self._corpus = corpus_text or ""
        self._built = False
        self._passages.clear()

    def use_embeddings(self, embeddings: Optional[WordEmbeddings]):
        self._embeddings = embeddings
        self._built = False
        self._passages.clear()

    def answer(self, query_tokens: Set[str], query_vec: Optional[np.ndarray]) -> Tuple[Optional[str], float]:
        self._ensure_built()
        if not self._passages or not query_tokens:
            return None, 0.0

        best: Optional[_Passage] = None
        best_score = 0.0

        for p in self._passages:
            score = combined(query_tokens, query_vec, p.tokens, p.vector)
            if score > best_score:
                best_score = score
                best = p

        if best is None or best_score < 0.15:
            return None, 0.0

        answer = self._extract_best_sentences(best, query_tokens, query_vec)
        return (answer, best_score) if answer and answer.strip() else (None, 0.0)

    def _ensure_built(self):
        if self._built:
            return
        self._built = True
        self._passages.clear()

        for paragraph in _split_paragraphs(self._corpus):
            if _count_words(paragraph) < MIN_PASSAGE_WORDS:
                continue
            tokens = content_tokens(paragraph)
            if not tokens:
                continue
            vec = sentence_vector(tokens, self._embeddings)
            sentences = _split_sentences(paragraph)
            self._passages.append(_Passage(paragraph, tokens, vec, sentences))
            if len(self._passages) >= MAX_PASSAGES:
                break

    def _extract_best_sentences(self, passage: _Passage, query_tokens: Set[str],
                                query_vec: Optional[np.ndarray]) -> str:
        if not passage.sentences:
            return passage.text

        best_idx = 0
        best_score = -1.0
        for i, s in enumerate(passage.sentences):
            st = content_tokens(s)
            sv = sentence_vector(st, self._embeddings)
            score = combined(query_tokens, query_vec, st, sv)
            if score > best_score:
                best_score = score
                best_idx = i

        answer = passage.sentences[best_idx].strip()
        if _count_words(answer) < 10 and best_idx + 1 < len(passage.sentences):
            answer = answer + " " + passage.sentences[best_idx + 1].strip()

        return answer.strip()


def _split_paragraphs(text: str):
    current: List[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            if current:
                yield " ".join(current)
                current.clear()
            continue
        current.append(line)
    if current:
        yield " ".join(current)


def _split_sentences(paragraph: str) -> List[str]:
    result = []
    cur = []
    for c in paragraph:
        cur.append(c)
        if c in ".!?…":
            s = "".join(cur).strip()
            if s:
                result.append(s)
            cur.clear()
    tail = "".join(cur).strip()
    if tail:
        result.append(tail)
    return result


def _count_words(s: str) -> int:
    return len(s.split())