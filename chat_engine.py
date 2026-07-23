import random
from typing import List, Optional, Set
from text_similarity import (
    content_tokens, content_sequence, sentence_vector, WordEmbeddings
)
from response_source import ResponseSource, ResponseCandidate
from neural_code import NeuralCode
from knowledge_base import KnowledgeBase
from frontal_lobe import FrontalLobe


GREETINGS = ["hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "hey", "holi"]
FAREWELLS = ["adiós", "adios", "chao", "hasta luego", "nos vemos", "hasta pronto", "bye"]
THANKS = ["gracias", "muchas gracias", "te lo agradezco", "thanks"]

KNOW_MIN_SCORE = 0.22
GEN_COHERENT_MIN_WORDS = 4
GEN_UNIQUE_RATIO = 0.6

FALLBACKS = [
    "Cuéntame un poco más y sigo el hilo.",
    "Interesante… ¿me lo explicas con otras palabras?",
    "Todavía estoy aprendiendo de eso. ¿Me das más contexto?",
    "No estoy seguro de haberte entendido. ¿Puedes reformularlo?",
]


class ChatEngine:
    def __init__(self, model=None):
        self._model = model
        self._embeddings: Optional[WordEmbeddings] = None
        self._neural_code = NeuralCode()
        self._knowledge = KnowledgeBase()
        self._frontal_lobe = FrontalLobe()
        self._last_tokens: Set[str] = set()

        if hasattr(model, "get_embeddings"):
            self._embeddings = model.get_embeddings()
            self._neural_code.use_embeddings(self._embeddings)
            self._knowledge.use_embeddings(self._embeddings)

    @property
    def last_source(self) -> ResponseSource:
        return self._frontal_lobe.last_source

    @property
    def last_source_label(self) -> str:
        return self._frontal_lobe.last_source_label

    def load_knowledge(self, corpus_text: str):
        self._knowledge.load(corpus_text)

    def respond(self, user_message: str) -> str:
        trimmed = user_message.strip()
        if not trimmed:
            return "¿Me decías algo?"

        user_tokens = content_tokens(trimmed)
        match_tokens: Set[str] = user_tokens
        if len(user_tokens) <= 2 and self._last_tokens:
            match_tokens = set(user_tokens)
            match_tokens.update(self._last_tokens)

        user_vec = sentence_vector(match_tokens, self._embeddings)
        candidates: List[ResponseCandidate] = []

        intent = self._match_intent(trimmed.lower())
        if intent is not None:
            candidates.append(ResponseCandidate(intent, 0.9, ResponseSource.Intent))

        code_text, code_score = self._neural_code.answer(match_tokens, user_vec)
        if code_text is not None:
            if self._neural_code.looks_like_code(match_tokens):
                code_score = max(code_score, 0.5)
            candidates.append(ResponseCandidate(code_text, code_score, ResponseSource.Code))

        know_text, know_score = self._knowledge.answer(match_tokens, user_vec)
        if know_text is not None and know_score >= KNOW_MIN_SCORE:
            know_conf = min(0.9, 0.35 + know_score)
            candidates.append(ResponseCandidate(know_text, know_conf, ResponseSource.Knowledge))

        if self._model is not None and hasattr(self._model, "is_trained") and self._model.is_trained:
            gen = self._generate_reply(self._seed_from(trimmed))
            if self._is_coherent(gen):
                candidates.append(ResponseCandidate(gen, 0.46, ResponseSource.Generation))

        candidates.append(ResponseCandidate(self._fallback(), 0.1, ResponseSource.Fallback))

        response = self._frontal_lobe.decide(candidates)

        if len(user_tokens) >= 2:
            self._last_tokens = user_tokens

        return response

    def _generate_reply(self, seed: str) -> str:
        if self._model is None:
            return ""
        try:
            return self._model.generate(seed, max_chars=200, temperature=0.8, top_k=20)
        except Exception:
            return ""

    @staticmethod
    def _seed_from(message: str) -> str:
        seq = content_sequence(message)
        if len(seq) <= 3:
            return " ".join(seq)
        return " ".join(seq[-3:])

    @staticmethod
    def _is_coherent(text: str) -> bool:
        words = [
            w for w in text.lower().split()
            if w and w[0].isalpha()
        ]
        if len(words) < GEN_COHERENT_MIN_WORDS:
            return False
        unique = len(set(words))
        return unique / len(words) >= GEN_UNIQUE_RATIO

    @staticmethod
    def _match_intent(lower: str):
        if _matches_any(lower, GREETINGS):
            return "¡Hola! ¿De qué te gustaría hablar?"
        if _matches_any(lower, FAREWELLS):
            return "¡Hasta luego! Fue un gusto charlar contigo."
        if _matches_any(lower, THANKS):
            return "¡De nada! Estoy aquí para ayudarte."
        return None

    @staticmethod
    def _fallback() -> str:
        return random.choice(FALLBACKS)


def _matches_any(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if text == p or text.startswith(p + " ") or text == p + "!":
            return True
    return False