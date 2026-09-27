"""C36 — Retrieval-Augmented Knowledge Fabric pipeline.

Pure local retrieval: deterministic lexical scoring over C34's SQLite store.
No LLM/API is required. The design leaves a clean seam for a future semantic
retriever without changing C35's contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .c34 import FabricRecord, KnowledgeFabricLoader


@dataclass(slots=True)
class RAGItem:
    record: FabricRecord
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RAGContext:
    query: str
    items: list[RAGItem] = field(default_factory=list)
    retrieval_method: str = "lexical"

    @property
    def records(self) -> list[FabricRecord]:
        return [x.record for x in self.items]


class RAGPipeline:
    """Deterministic retrieval with field-aware scoring and deduplication."""

    STOPWORDS = {"the", "and", "for", "with", "how", "what", "from", "into", "that", "this", "are", "can", "you", "use", "using"}

    def __init__(self, loader: KnowledgeFabricLoader):
        self.loader = loader

    def retrieve(self, query: str, *, top_k: int = 5,
                 language: str | None = None, concept: str | None = None) -> RAGContext:
        query = (query or "").strip()
        if not query or top_k <= 0:
            return RAGContext(query=query)
        tokens = self._tokens(query)
        if not tokens:
            return RAGContext(query=query)
        candidates = self.loader.get_batch(batch_size=max(1000, top_k * 50), language=language, concept=concept)
        # If the first page is not enough, use the loader's deterministic search to expand candidates.
        candidates.extend(self.loader.search(query, limit=max(top_k * 20, 50), language=language))
        unique: dict[str, FabricRecord] = {r.record_id: r for r in candidates}
        ranked: list[RAGItem] = []
        for record in unique.values():
            score, matched = self._score(record, tokens)
            if score > 0:
                ranked.append(RAGItem(record, score, matched))
        ranked.sort(key=lambda x: (-x.score, x.record.record_id))
        return RAGContext(query=query, items=ranked[:top_k])

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [t for t in re.findall(r"[A-Za-z0-9_+#.-]+", text.lower()) if len(t) >= 2 and t not in cls.STOPWORDS]

    @staticmethod
    def _score(record: FabricRecord, tokens: list[str]) -> tuple[float, list[str]]:
        fields = {
            "concept": (record.concept.lower(), 5.0),
            "topic": (record.topic.lower(), 4.0),
            "question": (record.question.lower(), 3.0),
            "answer": (record.answer.lower(), 2.0),
            "explanation": (record.explanation.lower(), 1.5),
            "tags": (" ".join(record.tags).lower(), 2.5),
            "language": (record.language.lower(), 1.0),
            "framework": (record.framework.lower(), 1.5),
        }
        score = 0.0; matched: list[str] = []
        for token in tokens:
            best = 0.0
            for text, weight in fields.values():
                if token in text:
                    best = max(best, weight)
            if best:
                matched.append(token); score += best
        return score, matched
