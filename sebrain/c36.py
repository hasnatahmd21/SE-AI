"""C36 — Deterministic local Retrieval-Augmented Knowledge Fabric pipeline.

C36 is the retrieval layer between C34's persistent Knowledge Fabric store and
C35's Brain/Dataset Bridge. Retrieval is local and deterministic: no external
LLM, vector service, embedding API or paid API is required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .c34 import FabricRecord, KnowledgeFabricLoader


@dataclass(slots=True)
class RAGItem:
    record: FabricRecord
    score: float
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record.record_id,
            "dataset_id": self.record.dataset_id,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
            "source_file": self.record.source_file,
            "source_line": self.record.source_line,
            "content_hash": self.record.content_hash,
        }


@dataclass(slots=True)
class RAGContext:
    query: str
    items: list[RAGItem] = field(default_factory=list)
    retrieval_method: str = "lexical"

    @property
    def records(self) -> list[FabricRecord]:
        return [x.record for x in self.items]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "retrieval_method": self.retrieval_method,
            "items": [item.to_dict() for item in self.items],
        }


class RAGPipeline:
    """Deterministic field-aware retrieval with explicit provenance trace."""

    STOPWORDS = {
        "the", "and", "for", "with", "how", "what", "from", "into", "that",
        "this", "are", "can", "you", "use", "using", "why", "when", "where",
        "which",
    }

    def __init__(self, loader: KnowledgeFabricLoader):
        self.loader = loader

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        language: str | None = None,
        dataset_id: str | None = None,
        concept: str | None = None,
    ) -> RAGContext:
        query = (query or "").strip()
        if not query or top_k <= 0:
            return RAGContext(query=query)
        tokens = self._tokens(query)
        if not tokens:
            return RAGContext(query=query)

        # Search the whole indexed fabric rather than sampling only the first
        # record page. C34 performs the broad candidate lookup; C36 re-ranks it.
        candidate_limit = max(200, top_k * 50)
        candidates = self.loader.search(
            query,
            limit=candidate_limit,
            language=language,
            dataset_id=dataset_id,
            concept=concept,
        )
        unique = {record.record_id: record for record in candidates}

        ranked: list[RAGItem] = []
        for record in unique.values():
            score, matched = self._score(record, tokens)
            if score > 0:
                ranked.append(RAGItem(record=record, score=score, matched_terms=matched))

        ranked.sort(key=lambda item: (-item.score, item.record.record_id))
        return RAGContext(
            query=query,
            items=ranked[:top_k],
            retrieval_method="lexical",
        )

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[A-Za-z0-9_+#.-]+", text.lower())
            if len(token) >= 2 and token not in cls.STOPWORDS
        ]

    @staticmethod
    def _score(
        record: FabricRecord, tokens: list[str]
    ) -> tuple[float, list[str]]:
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
        score = 0.0
        matched: list[str] = []
        for token in tokens:
            best = 0.0
            for field_text, weight in fields.values():
                if token in field_text:
                    best = max(best, weight)
            if best:
                matched.append(token)
                score += best
        return score, matched
