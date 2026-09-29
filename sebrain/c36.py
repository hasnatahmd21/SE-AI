"""C36 — Deterministic local Retrieval-Augmented Knowledge Fabric pipeline.

C36 is the retrieval layer between C34's persistent Knowledge Fabric store and
C35's Brain/Dataset Bridge. Retrieval is local and deterministic: no external
LLM, vector service, embedding API or paid API is required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .c34 import FabricRecord, KnowledgeFabricLoader


@dataclass(slots=True)
class RAGItem:
    record: FabricRecord
    score: float
    matched_terms: list[str] = field(default_factory=list)
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    retrieval_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record.record_id,
            "dataset_id": self.record.dataset_id,
            "score": self.score,
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
            "matched_terms": list(self.matched_terms),
            "retrieval_sources": list(self.retrieval_sources),
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


class _LexicalRAGPipeline:
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

        # Retrieve every lexical candidate before C36 scoring. C34's previous
        # fixed candidate window could discard a genuinely relevant record
        # before field-aware reranking when the fabric grows large. Correctness
        # takes precedence here; C34 already performs the indexed persistence
        # lookup and C36 applies the final top-k bound after scoring.
        candidates = self.loader.search(
            query,
            limit=None,
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

class RAGPipeline(_LexicalRAGPipeline):
    """C36 hybrid layer; lexical retrieval remains the safe fallback."""

    def __init__(self, loader: KnowledgeFabricLoader, *, semantic_encoder: Callable[[Sequence[str]], Any] | None = None, semantic_model_name: str | None = None, semantic_weight: float = 0.45, semantic_top_k: int = 50):
        super().__init__(loader)
        self.semantic_weight = max(0.0, min(1.0, float(semantic_weight)))
        self.semantic_top_k = max(1, int(semantic_top_k))
        self._semantic_encoder = semantic_encoder
        self._semantic_model_name = semantic_model_name
        self._semantic_model = None
        self._semantic_records = None
        self._semantic_matrix = None

    @property
    def semantic_enabled(self) -> bool:
        return self._semantic_encoder is not None or self._semantic_model_name is not None

    def enable_semantic(self, *, model_name: str = "BAAI/bge-small-en-v1.5", semantic_weight: float | None = None, semantic_top_k: int | None = None) -> None:
        self._semantic_model_name = model_name
        self._semantic_encoder = None
        self._semantic_model = None
        self._semantic_records = None
        self._semantic_matrix = None
        if semantic_weight is not None:
            self.semantic_weight = max(0.0, min(1.0, float(semantic_weight)))
        if semantic_top_k is not None:
            self.semantic_top_k = max(1, int(semantic_top_k))

    def disable_semantic(self) -> None:
        self._semantic_encoder = None
        self._semantic_model_name = None
        self._semantic_model = None
        self._semantic_records = None
        self._semantic_matrix = None

    def retrieve(self, query: str, *, top_k: int = 5, language: str | None = None, dataset_id: str | None = None, concept: str | None = None) -> RAGContext:
        if not self.semantic_enabled:
            return super().retrieve(query, top_k=top_k, language=language, dataset_id=dataset_id, concept=concept)
        try:
            lexical = super().retrieve(query, top_k=max(top_k, self.semantic_top_k), language=language, dataset_id=dataset_id, concept=concept)
            semantic = self._semantic_search(query, language=language, dataset_id=dataset_id, concept=concept)
        except Exception:
            return super().retrieve(query, top_k=top_k, language=language, dataset_id=dataset_id, concept=concept)
        lex = {x.record.record_id: x for x in lexical.items}
        sem = {r.record_id: (r, s) for r, s in semantic}
        max_l = max((x.score for x in lexical.items), default=1.0)
        max_s = max((s for _, s in semantic), default=1.0)
        ranked = []
        for rid in set(lex) | set(sem):
            li = lex.get(rid); si = sem.get(rid)
            record = li.record if li else si[0]
            ls = li.score if li else 0.0; ss = si[1] if si else 0.0
            score = (1-self.semantic_weight)*(ls/max_l if ls else 0.0) + self.semantic_weight*(ss/max_s if ss else 0.0)
            item = RAGItem(record=record, score=score, matched_terms=list(li.matched_terms) if li else [])
            item.lexical_score = ls; item.semantic_score = ss
            item.retrieval_sources = (["lexical"] if li else []) + (["semantic"] if si else [])
            ranked.append(item)
        ranked.sort(key=lambda x: (-x.score, -x.semantic_score, -x.lexical_score, x.record.record_id))
        return RAGContext(query=query, items=ranked[:top_k], retrieval_method="hybrid")

    def _semantic_search(self, query: str, *, language: str | None, dataset_id: str | None, concept: str | None):
        import numpy as np
        records = self._get_semantic_records()
        encode = self._get_semantic_encoder()
        if self._semantic_matrix is None:
            self._semantic_matrix = self._normalize(
                np.asarray(
                    encode([self._record_text(r) for r in records]),
                    dtype="float32",
                )
            )
        matrix = self._semantic_matrix
        q = self._normalize(np.asarray(encode([query]), dtype="float32"))[0]
        scored = []
        for similarity, record in zip(matrix @ q, records):
            if language and record.language.lower() != language.lower(): continue
            if dataset_id and record.dataset_id.upper() != dataset_id.upper(): continue
            if concept and record.concept.lower() != concept.lower(): continue
            if float(similarity) > 0: scored.append((record, float(similarity)))
        scored.sort(key=lambda x: (-x[1], x[0].record_id))
        return scored[:self.semantic_top_k]

    def _get_semantic_records(self) -> list[FabricRecord]:
        if self._semantic_records is None:
            records = []
            for batch in self.loader.iterate_batches(batch_size=1000): records.extend(batch)
            self._semantic_records = list({r.record_id: r for r in records}.values())
        return self._semantic_records

    def _get_semantic_encoder(self):
        if self._semantic_encoder is not None: return self._semantic_encoder
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Semantic retrieval requires sentence-transformers") from exc
            self._semantic_model = SentenceTransformer(self._semantic_model_name)
        return lambda texts: self._semantic_model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    @staticmethod
    def _normalize(vectors):
        import numpy as np
        if vectors.ndim != 2: raise ValueError("semantic encoder must return a 2-D matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    @staticmethod
    def _record_text(record: FabricRecord) -> str:
        return "\n".join(part for part in (record.concept, record.topic, record.question, record.answer, record.explanation, record.knowledge_type, " ".join(record.tags), record.language, record.framework) if part)