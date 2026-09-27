"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C03 — KNOWLEDGE RETRIEVAL ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01 (`sebrain/c01.py`) and C02 (`sebrain/c02.py`).

Purpose:
    Retrieve knowledge from a structured local store with:
      - exact matching
      - keyword retrieval (real inverted index, BM25-style ranking)
      - metadata filtering
      - language / framework / version filtering
      - confidence filtering (exact + minimum)
      - provenance filtering
      - kind filtering (Knowledge / Evidence / Inference / Assumption / Unknown)
      - tag filtering
      - relevance scoring + ranking
      - provenance preserved on every hit

Design rules honored:
  - NO external LLM. Pure deterministic local retrieval.
  - NO graph DB. Two SQL tables: `knowledge_items` + `knowledge_terms`.
  - Every item carries Provenance (from C02) + Confidence (from C01).
  - Unknown ≠ Assumption ≠ Inference ≠ Evidence ≠ Knowledge — never collapsed.
  - Filter-then-score: SQL filter narrows candidates; BM25-style scoring ranks.
  - Confidence affects ranking (VERIFIED > HIGH > MEDIUM > LOW > ASSUMPTION > UNKNOWN).
  - Transactional writes: insert + index update in one transaction.

Contents:
  1.  Tokenizer (code-aware: snake_case + camelCase + stopwords)
  2.  KnowledgeKind enum (Knowledge / Evidence / Inference / Assumption / Unknown)
  3.  KnowledgeItem dataclass
  4.  RetrievalQuery dataclass
  5.  ScoredItem + RetrievalResult
  6.  Migration 003 (knowledge_items + knowledge_terms + indices)
  7.  KnowledgeStore (add/get/update/archive/delete/find/count/stats/integrity)
  8.  KnowledgeRetriever (retrieve + retrieve_text, BM25-ish scoring)
  9.  __main__ demo + self-tests

Run as script:
    python -m sebrain.c03            # demo
    python -m sebrain.c03 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import json
import math
import re
import sys
import tempfile
import traceback
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from sebrain.c01 import (
    Confidence,
    Config,
    Migration,
    MigrationRunner,
    MIGRATIONS as C01_MIGRATIONS,
    SEBrainApp,
    SQLiteStorage,
    ValidationError,
    execution_scope,
    get_logger,
)
from sebrain.c02 import (
    C02_MIGRATIONS,
    Provenance,
    ProvenanceType,
)


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


# ════════════════════════════════════════════════════════════════════════════
# 1. TOKENIZER
# ════════════════════════════════════════════════════════════════════════════
# Matches words and numbers. Underscores allowed inside identifiers.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+")
# Splits camelCase / PascalCase / UPPER into component words.
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")

# Small, curated stopword set — expanded for code/doc search context.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have he her him his i in is it its
    of on or that the their them they this to was were will with you your we
    our do does did how what when where which who why use used using
    """.split()
)


def _split_identifier(raw: str) -> list[str]:
    """Split `user_id`, `getUserById`, `HTTPServer` into component words.

    Returns lowercase components, deduplicated.
    """
    parts: set[str] = {raw.lower()}
    # snake_case components
    for p in raw.split("_"):
        if p:
            parts.add(p.lower())
    # camelCase / PascalCase components
    for m in _CAMEL_RE.findall(raw):
        parts.add(m.lower())
    return list(parts)


def tokenize(text: str) -> list[str]:
    """Return lowercase tokens (with duplicates preserved).

    Stopwords are NOT removed here — removal happens in `_term_counts`.
    Filters out tokens shorter than 2 characters.
    """
    if not text:
        return []
    result: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        for part in _split_identifier(raw):
            if len(part) >= 2:
                result.append(part)
    return result


def _term_counts(text: str) -> dict[str, int]:
    """Count non-stopword token frequencies in `text`."""
    counts: dict[str, int] = {}
    for t in tokenize(text):
        if t in STOPWORDS:
            continue
        counts[t] = counts.get(t, 0) + 1
    return counts


def query_terms(text: str) -> list[str]:
    """Tokenize a query, dropping stopwords. Preserves first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokenize(text):
        if t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 2. KNOWLEDGE KIND
# ════════════════════════════════════════════════════════════════════════════
class KnowledgeKind(str, Enum):
    """Never collapse these. They mean fundamentally different things.

    KNOWLEDGE  — an established fact/claim about the world or codebase
    EVIDENCE   — measured/observed artifact supporting or refuting a claim
    INFERENCE  — derived by reasoning (not directly observed)
    ASSUMPTION — explicitly stated assumption (not verified)
    UNKNOWN    — explicitly unknown — must not be treated as fact
    """
    KNOWLEDGE = "knowledge"
    EVIDENCE = "evidence"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"


# Confidence ranking for `min_confidence` filtering.
_CONF_RANK: dict[Confidence, int] = {
    Confidence.UNKNOWN: 0,
    Confidence.ASSUMPTION: 1,
    Confidence.LOW: 2,
    Confidence.MEDIUM: 3,
    Confidence.HIGH: 4,
    Confidence.VERIFIED: 5,
}

# Score multiplier applied to BM25 base — verified knowledge outranks guesses.
_CONF_WEIGHT: dict[Confidence, float] = {
    Confidence.VERIFIED: 1.00,
    Confidence.HIGH: 0.90,
    Confidence.MEDIUM: 0.70,
    Confidence.LOW: 0.50,
    Confidence.ASSUMPTION: 0.30,
    Confidence.UNKNOWN: 0.20,
}


# ════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE ITEM
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class KnowledgeItem:
    kind: KnowledgeKind
    title: str
    content: str
    id: str = field(default_factory=_new_id)
    language: str | None = None
    framework: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    related_entity_ids: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    status: str = "active"                    # active | archived
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "title": self.title,
            "content": self.content,
            "language": self.language,
            "framework": self.framework,
            "version": self.version,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "related_entity_ids": list(self.related_entity_ids),
            "provenance": self.provenance.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. RETRIEVAL QUERY
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class RetrievalQuery:
    """Structured retrieval request.

    Filters narrow candidates (SQL WHERE). Text is scored after filtering.
    """
    text: str | None = None
    kind: KnowledgeKind | str | None = None
    language: str | None = None
    framework: str | None = None
    version: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata_equals: dict[str, Any] = field(default_factory=dict)

    confidence: Confidence | str | None = None          # exact match
    min_confidence: Confidence | str | None = None       # >= threshold
    provenance_source: str | None = None
    provenance_type: ProvenanceType | str | None = None

    related_entity_id: str | None = None

    top_k: int = 10
    min_score: float = 0.0


# ════════════════════════════════════════════════════════════════════════════
# 5. SCORED ITEM + RESULT
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class ScoredItem:
    item: KnowledgeItem
    score: float
    matched_terms: list[str] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalResult:
    items: list[ScoredItem]
    total_candidates: int
    notes: str = ""

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


# ════════════════════════════════════════════════════════════════════════════
# 6. MIGRATION 003
# ════════════════════════════════════════════════════════════════════════════
def _migration_003_knowledge(storage: SQLiteStorage) -> None:
    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id                  TEXT PRIMARY KEY,
            kind                TEXT NOT NULL,
            title               TEXT NOT NULL,
            content             TEXT NOT NULL,
            language            TEXT,
            framework           TEXT,
            version             TEXT,
            tags                TEXT NOT NULL DEFAULT '[]',
            metadata            TEXT NOT NULL DEFAULT '{}',
            related_entity_ids  TEXT NOT NULL DEFAULT '[]',
            prov_source         TEXT NOT NULL,
            prov_type           TEXT NOT NULL,
            prov_ref            TEXT,
            prov_conf           TEXT NOT NULL,
            prov_notes          TEXT NOT NULL DEFAULT '',
            status              TEXT NOT NULL DEFAULT 'active',
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL
        );
        """
    )
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_kind   ON knowledge_items(kind);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_lang   ON knowledge_items(language);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_fw     ON knowledge_items(framework);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_ver    ON knowledge_items(version);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_status ON knowledge_items(status);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_conf   ON knowledge_items(prov_conf);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_know_ptype  ON knowledge_items(prov_type);")

    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_terms (
            term    TEXT NOT NULL,
            item_id TEXT NOT NULL,
            field   TEXT NOT NULL,       -- 'title' | 'content' | 'tags'
            tf      INTEGER NOT NULL,
            PRIMARY KEY (term, item_id, field),
            FOREIGN KEY (item_id) REFERENCES knowledge_items(id) ON DELETE CASCADE
        );
        """
    )
    storage.execute("CREATE INDEX IF NOT EXISTS idx_kterms_term ON knowledge_terms(term);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_kterms_item ON knowledge_terms(item_id);")


C03_MIGRATIONS: list[Migration] = [
    Migration(version=3, name="knowledge", up=_migration_003_knowledge),
]

_ALL_MIGRATIONS: list[Migration] = sorted(
    C01_MIGRATIONS + C02_MIGRATIONS + C03_MIGRATIONS, key=lambda m: m.version
)


# ════════════════════════════════════════════════════════════════════════════
# 7. KNOWLEDGE STORE
# ════════════════════════════════════════════════════════════════════════════
class KnowledgeStore:
    """Authoritative structured knowledge store.

    Every item is written with a corresponding inverted index entry per field.
    Deletes cascade to index. Updates reindex transactionally.
    """

    def __init__(self, storage: SQLiteStorage, *, auto_migrate: bool = True) -> None:
        self.storage = storage
        self._initialized = False
        if auto_migrate:
            self.initialize()

    # ---- lifecycle ----
    def initialize(self) -> None:
        if self._initialized:
            return
        MigrationRunner(self.storage, migrations=_ALL_MIGRATIONS).run()
        self._initialized = True

    def health(self) -> dict[str, Any]:
        try:
            active = self._count_active()
            terms = self.storage.query_one(
                "SELECT COUNT(*) AS c FROM knowledge_terms;"
            )
            return {
                "component": "knowledge_store",
                "ok": True,
                "items_active": active,
                "index_terms": int(terms["c"]) if terms else 0,
            }
        except Exception as exc:
            return {"component": "knowledge_store", "ok": False, "error": str(exc)}

    # ══════════════════════════════════════════════════════════════════════
    # WRITE
    # ══════════════════════════════════════════════════════════════════════
    def add(
        self,
        kind: KnowledgeKind | str,
        title: str,
        content: str,
        *,
        language: str | None = None,
        framework: str | None = None,
        version: str | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        related_entity_ids: Iterable[str] | None = None,
        provenance: Provenance | None = None,
    ) -> KnowledgeItem:
        kind_e = self._coerce_kind(kind)
        if not title or not title.strip():
            raise ValidationError("knowledge title must be non-empty")
        if content is None:
            raise ValidationError("knowledge content must be provided")

        item = KnowledgeItem(
            kind=kind_e,
            title=title.strip(),
            content=content,
            language=language,
            framework=framework,
            version=version,
            tags=sorted({str(t) for t in (tags or []) if str(t).strip()}),
            metadata=dict(metadata or {}),
            related_entity_ids=sorted({str(x) for x in (related_entity_ids or [])}),
            provenance=provenance or Provenance(),
        )
        with self.storage.transaction():
            self._insert(item)
            self._index_item(item)
        log.debug("knowledge.added", id=item.id, kind=kind_e.value)
        return item

    def update(
        self,
        item_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        language: str | None = None,
        framework: str | None = None,
        version: str | None = None,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        related_entity_ids: Iterable[str] | None = None,
    ) -> KnowledgeItem:
        current = self.get(item_id)
        if current is None:
            raise ValidationError(f"knowledge item not found: {item_id}")
        if current.status == "archived":
            raise ValidationError("cannot update archived knowledge item")

        new_title = (title.strip() if title else current.title)
        if not new_title:
            raise ValidationError("title must be non-empty")
        new_content = content if content is not None else current.content

        new_meta = dict(current.metadata)
        if metadata is not None:
            new_meta.update(metadata)

        updated = KnowledgeItem(
            id=current.id,
            kind=current.kind,
            title=new_title,
            content=new_content,
            language=language if language is not None else current.language,
            framework=framework if framework is not None else current.framework,
            version=version if version is not None else current.version,
            tags=current.tags if tags is None else sorted({str(t) for t in tags if str(t).strip()}),
            metadata=new_meta,
            related_entity_ids=(
                current.related_entity_ids
                if related_entity_ids is None
                else sorted({str(x) for x in related_entity_ids})
            ),
            provenance=current.provenance,
            status=current.status,
            created_at=current.created_at,
            updated_at=now_iso(),
        )
        with self.storage.transaction():
            self._replace(updated)
            self._index_item(updated)          # reindex always
        return updated

    def archive(self, item_id: str) -> KnowledgeItem:
        current = self.get(item_id)
        if current is None:
            raise ValidationError(f"knowledge item not found: {item_id}")
        if current.status == "archived":
            return current
        archived = KnowledgeItem(
            id=current.id, kind=current.kind, title=current.title,
            content=current.content, language=current.language,
            framework=current.framework, version=current.version,
            tags=current.tags, metadata=current.metadata,
            related_entity_ids=current.related_entity_ids,
            provenance=current.provenance, status="archived",
            created_at=current.created_at, updated_at=now_iso(),
        )
        with self.storage.transaction():
            self._replace(archived)
        return archived

    def delete(self, item_id: str) -> bool:
        row = self.storage.query_one(
            "SELECT id FROM knowledge_items WHERE id=?", (item_id,)
        )
        if row is None:
            return False
        with self.storage.transaction():
            self.storage.execute(
                "DELETE FROM knowledge_items WHERE id=?", (item_id,)
            )
            # FK CASCADE handles knowledge_terms
        return True

    # ══════════════════════════════════════════════════════════════════════
    # READ
    # ══════════════════════════════════════════════════════════════════════
    def get(self, item_id: str) -> KnowledgeItem | None:
        row = self.storage.query_one(
            "SELECT * FROM knowledge_items WHERE id=?", (item_id,)
        )
        return _row_to_item(row) if row else None

    def find(
        self,
        *,
        kind: KnowledgeKind | str | None = None,
        language: str | None = None,
        framework: str | None = None,
        version: str | None = None,
        tags: Iterable[str] | None = None,
        confidence: Confidence | str | None = None,
        min_confidence: Confidence | str | None = None,
        provenance_source: str | None = None,
        provenance_type: ProvenanceType | str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[KnowledgeItem]:
        """Pure filter — no text scoring."""
        q = RetrievalQuery(
            kind=kind, language=language, framework=framework, version=version,
            tags=list(tags or []), confidence=confidence, min_confidence=min_confidence,
            provenance_source=provenance_source, provenance_type=provenance_type,
        )
        ids = self._candidate_ids(q, include_archived=include_archived)
        items: list[KnowledgeItem] = []
        for cid in ids:
            it = self.get(cid)
            if it is not None:
                items.append(it)
        items.sort(key=lambda i: i.created_at, reverse=True)
        if offset:
            items = items[offset:]
        if limit is not None:
            items = items[:limit]
        return items

    def count(
        self, *, include_archived: bool = False,
        kind: KnowledgeKind | str | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) AS c FROM knowledge_items WHERE 1=1"
        params: list[Any] = []
        if not include_archived:
            sql += " AND status='active'"
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(kind).value)
        row = self.storage.query_one(sql + ";", params)
        return int(row["c"]) if row else 0

    def stats(self) -> dict[str, Any]:
        by_kind = self.storage.query(
            "SELECT kind, COUNT(*) AS c FROM knowledge_items "
            "WHERE status='active' GROUP BY kind;"
        )
        by_lang = self.storage.query(
            "SELECT language, COUNT(*) AS c FROM knowledge_items "
            "WHERE status='active' AND language IS NOT NULL GROUP BY language;"
        )
        terms = self.storage.query_one("SELECT COUNT(DISTINCT term) AS c FROM knowledge_terms;")
        return {
            "items_active": self._count_active(),
            "items_archived": self.count(include_archived=True) - self._count_active(),
            "by_kind": {r["kind"]: int(r["c"]) for r in by_kind},
            "by_language": {r["language"]: int(r["c"]) for r in by_lang},
            "distinct_terms": int(terms["c"]) if terms else 0,
        }

    # ══════════════════════════════════════════════════════════════════════
    # INTEGRITY
    # ══════════════════════════════════════════════════════════════════════
    def verify_integrity(self) -> dict[str, Any]:
        """Check for index orphans, bad JSON, invalid kinds."""
        issues: list[dict[str, Any]] = []

        # Orphan index rows (should not happen with FK CASCADE, but verify)
        orphan_terms = self.storage.query(
            """
            SELECT t.item_id FROM knowledge_terms t
            LEFT JOIN knowledge_items i ON t.item_id = i.id
            WHERE i.id IS NULL;
            """
        )
        for r in orphan_terms:
            issues.append({"type": "orphan_index_term", "item_id": r["item_id"]})

        # Bad JSON in tags/metadata/related
        for row in self.storage.query(
            "SELECT id, tags, metadata, related_entity_ids FROM knowledge_items;"
        ):
            for field_name in ("tags", "metadata", "related_entity_ids"):
                try:
                    json.loads(row[field_name] or "null")
                except json.JSONDecodeError as exc:
                    issues.append({
                        "type": "bad_json",
                        "item_id": row["id"],
                        "field": field_name,
                        "error": str(exc),
                    })

        # Invalid kind values
        rows = self.storage.query("SELECT DISTINCT kind FROM knowledge_items;")
        valid_kinds = {k.value for k in KnowledgeKind}
        for r in rows:
            if r["kind"] not in valid_kinds:
                issues.append({"type": "invalid_kind", "kind": r["kind"]})

        return {
            "ok": not issues,
            "issues": issues,
            "issue_count": len(issues),
        }

    # ══════════════════════════════════════════════════════════════════════
    # INTERNALS
    # ══════════════════════════════════════════════════════════════════════
    def _count_active(self) -> int:
        row = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM knowledge_items WHERE status='active';"
        )
        return int(row["c"]) if row else 0

    def _insert(self, item: KnowledgeItem) -> None:
        self.storage.execute(
            """
            INSERT INTO knowledge_items
                (id, kind, title, content, language, framework, version,
                 tags, metadata, related_entity_ids,
                 prov_source, prov_type, prov_ref, prov_conf, prov_notes,
                 status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            _item_to_row(item),
        )

    def _replace(self, item: KnowledgeItem) -> None:
        self.storage.execute(
            """
            UPDATE knowledge_items SET
                kind=?, title=?, content=?, language=?, framework=?, version=?,
                tags=?, metadata=?, related_entity_ids=?,
                prov_source=?, prov_type=?, prov_ref=?, prov_conf=?, prov_notes=?,
                status=?, updated_at=?
            WHERE id=?;
            """,
            (
                item.kind.value, item.title, item.content,
                item.language, item.framework, item.version,
                json.dumps(item.tags, ensure_ascii=False),
                json.dumps(item.metadata, ensure_ascii=False, default=str),
                json.dumps(item.related_entity_ids, ensure_ascii=False),
                item.provenance.source, item.provenance.source_type.value,
                item.provenance.reference, item.provenance.confidence.value,
                item.provenance.notes,
                item.status, item.updated_at,
                item.id,
            ),
        )

    def _index_item(self, item: KnowledgeItem) -> None:
        """Rebuild the inverted-index rows for one item (idempotent)."""
        self.storage.execute(
            "DELETE FROM knowledge_terms WHERE item_id=?;", (item.id,)
        )

        # title
        for term, tf in _term_counts(item.title).items():
            self.storage.execute(
                "INSERT INTO knowledge_terms(term, item_id, field, tf) "
                "VALUES (?, ?, 'title', ?);",
                (term, item.id, tf),
            )
        # content
        for term, tf in _term_counts(item.content).items():
            self.storage.execute(
                "INSERT INTO knowledge_terms(term, item_id, field, tf) "
                "VALUES (?, ?, 'content', ?);",
                (term, item.id, tf),
            )
        # tags (aggregate tf across all tags)
        tag_counts: dict[str, int] = {}
        for tag in item.tags:
            for term in tokenize(str(tag)):
                if term in STOPWORDS:
                    continue
                tag_counts[term] = tag_counts.get(term, 0) + 1
        for term, tf in tag_counts.items():
            self.storage.execute(
                "INSERT INTO knowledge_terms(term, item_id, field, tf) "
                "VALUES (?, ?, 'tags', ?);",
                (term, item.id, tf),
            )

    def _candidate_ids(
        self, q: RetrievalQuery, *, include_archived: bool = False
    ) -> list[str]:
        """SQL filter → list of item ids. No scoring here."""
        sql = "SELECT id FROM knowledge_items WHERE 1=1"
        params: list[Any] = []
        if not include_archived:
            sql += " AND status='active'"
        if q.kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(q.kind).value)
        if q.language is not None:
            sql += " AND language=?"
            params.append(q.language)
        if q.framework is not None:
            sql += " AND framework=?"
            params.append(q.framework)
        if q.version is not None:
            sql += " AND version=?"
            params.append(q.version)
        if q.confidence is not None:
            sql += " AND prov_conf=?"
            params.append(self._coerce_conf(q.confidence).value)
        if q.provenance_source is not None:
            sql += " AND prov_source=?"
            params.append(q.provenance_source)
        if q.provenance_type is not None:
            sql += " AND prov_type=?"
            params.append(self._coerce_prov_type(q.provenance_type).value)
        if q.related_entity_id is not None:
            # related_entity_ids stored as JSON array — LIKE is unsafe; filter in Python.
            pass
        rows = self.storage.query(sql + ";", params)
        ids = [r["id"] for r in rows]

        # Post-filter: tags, metadata, min_confidence, related_entity_id
        if ids and (q.tags or q.metadata_equals or q.min_confidence is not None
                    or q.related_entity_id is not None):
            placeholders = ",".join("?" * len(ids))
            info_rows = self.storage.query(
                f"SELECT id, tags, metadata, prov_conf, related_entity_ids "
                f"FROM knowledge_items WHERE id IN ({placeholders});",
                ids,
            )
            info = {r["id"]: r for r in info_rows}
            want_tags = set(q.tags or [])
            min_rank = (
                _CONF_RANK[self._coerce_conf(q.min_confidence)]
                if q.min_confidence is not None else None
            )

            def keep(iid: str) -> bool:
                row = info.get(iid)
                if row is None:
                    return False
                if want_tags:
                    try:
                        have = set(json.loads(row["tags"] or "[]"))
                    except json.JSONDecodeError:
                        have = set()
                    if not want_tags.issubset(have):
                        return False
                if q.metadata_equals:
                    try:
                        meta = json.loads(row["metadata"] or "{}")
                    except json.JSONDecodeError:
                        meta = {}
                    for k, v in q.metadata_equals.items():
                        if meta.get(k) != v:
                            return False
                if min_rank is not None:
                    try:
                        rank = _CONF_RANK[Confidence(row["prov_conf"])]
                    except (ValueError, KeyError):
                        return False
                    if rank < min_rank:
                        return False
                if q.related_entity_id is not None:
                    try:
                        rels = set(json.loads(row["related_entity_ids"] or "[]"))
                    except json.JSONDecodeError:
                        rels = set()
                    if q.related_entity_id not in rels:
                        return False
                return True

            ids = [i for i in ids if keep(i)]
        return ids

    # ---- coercion helpers ----
    @staticmethod
    def _coerce_kind(kind: KnowledgeKind | str) -> KnowledgeKind:
        if isinstance(kind, KnowledgeKind):
            return kind
        try:
            return KnowledgeKind(kind)
        except ValueError as exc:
            raise ValidationError(f"unknown knowledge kind: {kind!r}") from exc

    @staticmethod
    def _coerce_conf(conf: Confidence | str) -> Confidence:
        if isinstance(conf, Confidence):
            return conf
        try:
            return Confidence(conf)
        except ValueError as exc:
            raise ValidationError(f"unknown confidence: {conf!r}") from exc

    @staticmethod
    def _coerce_prov_type(pt: ProvenanceType | str) -> ProvenanceType:
        if isinstance(pt, ProvenanceType):
            return pt
        try:
            return ProvenanceType(pt)
        except ValueError as exc:
            raise ValidationError(f"unknown provenance type: {pt!r}") from exc


# ════════════════════════════════════════════════════════════════════════════
# 8. RETRIEVER
# ════════════════════════════════════════════════════════════════════════════
_FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "tags": 2.5,
    "content": 1.0,
}
_BM25_K1 = 1.2


class KnowledgeRetriever:
    """BM25-style ranker over `KnowledgeStore`.

    Flow:
      1. SQL filter narrows candidates (filters from RetrievalQuery).
      2. If text provided: tokenize → compute IDF per query term →
         fetch term frequencies for candidates → score → filter min_score.
      3. If no text: return candidates with uniform score 1.0 (filter-only).
      4. Rank by (score desc, created_at desc).
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    # ---- public API ----
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if query.top_k <= 0:
            raise ValidationError("top_k must be >= 1")

        candidates = self.store._candidate_ids(query)
        if not candidates:
            return RetrievalResult(
                items=[], total_candidates=0,
                notes="no candidates after filtering",
            )

        text = (query.text or "").strip()
        if not text:
            return self._filter_only(candidates, query)

        q_terms = query_terms(text)
        if not q_terms:
            return self._filter_only(
                candidates, query,
                notes="query text contained only stopwords; filter-only",
            )

        N = self.store._count_active()
        idf = self._idf(q_terms, N)
        if not idf:
            return RetrievalResult(
                items=[], total_candidates=len(candidates),
                notes="no query term present in the corpus",
            )

        # Fetch term frequencies + doc lengths for candidates
        placeholders_c = ",".join("?" * len(candidates))
        placeholders_t = ",".join("?" * len(idf))
        term_rows = self.store.storage.query(
            f"SELECT item_id, field, term, tf FROM knowledge_terms "
            f"WHERE item_id IN ({placeholders_c}) AND term IN ({placeholders_t});",
            candidates + list(idf.keys()),
        )
        doc_terms: dict[str, dict[tuple[str, str], int]] = {}
        for r in term_rows:
            d = doc_terms.setdefault(r["item_id"], {})
            d[(r["field"], r["term"])] = int(r["tf"])

        len_rows = self.store.storage.query(
            f"SELECT item_id, SUM(tf) AS total FROM knowledge_terms "
            f"WHERE item_id IN ({placeholders_c}) GROUP BY item_id;",
            candidates,
        )
        doc_lens = {r["item_id"]: int(r["total"] or 0) for r in len_rows}
        avgdl = (sum(doc_lens.values()) / len(doc_lens)) if doc_lens else 1.0
        if avgdl <= 0:
            avgdl = 1.0

        scored: list[ScoredItem] = []
        for cid in candidates:
            dterms = doc_terms.get(cid)
            if not dterms:
                continue
            item = self.store.get(cid)
            if item is None:
                continue
            dlen = doc_lens.get(cid, 1)
            conf_w = _CONF_WEIGHT.get(item.provenance.confidence, 0.5)
            score, matched = _score_doc(
                idf, dterms, dlen, avgdl, conf_w,
            )
            if score < query.min_score:
                continue
            matched_fields = sorted({
                field for (field, term) in dterms if term in idf
            })
            scored.append(ScoredItem(
                item=item, score=score,
                matched_terms=matched, matched_fields=matched_fields,
            ))

        # Rank: score desc, then recency desc
        scored.sort(key=lambda s: s.item.created_at, reverse=True)
        scored.sort(key=lambda s: s.score, reverse=True)
        scored = scored[: query.top_k]
        return RetrievalResult(
            items=scored, total_candidates=len(candidates),
            notes=f"query_terms={len(idf)}",
        )

    def retrieve_text(self, text: str, **kwargs: Any) -> RetrievalResult:
        """Convenience: `retrieve(RetrievalQuery(text=text, **kwargs))`."""
        return self.retrieve(RetrievalQuery(text=text, **kwargs))

    # ---- internals ----
    def _filter_only(
        self, candidates: list[str], query: RetrievalQuery, notes: str = ""
    ) -> RetrievalResult:
        items: list[ScoredItem] = []
        for cid in candidates:
            it = self.store.get(cid)
            if it is None:
                continue
            items.append(ScoredItem(item=it, score=1.0))
        items.sort(key=lambda s: s.item.created_at, reverse=True)
        items = items[: query.top_k]
        return RetrievalResult(
            items=items, total_candidates=len(candidates),
            notes=notes or "filter-only",
        )

    def _idf(self, terms: list[str], N: int) -> dict[str, float]:
        """Inverse Document Frequency for query terms present in the corpus."""
        if not terms:
            return {}
        placeholders = ",".join("?" * len(terms))
        rows = self.store.storage.query(
            f"SELECT term, COUNT(DISTINCT item_id) AS df "
            f"FROM knowledge_terms WHERE term IN ({placeholders}) "
            f"GROUP BY term;",
            terms,
        )
        df_map = {r["term"]: int(r["df"]) for r in rows}
        out: dict[str, float] = {}
        for t in terms:
            df = df_map.get(t, 0)
            if df == 0:
                continue
            out[t] = math.log(1.0 + N / (1.0 + df))
        return out


# ---- scoring function (module-level, pure, testable) ----
def _score_doc(
    term_idf: dict[str, float],
    doc_terms: dict[tuple[str, str], int],
    doc_total_tokens: int,
    avgdl: float,
    confidence_weight: float,
) -> tuple[float, list[str]]:
    """Return (score, matched_terms).

    BM25-ish: saturating TF per term, field weighting, length normalization,
    then multiplied by a confidence weight (VERIFIED → 1.0, UNKNOWN → 0.2).
    """
    if not term_idf:
        return (1.0, [])

    score_raw = 0.0
    matched: list[str] = []

    for term, idf in term_idf.items():
        tf_weighted = 0.0
        for (field, t), tf in doc_terms.items():
            if t == term:
                tf_weighted += _FIELD_WEIGHTS.get(field, 1.0) * tf
        if tf_weighted <= 0:
            continue
        sat = (tf_weighted * (_BM25_K1 + 1.0)) / (tf_weighted + _BM25_K1)
        score_raw += idf * sat
        matched.append(term)

    if score_raw <= 0:
        return (0.0, [])

    if avgdl > 0:
        norm = max(0.5, min(doc_total_tokens / avgdl, 2.0))
        score_raw /= norm

    score_raw *= confidence_weight
    return (score_raw, matched)


# ════════════════════════════════════════════════════════════════════════════
# ROW ↔ DATACLASS
# ════════════════════════════════════════════════════════════════════════════
def _item_to_row(item: KnowledgeItem) -> tuple:
    return (
        item.id, item.kind.value, item.title, item.content,
        item.language, item.framework, item.version,
        json.dumps(item.tags, ensure_ascii=False),
        json.dumps(item.metadata, ensure_ascii=False, default=str),
        json.dumps(item.related_entity_ids, ensure_ascii=False),
        item.provenance.source, item.provenance.source_type.value,
        item.provenance.reference, item.provenance.confidence.value,
        item.provenance.notes,
        item.status, item.created_at, item.updated_at,
    )


def _row_to_item(row: dict[str, Any]) -> KnowledgeItem:
    return KnowledgeItem(
        id=row["id"],
        kind=KnowledgeKind(row["kind"]),
        title=row["title"],
        content=row["content"],
        language=row["language"],
        framework=row["framework"],
        version=row["version"],
        tags=json.loads(row["tags"] or "[]"),
        metadata=json.loads(row["metadata"] or "{}"),
        related_entity_ids=json.loads(row["related_entity_ids"] or "[]"),
        provenance=Provenance(
            source=row["prov_source"],
            source_type=ProvenanceType(row["prov_type"]),
            reference=row["prov_ref"],
            confidence=Confidence(row["prov_conf"]),
            notes=row["prov_notes"] or "",
        ),
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ════════════════════════════════════════════════════════════════════════════
# 9. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _run_self_tests() -> int:
    failures: list[str] = []
    passed = 0

    def check(name: str, fn: Callable[[], None]) -> None:
        nonlocal passed
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception:
            failures.append(name)
            print(f"  ✗ {name}")
            traceback.print_exc()

    def fresh() -> tuple[KnowledgeStore, KnowledgeRetriever, tempfile.TemporaryDirectory]:
        td = tempfile.TemporaryDirectory()
        s = SQLiteStorage(Path(td.name) / "kb.sqlite3")
        s.initialize()
        store = KnowledgeStore(s)
        return store, KnowledgeRetriever(store), td

    print("Running C03 self-tests…")

    # ---- tokenizer ----
    def t_tokenizer_basic() -> None:
        assert "hello" in tokenize("Hello World")
        assert "world" in tokenize("Hello World")

    def t_tokenizer_snake() -> None:
        ts = tokenize("get_user_by_id")
        assert "get_user_by_id" in ts
        for p in ("get", "user", "by", "id"):
            assert p in ts, (p, ts)

    def t_tokenizer_camel() -> None:
        ts = tokenize("getUserById")
        for p in ("getuserbyid", "get", "user", "by", "id"):
            assert p in ts, (p, ts)

    def t_tokenizer_upper() -> None:
        ts = tokenize("HTTPServer")
        assert "http" in ts and "server" in ts, ts

    def t_tokenizer_short_dropped() -> None:
        ts = tokenize("a I x yz")
        # "a" and "I" are <2 chars → dropped; "yz" kept
        assert "yz" in ts
        assert "a" not in ts

    def t_query_terms_stopwords() -> None:
        q = query_terms("how do I use FastAPI for REST")
        assert "fastapi" in q
        assert "rest" in q
        assert "how" not in q
        assert "do" not in q

    check("tokenizer basic", t_tokenizer_basic)
    check("tokenizer snake_case", t_tokenizer_snake)
    check("tokenizer camelCase", t_tokenizer_camel)
    check("tokenizer UPPER", t_tokenizer_upper)
    check("tokenizer drops short tokens", t_tokenizer_short_dropped)
    check("query_terms removes stopwords", t_query_terms_stopwords)

    # ---- add / get ----
    def t_add_get() -> None:
        store, _, td = fresh()
        try:
            it = store.add(
                KnowledgeKind.KNOWLEDGE, "FastAPI uses Pydantic",
                "FastAPI integrates with Pydantic for validation.",
                language="python", framework="fastapi",
                tags=["api", "validation"],
            )
            assert it.id
            got = store.get(it.id)
            assert got is not None
            assert got.title == "FastAPI uses Pydantic"
            assert got.language == "python"
            assert "api" in got.tags
        finally:
            td.cleanup()

    def t_add_validates() -> None:
        store, _, td = fresh()
        try:
            try:
                store.add(KnowledgeKind.KNOWLEDGE, "", "x")
            except ValidationError:
                pass
            else:
                raise AssertionError("expected ValidationError")
            try:
                store.add(KnowledgeKind.KNOWLEDGE, "t", None)  # type: ignore[arg-type]
            except ValidationError:
                pass
            else:
                raise AssertionError("expected ValidationError")
            try:
                store.add("not_a_kind", "t", "c")
            except ValidationError:
                pass
            else:
                raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("add + get", t_add_get)
    check("add validates", t_add_validates)

    # ---- update / reindex ----
    def t_update_reindexes() -> None:
        store, retr, td = fresh()
        try:
            # "Old title"/"Brand new title" share the word "title", which
            # would still (correctly) match after reindexing and mask a
            # stale-index bug — use disjoint vocabulary so the assertions
            # actually isolate whether update() drops the old terms.
            it = store.add(KnowledgeKind.KNOWLEDGE, "Zebra Widget", "irrelevant words")
            r1 = retr.retrieve_text("Zebra Widget")
            assert len(r1) == 1
            store.update(it.id, title="Quantum Sprocket", content="completely different")
            r2 = retr.retrieve_text("Zebra Widget")
            assert len(r2) == 0
            r3 = retr.retrieve_text("Quantum Sprocket")
            assert len(r3) == 1
        finally:
            td.cleanup()

    check("update reindexes terms", t_update_reindexes)

    # ---- archive / delete ----
    def t_archive_excludes() -> None:
        store, retr, td = fresh()
        try:
            it = store.add(KnowledgeKind.KNOWLEDGE, "X", "X content here")
            assert len(retr.retrieve_text("X")) == 1
            store.archive(it.id)
            assert len(retr.retrieve_text("X")) == 0
            # find default excludes archived
            assert store.find() == []
            # but with include_archived it appears
            assert len(store.find(include_archived=True)) == 1
        finally:
            td.cleanup()

    def t_delete_cascades() -> None:
        store, _, td = fresh()
        try:
            it = store.add(KnowledgeKind.KNOWLEDGE, "X", "content")
            row = store.storage.query_one(
                "SELECT COUNT(*) AS c FROM knowledge_terms WHERE item_id=?;",
                (it.id,),
            )
            assert int(row["c"]) > 0
            assert store.delete(it.id) is True
            row2 = store.storage.query_one(
                "SELECT COUNT(*) AS c FROM knowledge_terms WHERE item_id=?;",
                (it.id,),
            )
            assert int(row2["c"]) == 0
        finally:
            td.cleanup()

    check("archive excludes from default views", t_archive_excludes)
    check("delete cascades to index", t_delete_cascades)

    # ---- exact match via title ----
    def t_exact_title_boost() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "SQLite WAL mode", "noise noise noise")
            store.add(KnowledgeKind.KNOWLEDGE, "Some other doc", "SQLite WAL appears in body only")
            res = retr.retrieve_text("SQLite WAL")
            assert len(res) >= 1
            # title-matched doc should rank first
            assert res.items[0].item.title == "SQLite WAL mode"
        finally:
            td.cleanup()

    check("title match outranks content match", t_exact_title_boost)

    # ---- keyword ranking ----
    def t_keyword_ranking() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "Python list comprehension",
                      "A list comprehension in Python is a concise way to create lists")
            store.add(KnowledgeKind.KNOWLEDGE, "Java streams",
                      "Java streams provide functional-style operations on collections")
            res = retr.retrieve_text("python list")
            assert len(res) >= 1
            assert "Python" in res.items[0].item.title
        finally:
            td.cleanup()

    check("keyword ranking picks relevant doc", t_keyword_ranking)

    # ---- language / framework / version filters ----
    def t_language_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "python stuff", language="python")
            store.add(KnowledgeKind.KNOWLEDGE, "B", "js stuff", language="javascript")
            res = retr.retrieve(RetrievalQuery(language="python"))
            assert len(res) == 1
            assert res.items[0].item.language == "python"
        finally:
            td.cleanup()

    def t_framework_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "content", framework="fastapi")
            store.add(KnowledgeKind.KNOWLEDGE, "B", "content", framework="flask")
            res = retr.retrieve(RetrievalQuery(framework="fastapi"))
            assert len(res) == 1
            assert res.items[0].item.framework == "fastapi"
        finally:
            td.cleanup()

    def t_version_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c", version="3.11")
            store.add(KnowledgeKind.KNOWLEDGE, "B", "c", version="3.12")
            res = retr.retrieve(RetrievalQuery(version="3.12"))
            assert len(res) == 1
            assert res.items[0].item.version == "3.12"
        finally:
            td.cleanup()

    check("language filter", t_language_filter)
    check("framework filter", t_framework_filter)
    check("version filter", t_version_filter)

    # ---- tag filter ----
    def t_tag_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c", tags=["api", "rest"])
            store.add(KnowledgeKind.KNOWLEDGE, "B", "c", tags=["api"])
            store.add(KnowledgeKind.KNOWLEDGE, "C", "c", tags=["db"])
            res = retr.retrieve(RetrievalQuery(tags=["api", "rest"]))
            assert len(res) == 1
            assert res.items[0].item.title == "A"
        finally:
            td.cleanup()

    check("tag filter (subset match)", t_tag_filter)

    # ---- kind filter ----
    def t_kind_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "K", "c")
            store.add(KnowledgeKind.EVIDENCE, "E", "c")
            store.add(KnowledgeKind.INFERENCE, "I", "c")
            store.add(KnowledgeKind.ASSUMPTION, "A", "c")
            store.add(KnowledgeKind.UNKNOWN, "U", "c")
            assert len(retr.retrieve(RetrievalQuery(kind=KnowledgeKind.EVIDENCE))) == 1
            assert len(retr.retrieve(RetrievalQuery(kind="inference"))) == 1
        finally:
            td.cleanup()

    check("kind filter distinguishes all 5 kinds", t_kind_filter)

    # ---- confidence filter ----
    def t_confidence_exact() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "high", "c",
                      provenance=Provenance(confidence=Confidence.HIGH))
            store.add(KnowledgeKind.KNOWLEDGE, "low", "c",
                      provenance=Provenance(confidence=Confidence.LOW))
            res = retr.retrieve(RetrievalQuery(confidence=Confidence.HIGH))
            assert len(res) == 1
            assert res.items[0].item.title == "high"
        finally:
            td.cleanup()

    def t_confidence_min() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "verified", "c",
                      provenance=Provenance(confidence=Confidence.VERIFIED))
            store.add(KnowledgeKind.KNOWLEDGE, "medium", "c",
                      provenance=Provenance(confidence=Confidence.MEDIUM))
            store.add(KnowledgeKind.KNOWLEDGE, "low", "c",
                      provenance=Provenance(confidence=Confidence.LOW))
            res = retr.retrieve(RetrievalQuery(min_confidence=Confidence.HIGH))
            assert len(res) == 1
            assert res.items[0].item.title == "verified"
        finally:
            td.cleanup()

    check("confidence exact filter", t_confidence_exact)
    check("min_confidence threshold", t_confidence_min)

    # ---- provenance filter ----
    def t_provenance_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "user-sourced", "c",
                      provenance=Provenance(
                          source="alice", source_type=ProvenanceType.USER,
                      ))
            store.add(KnowledgeKind.KNOWLEDGE, "imported", "c",
                      provenance=Provenance(
                          source="docs", source_type=ProvenanceType.IMPORT,
                      ))
            res = retr.retrieve(RetrievalQuery(provenance_source="alice"))
            assert len(res) == 1
            res2 = retr.retrieve(RetrievalQuery(provenance_type=ProvenanceType.IMPORT))
            assert len(res2) == 1
        finally:
            td.cleanup()

    check("provenance source + type filter", t_provenance_filter)

    # ---- metadata filter ----
    def t_metadata_equals() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c",
                      metadata={"severity": "high"})
            store.add(KnowledgeKind.KNOWLEDGE, "B", "c",
                      metadata={"severity": "low"})
            res = retr.retrieve(RetrievalQuery(metadata_equals={"severity": "high"}))
            assert len(res) == 1
            assert res.items[0].item.title == "A"
        finally:
            td.cleanup()

    check("metadata equals filter", t_metadata_equals)

    # ---- related entity filter ----
    def t_related_entity_filter() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c",
                      related_entity_ids=["ent-1"])
            store.add(KnowledgeKind.KNOWLEDGE, "B", "c",
                      related_entity_ids=["ent-2"])
            res = retr.retrieve(RetrievalQuery(related_entity_id="ent-1"))
            assert len(res) == 1
            assert res.items[0].item.title == "A"
        finally:
            td.cleanup()

    check("related entity id filter", t_related_entity_filter)

    # ---- scoring: more matches → higher score ----
    def t_score_monotonic() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "Only python",
                      "python is here")
            store.add(KnowledgeKind.KNOWLEDGE, "python and fastapi",
                      "python fastapi together")
            res = retr.retrieve_text("python fastapi")
            assert len(res) == 2
            # doc matching both terms should beat doc matching one
            assert res.items[0].item.title == "python and fastapi"
        finally:
            td.cleanup()

    check("score monotonic in matched terms", t_score_monotonic)

    # ---- confidence affects ranking ----
    def t_confidence_affects_ranking() -> None:
        store, retr, td = fresh()
        try:
            # Same title, different confidence
            store.add(KnowledgeKind.KNOWLEDGE, "same title here", "content",
                      provenance=Provenance(confidence=Confidence.VERIFIED))
            store.add(KnowledgeKind.KNOWLEDGE, "same title here", "content",
                      provenance=Provenance(confidence=Confidence.UNKNOWN))
            res = retr.retrieve_text("same title here")
            assert len(res) == 2
            assert res.items[0].item.provenance.confidence is Confidence.VERIFIED
            assert res.items[1].item.provenance.confidence is Confidence.UNKNOWN
        finally:
            td.cleanup()

    check("confidence weight affects ranking", t_confidence_affects_ranking)

    # ---- min_score ----
    def t_min_score() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "alpha beta", "alpha beta gamma")
            store.add(KnowledgeKind.KNOWLEDGE, "only alpha", "alpha only")
            res_all = retr.retrieve_text("alpha beta")
            assert len(res_all) >= 1
            high = res_all.items[0].score
            res_hi = retr.retrieve(RetrievalQuery(text="alpha beta", min_score=high + 0.001))
            assert len(res_hi) == 0
        finally:
            td.cleanup()

    check("min_score threshold filters", t_min_score)

    # ---- top_k ----
    def t_top_k() -> None:
        store, retr, td = fresh()
        try:
            for i in range(5):
                store.add(KnowledgeKind.KNOWLEDGE, f"doc {i} python", "content")
            res = retr.retrieve_text("python", top_k=3)
            assert len(res) == 3
        finally:
            td.cleanup()

    check("top_k limits results", t_top_k)

    # ---- filter-only mode ----
    def t_filter_only() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c", language="python")
            store.add(KnowledgeKind.KNOWLEDGE, "B", "c", language="python")
            res = retr.retrieve(RetrievalQuery(language="python"))
            assert len(res) == 2
            assert all(s.score == 1.0 for s in res.items)
            assert "filter-only" in res.notes
        finally:
            td.cleanup()

    def t_stopwords_only_query() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c")
            res = retr.retrieve_text("the and or")
            assert len(res) == 1  # falls back to filter-only
            assert "stopwords" in res.notes
        finally:
            td.cleanup()

    check("filter-only mode returns all candidates", t_filter_only)
    check("stopwords-only query falls back to filter-only", t_stopwords_only_query)

    # ---- empty / no-match ----
    def t_no_candidates() -> None:
        store, retr, td = fresh()
        try:
            res = retr.retrieve(RetrievalQuery(language="rust"))
            assert len(res) == 0
            assert res.total_candidates == 0
            assert "no candidates" in res.notes
        finally:
            td.cleanup()

    def t_no_matching_terms() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "hello world", "some content here")
            res = retr.retrieve_text("completelyunrelatedterm xyz")
            assert len(res) == 0
        finally:
            td.cleanup()

    check("no candidates → empty result", t_no_candidates)
    check("query with no corpus match → empty", t_no_matching_terms)

    # ---- matched fields ----
    def t_matched_fields() -> None:
        store, retr, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "python code",
                      "this is a python content", tags=["python"])
            res = retr.retrieve_text("python")
            assert len(res) == 1
            fields = set(res.items[0].matched_fields)
            assert "title" in fields
            assert "content" in fields
            assert "tags" in fields
        finally:
            td.cleanup()

    check("matched_fields populated", t_matched_fields)

    # ---- count / stats ----
    def t_count_stats() -> None:
        store, _, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "K1", "c", language="python")
            store.add(KnowledgeKind.EVIDENCE, "E1", "c", language="python")
            store.add(KnowledgeKind.INFERENCE, "I1", "c", language="js")
            it = store.add(KnowledgeKind.KNOWLEDGE, "K2", "c")
            store.archive(it.id)
            assert store.count() == 3
            assert store.count(include_archived=True) == 4
            assert store.count(kind=KnowledgeKind.KNOWLEDGE) == 1
            st = store.stats()
            assert st["items_active"] == 3
            assert st["items_archived"] == 1
        finally:
            td.cleanup()

    check("count + stats", t_count_stats)

    # ---- integrity ----
    def t_integrity_clean() -> None:
        store, _, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "A", "c")
            rep = store.verify_integrity()
            assert rep["ok"] is True
            assert rep["issue_count"] == 0
        finally:
            td.cleanup()

    check("integrity check clean", t_integrity_clean)

    # ---- health ----
    def t_health() -> None:
        store, _, td = fresh()
        try:
            store.add(KnowledgeKind.KNOWLEDGE, "Auth Pattern", "OAuth2 token refresh flow")
            h = store.health()
            assert h["ok"] is True
            assert h["items_active"] == 1
            assert h["index_terms"] > 0
        finally:
            td.cleanup()

    check("knowledge store health", t_health)

    # ---- provenance preserved ----
    def t_provenance_preserved() -> None:
        store, retr, td = fresh()
        try:
            prov = Provenance(
                source="user:bob",
                source_type=ProvenanceType.USER,
                reference="chat://msg/7",
                confidence=Confidence.HIGH,
                notes="explicitly stated",
            )
            store.add(KnowledgeKind.KNOWLEDGE, "R1", "content", provenance=prov)
            res = retr.retrieve_text("R1")
            assert len(res) == 1
            got = res.items[0].item.provenance
            assert got.source == "user:bob"
            assert got.source_type is ProvenanceType.USER
            assert got.reference == "chat://msg/7"
            assert got.confidence is Confidence.HIGH
        finally:
            td.cleanup()

    check("provenance preserved through retrieval", t_provenance_preserved)

    # ---- e2e ----
    def t_e2e() -> None:
        store, retr, td = fresh()
        try:
            # Populate a small corpus
            store.add(KnowledgeKind.KNOWLEDGE,
                      "FastAPI request validation with Pydantic",
                      "FastAPI uses Pydantic models for request validation.",
                      language="python", framework="fastapi",
                      tags=["api", "validation", "pydantic"],
                      provenance=Provenance(source="docs", source_type=ProvenanceType.IMPORT,
                                            confidence=Confidence.HIGH))
            store.add(KnowledgeKind.KNOWLEDGE,
                      "Flask routing basics",
                      "Flask uses decorators for routing.",
                      language="python", framework="flask",
                      tags=["api", "routing"],
                      provenance=Provenance(source="docs", source_type=ProvenanceType.IMPORT,
                                            confidence=Confidence.HIGH))
            store.add(KnowledgeKind.EVIDENCE,
                      "Benchmark: FastAPI 10k req/s",
                      "Measured 10,000 requests per second on m5.large.",
                      language="python", framework="fastapi",
                      tags=["benchmark", "performance"],
                      provenance=Provenance(source="bench", source_type=ProvenanceType.EXPERIMENT,
                                            confidence=Confidence.VERIFIED))
            store.add(KnowledgeKind.ASSUMPTION,
                      "Assume Python 3.11+ available",
                      "Assumption: user environment has Python 3.11 or newer.",
                      language="python",
                      provenance=Provenance(source="agent:planner",
                                            source_type=ProvenanceType.AGENT,
                                            confidence=Confidence.ASSUMPTION))

            # Query 1: framework-specific retrieval
            r1 = retr.retrieve(RetrievalQuery(
                text="validation", framework="fastapi", top_k=5,
            ))
            assert len(r1) >= 1
            assert all(s.item.framework == "fastapi" for s in r1.items)

            # Query 2: benchmark evidence only
            r2 = retr.retrieve(RetrievalQuery(
                text="requests second",
                kind=KnowledgeKind.EVIDENCE,
                min_confidence=Confidence.HIGH,
            ))
            assert len(r2) == 1
            assert r2.items[0].item.kind is KnowledgeKind.EVIDENCE

            # Query 3: assumptions are never treated as knowledge
            r3 = retr.retrieve(RetrievalQuery(kind=KnowledgeKind.KNOWLEDGE,
                                              text="Python"))
            assert all(s.item.kind is KnowledgeKind.KNOWLEDGE for s in r3.items)
            r4 = retr.retrieve(RetrievalQuery(kind=KnowledgeKind.ASSUMPTION))
            assert len(r4) == 1

            # Query 4: tag intersection
            r5 = retr.retrieve(RetrievalQuery(tags=["api"]))
            assert len(r5) == 2
        finally:
            td.cleanup()

    check("e2e: multi-strategy retrieval across kinds", t_e2e)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C03 — Knowledge Retrieval Engine")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                store = KnowledgeStore(app.storage)
                retr = KnowledgeRetriever(store)

                print("\n[1] Seeding knowledge base…")
                store.add(
                    KnowledgeKind.KNOWLEDGE,
                    "FastAPI request validation with Pydantic",
                    "FastAPI uses Pydantic models for request body/query validation.",
                    language="python", framework="fastapi",
                    tags=["api", "validation", "pydantic"],
                    provenance=Provenance(source="docs", source_type=ProvenanceType.IMPORT,
                                          confidence=Confidence.HIGH),
                )
                store.add(
                    KnowledgeKind.KNOWLEDGE,
                    "Flask routing with decorators",
                    "Flask uses @app.route decorators for HTTP routing.",
                    language="python", framework="flask",
                    tags=["api", "routing"],
                    provenance=Provenance(source="docs", source_type=ProvenanceType.IMPORT,
                                          confidence=Confidence.HIGH),
                )
                store.add(
                    KnowledgeKind.EVIDENCE,
                    "FastAPI benchmark: 10k req/s",
                    "Measured throughput: 10,000 requests/second on m5.large.",
                    language="python", framework="fastapi",
                    tags=["benchmark", "performance"],
                    provenance=Provenance(source="bench", source_type=ProvenanceType.EXPERIMENT,
                                          confidence=Confidence.VERIFIED),
                )
                store.add(
                    KnowledgeKind.ASSUMPTION,
                    "Assume Python 3.11+ runtime",
                    "Assumption (not verified): the deployment env has Python 3.11+.",
                    language="python",
                    provenance=Provenance(source="agent:planner",
                                          source_type=ProvenanceType.AGENT,
                                          confidence=Confidence.ASSUMPTION),
                )
                print(f"    {store.health()}")

                print("\n[2] Query: 'validation' [framework=fastapi]")
                for s in retr.retrieve(RetrievalQuery(text="validation",
                                                      framework="fastapi")):
                    print(f"    {s.score:.3f}  {s.item.title}  "
                          f"(fields={s.matched_fields})")

                print("\n[3] Query: 'requests second' [kind=EVIDENCE, min_conf=HIGH]")
                for s in retr.retrieve(RetrievalQuery(
                    text="requests second",
                    kind=KnowledgeKind.EVIDENCE,
                    min_confidence=Confidence.HIGH,
                )):
                    print(f"    {s.score:.3f}  {s.item.title}")

                print("\n[4] Query: 'python' [kind=KNOWLEDGE only]")
                for s in retr.retrieve(RetrievalQuery(
                    text="python", kind=KnowledgeKind.KNOWLEDGE,
                )):
                    print(f"    {s.score:.3f}  [{s.item.kind.value}] {s.item.title}")

                print("\n[5] Query: assumptions only")
                for s in retr.retrieve(RetrievalQuery(kind=KnowledgeKind.ASSUMPTION)):
                    print(f"    {s.score:.3f}  {s.item.title}")

                print("\n[6] Stats:")
                for k, v in sorted(store.stats().items()):
                    print(f"    {k}: {v}")

                print("\n[7] Integrity:", store.verify_integrity())
        finally:
            app.stop()
    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
