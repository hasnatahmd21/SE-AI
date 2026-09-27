"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C02 — SOFTWARE ONTOLOGY ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01: place this file in the same directory as `sebrain/c01.py`.

Provides the AUTHORITATIVE ontology for the SE Brain — the single source of
truth for every entity and relationship the system will reason about.

Contents:
  1.  ProvenanceType — where a fact came from
  2.  EntityKind     — 36 entity kinds (Project, Requirement, Bug, Test, ...)
  3.  RelationKind   — 22 relation kinds (CONTAINS, DEPENDS_ON, FIXES, ...)
  4.  Provenance     — source + confidence attached to every node/edge
  5.  Entity         — generic node with id/kind/name/version/status/attrs/tags
  6.  Relation       — typed edge with attributes + provenance
  7.  Neighbor       — (edge, node, direction) result of graph queries
  8.  Migration 002  — ontology_entities + ontology_relations tables
  9.  Ontology       — facade: add/get/update/supersede/archive/delete,
                       link/unlink/neighbors/traverse, find/count/stats,
                       verify_integrity, health
  10. __main__ demo + self-tests

Design rules honored:
  - NO graph database. Two SQL tables + indices. That's it.
  - Every entity and relation carries provenance + confidence.
  - Lifecycle: active → superseded | archived. Hard delete is explicit.
  - Versioning: `update()` bumps version + updated_at.
  - Timestamps: ISO-8601 UTC.
  - Foreign keys enforce referential integrity; ON DELETE CASCADE on relations.
  - Deterministic. No global mutable state.

Run as script:
    python -m sebrain.c02            # demo
    python -m sebrain.c02 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import json
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

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


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


# ════════════════════════════════════════════════════════════════════════════
# 1. PROVENANCE TYPE
# ════════════════════════════════════════════════════════════════════════════
class ProvenanceType(str, Enum):
    """Where an entity/relation/attribute came from.

    Rule: never collapse these. `USER` statements and `INFERENCE` are
    not equal, and both differ from `RETRIEVAL` or `EXPERIMENT`.
    """
    USER = "user"
    AGENT = "agent"
    INFERENCE = "inference"
    RETRIEVAL = "retrieval"
    EXPERIMENT = "experiment"
    IMPORT = "import"
    SYSTEM = "system"


# ════════════════════════════════════════════════════════════════════════════
# 2. ENTITY KINDS (36)
# ════════════════════════════════════════════════════════════════════════════
class EntityKind(str, Enum):
    # Product / project context
    PROJECT = "project"
    PRODUCT = "product"
    ARCHITECTURE = "architecture"
    COMPONENT = "component"

    # Requirements / constraints
    REQUIREMENT = "requirement"
    CONSTRAINT = "constraint"
    GOAL = "goal"

    # Code structure
    MODULE = "module"
    PACKAGE = "package"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    TYPE = "type"
    INTERFACE = "interface"

    # Interfaces & data
    API = "api"
    DATABASE = "database"
    SCHEMA = "schema"
    DEPENDENCY = "dependency"

    # Testing / quality
    TEST = "test"
    TEST_RESULT = "test_result"

    # Failures & repairs
    BUG = "bug"
    FAILURE = "failure"
    ROOT_CAUSE = "root_cause"
    FIX = "fix"

    # Reasoning / decisions
    DECISION = "decision"
    ALTERNATIVE = "alternative"
    EVIDENCE = "evidence"

    # Outcomes & learning
    OUTCOME = "outcome"
    EXPERIENCE = "experience"

    # Orchestration
    AGENT = "agent"
    TASK = "task"
    PLAN = "plan"
    EXECUTION = "execution"
    VERIFICATION = "verification"


# ════════════════════════════════════════════════════════════════════════════
# 3. RELATION KINDS (22)
# ════════════════════════════════════════════════════════════════════════════
class RelationKind(str, Enum):
    # Structural
    CONTAINS = "contains"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"

    # Code-level
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    USES = "uses"

    # Testing / quality
    TESTS = "tests"
    VERIFIES = "verifies"
    FAILS_WITH = "fails_with"

    # Reasoning
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFUTES = "refutes"
    DERIVED_FROM = "derived_from"

    # Lifecycle / causality
    SUPERSEDES = "supersedes"
    PRODUCES = "produces"
    FIXES = "fixes"
    RELATES_TO = "relates_to"
    DOCUMENTS = "documents"


# ════════════════════════════════════════════════════════════════════════════
# 4. PROVENANCE
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a piece of knowledge came from."""
    source: str = "system"
    source_type: ProvenanceType = ProvenanceType.SYSTEM
    reference: str | None = None         # URL, file path, entity id, ...
    confidence: Confidence = Confidence.UNKNOWN
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type.value,
            "reference": self.reference,
            "confidence": self.confidence.value,
            "notes": self.notes,
        }


# ════════════════════════════════════════════════════════════════════════════
# 5. ENTITY
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Entity:
    """A single node in the ontology.

    Invariants enforced by Ontology.add():
      - kind is a valid EntityKind
      - name is non-empty
      - id is unique
    """
    kind: EntityKind
    name: str
    id: str = field(default_factory=_new_id)
    version: int = 1
    status: str = "active"          # "active" | "superseded" | "archived"
    attributes: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    superseded_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "version": self.version,
            "status": self.status,
            "attributes": self.attributes,
            "tags": list(self.tags),
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "superseded_by": self.superseded_by,
        }


# ════════════════════════════════════════════════════════════════════════════
# 6. RELATION
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Relation:
    kind: RelationKind
    source_id: str
    target_id: str
    id: str = field(default_factory=_new_id)
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "attributes": self.attributes,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }


# ════════════════════════════════════════════════════════════════════════════
# 7. NEIGHBOR RESULT
# ════════════════════════════════════════════════════════════════════════════
Direction = Literal["out", "in", "both"]


@dataclass(slots=True)
class Neighbor:
    edge: Relation
    node: Entity
    direction: Literal["out", "in"]   # "out" = edge points from query node, "in" = to it


# ════════════════════════════════════════════════════════════════════════════
# 8. MIGRATION 002 — ONTOLOGY TABLES
# ════════════════════════════════════════════════════════════════════════════
def _migration_002_ontology(storage: SQLiteStorage) -> None:
    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS ontology_entities (
            id            TEXT PRIMARY KEY,
            kind          TEXT NOT NULL,
            name          TEXT NOT NULL,
            version       INTEGER NOT NULL DEFAULT 1,
            status        TEXT NOT NULL DEFAULT 'active',
            attributes    TEXT NOT NULL DEFAULT '{}',
            tags          TEXT NOT NULL DEFAULT '[]',
            prov_source   TEXT NOT NULL,
            prov_type     TEXT NOT NULL,
            prov_ref      TEXT,
            prov_conf     TEXT NOT NULL,
            prov_notes    TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL,
            superseded_by TEXT
        );
        """
    )
    storage.execute("CREATE INDEX IF NOT EXISTS idx_ont_kind   ON ontology_entities(kind);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_ont_name   ON ontology_entities(name);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_ont_status ON ontology_entities(status);")

    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS ontology_relations (
            id          TEXT PRIMARY KEY,
            kind        TEXT NOT NULL,
            source_id   TEXT NOT NULL,
            target_id   TEXT NOT NULL,
            attributes  TEXT NOT NULL DEFAULT '{}',
            prov_source TEXT NOT NULL,
            prov_type   TEXT NOT NULL,
            prov_ref    TEXT,
            prov_conf   TEXT NOT NULL,
            prov_notes  TEXT NOT NULL DEFAULT '',
            created_at  TEXT NOT NULL,
            UNIQUE(kind, source_id, target_id),
            FOREIGN KEY(source_id) REFERENCES ontology_entities(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES ontology_entities(id) ON DELETE CASCADE
        );
        """
    )
    storage.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON ontology_relations(source_id);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON ontology_relations(target_id);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_rel_kind   ON ontology_relations(kind);")


C02_MIGRATIONS: list[Migration] = [
    Migration(version=2, name="ontology", up=_migration_002_ontology),
]

_ALL_MIGRATIONS = sorted(C01_MIGRATIONS + C02_MIGRATIONS, key=lambda m: m.version)


# ════════════════════════════════════════════════════════════════════════════
# 9. ONTOLOGY FACADE
# ════════════════════════════════════════════════════════════════════════════
class Ontology:
    """Authoritative software-engineering ontology.

    Every entity and relation carries provenance + confidence.
    No graph DB. Two SQL tables + indices.
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
            counts = self.count_by_kind()
            rel_count = self.storage.query_one(
                "SELECT COUNT(*) AS c FROM ontology_relations;"
            )
            return {
                "component": "ontology",
                "ok": True,
                "entities": sum(counts.values()),
                "relations": int(rel_count["c"]) if rel_count else 0,
            }
        except Exception as exc:
            return {"component": "ontology", "ok": False, "error": str(exc)}

    # ══════════════════════════════════════════════════════════════════════
    # ENTITY: WRITE
    # ══════════════════════════════════════════════════════════════════════
    def add(
        self,
        kind: EntityKind | str,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        tags: Iterable[str] | None = None,
        provenance: Provenance | None = None,
        status: str = "active",
    ) -> Entity:
        """Insert a new entity. Name need not be unique; callers use ids."""
        kind_e = self._coerce_kind(kind)
        if not name or not name.strip():
            raise ValidationError("entity name must be non-empty")
        if status not in ("active", "draft", "superseded", "archived"):
            raise ValidationError(f"invalid status: {status}")

        entity = Entity(
            kind=kind_e,
            name=name.strip(),
            attributes=dict(attributes or {}),
            tags=sorted({str(t) for t in (tags or [])}),
            provenance=provenance or Provenance(),
            status=status,
        )
        self._insert_entity(entity)
        log.debug("ontology.entity.added", kind=kind_e.value, id=entity.id, name=name)
        return entity

    def update(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
        tags: Iterable[str] | None = None,
        status: str | None = None,
    ) -> Entity:
        """Update mutable fields. Bumps version and updated_at."""
        current = self.get(entity_id)
        if current is None:
            raise ValidationError(f"entity not found: {entity_id}")
        if current.status == "archived":
            raise ValidationError("cannot update archived entity")

        new_name = (name.strip() if name else current.name)
        if not new_name:
            raise ValidationError("entity name must be non-empty")

        new_attrs = dict(current.attributes)
        if attributes is not None:
            new_attrs.update(attributes)

        new_tags = current.tags if tags is None else sorted({str(t) for t in tags})
        new_status = status if status is not None else current.status
        if new_status not in ("active", "draft", "superseded", "archived"):
            raise ValidationError(f"invalid status: {new_status}")

        updated = Entity(
            id=current.id,
            kind=current.kind,
            name=new_name,
            version=current.version + 1,
            status=new_status,
            attributes=new_attrs,
            tags=list(new_tags),
            provenance=current.provenance,
            created_at=current.created_at,
            updated_at=now_iso(),
            superseded_by=current.superseded_by,
        )
        self._replace_entity(updated)
        return updated

    def supersede(self, old_id: str, new_id: str) -> tuple[Entity, Entity]:
        """Mark old entity as superseded by new one. Both must exist.

        Sets old.status='superseded', old.superseded_by=new_id,
        and links them with a SUPERSEDES relation (new → old).
        """
        old = self.get(old_id)
        new = self.get(new_id)
        if old is None:
            raise ValidationError(f"entity not found: {old_id}")
        if new is None:
            raise ValidationError(f"entity not found: {new_id}")
        if old.id == new.id:
            raise ValidationError("cannot supersede an entity with itself")
        if old.status == "superseded":
            raise ValidationError(f"entity already superseded: {old_id}")

        new_old = Entity(
            id=old.id, kind=old.kind, name=old.name,
            version=old.version + 1, status="superseded",
            attributes=old.attributes, tags=old.tags,
            provenance=old.provenance,
            created_at=old.created_at, updated_at=now_iso(),
            superseded_by=new.id,
        )
        self._replace_entity(new_old)
        # Idempotent link (UNIQUE constraint on kind+source+target)
        try:
            self.link(RelationKind.SUPERSEDES, new.id, old.id,
                      provenance=old.provenance)
        except ValidationError:
            pass
        return new_old, new

    def archive(self, entity_id: str) -> Entity:
        """Soft-delete: mark as archived. Keeps the record."""
        current = self.get(entity_id)
        if current is None:
            raise ValidationError(f"entity not found: {entity_id}")
        if current.status == "archived":
            return current
        archived = Entity(
            id=current.id, kind=current.kind, name=current.name,
            version=current.version + 1, status="archived",
            attributes=current.attributes, tags=current.tags,
            provenance=current.provenance,
            created_at=current.created_at, updated_at=now_iso(),
            superseded_by=current.superseded_by,
        )
        self._replace_entity(archived)
        return archived

    def delete(self, entity_id: str) -> bool:
        """Hard delete. Cascades to relations via FK."""
        row = self.storage.query_one(
            "SELECT id FROM ontology_entities WHERE id=?", (entity_id,)
        )
        if row is None:
            return False
        with self.storage.transaction():
            self.storage.execute(
                "DELETE FROM ontology_entities WHERE id=?", (entity_id,)
            )
        return True

    # ══════════════════════════════════════════════════════════════════════
    # ENTITY: READ
    # ══════════════════════════════════════════════════════════════════════
    def get(self, entity_id: str) -> Entity | None:
        row = self.storage.query_one(
            "SELECT * FROM ontology_entities WHERE id=?", (entity_id,)
        )
        return _entity_from_row(row) if row else None

    def get_by_name(
        self, kind: EntityKind | str, name: str, *, include_archived: bool = False
    ) -> list[Entity]:
        kind_e = self._coerce_kind(kind)
        sql = "SELECT * FROM ontology_entities WHERE kind=? AND name=?"
        params: list[Any] = [kind_e.value, name]
        if not include_archived:
            sql += " AND status != 'archived'"
        sql += " ORDER BY created_at DESC;"
        rows = self.storage.query(sql, params)
        return [_entity_from_row(r) for r in rows]

    def find(
        self,
        *,
        kind: EntityKind | str | None = None,
        status: str | None = None,
        tag: str | None = None,
        name_like: str | None = None,
        include_archived: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Entity]:
        sql = "SELECT * FROM ontology_entities WHERE 1=1"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(kind).value)
        if status is not None:
            sql += " AND status=?"
            params.append(status)
        elif not include_archived:
            sql += " AND status != 'archived'"
        # tag and name_like are applied in Python after fetching: tags are
        # a JSON array (SQL substring match would be unsafe), and SQLite's
        # LIKE is case-insensitive for ASCII by default, which would
        # silently defeat a case-sensitive name search. limit/offset are
        # applied after these filters too, so they paginate the filtered
        # result rather than the pre-filter row set.
        sql += " ORDER BY created_at DESC"
        rows = self.storage.query(sql, params)
        entities = [_entity_from_row(r) for r in rows]
        if tag is not None:
            entities = [e for e in entities if tag in e.tags]
        if name_like is not None:
            entities = [e for e in entities if name_like in e.name]
        if limit is not None:
            entities = entities[offset:offset + limit]
        elif offset:
            entities = entities[offset:]
        return entities

    def count(self, *, kind: EntityKind | str | None = None, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS c FROM ontology_entities WHERE 1=1"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(kind).value)
        if status is not None:
            sql += " AND status=?"
            params.append(status)
        else:
            # Match find()/count_by_kind(): archived entities are hidden
            # by default unless a specific status is requested.
            sql += " AND status != 'archived'"
        row = self.storage.query_one(sql + ";", params)
        return int(row["c"]) if row else 0

    def count_by_kind(self) -> dict[str, int]:
        rows = self.storage.query(
            "SELECT kind, COUNT(*) AS c FROM ontology_entities "
            "WHERE status != 'archived' GROUP BY kind;"
        )
        return {r["kind"]: int(r["c"]) for r in rows}

    def stats(self) -> dict[str, Any]:
        return {
            "entities_total": self.count(),
            "entities_by_kind": self.count_by_kind(),
            "relations_total": self._count_relations(),
            "relations_by_kind": self._count_relations_by_kind(),
        }

    # ══════════════════════════════════════════════════════════════════════
    # RELATIONS
    # ══════════════════════════════════════════════════════════════════════
    def link(
        self,
        kind: RelationKind | str,
        source_id: str,
        target_id: str,
        *,
        attributes: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
    ) -> Relation:
        kind_e = self._coerce_relation_kind(kind)
        # FK will enforce existence, but we give a nicer error first.
        if self.get(source_id) is None:
            raise ValidationError(f"source entity not found: {source_id}")
        if self.get(target_id) is None:
            raise ValidationError(f"target entity not found: {target_id}")
        # Unique check (SQLite would enforce too, but explicit for clarity)
        existing = self.storage.query_one(
            "SELECT id FROM ontology_relations "
            "WHERE kind=? AND source_id=? AND target_id=?;",
            (kind_e.value, source_id, target_id),
        )
        if existing:
            raise ValidationError(
                f"relation already exists: {kind_e.value} "
                f"{source_id}->{target_id}"
            )

        rel = Relation(
            kind=kind_e,
            source_id=source_id,
            target_id=target_id,
            attributes=dict(attributes or {}),
            provenance=provenance or Provenance(),
        )
        self.storage.execute(
            """
            INSERT INTO ontology_relations
                (id, kind, source_id, target_id, attributes,
                 prov_source, prov_type, prov_ref, prov_conf, prov_notes,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                rel.id, rel.kind.value, rel.source_id, rel.target_id,
                json.dumps(rel.attributes, ensure_ascii=False, default=str),
                rel.provenance.source, rel.provenance.source_type.value,
                rel.provenance.reference, rel.provenance.confidence.value,
                rel.provenance.notes, rel.created_at,
            ),
        )
        return rel

    def get_relation(self, relation_id: str) -> Relation | None:
        row = self.storage.query_one(
            "SELECT * FROM ontology_relations WHERE id=?", (relation_id,)
        )
        return _relation_from_row(row) if row else None

    def unlink(self, relation_id: str) -> bool:
        row = self.storage.query_one(
            "SELECT id FROM ontology_relations WHERE id=?", (relation_id,)
        )
        if row is None:
            return False
        self.storage.execute(
            "DELETE FROM ontology_relations WHERE id=?", (relation_id,)
        )
        return True

    def relations(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: RelationKind | str | None = None,
        limit: int | None = None,
    ) -> list[Relation]:
        sql = "SELECT * FROM ontology_relations WHERE 1=1"
        params: list[Any] = []
        if source_id is not None:
            sql += " AND source_id=?"
            params.append(source_id)
        if target_id is not None:
            sql += " AND target_id=?"
            params.append(target_id)
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_relation_kind(kind).value)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self.storage.query(sql + ";", params)
        return [_relation_from_row(r) for r in rows]

    def neighbors(
        self,
        entity_id: str,
        *,
        kind: RelationKind | str | None = None,
        direction: Direction = "both",
    ) -> list[Neighbor]:
        """Return immediate neighbors of a node."""
        if self.get(entity_id) is None:
            raise ValidationError(f"entity not found: {entity_id}")
        kind_v = self._coerce_relation_kind(kind).value if kind is not None else None

        out: list[Neighbor] = []
        # Outgoing edges
        if direction in ("out", "both"):
            sql = "SELECT * FROM ontology_relations WHERE source_id=?"
            params: list[Any] = [entity_id]
            if kind_v:
                sql += " AND kind=?"
                params.append(kind_v)
            for row in self.storage.query(sql + ";", params):
                edge = _relation_from_row(row)
                node = self.get(edge.target_id)
                if node is not None:
                    out.append(Neighbor(edge=edge, node=node, direction="out"))
        # Incoming edges
        if direction in ("in", "both"):
            sql = "SELECT * FROM ontology_relations WHERE target_id=?"
            params = [entity_id]
            if kind_v:
                sql += " AND kind=?"
                params.append(kind_v)
            for row in self.storage.query(sql + ";", params):
                edge = _relation_from_row(row)
                node = self.get(edge.source_id)
                if node is not None:
                    out.append(Neighbor(edge=edge, node=node, direction="in"))
        return out

    def traverse(
        self,
        start_id: str,
        *,
        relation_kinds: Iterable[RelationKind | str] | None = None,
        direction: Direction = "out",
        max_depth: int = 3,
        max_nodes: int = 500,
        include_start: bool = False,
    ) -> list[Entity]:
        """BFS traversal with hard limits. Cycle-safe."""
        if max_depth < 0:
            raise ValidationError("max_depth must be >= 0")
        if max_nodes < 1:
            raise ValidationError("max_nodes must be >= 1")
        start = self.get(start_id)
        if start is None:
            raise ValidationError(f"entity not found: {start_id}")

        kind_set: set[str] | None = None
        if relation_kinds is not None:
            kind_set = {self._coerce_relation_kind(k).value for k in relation_kinds}

        visited: set[str] = {start.id}
        result: list[Entity] = [start] if include_start else []
        frontier: list[tuple[str, int]] = [(start.id, 0)]

        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for n in self.neighbors(current_id, direction=direction):
                if kind_set is not None and n.edge.kind.value not in kind_set:
                    continue
                if n.node.id in visited:
                    continue
                visited.add(n.node.id)
                result.append(n.node)
                if len(result) >= max_nodes:
                    return result
                frontier.append((n.node.id, depth + 1))
        return result

    # ══════════════════════════════════════════════════════════════════════
    # INTEGRITY
    # ══════════════════════════════════════════════════════════════════════
    def verify_integrity(self) -> dict[str, Any]:
        """Checks for orphan relations, dangling supersessions, bad JSON."""
        issues: list[dict[str, Any]] = []

        # Orphan relations (should not happen with FK ON)
        orphan_edges = self.storage.query(
            """
            SELECT r.id FROM ontology_relations r
            LEFT JOIN ontology_entities s ON r.source_id = s.id
            LEFT JOIN ontology_entities t ON r.target_id = t.id
            WHERE s.id IS NULL OR t.id IS NULL;
            """
        )
        for row in orphan_edges:
            issues.append({"type": "orphan_relation", "id": row["id"]})

        # Dangling supersede pointers
        dangling = self.storage.query(
            """
            SELECT e.id FROM ontology_entities e
            LEFT JOIN ontology_entities n ON e.superseded_by = n.id
            WHERE e.superseded_by IS NOT NULL AND n.id IS NULL;
            """
        )
        for row in dangling:
            issues.append({"type": "dangling_supersede", "id": row["id"]})

        # Bad JSON in attributes / tags
        for row in self.storage.query("SELECT id, attributes, tags FROM ontology_entities;"):
            try:
                json.loads(row["attributes"])
                json.loads(row["tags"])
            except json.JSONDecodeError as exc:
                issues.append({"type": "bad_json", "id": row["id"], "error": str(exc)})

        return {
            "ok": not issues,
            "issues": issues,
            "issue_count": len(issues),
        }

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL: row <-> dataclass
    # ══════════════════════════════════════════════════════════════════════
    def _insert_entity(self, e: Entity) -> None:
        self.storage.execute(
            """
            INSERT INTO ontology_entities
                (id, kind, name, version, status, attributes, tags,
                 prov_source, prov_type, prov_ref, prov_conf, prov_notes,
                 created_at, updated_at, superseded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            _entity_to_row(e),
        )

    def _replace_entity(self, e: Entity) -> None:
        self.storage.execute(
            """
            UPDATE ontology_entities SET
                kind=?, name=?, version=?, status=?, attributes=?, tags=?,
                prov_source=?, prov_type=?, prov_ref=?, prov_conf=?, prov_notes=?,
                updated_at=?, superseded_by=?
            WHERE id=?;
            """,
            (
                e.kind.value, e.name, e.version, e.status,
                json.dumps(e.attributes, ensure_ascii=False, default=str),
                json.dumps(e.tags, ensure_ascii=False),
                e.provenance.source, e.provenance.source_type.value,
                e.provenance.reference, e.provenance.confidence.value,
                e.provenance.notes, e.updated_at, e.superseded_by,
                e.id,
            ),
        )

    def _count_relations(self) -> int:
        row = self.storage.query_one("SELECT COUNT(*) AS c FROM ontology_relations;")
        return int(row["c"]) if row else 0

    def _count_relations_by_kind(self) -> dict[str, int]:
        rows = self.storage.query(
            "SELECT kind, COUNT(*) AS c FROM ontology_relations GROUP BY kind;"
        )
        return {r["kind"]: int(r["c"]) for r in rows}

    # ---- coercion helpers ----
    @staticmethod
    def _coerce_kind(kind: EntityKind | str) -> EntityKind:
        if isinstance(kind, EntityKind):
            return kind
        try:
            return EntityKind(kind)
        except ValueError as exc:
            raise ValidationError(f"unknown entity kind: {kind!r}") from exc

    @staticmethod
    def _coerce_relation_kind(kind: RelationKind | str) -> RelationKind:
        if isinstance(kind, RelationKind):
            return kind
        try:
            return RelationKind(kind)
        except ValueError as exc:
            raise ValidationError(f"unknown relation kind: {kind!r}") from exc


# ════════════════════════════════════════════════════════════════════════════
# ROW CONVERSION HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _entity_to_row(e: Entity) -> tuple:
    return (
        e.id, e.kind.value, e.name, e.version, e.status,
        json.dumps(e.attributes, ensure_ascii=False, default=str),
        json.dumps(e.tags, ensure_ascii=False),
        e.provenance.source, e.provenance.source_type.value,
        e.provenance.reference, e.provenance.confidence.value,
        e.provenance.notes,
        e.created_at, e.updated_at, e.superseded_by,
    )


def _entity_from_row(row: dict[str, Any]) -> Entity:
    return Entity(
        id=row["id"],
        kind=EntityKind(row["kind"]),
        name=row["name"],
        version=int(row["version"]),
        status=row["status"],
        attributes=json.loads(row["attributes"] or "{}"),
        tags=json.loads(row["tags"] or "[]"),
        provenance=Provenance(
            source=row["prov_source"],
            source_type=ProvenanceType(row["prov_type"]),
            reference=row["prov_ref"],
            confidence=Confidence(row["prov_conf"]),
            notes=row["prov_notes"] or "",
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        superseded_by=row["superseded_by"],
    )


def _relation_from_row(row: dict[str, Any]) -> Relation:
    return Relation(
        id=row["id"],
        kind=RelationKind(row["kind"]),
        source_id=row["source_id"],
        target_id=row["target_id"],
        attributes=json.loads(row["attributes"] or "{}"),
        provenance=Provenance(
            source=row["prov_source"],
            source_type=ProvenanceType(row["prov_type"]),
            reference=row["prov_ref"],
            confidence=Confidence(row["prov_conf"]),
            notes=row["prov_notes"] or "",
        ),
        created_at=row["created_at"],
    )


# ════════════════════════════════════════════════════════════════════════════
# 10. SELF-TESTS
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

    def fresh_ontology() -> tuple[Ontology, tempfile.TemporaryDirectory]:
        td = tempfile.TemporaryDirectory()
        s = SQLiteStorage(Path(td.name) / "ont.sqlite3")
        s.initialize()
        return Ontology(s), td

    print("Running C02 self-tests…")

    # ---- enums ----
    def t_kinds_complete() -> None:
        expected = {
            "project", "product", "architecture", "component",
            "requirement", "constraint", "goal",
            "module", "package", "file", "class", "function", "method",
            "variable", "type", "interface",
            "api", "database", "schema", "dependency",
            "test", "test_result",
            "bug", "failure", "root_cause", "fix",
            "decision", "alternative", "evidence",
            "outcome", "experience",
            "agent", "task", "plan", "execution", "verification",
        }
        got = {k.value for k in EntityKind}
        assert got == expected, f"missing/extra: {expected ^ got}"

    def t_relation_kinds_complete() -> None:
        expected = {
            "contains", "parent_of", "child_of",
            "imports", "calls", "inherits", "implements", "depends_on", "uses",
            "tests", "verifies", "fails_with",
            "supports", "contradicts", "refutes", "derived_from",
            "supersedes", "produces", "fixes", "relates_to", "documents",
        }
        got = {k.value for k in RelationKind}
        assert got == expected, f"missing/extra: {expected ^ got}"

    check("entity kinds complete (36)", t_kinds_complete)
    check("relation kinds complete (22)", t_relation_kinds_complete)

    # ---- add / get ----
    def t_add_get() -> None:
        ont, td = fresh_ontology()
        try:
            p = ont.add(EntityKind.PROJECT, "Task Manager", attributes={"lang": "python"})
            assert p.id
            assert p.version == 1
            assert p.status == "active"
            got = ont.get(p.id)
            assert got is not None
            assert got.name == "Task Manager"
            assert got.attributes == {"lang": "python"}
        finally:
            td.cleanup()

    def t_add_validates_name() -> None:
        ont, td = fresh_ontology()
        try:
            try:
                ont.add(EntityKind.PROJECT, "")
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    def t_add_validates_kind() -> None:
        ont, td = fresh_ontology()
        try:
            try:
                ont.add("not_a_kind", "x")
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("add + get entity", t_add_get)
    check("add rejects empty name", t_add_validates_name)
    check("add rejects bad kind", t_add_validates_kind)

    # ---- get_by_name ----
    def t_get_by_name() -> None:
        ont, td = fresh_ontology()
        try:
            ont.add(EntityKind.REQUIREMENT, "Login")
            ont.add(EntityKind.REQUIREMENT, "Login")  # duplicate name is OK
            ont.add(EntityKind.GOAL, "Login")
            reqs = ont.get_by_name(EntityKind.REQUIREMENT, "Login")
            assert len(reqs) == 2
            goals = ont.get_by_name(EntityKind.GOAL, "Login")
            assert len(goals) == 1
        finally:
            td.cleanup()

    check("get_by_name filter", t_get_by_name)

    # ---- update ----
    def t_update_bumps_version() -> None:
        ont, td = fresh_ontology()
        try:
            e = ont.add(EntityKind.REQUIREMENT, "R1", attributes={"priority": "low"})
            assert e.version == 1
            e2 = ont.update(e.id, attributes={"priority": "high"}, tags=["urgent"])
            assert e2.version == 2
            assert e2.attributes["priority"] == "high"
            assert "urgent" in e2.tags
            assert e2.created_at == e.created_at
            assert e2.updated_at >= e.updated_at
        finally:
            td.cleanup()

    def t_update_missing_entity() -> None:
        ont, td = fresh_ontology()
        try:
            try:
                ont.update("nope", name="x")
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("update bumps version + preserves created_at", t_update_bumps_version)
    check("update missing entity raises", t_update_missing_entity)

    # ---- supersede ----
    def t_supersede() -> None:
        ont, td = fresh_ontology()
        try:
            old = ont.add(EntityKind.DECISION, "Use Flask")
            new = ont.add(EntityKind.DECISION, "Use FastAPI")
            old2, _ = ont.supersede(old.id, new.id)
            assert old2.status == "superseded"
            assert old2.superseded_by == new.id
            # relation created
            rels = ont.relations(source_id=new.id, target_id=old.id,
                                 kind=RelationKind.SUPERSEDES)
            assert len(rels) == 1
        finally:
            td.cleanup()

    def t_supersede_self_invalid() -> None:
        ont, td = fresh_ontology()
        try:
            e = ont.add(EntityKind.DECISION, "x")
            try:
                ont.supersede(e.id, e.id)
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("supersede marks old + creates relation", t_supersede)
    check("supersede self rejected", t_supersede_self_invalid)

    # ---- archive ----
    def t_archive() -> None:
        ont, td = fresh_ontology()
        try:
            e = ont.add(EntityKind.BUG, "Null pointer")
            a = ont.archive(e.id)
            assert a.status == "archived"
            # archived hidden by default
            assert ont.count() == 0
            assert ont.count(status="archived") == 1
            # but get() still finds it
            assert ont.get(e.id) is not None
        finally:
            td.cleanup()

    check("archive (soft delete)", t_archive)

    # ---- hard delete + cascade ----
    def t_hard_delete_cascades() -> None:
        ont, td = fresh_ontology()
        try:
            a = ont.add(EntityKind.PROJECT, "A")
            b = ont.add(EntityKind.REQUIREMENT, "B")
            ont.link(RelationKind.CONTAINS, a.id, b.id)
            assert ont._count_relations() == 1
            ok = ont.delete(a.id)
            assert ok is True
            assert ont.get(a.id) is None
            # Relation should be gone (FK ON DELETE CASCADE)
            assert ont._count_relations() == 0
        finally:
            td.cleanup()

    check("hard delete cascades to relations", t_hard_delete_cascades)

    # ---- relations ----
    def t_link_unique() -> None:
        ont, td = fresh_ontology()
        try:
            a = ont.add(EntityKind.PROJECT, "A")
            b = ont.add(EntityKind.REQUIREMENT, "B")
            ont.link(RelationKind.CONTAINS, a.id, b.id)
            try:
                ont.link(RelationKind.CONTAINS, a.id, b.id)
            except ValidationError:
                return
            raise AssertionError("expected ValidationError for duplicate edge")
        finally:
            td.cleanup()

    def t_link_missing_endpoint() -> None:
        ont, td = fresh_ontology()
        try:
            a = ont.add(EntityKind.PROJECT, "A")
            try:
                ont.link(RelationKind.CONTAINS, a.id, "missing")
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("link rejects duplicates", t_link_unique)
    check("link rejects missing endpoints", t_link_missing_endpoint)

    # ---- neighbors ----
    def t_neighbors_both() -> None:
        ont, td = fresh_ontology()
        try:
            a = ont.add(EntityKind.PROJECT, "A")
            b = ont.add(EntityKind.MODULE, "B")
            c = ont.add(EntityKind.FILE, "C")
            ont.link(RelationKind.CONTAINS, a.id, b.id)
            ont.link(RelationKind.CONTAINS, b.id, c.id)
            nb = ont.neighbors(b.id, direction="both")
            kinds = {(n.node.name, n.direction) for n in nb}
            assert ("A", "in") in kinds
            assert ("C", "out") in kinds
        finally:
            td.cleanup()

    check("neighbors both directions", t_neighbors_both)

    # ---- traverse ----
    def t_traverse_bfs() -> None:
        ont, td = fresh_ontology()
        try:
            root = ont.add(EntityKind.PROJECT, "root")
            mid = ont.add(EntityKind.MODULE, "mid")
            leaf = ont.add(EntityKind.FILE, "leaf")
            ont.link(RelationKind.CONTAINS, root.id, mid.id)
            ont.link(RelationKind.CONTAINS, mid.id, leaf.id)
            reached = ont.traverse(root.id, relation_kinds=[RelationKind.CONTAINS], max_depth=5)
            names = {r.name for r in reached}
            assert names == {"mid", "leaf"}
        finally:
            td.cleanup()

    def t_traverse_depth() -> None:
        ont, td = fresh_ontology()
        try:
            root = ont.add(EntityKind.PROJECT, "root")
            a = ont.add(EntityKind.MODULE, "a")
            b = ont.add(EntityKind.FILE, "b")
            ont.link(RelationKind.CONTAINS, root.id, a.id)
            ont.link(RelationKind.CONTAINS, a.id, b.id)
            only_a = ont.traverse(root.id, max_depth=1)
            names = {r.name for r in only_a}
            assert names == {"a"}, names
        finally:
            td.cleanup()

    def t_traverse_cycle_safe() -> None:
        ont, td = fresh_ontology()
        try:
            a = ont.add(EntityKind.MODULE, "a")
            b = ont.add(EntityKind.MODULE, "b")
            ont.link(RelationKind.DEPENDS_ON, a.id, b.id)
            ont.link(RelationKind.DEPENDS_ON, b.id, a.id)
            reached = ont.traverse(a.id, max_depth=10)
            names = {r.name for r in reached}
            assert names == {"b"}  # a not re-visited
        finally:
            td.cleanup()

    check("traverse BFS", t_traverse_bfs)
    check("traverse respects depth", t_traverse_depth)
    check("traverse cycle-safe", t_traverse_cycle_safe)

    # ---- find ----
    def t_find_filters() -> None:
        ont, td = fresh_ontology()
        try:
            ont.add(EntityKind.REQUIREMENT, "User login")
            ont.add(EntityKind.REQUIREMENT, "User logout")
            ont.add(EntityKind.BUG, "Login bug")
            ont.add(EntityKind.BUG, "Other bug", tags=["critical"])
            assert len(ont.find(kind=EntityKind.BUG)) == 2
            assert len(ont.find(kind=EntityKind.BUG, tag="critical")) == 1
            assert len(ont.find(name_like="Login")) == 1  # case-sensitive: only "Login bug"
            assert len(ont.find(name_like="login")) == 1  # case-sensitive: only "User login" (not "Login bug")
            assert len(ont.find(kind=EntityKind.BUG, limit=1)) == 1
        finally:
            td.cleanup()

    check("find filters (kind/tag/name_like/limit)", t_find_filters)

    # ---- count / stats ----
    def t_count_stats() -> None:
        ont, td = fresh_ontology()
        try:
            ont.add(EntityKind.REQUIREMENT, "R1")
            ont.add(EntityKind.REQUIREMENT, "R2")
            ont.add(EntityKind.BUG, "B1")
            assert ont.count() == 3
            assert ont.count(kind=EntityKind.REQUIREMENT) == 2
            st = ont.stats()
            assert st["entities_total"] == 3
            assert st["entities_by_kind"]["requirement"] == 2
        finally:
            td.cleanup()

    check("count + stats", t_count_stats)

    # ---- provenance / tags / attributes ----
    def t_provenance_roundtrip() -> None:
        ont, td = fresh_ontology()
        try:
            prov = Provenance(
                source="user:alice",
                source_type=ProvenanceType.USER,
                reference="chat://msg/42",
                confidence=Confidence.HIGH,
                notes="stated explicitly",
            )
            e = ont.add(EntityKind.REQUIREMENT, "R", provenance=prov)
            got = ont.get(e.id)
            assert got is not None
            assert got.provenance.source == "user:alice"
            assert got.provenance.source_type is ProvenanceType.USER
            assert got.provenance.confidence is Confidence.HIGH
            assert got.provenance.reference == "chat://msg/42"
        finally:
            td.cleanup()

    def t_tags_roundtrip() -> None:
        ont, td = fresh_ontology()
        try:
            e = ont.add(EntityKind.REQUIREMENT, "R", tags=["a", "b", "a"])
            got = ont.get(e.id)
            assert got is not None
            assert got.tags == ["a", "b"]  # deduped + sorted
        finally:
            td.cleanup()

    def t_attributes_json() -> None:
        ont, td = fresh_ontology()
        try:
            attrs = {"nested": {"a": [1, 2, 3]}, "unicode": "üñïçø∂é", "n": 42}
            e = ont.add(EntityKind.CONSTRAINT, "C", attributes=attrs)
            got = ont.get(e.id)
            assert got is not None
            assert got.attributes == attrs
        finally:
            td.cleanup()

    check("provenance roundtrip", t_provenance_roundtrip)
    check("tags dedupe + sort", t_tags_roundtrip)
    check("attributes JSON roundtrip (incl. unicode)", t_attributes_json)

    # ---- integrity ----
    def t_integrity_clean() -> None:
        ont, td = fresh_ontology()
        try:
            a = ont.add(EntityKind.PROJECT, "A")
            b = ont.add(EntityKind.REQUIREMENT, "B")
            ont.link(RelationKind.CONTAINS, a.id, b.id)
            rep = ont.verify_integrity()
            assert rep["ok"] is True
            assert rep["issue_count"] == 0
        finally:
            td.cleanup()

    check("integrity check clean", t_integrity_clean)

    # ---- migrations idempotent ----
    def t_migrations_idempotent() -> None:
        ont, td = fresh_ontology()
        try:
            ont.initialize()
            ont.initialize()  # should be a no-op
            assert ont.count() == 0
        finally:
            td.cleanup()

    check("ontology re-init idempotent", t_migrations_idempotent)

    # ---- health ----
    def t_health() -> None:
        ont, td = fresh_ontology()
        try:
            ont.add(EntityKind.PROJECT, "p")
            h = ont.health()
            assert h["ok"] is True
            assert h["entities"] == 1
            assert h["relations"] == 0
        finally:
            td.cleanup()

    check("ontology health", t_health)

    # ---- end-to-end demo scenario ----
    def t_e2e_flow() -> None:
        ont, td = fresh_ontology()
        try:
            user_prov = Provenance(
                source="user:demo",
                source_type=ProvenanceType.USER,
                confidence=Confidence.HIGH,
            )
            proj = ont.add(EntityKind.PROJECT, "Task Manager API", provenance=user_prov)
            req1 = ont.add(EntityKind.REQUIREMENT, "CRUD tasks",
                           attributes={"priority": "must"}, provenance=user_prov)
            req2 = ont.add(EntityKind.REQUIREMENT, "Auth",
                           attributes={"priority": "must"}, provenance=user_prov)
            arch = ont.add(EntityKind.ARCHITECTURE, "Layered",
                           attributes={"style": "3-tier"},
                           provenance=Provenance(
                               source="agent:architect",
                               source_type=ProvenanceType.AGENT,
                               confidence=Confidence.MEDIUM,
                           ))
            dec = ont.add(EntityKind.DECISION, "Use FastAPI",
                          attributes={"rationale": "async + pydantic"})
            alt = ont.add(EntityKind.ALTERNATIVE, "Use Flask")

            ont.link(RelationKind.CONTAINS, proj.id, req1.id)
            ont.link(RelationKind.CONTAINS, proj.id, req2.id)
            ont.link(RelationKind.CONTAINS, proj.id, arch.id)
            ont.link(RelationKind.SUPPORTS, dec.id, arch.id)
            ont.link(RelationKind.CONTRADICTS, alt.id, dec.id)

            # Query: everything in the project
            members = ont.traverse(proj.id, relation_kinds=[RelationKind.CONTAINS])
            assert {m.name for m in members} == {"CRUD tasks", "Auth", "Layered"}

            # Query: what supports the chosen architecture?
            nb = ont.neighbors(arch.id, kind=RelationKind.SUPPORTS, direction="in")
            assert len(nb) == 1
            assert nb[0].node.name == "Use FastAPI"

            # stats
            st = ont.stats()
            assert st["entities_total"] == 6  # proj, req1, req2, arch, dec, alt
            assert st["relations_total"] == 5
        finally:
            td.cleanup()

    check("e2e: project → requirements → architecture → decisions", t_e2e_flow)

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
    print("SE Brain C02 — Software Ontology Engine")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                ont = Ontology(app.storage)
                print("\n[1] Ontology initialized")
                print(f"    health: {ont.health()}")

                print("\n[2] Adding entities…")
                user_prov = Provenance(
                    source="user:demo",
                    source_type=ProvenanceType.USER,
                    confidence=Confidence.HIGH,
                )
                proj = ont.add(EntityKind.PROJECT, "Task API", provenance=user_prov)
                r1 = ont.add(EntityKind.REQUIREMENT, "CRUD tasks",
                             attributes={"priority": "must"}, provenance=user_prov)
                r2 = ont.add(EntityKind.REQUIREMENT, "Auth",
                             attributes={"priority": "must"}, provenance=user_prov)
                dec = ont.add(EntityKind.DECISION, "Use FastAPI",
                              attributes={"rationale": "async + types"})
                arch = ont.add(EntityKind.ARCHITECTURE, "3-tier")

                print("\n[3] Linking…")
                ont.link(RelationKind.CONTAINS, proj.id, r1.id)
                ont.link(RelationKind.CONTAINS, proj.id, r2.id)
                ont.link(RelationKind.CONTAINS, proj.id, arch.id)
                ont.link(RelationKind.SUPPORTS, dec.id, arch.id)

                print("\n[4] Stats:")
                for k, v in sorted(ont.stats().items()):
                    print(f"    {k}: {v}")

                print("\n[5] Traverse from project (contains):")
                for e in ont.traverse(proj.id, relation_kinds=[RelationKind.CONTAINS]):
                    print(f"    - [{e.kind.value}] {e.name}")

                print("\n[6] Integrity:", ont.verify_integrity())

                print("\n[7] Superseding decision…")
                new_dec = ont.add(EntityKind.DECISION, "Use FastAPI + SQLModel")
                ont.supersede(dec.id, new_dec.id)
                print(f"    old status: {ont.get(dec.id).status}")
                print(f"    superseded_by: {ont.get(dec.id).superseded_by}")
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
