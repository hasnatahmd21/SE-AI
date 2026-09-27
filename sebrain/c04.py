"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C04 — PERSISTENT MEMORY (Single-File Complete Implementation)
================================================================================

Depends on C01 (`sebrain/c01.py`), C02 (`sebrain/c02.py`), C03 (`sebrain/c03.py`).

Provides six distinct memory kinds with a shared, versioned, provenance-aware
storage engine:

    WORKING    — current task state (task-scoped, optional TTL)
    PROJECT    — project facts (architecture, requirements, decisions history)
    LONG_TERM  — reusable knowledge (global)
    EXPERIENCE — past verified outcomes (project or global)
    DECISION   — why a decision was made (rationale + alternatives)
    FAILURE    — what failed and why (root cause + fix attempts)

Lifecycle:
    create → validate → store → retrieve → update → supersede → archive → forget

Invariants honored:
  - Historical facts are NEVER silently mutated. Every update creates a new
    version; the old version is marked `superseded` and PRESERVED.
  - `forget` is the ONLY destructive op, and it writes a tombstone (audit row).
  - Working memory supports TTL; expired entries are marked, not deleted.
  - Every entry carries Provenance (C02) + Confidence (C01).
  - Kind↔Scope combos are validated (`WORKING` is task-only, etc.).
  - No external LLM. Deterministic. SQLite-backed.

Run as script:
    python -m sebrain.c04            # demo
    python -m sebrain.c04 --test     # self-tests
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
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

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
from sebrain.c03 import C03_MIGRATIONS


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ════════════════════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════════════════════
class MemoryKind(str, Enum):
    WORKING = "working"
    PROJECT = "project"
    LONG_TERM = "long_term"
    EXPERIENCE = "experience"
    DECISION = "decision"
    FAILURE = "failure"


class MemoryScope(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"
    TASK = "task"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    EXPIRED = "expired"
    FORGOTTEN = "forgotten"     # recorded only in tombstones


# Kind ↔ allowed scope combinations
_ALLOWED_SCOPES: dict[MemoryKind, set[MemoryScope]] = {
    MemoryKind.WORKING: {MemoryScope.TASK},
    MemoryKind.PROJECT: {MemoryScope.PROJECT},
    MemoryKind.LONG_TERM: {MemoryScope.GLOBAL},
    MemoryKind.EXPERIENCE: {MemoryScope.PROJECT, MemoryScope.GLOBAL},
    MemoryKind.DECISION: {MemoryScope.PROJECT, MemoryScope.GLOBAL},
    MemoryKind.FAILURE: {MemoryScope.PROJECT, MemoryScope.GLOBAL},
}


# ════════════════════════════════════════════════════════════════════════════
# MemoryEntry
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class MemoryEntry:
    kind: MemoryKind
    key: str
    content: dict[str, Any]
    id: str = field(default_factory=_new_id)
    scope_type: MemoryScope = MemoryScope.GLOBAL
    scope_id: str | None = None
    note: str = ""
    tags: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    supersedes_id: str | None = None
    superseded_by: str | None = None
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    expires_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "key": self.key,
            "content": self.content,
            "scope_type": self.scope_type.value,
            "scope_id": self.scope_id,
            "note": self.note,
            "tags": list(self.tags),
            "attributes": self.attributes,
            "provenance": self.provenance.to_dict(),
            "status": self.status.value,
            "version": self.version,
            "supersedes_id": self.supersedes_id,
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }

    def is_expired(self, *, at: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        when = at or datetime.now(timezone.utc)
        return _parse_iso(self.expires_at) <= when


# ════════════════════════════════════════════════════════════════════════════
# Migration 004
# ════════════════════════════════════════════════════════════════════════════
def _migration_004_memory(storage: SQLiteStorage) -> None:
    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            id             TEXT PRIMARY KEY,
            kind           TEXT NOT NULL,
            scope_type     TEXT NOT NULL,
            scope_id       TEXT,
            key            TEXT NOT NULL,
            content        TEXT NOT NULL DEFAULT '{}',
            note           TEXT NOT NULL DEFAULT '',
            tags           TEXT NOT NULL DEFAULT '[]',
            attributes     TEXT NOT NULL DEFAULT '{}',
            prov_source    TEXT NOT NULL,
            prov_type      TEXT NOT NULL,
            prov_ref       TEXT,
            prov_conf      TEXT NOT NULL,
            prov_notes     TEXT NOT NULL DEFAULT '',
            status         TEXT NOT NULL DEFAULT 'active',
            version        INTEGER NOT NULL DEFAULT 1,
            supersedes_id  TEXT,
            superseded_by  TEXT,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            expires_at     TEXT,
            FOREIGN KEY(supersedes_id) REFERENCES memory_entries(id) ON DELETE SET NULL,
            FOREIGN KEY(superseded_by) REFERENCES memory_entries(id) ON DELETE SET NULL
        );
        """
    )
    storage.execute("CREATE INDEX IF NOT EXISTS idx_mem_kind    ON memory_entries(kind);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_mem_scope   ON memory_entries(scope_type, scope_id);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_mem_key     ON memory_entries(key);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_mem_status  ON memory_entries(status);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_mem_conf    ON memory_entries(prov_conf);")
    storage.execute("CREATE INDEX IF NOT EXISTS idx_mem_expires ON memory_entries(expires_at);")

    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_tombstones (
            id           TEXT PRIMARY KEY,
            kind         TEXT NOT NULL,
            scope_type   TEXT NOT NULL,
            scope_id     TEXT,
            key          TEXT NOT NULL,
            reason       TEXT NOT NULL DEFAULT '',
            forgotten_at TEXT NOT NULL
        );
        """
    )
    storage.execute("CREATE INDEX IF NOT EXISTS idx_tomb_kind ON memory_tombstones(kind);")


C04_MIGRATIONS: list[Migration] = [
    Migration(version=4, name="memory", up=_migration_004_memory),
]

_ALL_MIGRATIONS: list[Migration] = sorted(
    C01_MIGRATIONS + C02_MIGRATIONS + C03_MIGRATIONS + C04_MIGRATIONS,
    key=lambda m: m.version,
)


# ════════════════════════════════════════════════════════════════════════════
# MemoryStore
# ════════════════════════════════════════════════════════════════════════════
class MemoryStore:
    """Authoritative persistent memory engine.

    All writes are transactional. Updates create new versions and preserve
    history. Only `forget()` deletes, and it always writes a tombstone.
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
            active = self._scalar(
                "SELECT COUNT(*) AS c FROM memory_entries WHERE status='active';"
            )
            all_ = self._scalar("SELECT COUNT(*) AS c FROM memory_entries;")
            tombs = self._scalar("SELECT COUNT(*) AS c FROM memory_tombstones;")
            return {
                "component": "memory_store",
                "ok": True,
                "active": active,
                "total": all_,
                "tombstones": tombs,
            }
        except Exception as exc:
            return {"component": "memory_store", "ok": False, "error": str(exc)}

    # ══════════════════════════════════════════════════════════════════════
    # CREATE
    # ══════════════════════════════════════════════════════════════════════
    def create(
        self,
        kind: MemoryKind | str,
        key: str,
        content: dict[str, Any] | None = None,
        *,
        scope_type: MemoryScope | str = MemoryScope.GLOBAL,
        scope_id: str | None = None,
        note: str = "",
        tags: Iterable[str] | None = None,
        attributes: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
        ttl_seconds: int | None = None,
        expires_at: str | None = None,
    ) -> MemoryEntry:
        """Create a fresh entry. Errors if an active entry exists for the
        same (kind, scope, key)."""
        kind_e = self._coerce_kind(kind)
        scope_e = self._coerce_scope(scope_type)
        self._validate_scope(kind_e, scope_e, scope_id)
        self._validate_key(key)

        current = self.get_current(kind_e, key, scope_type=scope_e, scope_id=scope_id)
        if current is not None and current.status is MemoryStatus.ACTIVE:
            raise ValidationError(
                f"active entry already exists for kind={kind_e.value} "
                f"key={key!r} scope={scope_e.value}:{scope_id}"
            )
        return self._append_version(
            kind=kind_e, key=key, content=dict(content or {}),
            scope_type=scope_e, scope_id=scope_id,
            note=note, tags=tags, attributes=attributes,
            provenance=provenance, ttl_seconds=ttl_seconds, expires_at=expires_at,
        )

    # ══════════════════════════════════════════════════════════════════════
    # UPDATE (creates new version, preserves old)
    # ══════════════════════════════════════════════════════════════════════
    def update(
        self,
        entry_id: str,
        *,
        content: dict[str, Any] | None = None,
        note: str | None = None,
        tags: Iterable[str] | None = None,
        attributes: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
        ttl_seconds: int | None = None,
        expires_at: str | None = None,
    ) -> MemoryEntry:
        """Create a new version. Old version marked `superseded` — never
        mutated or deleted."""
        current = self.get(entry_id)
        if current is None:
            raise ValidationError(f"memory entry not found: {entry_id}")
        if current.status is not MemoryStatus.ACTIVE:
            raise ValidationError(
                f"cannot update entry with status={current.status.value}"
            )

        new_content = dict(current.content)
        if content is not None:
            new_content.update(content)

        new_note = note if note is not None else current.note
        new_tags = sorted({str(t) for t in tags}) if tags is not None else current.tags
        new_attrs = dict(current.attributes)
        if attributes is not None:
            new_attrs.update(attributes)

        new_prov = provenance if provenance is not None else current.provenance

        return self._append_version(
            kind=current.kind, key=current.key, content=new_content,
            scope_type=current.scope_type, scope_id=current.scope_id,
            note=new_note, tags=new_tags, attributes=new_attrs,
            provenance=new_prov,
            ttl_seconds=ttl_seconds, expires_at=expires_at,
            _supersedes=current,
        )

    def upsert(
        self,
        kind: MemoryKind | str,
        key: str,
        content: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> MemoryEntry:
        """Create if no active entry exists, else update."""
        kind_e = self._coerce_kind(kind)
        scope_e = self._coerce_scope(kwargs.get("scope_type", MemoryScope.GLOBAL))
        scope_id = kwargs.get("scope_id")
        self._validate_scope(kind_e, scope_e, scope_id)
        self._validate_key(key)
        current = self.get_current(kind_e, key, scope_type=scope_e, scope_id=scope_id)
        if current is not None and current.status is MemoryStatus.ACTIVE:
            # update() derives scope from the existing entry and does not
            # accept scope_type/scope_id (they aren't changeable via
            # update) — strip them so we only forward args update() knows.
            _UPDATE_KWARGS = {
                "note", "tags", "attributes", "provenance",
                "ttl_seconds", "expires_at",
            }
            update_kwargs = {k: v for k, v in kwargs.items() if k in _UPDATE_KWARGS}
            return self.update(current.id, content=content, **update_kwargs)
        return self.create(kind_e, key, content, **kwargs)

    # ══════════════════════════════════════════════════════════════════════
    # SUPERSEDE (explicit cross-key replacement)
    # ══════════════════════════════════════════════════════════════════════
    def supersede(self, old_id: str, new_id: str) -> tuple[MemoryEntry, MemoryEntry]:
        """Mark `old` as superseded by `new`. Both must exist; `new` must be
        a fresh entry (no prior history) and have the same kind."""
        if old_id == new_id:
            raise ValidationError("cannot supersede an entry with itself")
        old = self.get(old_id)
        new = self.get(new_id)
        if old is None:
            raise ValidationError(f"memory entry not found: {old_id}")
        if new is None:
            raise ValidationError(f"memory entry not found: {new_id}")
        if old.kind is not new.kind:
            raise ValidationError(
                f"kind mismatch: {old.kind.value} vs {new.kind.value}"
            )
        if old.status is MemoryStatus.ARCHIVED:
            raise ValidationError("cannot supersede an archived entry")
        if old.superseded_by is not None:
            raise ValidationError(f"entry already superseded: {old_id}")
        if new.supersedes_id is not None:
            raise ValidationError(
                "replacement entry must not already have its own history"
            )

        ts = now_iso()
        with self.storage.transaction():
            self.storage.execute(
                "UPDATE memory_entries SET status=?, superseded_by=?, updated_at=? "
                "WHERE id=?;",
                (MemoryStatus.SUPERSEDED.value, new.id, ts, old.id),
            )
            self.storage.execute(
                "UPDATE memory_entries SET supersedes_id=?, updated_at=? WHERE id=?;",
                (old.id, ts, new.id),
            )
        old.status = MemoryStatus.SUPERSEDED
        old.superseded_by = new.id
        old.updated_at = ts
        new.supersedes_id = old.id
        new.updated_at = ts
        return old, new

    # ══════════════════════════════════════════════════════════════════════
    # ARCHIVE
    # ══════════════════════════════════════════════════════════════════════
    def archive(self, entry_id: str) -> MemoryEntry:
        current = self.get(entry_id)
        if current is None:
            raise ValidationError(f"memory entry not found: {entry_id}")
        if current.status is MemoryStatus.ARCHIVED:
            return current
        if current.status is MemoryStatus.SUPERSEDED:
            raise ValidationError("cannot archive a superseded entry")
        ts = now_iso()
        self.storage.execute(
            "UPDATE memory_entries SET status=?, updated_at=? WHERE id=?;",
            (MemoryStatus.ARCHIVED.value, ts, entry_id),
        )
        current.status = MemoryStatus.ARCHIVED
        current.updated_at = ts
        return current

    # ══════════════════════════════════════════════════════════════════════
    # FORGET (hard delete + tombstone)
    # ══════════════════════════════════════════════════════════════════════
    def forget(self, entry_id: str, *, reason: str = "") -> bool:
        """Hard delete. Writes a tombstone for audit. Only op that removes."""
        row = self.storage.query_one(
            "SELECT * FROM memory_entries WHERE id=?;", (entry_id,)
        )
        if row is None:
            return False
        with self.storage.transaction():
            self.storage.execute(
                """
                INSERT INTO memory_tombstones(id, kind, scope_type, scope_id,
                                              key, reason, forgotten_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    row["id"], row["kind"], row["scope_type"], row["scope_id"],
                    row["key"], reason, now_iso(),
                ),
            )
            self.storage.execute(
                "DELETE FROM memory_entries WHERE id=?;", (entry_id,)
            )
        return True

    def tombstones(
        self, *, kind: MemoryKind | str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memory_tombstones WHERE 1=1"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(kind).value)
        sql += " ORDER BY forgotten_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self.storage.query(sql + ";", params)

    # ══════════════════════════════════════════════════════════════════════
    # RETRIEVE
    # ══════════════════════════════════════════════════════════════════════
    def get(self, entry_id: str) -> MemoryEntry | None:
        row = self.storage.query_one(
            "SELECT * FROM memory_entries WHERE id=?;", (entry_id,)
        )
        if row is None:
            return None
        return _row_to_entry(row)

    def get_current(
        self,
        kind: MemoryKind | str,
        key: str,
        *,
        scope_type: MemoryScope | str = MemoryScope.GLOBAL,
        scope_id: str | None = None,
    ) -> MemoryEntry | None:
        """Latest active version for (kind, scope, key). Returns None if
        none active. Expired entries are auto-marked `expired` and returned
        with that status."""
        kind_e = self._coerce_kind(kind)
        scope_e = self._coerce_scope(scope_type)
        row = self._fetch_current_row(kind_e, scope_e, scope_id, key)
        if row is None:
            return None
        entry = _row_to_entry(row)
        if entry.status is MemoryStatus.ACTIVE and entry.is_expired():
            self._mark_expired(entry.id)
            entry.status = MemoryStatus.EXPIRED
        return entry

    def history(
        self,
        kind: MemoryKind | str,
        key: str,
        *,
        scope_type: MemoryScope | str = MemoryScope.GLOBAL,
        scope_id: str | None = None,
    ) -> list[MemoryEntry]:
        """All versions of a key, oldest first. Includes superseded/archived."""
        kind_e = self._coerce_kind(kind)
        scope_e = self._coerce_scope(scope_type)
        params: list[Any] = [kind_e.value, scope_e.value, key]
        sql = "SELECT * FROM memory_entries WHERE kind=? AND scope_type=? AND key=?"
        if scope_id is None:
            sql += " AND scope_id IS NULL"
        else:
            sql += " AND scope_id=?"
            params.append(scope_id)
        sql += " ORDER BY version ASC;"
        return [_row_to_entry(r) for r in self.storage.query(sql, params)]

    def find(
        self,
        *,
        kind: MemoryKind | str | None = None,
        scope_type: MemoryScope | str | None = None,
        scope_id: str | None = None,
        status: MemoryStatus | str | None = None,
        tag: str | None = None,
        key_like: str | None = None,
        confidence: Confidence | str | None = None,
        min_confidence: Confidence | str | None = None,
        include_superseded: bool = False,
        include_archived: bool = False,
        include_expired: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        sql = "SELECT * FROM memory_entries WHERE 1=1"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(kind).value)
        if scope_type is not None:
            sql += " AND scope_type=?"
            params.append(self._coerce_scope(scope_type).value)
        if scope_id is not None:
            sql += " AND scope_id=?"
            params.append(scope_id)
        if status is not None:
            if isinstance(status, MemoryStatus):
                sql += " AND status=?"
                params.append(status.value)
            else:
                sql += " AND status=?"
                params.append(str(status))
        else:
            allowed = ["active"]
            if include_superseded:
                allowed.append("superseded")
            if include_archived:
                allowed.append("archived")
            if include_expired:
                allowed.append("expired")
            sql += " AND status IN (" + ",".join("?" * len(allowed)) + ")"
            params.extend(allowed)
        if key_like is not None:
            sql += " AND key LIKE ?"
            params.append(f"%{key_like}%")
        if confidence is not None:
            sql += " AND prov_conf=?"
            params.append(self._coerce_conf(confidence).value)
        if min_confidence is not None:
            rank = _CONF_RANK[self._coerce_conf(min_confidence)]
            allowed = [c.value for c, r in _CONF_RANK.items() if r >= rank]
            sql += " AND prov_conf IN (" + ",".join("?" * len(allowed)) + ")"
            params.extend(allowed)
        sql += " ORDER BY updated_at DESC"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])
        rows = self.storage.query(sql + ";", params)
        entries = [_row_to_entry(r) for r in rows]
        if tag is not None:
            entries = [e for e in entries if tag in e.tags]
        return entries

    def count(
        self,
        *,
        kind: MemoryKind | str | None = None,
        status: MemoryStatus | str | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) AS c FROM memory_entries WHERE 1=1"
        params: list[Any] = []
        if kind is not None:
            sql += " AND kind=?"
            params.append(self._coerce_kind(kind).value)
        if status is not None:
            sql += " AND status=?"
            params.append(status.value if isinstance(status, MemoryStatus) else str(status))
        return self._scalar(sql + ";", params)

    def stats(self) -> dict[str, Any]:
        by_kind = self.storage.query(
            "SELECT kind, COUNT(*) AS c FROM memory_entries GROUP BY kind;"
        )
        by_status = self.storage.query(
            "SELECT status, COUNT(*) AS c FROM memory_entries GROUP BY status;"
        )
        return {
            "total": self._scalar("SELECT COUNT(*) AS c FROM memory_entries;"),
            "by_kind": {r["kind"]: int(r["c"]) for r in by_kind},
            "by_status": {r["status"]: int(r["c"]) for r in by_status},
            "tombstones": self._scalar("SELECT COUNT(*) AS c FROM memory_tombstones;"),
        }

    # ══════════════════════════════════════════════════════════════════════
    # WORKING MEMORY (TTL-aware)
    # ══════════════════════════════════════════════════════════════════════
    def set_working(
        self,
        task_id: str,
        key: str,
        content: dict[str, Any] | None = None,
        *,
        ttl_seconds: int | None = None,
        note: str = "",
        tags: Iterable[str] | None = None,
        provenance: Provenance | None = None,
    ) -> MemoryEntry:
        return self.upsert(
            MemoryKind.WORKING, key, content,
            scope_type=MemoryScope.TASK, scope_id=task_id,
            note=note, tags=tags, provenance=provenance,
            ttl_seconds=ttl_seconds,
        )

    def get_working(self, task_id: str, key: str) -> MemoryEntry | None:
        return self.get_current(
            MemoryKind.WORKING, key,
            scope_type=MemoryScope.TASK, scope_id=task_id,
        )

    def list_working(self, task_id: str) -> list[MemoryEntry]:
        return self.find(
            kind=MemoryKind.WORKING,
            scope_type=MemoryScope.TASK, scope_id=task_id,
            status=MemoryStatus.ACTIVE,
        )

    def clear_working(self, task_id: str) -> int:
        """Archive all active working-memory entries for a task."""
        entries = self.list_working(task_id)
        n = 0
        for e in entries:
            self.archive(e.id)
            n += 1
        return n

    def cleanup_expired(self) -> int:
        """Mark all expired-active entries as `expired`."""
        now = now_iso()
        row = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM memory_entries "
            "WHERE status='active' AND expires_at IS NOT NULL AND expires_at <= ?;",
            (now,),
        )
        n = int(row["c"]) if row else 0
        if n:
            self.storage.execute(
                "UPDATE memory_entries SET status=?, updated_at=? "
                "WHERE status='active' AND expires_at IS NOT NULL AND expires_at <= ?;",
                (MemoryStatus.EXPIRED.value, now, now),
            )
        return n

    # ══════════════════════════════════════════════════════════════════════
    # DOMAIN HELPERS
    # ══════════════════════════════════════════════════════════════════════
    def record_experience(
        self,
        key: str,
        *,
        problem: str,
        approach: str,
        outcome: str,
        lesson: str,
        scope_type: MemoryScope | str = MemoryScope.PROJECT,
        scope_id: str | None = None,
        tags: Iterable[str] | None = None,
        provenance: Provenance | None = None,
        confidence: Confidence = Confidence.MEDIUM,
    ) -> MemoryEntry:
        prov = provenance or Provenance(
            source="agent:experience", source_type=ProvenanceType.AGENT,
            confidence=confidence,
        )
        return self.upsert(
            MemoryKind.EXPERIENCE, key,
            {"problem": problem, "approach": approach,
             "outcome": outcome, "lesson": lesson},
            scope_type=scope_type, scope_id=scope_id,
            tags=tags, provenance=prov,
        )

    def record_decision(
        self,
        key: str,
        *,
        decision: str,
        rationale: str,
        alternatives: list[str] | None = None,
        scope_type: MemoryScope | str = MemoryScope.PROJECT,
        scope_id: str | None = None,
        tags: Iterable[str] | None = None,
        provenance: Provenance | None = None,
        confidence: Confidence = Confidence.MEDIUM,
    ) -> MemoryEntry:
        prov = provenance or Provenance(
            source="agent:decision", source_type=ProvenanceType.AGENT,
            confidence=confidence,
        )
        return self.upsert(
            MemoryKind.DECISION, key,
            {"decision": decision, "rationale": rationale,
             "alternatives": list(alternatives or [])},
            scope_type=scope_type, scope_id=scope_id,
            tags=tags, provenance=prov,
        )

    def record_failure(
        self,
        key: str,
        *,
        what: str,
        root_cause: str,
        fix: str | None = None,
        scope_type: MemoryScope | str = MemoryScope.PROJECT,
        scope_id: str | None = None,
        tags: Iterable[str] | None = None,
        provenance: Provenance | None = None,
        confidence: Confidence = Confidence.MEDIUM,
    ) -> MemoryEntry:
        prov = provenance or Provenance(
            source="agent:debugger", source_type=ProvenanceType.AGENT,
            confidence=confidence,
        )
        return self.upsert(
            MemoryKind.FAILURE, key,
            {"what": what, "root_cause": root_cause, "fix": fix},
            scope_type=scope_type, scope_id=scope_id,
            tags=tags, provenance=prov,
        )

    # ══════════════════════════════════════════════════════════════════════
    # INTEGRITY
    # ══════════════════════════════════════════════════════════════════════
    def verify_integrity(self) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []

        # Dangling supersede pointers
        dangling = self.storage.query(
            """
            SELECT e.id, e.superseded_by, e.supersedes_id FROM memory_entries e
            LEFT JOIN memory_entries n1 ON e.superseded_by = n1.id
            LEFT JOIN memory_entries n2 ON e.supersedes_id = n2.id
            WHERE (e.superseded_by IS NOT NULL AND n1.id IS NULL)
               OR (e.supersedes_id IS NOT NULL AND n2.id IS NULL);
            """
        )
        for r in dangling:
            issues.append({"type": "dangling_pointer", "id": r["id"]})

        # Multiple active versions of the same (kind, scope, key)
        dup = self.storage.query(
            """
            SELECT kind, scope_type, scope_id, key, COUNT(*) AS c
            FROM memory_entries
            WHERE status='active'
            GROUP BY kind, scope_type, scope_id, key
            HAVING c > 1;
            """
        )
        for r in dup:
            issues.append({
                "type": "multiple_active",
                "kind": r["kind"], "key": r["key"],
                "scope_type": r["scope_type"], "scope_id": r["scope_id"],
                "count": int(r["c"]),
            })

        # Bad JSON
        for row in self.storage.query(
            "SELECT id, content, tags, attributes FROM memory_entries;"
        ):
            for col in ("content", "tags", "attributes"):
                try:
                    json.loads(row[col] or "null")
                except json.JSONDecodeError as exc:
                    issues.append({
                        "type": "bad_json", "id": row["id"],
                        "field": col, "error": str(exc),
                    })

        # Status consistency
        bad = self.storage.query(
            "SELECT id, status, superseded_by FROM memory_entries "
            "WHERE (status='superseded' AND superseded_by IS NULL) "
            "   OR (status='active' AND superseded_by IS NOT NULL);"
        )
        for r in bad:
            issues.append({
                "type": "status_inconsistent", "id": r["id"], "status": r["status"],
            })

        return {"ok": not issues, "issues": issues, "issue_count": len(issues)}

    # ══════════════════════════════════════════════════════════════════════
    # INTERNALS
    # ══════════════════════════════════════════════════════════════════════
    def _append_version(
        self,
        *,
        kind: MemoryKind,
        key: str,
        content: dict[str, Any],
        scope_type: MemoryScope,
        scope_id: str | None,
        note: str = "",
        tags: Iterable[str] | None = None,
        attributes: dict[str, Any] | None = None,
        provenance: Provenance | None = None,
        ttl_seconds: int | None = None,
        expires_at: str | None = None,
        _supersedes: MemoryEntry | None = None,
    ) -> MemoryEntry:
        if _supersedes is None:
            current_row = self._fetch_current_row(kind, scope_type, scope_id, key)
            if current_row is not None:
                _supersedes = _row_to_entry(current_row)

        version = 1
        supersedes_id: str | None = None
        if _supersedes is not None:
            version = _supersedes.version + 1
            supersedes_id = _supersedes.id

        if expires_at is None and ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValidationError("ttl_seconds must be > 0")
            expires_at = (datetime.now(timezone.utc)
                          + timedelta(seconds=ttl_seconds)).isoformat(timespec="microseconds")

        entry = MemoryEntry(
            kind=kind, key=key, content=content,
            scope_type=scope_type, scope_id=scope_id,
            note=note,
            tags=sorted({str(t) for t in (tags or []) if str(t).strip()}),
            attributes=dict(attributes or {}),
            provenance=provenance or Provenance(),
            status=MemoryStatus.ACTIVE,
            version=version,
            supersedes_id=supersedes_id,
            expires_at=expires_at,
        )

        with self.storage.transaction():
            # Insert the new row FIRST: memory_entries.superseded_by has a
            # FOREIGN KEY on memory_entries(id), so pointing the old row at
            # the new row's id before that row exists violates the
            # constraint. Insert-then-update keeps the FK always valid.
            self._insert(entry)
            if _supersedes is not None:
                self.storage.execute(
                    "UPDATE memory_entries SET status=?, superseded_by=?, updated_at=? "
                    "WHERE id=?;",
                    (MemoryStatus.SUPERSEDED.value, entry.id, entry.created_at,
                     _supersedes.id),
                )
        return entry


    def _insert(self, e: MemoryEntry) -> None:
        self.storage.execute(
            """
            INSERT INTO memory_entries
                (id, kind, scope_type, scope_id, key, content, note, tags,
                 attributes,
                 prov_source, prov_type, prov_ref, prov_conf, prov_notes,
                 status, version, supersedes_id, superseded_by,
                 created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            _entry_to_row(e),
        )

    def _mark_expired(self, entry_id: str) -> None:
        self.storage.execute(
            "UPDATE memory_entries SET status=?, updated_at=? WHERE id=?;",
            (MemoryStatus.EXPIRED.value, now_iso(), entry_id),
        )

    def _fetch_current_row(
        self,
        kind: MemoryKind,
        scope_type: MemoryScope,
        scope_id: str | None,
        key: str,
    ) -> dict[str, Any] | None:
        params: list[Any] = [kind.value, scope_type.value, key]
        sql = ("SELECT * FROM memory_entries WHERE kind=? AND scope_type=? AND key=? "
               "AND status IN ('active','expired')")
        if scope_id is None:
            sql += " AND scope_id IS NULL"
        else:
            sql += " AND scope_id=?"
            params.append(scope_id)
        sql += " ORDER BY version DESC LIMIT 1;"
        return self.storage.query_one(sql, params)

    def _scalar(self, sql: str, params: Iterable[Any] = ()) -> int:
        row = self.storage.query_one(sql, list(params))
        return int(row["c"]) if row else 0

    # ---- coercion ----
    @staticmethod
    def _coerce_kind(kind: MemoryKind | str) -> MemoryKind:
        if isinstance(kind, MemoryKind):
            return kind
        try:
            return MemoryKind(kind)
        except ValueError as exc:
            raise ValidationError(f"unknown memory kind: {kind!r}") from exc

    @staticmethod
    def _coerce_scope(scope: MemoryScope | str) -> MemoryScope:
        if isinstance(scope, MemoryScope):
            return scope
        try:
            return MemoryScope(scope)
        except ValueError as exc:
            raise ValidationError(f"unknown memory scope: {scope!r}") from exc

    @staticmethod
    def _coerce_conf(conf: Confidence | str) -> Confidence:
        if isinstance(conf, Confidence):
            return conf
        try:
            return Confidence(conf)
        except ValueError as exc:
            raise ValidationError(f"unknown confidence: {conf!r}") from exc

    @staticmethod
    def _validate_scope(kind: MemoryKind, scope: MemoryScope, scope_id: str | None) -> None:
        if scope not in _ALLOWED_SCOPES[kind]:
            allowed = sorted(s.value for s in _ALLOWED_SCOPES[kind])
            raise ValidationError(
                f"kind {kind.value} does not allow scope {scope.value}; "
                f"allowed: {allowed}"
            )
        if scope is MemoryScope.GLOBAL and scope_id is not None:
            raise ValidationError("global-scoped memory must not have scope_id")
        if scope is not MemoryScope.GLOBAL and not scope_id:
            raise ValidationError(
                f"scope {scope.value} requires a non-empty scope_id"
            )

    @staticmethod
    def _validate_key(key: str) -> None:
        if not key or not key.strip():
            raise ValidationError("memory key must be non-empty")
        if len(key) > 512:
            raise ValidationError("memory key too long (>512 chars)")


# ════════════════════════════════════════════════════════════════════════════
# Confidence ranking (used by find)
# ════════════════════════════════════════════════════════════════════════════
_CONF_RANK: dict[Confidence, int] = {
    Confidence.UNKNOWN: 0,
    Confidence.ASSUMPTION: 1,
    Confidence.LOW: 2,
    Confidence.MEDIUM: 3,
    Confidence.HIGH: 4,
    Confidence.VERIFIED: 5,
}


# ════════════════════════════════════════════════════════════════════════════
# Row ↔ dataclass
# ════════════════════════════════════════════════════════════════════════════
def _entry_to_row(e: MemoryEntry) -> tuple:
    return (
        e.id, e.kind.value, e.scope_type.value, e.scope_id, e.key,
        json.dumps(e.content, ensure_ascii=False, default=str),
        e.note,
        json.dumps(e.tags, ensure_ascii=False),
        json.dumps(e.attributes, ensure_ascii=False, default=str),
        e.provenance.source, e.provenance.source_type.value,
        e.provenance.reference, e.provenance.confidence.value,
        e.provenance.notes,
        e.status.value, e.version, e.supersedes_id, e.superseded_by,
        e.created_at, e.updated_at, e.expires_at,
    )


def _row_to_entry(row: dict[str, Any]) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        kind=MemoryKind(row["kind"]),
        scope_type=MemoryScope(row["scope_type"]),
        scope_id=row["scope_id"],
        key=row["key"],
        content=json.loads(row["content"] or "{}"),
        note=row["note"] or "",
        tags=json.loads(row["tags"] or "[]"),
        attributes=json.loads(row["attributes"] or "{}"),
        provenance=Provenance(
            source=row["prov_source"],
            source_type=ProvenanceType(row["prov_type"]),
            reference=row["prov_ref"],
            confidence=Confidence(row["prov_conf"]),
            notes=row["prov_notes"] or "",
        ),
        status=MemoryStatus(row["status"]),
        version=int(row["version"]),
        supersedes_id=row["supersedes_id"],
        superseded_by=row["superseded_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        expires_at=row["expires_at"],
    )


# ════════════════════════════════════════════════════════════════════════════
# SELF-TESTS
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

    def fresh() -> tuple[MemoryStore, tempfile.TemporaryDirectory]:
        td = tempfile.TemporaryDirectory()
        s = SQLiteStorage(Path(td.name) / "mem.sqlite3")
        s.initialize()
        return MemoryStore(s), td

    print("Running C04 self-tests…")

    # ---- enums ----
    def t_kinds() -> None:
        assert {k.value for k in MemoryKind} == {
            "working", "project", "long_term",
            "experience", "decision", "failure",
        }

    def t_scopes() -> None:
        assert {s.value for s in MemoryScope} == {"global", "project", "task"}

    check("6 memory kinds", t_kinds)
    check("3 memory scopes", t_scopes)

    # ---- create / get ----
    def t_create_get() -> None:
        store, td = fresh()
        try:
            e = store.create(MemoryKind.PROJECT, "architecture", {"style": "3-tier"},
                             scope_type=MemoryScope.PROJECT, scope_id="p1")
            assert e.id
            assert e.version == 1
            assert e.status is MemoryStatus.ACTIVE
            got = store.get(e.id)
            assert got is not None
            assert got.content == {"style": "3-tier"}
        finally:
            td.cleanup()

    def t_create_duplicate_active() -> None:
        store, td = fresh()
        try:
            store.create(MemoryKind.PROJECT, "k", {}, scope_type=MemoryScope.PROJECT, scope_id="p1")
            try:
                store.create(MemoryKind.PROJECT, "k", {}, scope_type=MemoryScope.PROJECT, scope_id="p1")
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("create + get", t_create_get)
    check("create rejects duplicate active", t_create_duplicate_active)

    # ---- scope validation ----
    def t_scope_working_task_only() -> None:
        store, td = fresh()
        try:
            try:
                store.create(MemoryKind.WORKING, "k", {}, scope_type=MemoryScope.PROJECT, scope_id="p")
            except ValidationError:
                pass
            else:
                raise AssertionError("WORKING must be task-scoped")
            store.create(MemoryKind.WORKING, "k", {},
                         scope_type=MemoryScope.TASK, scope_id="t1")
        finally:
            td.cleanup()

    def t_scope_longterm_global_only() -> None:
        store, td = fresh()
        try:
            try:
                store.create(MemoryKind.LONG_TERM, "k", {},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            except ValidationError:
                pass
            else:
                raise AssertionError("LONG_TERM must be global")
            store.create(MemoryKind.LONG_TERM, "k", {})
        finally:
            td.cleanup()

    def t_scope_global_no_id() -> None:
        store, td = fresh()
        try:
            try:
                store.create(MemoryKind.LONG_TERM, "k", {},
                             scope_type=MemoryScope.GLOBAL, scope_id="x")
            except ValidationError:
                return
            raise AssertionError("global scope must not have scope_id")
        finally:
            td.cleanup()

    def t_scope_non_global_requires_id() -> None:
        store, td = fresh()
        try:
            try:
                store.create(MemoryKind.PROJECT, "k", {},
                             scope_type=MemoryScope.PROJECT, scope_id=None)
            except ValidationError:
                return
            raise AssertionError("non-global scope requires scope_id")
        finally:
            td.cleanup()

    check("WORKING scope locked to TASK", t_scope_working_task_only)
    check("LONG_TERM scope locked to GLOBAL", t_scope_longterm_global_only)
    check("GLOBAL scope rejects scope_id", t_scope_global_no_id)
    check("non-GLOBAL scope requires scope_id", t_scope_non_global_requires_id)

    # ---- update preserves history ----
    def t_update_preserves_old() -> None:
        store, td = fresh()
        try:
            e1 = store.create(MemoryKind.PROJECT, "arch", {"layers": 1},
                              scope_type=MemoryScope.PROJECT, scope_id="p")
            e2 = store.update(e1.id, content={"layers": 2})
            assert e2.version == 2
            # old preserved
            old = store.get(e1.id)
            assert old is not None
            assert old.status is MemoryStatus.SUPERSEDED
            assert old.superseded_by == e2.id
            assert old.content == {"layers": 1}
            # new links back
            assert e2.supersedes_id == e1.id
            # current returns new
            cur = store.get_current(MemoryKind.PROJECT, "arch",
                                    scope_type=MemoryScope.PROJECT, scope_id="p")
            assert cur is not None and cur.id == e2.id
        finally:
            td.cleanup()

    def t_history_order() -> None:
        store, td = fresh()
        try:
            e1 = store.create(MemoryKind.PROJECT, "k", {"v": 1},
                              scope_type=MemoryScope.PROJECT, scope_id="p")
            e2 = store.update(e1.id, content={"v": 2})
            e3 = store.update(e2.id, content={"v": 3})
            hist = store.history(MemoryKind.PROJECT, "k",
                                 scope_type=MemoryScope.PROJECT, scope_id="p")
            assert [h.version for h in hist] == [1, 2, 3]
            assert hist[0].content == {"v": 1}
            assert hist[1].content == {"v": 2}
            assert hist[2].content == {"v": 3}
            assert hist[0].status is MemoryStatus.SUPERSEDED
            assert hist[2].status is MemoryStatus.ACTIVE
        finally:
            td.cleanup()

    def t_update_archived_rejected() -> None:
        store, td = fresh()
        try:
            e = store.create(MemoryKind.PROJECT, "k", {},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            store.archive(e.id)
            try:
                store.update(e.id, content={"x": 1})
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("update preserves old version", t_update_preserves_old)
    check("history returns all versions in order", t_history_order)
    check("update rejects archived entry", t_update_archived_rejected)

    # ---- upsert ----
    def t_upsert() -> None:
        store, td = fresh()
        try:
            e1 = store.upsert(MemoryKind.PROJECT, "k", {"v": 1},
                              scope_type=MemoryScope.PROJECT, scope_id="p")
            assert e1.version == 1
            e2 = store.upsert(MemoryKind.PROJECT, "k", {"v": 2},
                              scope_type=MemoryScope.PROJECT, scope_id="p")
            assert e2.version == 2
            assert e2.id != e1.id
        finally:
            td.cleanup()

    check("upsert creates or updates", t_upsert)

    # ---- supersede (cross-key) ----
    def t_supersede() -> None:
        store, td = fresh()
        try:
            old = store.create(MemoryKind.DECISION, "db-choice",
                               {"db": "sqlite"},
                               scope_type=MemoryScope.PROJECT, scope_id="p")
            new = store.create(MemoryKind.DECISION, "db-choice-v2",
                               {"db": "postgres"},
                               scope_type=MemoryScope.PROJECT, scope_id="p")
            old2, new2 = store.supersede(old.id, new.id)
            assert old2.status is MemoryStatus.SUPERSEDED
            assert old2.superseded_by == new.id
            assert new2.supersedes_id == old.id
        finally:
            td.cleanup()

    def t_supersede_kind_mismatch() -> None:
        store, td = fresh()
        try:
            a = store.create(MemoryKind.DECISION, "a", {},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            b = store.create(MemoryKind.FAILURE, "b", {},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            try:
                store.supersede(a.id, b.id)
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    def t_supersede_with_historical_new_rejected() -> None:
        store, td = fresh()
        try:
            a = store.create(MemoryKind.DECISION, "a", {},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            b1 = store.create(MemoryKind.DECISION, "b", {},
                              scope_type=MemoryScope.PROJECT, scope_id="p")
            b2 = store.update(b1.id, content={"x": 1})
            try:
                store.supersede(a.id, b2.id)
            except ValidationError:
                return
            raise AssertionError("expected ValidationError")
        finally:
            td.cleanup()

    check("supersede cross-key", t_supersede)
    check("supersede kind mismatch rejected", t_supersede_kind_mismatch)
    check("supersede requires fresh replacement", t_supersede_with_historical_new_rejected)

    # ---- archive ----
    def t_archive() -> None:
        store, td = fresh()
        try:
            e = store.create(MemoryKind.FAILURE, "f", {},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            a = store.archive(e.id)
            assert a.status is MemoryStatus.ARCHIVED
            assert store.get(e.id).status is MemoryStatus.ARCHIVED  # preserved
            assert store.get_current(MemoryKind.FAILURE, "f",
                                     scope_type=MemoryScope.PROJECT, scope_id="p") is None
        finally:
            td.cleanup()

    check("archive preserves row", t_archive)

    # ---- forget + tombstone ----
    def t_forget_writes_tombstone() -> None:
        store, td = fresh()
        try:
            e = store.create(MemoryKind.FAILURE, "to-forget", {"x": 1},
                             scope_type=MemoryScope.PROJECT, scope_id="p")
            assert store.forget(e.id, reason="user requested") is True
            assert store.get(e.id) is None
            tombs = store.tombstones()
            assert len(tombs) == 1
            assert tombs[0]["id"] == e.id
            assert tombs[0]["reason"] == "user requested"
            assert tombs[0]["key"] == "to-forget"
        finally:
            td.cleanup()

    def t_forget_missing() -> None:
        store, td = fresh()
        try:
            assert store.forget("nope") is False
        finally:
            td.cleanup()

    check("forget writes tombstone", t_forget_writes_tombstone)
    check("forget missing returns False", t_forget_missing)

    # ---- working memory + TTL ----
    def t_working_set_get() -> None:
        store, td = fresh()
        try:
            store.set_working("task-1", "current_step", {"step": 3})
            e = store.get_working("task-1", "current_step")
            assert e is not None
            assert e.content == {"step": 3}
            assert e.scope_type is MemoryScope.TASK
            assert e.scope_id == "task-1"
            # different task — isolated
            assert store.get_working("task-2", "current_step") is None
        finally:
            td.cleanup()

    def t_working_ttl() -> None:
        store, td = fresh()
        try:
            e = store.set_working("t", "k", {"v": 1}, ttl_seconds=3600)
            assert e.expires_at is not None
            assert not e.is_expired()
            # simulate expiration
            from datetime import timedelta
            past = (datetime.now(timezone.utc) - timedelta(seconds=5))\
                .isoformat(timespec="microseconds")
            store.storage.execute(
                "UPDATE memory_entries SET expires_at=? WHERE id=?;", (past, e.id)
            )
            got = store.get_working("t", "k")
            assert got is not None
            assert got.status is MemoryStatus.EXPIRED
            assert store.get_working("t", "k").status is MemoryStatus.EXPIRED
        finally:
            td.cleanup()

    def t_cleanup_expired() -> None:
        store, td = fresh()
        try:
            e = store.set_working("t", "k", {}, ttl_seconds=3600)
            from datetime import timedelta
            past = (datetime.now(timezone.utc) - timedelta(seconds=5))\
                .isoformat(timespec="microseconds")
            store.storage.execute(
                "UPDATE memory_entries SET expires_at=? WHERE id=?;", (past, e.id)
            )
            n = store.cleanup_expired()
            assert n == 1
            row = store.get(e.id)
            assert row is not None and row.status is MemoryStatus.EXPIRED
        finally:
            td.cleanup()

    def t_clear_working() -> None:
        store, td = fresh()
        try:
            store.set_working("task-x", "a", {"v": 1})
            store.set_working("task-x", "b", {"v": 2})
            store.set_working("task-y", "a", {"v": 3})
            n = store.clear_working("task-x")
            assert n == 2
            assert store.list_working("task-x") == []
            assert len(store.list_working("task-y")) == 1
        finally:
            td.cleanup()

    check("working set/get (task-isolated)", t_working_set_get)
    check("working TTL expiration", t_working_ttl)
    check("cleanup_expired marks entries", t_cleanup_expired)
    check("clear_working archives only that task", t_clear_working)

    # ---- domain helpers ----
    def t_record_experience() -> None:
        store, td = fresh()
        try:
            e = store.record_experience(
                "exp-1",
                problem="Slow query",
                approach="Added index",
                outcome="10x faster",
                lesson="Index hot paths",
                scope_id="proj-a",
            )
            assert e.kind is MemoryKind.EXPERIENCE
            assert e.content["problem"] == "Slow query"
            assert e.content["lesson"] == "Index hot paths"
        finally:
            td.cleanup()

    def t_record_decision() -> None:
        store, td = fresh()
        try:
            e = store.record_decision(
                "db",
                decision="Use SQLite",
                rationale="Local-first, no server",
                alternatives=["Postgres", "MySQL"],
                scope_id="proj-a",
            )
            assert e.kind is MemoryKind.DECISION
            assert e.content["alternatives"] == ["Postgres", "MySQL"]
        finally:
            td.cleanup()

    def t_record_failure() -> None:
        store, td = fresh()
        try:
            e = store.record_failure(
                "bug-42",
                what="Null deref on empty list",
                root_cause="Missing guard in loop",
                fix="Added length check",
                scope_id="proj-a",
            )
            assert e.kind is MemoryKind.FAILURE
            assert e.content["root_cause"] == "Missing guard in loop"
        finally:
            td.cleanup()

    check("record_experience", t_record_experience)
    check("record_decision", t_record_decision)
    check("record_failure", t_record_failure)

    # ---- find / count / stats ----
    def t_find_filters() -> None:
        store, td = fresh()
        try:
            store.create(MemoryKind.PROJECT, "a", {}, scope_type=MemoryScope.PROJECT, scope_id="p1")
            store.create(MemoryKind.PROJECT, "b", {}, scope_type=MemoryScope.PROJECT, scope_id="p2")
            store.create(MemoryKind.LONG_TERM, "c", {})
            assert len(store.find(kind=MemoryKind.PROJECT)) == 2
            assert len(store.find(kind=MemoryKind.PROJECT, scope_id="p1")) == 1
            assert len(store.find(scope_type=MemoryScope.GLOBAL)) == 1
            assert len(store.find(key_like="a")) == 1
            assert store.count() == 3
            assert store.count(kind=MemoryKind.PROJECT) == 2
        finally:
            td.cleanup()

    def t_find_confidence() -> None:
        store, td = fresh()
        try:
            store.create(MemoryKind.LONG_TERM, "high", {},
                         provenance=Provenance(confidence=Confidence.HIGH))
            store.create(MemoryKind.LONG_TERM, "low", {},
                         provenance=Provenance(confidence=Confidence.LOW))
            assert len(store.find(confidence=Confidence.HIGH)) == 1
            assert len(store.find(min_confidence=Confidence.HIGH)) == 1
        finally:
            td.cleanup()

    def t_stats() -> None:
        store, td = fresh()
        try:
            store.create(MemoryKind.PROJECT, "a", {}, scope_type=MemoryScope.PROJECT, scope_id="p")
            store.create(MemoryKind.PROJECT, "b", {}, scope_type=MemoryScope.PROJECT, scope_id="p")
            e = store.create(MemoryKind.LONG_TERM, "c", {})
            store.archive(e.id)
            st = store.stats()
            assert st["total"] == 3
            assert st["by_kind"]["project"] == 2
            assert st["by_status"]["archived"] == 1
        finally:
            td.cleanup()

    check("find filters", t_find_filters)
    check("find by confidence", t_find_confidence)
    check("stats", t_stats)

    # ---- provenance preserved ----
    def t_provenance_preserved() -> None:
        store, td = fresh()
        try:
            prov = Provenance(
                source="user:carol", source_type=ProvenanceType.USER,
                reference="chat://7", confidence=Confidence.HIGH,
                notes="explicit",
            )
            e = store.create(MemoryKind.PROJECT, "k", {}, scope_type=MemoryScope.PROJECT,
                             scope_id="p", provenance=prov)
            got = store.get(e.id)
            assert got.provenance.source == "user:carol"
            assert got.provenance.source_type is ProvenanceType.USER
            assert got.provenance.confidence is Confidence.HIGH
        finally:
            td.cleanup()

    check("provenance preserved", t_provenance_preserved)

    # ---- integrity ----
    def t_integrity_clean() -> None:
        store, td = fresh()
        try:
            e1 = store.create(MemoryKind.PROJECT, "k", {},
                              scope_type=MemoryScope.PROJECT, scope_id="p")
            store.update(e1.id, content={"v": 2})
            rep = store.verify_integrity()
            assert rep["ok"] is True, rep
        finally:
            td.cleanup()

    def t_integrity_detects_duplicate_active() -> None:
        store, td = fresh()
        try:
            # Directly insert two actives (bypassing API) to test detection
            store.storage.execute(
                """
                INSERT INTO memory_entries
                  (id, kind, scope_type, scope_id, key, content, note, tags, attributes,
                   prov_source, prov_type, prov_ref, prov_conf, prov_notes,
                   status, version, supersedes_id, superseded_by,
                   created_at, updated_at, expires_at)
                VALUES ('dup1', 'project', 'project', 'p', 'k', '{}', '', '[]', '{}',
                        's', 'system', NULL, 'unknown', '',
                        'active', 1, NULL, NULL, '2024-01-01T00:00:00', '2024-01-01T00:00:00', NULL),
                       ('dup2', 'project', 'project', 'p', 'k', '{}', '', '[]', '{}',
                        's', 'system', NULL, 'unknown', '',
                        'active', 1, NULL, NULL, '2024-01-01T00:00:00', '2024-01-01T00:00:00', NULL);
                """
            )
            rep = store.verify_integrity()
            assert rep["ok"] is False
            types = {i["type"] for i in rep["issues"]}
            assert "multiple_active" in types
        finally:
            td.cleanup()

    check("integrity: clean state", t_integrity_clean)
    check("integrity: detects duplicate active", t_integrity_detects_duplicate_active)

    # ---- health ----
    def t_health() -> None:
        store, td = fresh()
        try:
            store.create(MemoryKind.PROJECT, "a", {},
                         scope_type=MemoryScope.PROJECT, scope_id="p")
            h = store.health()
            assert h["ok"] is True
            assert h["active"] == 1
        finally:
            td.cleanup()

    check("health check", t_health)

    # ---- e2e ----
    def t_e2e_full_lifecycle() -> None:
        store, td = fresh()
        try:
            # 1. project memory: architecture
            arch = store.create(
                MemoryKind.PROJECT, "architecture",
                {"style": "layered", "layers": ["api", "domain", "data"]},
                scope_type=MemoryScope.PROJECT, scope_id="task-api",
                provenance=Provenance(source="agent:arch", source_type=ProvenanceType.AGENT,
                                      confidence=Confidence.MEDIUM),
            )
            # revise it
            arch2 = store.update(arch.id, content={"style": "layered", "layers": 3})

            # 2. decision memory
            store.record_decision(
                "web-framework",
                decision="FastAPI",
                rationale="Async + Pydantic",
                alternatives=["Flask", "Django"],
                scope_id="task-api",
                confidence=Confidence.HIGH,
            )

            # 3. working memory
            store.set_working("task-api", "current_milestone",
                              {"name": "implement CRUD", "progress": 0.2},
                              ttl_seconds=3600)

            # 4. failure memory
            store.record_failure(
                "test-fail-1",
                what="Test test_create_task failed",
                root_cause="Fixture missing db session",
                fix="Added pytest fixture",
                scope_id="task-api",
            )

            # 5. experience memory
            store.record_experience(
                "exp-fastapi-crud",
                problem="CRUD API design",
                approach="FastAPI + SQLModel",
                outcome="All tests passed",
                lesson="FastAPI + SQLModel is productive for CRUD",
                scope_id="task-api",
                confidence=Confidence.HIGH,
            )

            # 6. long-term knowledge
            store.create(MemoryKind.LONG_TERM, "fastapi.pattern.crud",
                         {"note": "Use dependency injection for DB sessions"})

            # 7. Verify all present and isolated
            hist = store.history(MemoryKind.PROJECT, "architecture",
                                 scope_type=MemoryScope.PROJECT, scope_id="task-api")
            assert len(hist) == 2
            assert store.get_current(MemoryKind.DECISION, "web-framework",
                                     scope_type=MemoryScope.PROJECT,
                                     scope_id="task-api").content["decision"] == "FastAPI"
            assert store.get_working("task-api", "current_milestone").content["progress"] == 0.2
            assert len(store.find(kind=MemoryKind.FAILURE)) == 1
            assert len(store.find(kind=MemoryKind.EXPERIENCE)) == 1
            assert len(store.find(kind=MemoryKind.LONG_TERM)) == 1

            # 8. Integrity clean
            assert store.verify_integrity()["ok"] is True
            _ = arch2  # used
        finally:
            td.cleanup()

    check("e2e: full lifecycle across all 6 kinds", t_e2e_full_lifecycle)

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
    print("SE Brain C04 — Persistent Memory")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo", task_id="t-1"):
                mem = MemoryStore(app.storage)

                print("\n[1] health:", mem.health())

                print("\n[2] Project memory — create + update")
                e1 = mem.create(
                    MemoryKind.PROJECT, "architecture",
                    {"style": "layered", "layers": 2},
                    scope_type=MemoryScope.PROJECT, scope_id="task-api",
                )
                e2 = mem.update(e1.id, content={"layers": 3})
                print(f"    v1 {e1.id[:8]} status={mem.get(e1.id).status.value}")
                print(f"    v2 {e2.id[:8]} version={e2.version}")

                print("\n[3] Working memory with TTL")
                w = mem.set_working("task-api", "step", {"n": 1}, ttl_seconds=600)
                print(f"    expires_at={w.expires_at}")
                print(f"    current={mem.get_working('task-api', 'step').content}")

                print("\n[4] Domain helpers")
                d = mem.record_decision(
                    "web-fw", decision="FastAPI",
                    rationale="Async + types",
                    alternatives=["Flask"],
                    scope_id="task-api",
                )
                print(f"    decision: {d.content}")
                f = mem.record_failure(
                    "bug-1", what="Test failed",
                    root_cause="Missing fixture",
                    fix="Added fixture",
                    scope_id="task-api",
                )
                print(f"    failure : {f.content}")
                x = mem.record_experience(
                    "exp-1", problem="CRUD design",
                    approach="FastAPI", outcome="Tests pass",
                    lesson="Use SQLModel",
                    scope_id="task-api",
                )
                print(f"    exp     : {x.content['lesson']}")

                print("\n[5] Long-term memory")
                lt = mem.create(MemoryKind.LONG_TERM, "fastapi.di",
                                {"note": "Inject sessions via Depends()"})
                print(f"    {lt.content}")

                print("\n[6] History of architecture")
                for h in mem.history(MemoryKind.PROJECT, "architecture",
                                     scope_type=MemoryScope.PROJECT,
                                     scope_id="task-api"):
                    print(f"    v{h.version} status={h.status.value} content={h.content}")

                print("\n[7] Stats:")
                for k, v in sorted(mem.stats().items()):
                    print(f"    {k}: {v}")

                print("\n[8] Integrity:", mem.verify_integrity()["ok"])
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
