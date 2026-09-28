"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C31 — CROSS-PROJECT KNOWLEDGE ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04.

Purpose:
    Provide a tenant/project-isolated knowledge store with an explicit
    promotion/demotion pipeline and strict leak prevention. Project-private
    information MUST NOT leak into another project.

Two scopes:
    PROJECT  — owner_project_id is the project; only visible to that project
    SHARED   — owner_project_id is ""; visible to every project

Capabilities:
    1. Isolation-first store:
         * put_project(project_id, key, ...)
         * put_shared(key, ...)
         * resolve(project_id, key)      — project-local shadows shared
         * search(project_id, filters)   — never returns other projects' rows
    2. Leak prevention:
         * leak_check(content, source_project_id, target_project_id)
         * default policy rule "no_project_identifiers" refuses promotion
           when the payload contains the source project id / tags
         * a hard guarantee: no query ever returns another project's row
    3. Promotion pipeline (project → shared):
         Change candidate → ReusePolicy evaluation → verdict
              PROMOTE | HOLD | REJECT
         * Default policy: verified/high confidence, no PII tags,
           cross-project-safe keys, fresh, no leak
         * On PROMOTE: a shared copy is created; original preserved;
           provenance chain extended; a transition is logged.
    4. Demotion pipeline (shared → project):
         * to_project(key, target_project_id) — shared archived + copy
           created for target; log written
         * forget(key) — shared archived, reason required
    5. Version compatibility:
         * record carries version_req (e.g. {"python": ">=3.11"})
         * check_version_compat(record, env) → (ok, reasons)
    6. Audit trail:
         * every resolve/search writes an access log entry
         * every promotion/demotion writes a transition entry
         * history(key) returns all transitions for a key

Invariants honored:
    - NO external LLM. Deterministic.
    - Tenant isolation is enforced by SQL, not by caller discipline.
    - Project-private rows are NEVER returned to another project.
    - Promotion/demotion are explicit; no silent scope changes.
    - Every transition records actor + rationale + policy verdict.
    - Bounded (max rows per scope, max results per query).
    - Same inputs → same decisions.

Explicit limitations (Rule #59):
    - Leak check is regex-based; it scans for the source project_id and for
      "project:<id>" tags. It does NOT do general PII detection — callers
      should tag sensitive entries themselves.
    - Version compatibility uses a simple constraint parser (`>=`, `<=`,
      `==`, `!=`, `<`, `>`). No semver ranges, no pre-release ordering.
    - Promotion is manual (caller invokes promote()); no auto-promotion.
    - Audit log rows are append-only but not cryptographically signed.

Contents:
  1.  Enums: KnowledgeScope, PromotionAction, ReuseDecision, LeakVerdict
  2.  Dataclasses: KnowledgeRecord, ReuseRule, ReuseEvaluation,
                   PromotionRecord, TransitionRecord, AuditEntry
  3.  Migration 031 (three tables: knowledge, transitions, audit)
  4.  ReusePolicy (default rules)
  5.  Version compatibility
  6.  Leak checker
  7.  CrossProjectKnowledgeStore (facade)
  8.  Persistence convenience + history query
  9.  Self-tests (~40)
 10.  Demo

Run as script:
    python -m sebrain.c31            # demo
    python -m sebrain.c31 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from sebrain.c01 import (
    Confidence, Config, Migration, MigrationRunner, MIGRATIONS as C01_MIGRATIONS,
    SEBrainApp, SQLiteStorage, ValidationError, execution_scope, get_logger,
)
from sebrain.c02 import (
    EntityKind, Ontology, Provenance, ProvenanceType, RelationKind,
    C02_MIGRATIONS,
)
from sebrain.c04 import C04_MIGRATIONS, MemoryKind, MemoryScope, MemoryStore


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _short(s: str, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _enum_val(x: Any) -> str:
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)


_SENSITIVE_TAG_HINTS = frozenset({
    "pii", "secret", "internal", "internal-only", "confidential",
    "sensitive", "private", "customer-data",
})


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class KnowledgeScope(str, Enum):
    PROJECT = "project"
    SHARED = "shared"


class PromotionAction(str, Enum):
    PROMOTE = "promote"      # project → shared
    HOLD = "hold"            # kept project-private for now
    REJECT = "reject"        # not reusable


class ReuseDecision(str, Enum):
    ALLOW = "allow"
    CAUTION = "caution"
    DENY = "deny"
    NOT_APPLICABLE = "not_applicable"


class LeakVerdict(str, Enum):
    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    LEAK = "leak"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class KnowledgeRecord:
    id: str
    scope: KnowledgeScope
    owner_project_id: str               # "" for shared
    key: str
    content: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    version_req: dict[str, str] = field(default_factory=dict)
    confidence: Confidence = Confidence.UNKNOWN
    provenance: Provenance = field(default_factory=Provenance)
    rationale: str = ""
    status: str = "active"              # active | archived
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def is_shared(self) -> bool:
        return self.scope is KnowledgeScope.SHARED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope.value,
            "owner_project_id": self.owner_project_id,
            "key": self.key,
            "content": dict(self.content),
            "tags": list(self.tags),
            "version_req": dict(self.version_req),
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict(),
            "rationale": self.rationale,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(slots=True)
class ReuseRule:
    name: str
    description: str
    # returns (decision, reason, detail)
    check: Callable[["KnowledgeRecord", dict[str, Any]],
                     tuple[ReuseDecision, str, dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description}


@dataclass(slots=True)
class ReuseEvaluation:
    key: str
    owner_project_id: str
    overall: PromotionAction = PromotionAction.HOLD
    rules: list[dict[str, Any]] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "owner_project_id": self.owner_project_id,
            "overall": self.overall.value,
            "rules": list(self.rules),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class PromotionRecord:
    action: PromotionAction
    from_scope: KnowledgeScope
    to_scope: KnowledgeScope
    owner_project_id: str
    key: str
    actor: str
    rationale: str
    id: str = field(default_factory=_new_id)
    ts: str = field(default_factory=now_iso)
    policy_verdict: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "ts": self.ts, "action": self.action.value,
            "from_scope": self.from_scope.value,
            "to_scope": self.to_scope.value,
            "owner_project_id": self.owner_project_id,
            "key": self.key, "actor": self.actor,
            "rationale": self.rationale,
            "policy_verdict": self.policy_verdict,
            "evidence": dict(self.evidence),
        }


@dataclass(slots=True)
class TransitionRecord(PromotionRecord):
    """Alias — a promotion/demotion IS a transition."""
    pass


@dataclass(slots=True)
class AuditEntry:
    id: str = field(default_factory=_new_id)
    ts: str = field(default_factory=now_iso)
    project_id: str = ""
    action: str = ""            # put | get | search | promote | demote | ...
    scope: str = ""
    key: str = ""
    result_count: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "ts": self.ts,
            "project_id": self.project_id, "action": self.action,
            "scope": self.scope, "key": self.key,
            "result_count": self.result_count, "detail": self.detail,
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. MIGRATION 031
# ════════════════════════════════════════════════════════════════════════════
def _migration_031_cross_project(storage: SQLiteStorage) -> None:
    storage.execute("""
        CREATE TABLE IF NOT EXISTS c31_knowledge (
            id                TEXT PRIMARY KEY,
            scope             TEXT NOT NULL,
            owner_project_id  TEXT NOT NULL DEFAULT '',
            key               TEXT NOT NULL,
            content           TEXT NOT NULL DEFAULT '{}',
            tags              TEXT NOT NULL DEFAULT '[]',
            version_req       TEXT NOT NULL DEFAULT '{}',
            confidence        TEXT NOT NULL DEFAULT 'unknown',
            provenance_json   TEXT NOT NULL DEFAULT '{}',
            rationale         TEXT NOT NULL DEFAULT '',
            status            TEXT NOT NULL DEFAULT 'active',
            created_at        TEXT NOT NULL,
            updated_at        TEXT NOT NULL,
            UNIQUE(scope, owner_project_id, key)
        );
    """)
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_c31_scope ON c31_knowledge(scope);"
    )
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_c31_owner ON c31_knowledge(owner_project_id);"
    )
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_c31_key ON c31_knowledge(key);"
    )
    storage.execute("""
        CREATE TABLE IF NOT EXISTS c31_transitions (
            id                TEXT PRIMARY KEY,
            ts                TEXT NOT NULL,
            action            TEXT NOT NULL,
            from_scope        TEXT NOT NULL,
            to_scope          TEXT NOT NULL,
            owner_project_id  TEXT NOT NULL DEFAULT '',
            key               TEXT NOT NULL,
            actor             TEXT NOT NULL,
            rationale         TEXT NOT NULL,
            policy_verdict    TEXT NOT NULL DEFAULT '',
            evidence_json     TEXT NOT NULL DEFAULT '{}'
        );
    """)
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_c31_trans_key ON c31_transitions(key);"
    )
    storage.execute("""
        CREATE TABLE IF NOT EXISTS c31_audit (
            id            TEXT PRIMARY KEY,
            ts            TEXT NOT NULL,
            project_id    TEXT NOT NULL DEFAULT '',
            action        TEXT NOT NULL,
            scope         TEXT NOT NULL DEFAULT '',
            key           TEXT NOT NULL DEFAULT '',
            result_count  INTEGER NOT NULL DEFAULT 0,
            detail        TEXT NOT NULL DEFAULT ''
        );
    """)
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_c31_audit_project ON c31_audit(project_id);"
    )


C31_MIGRATIONS: list[Migration] = [
    Migration(version=31, name="cross_project_knowledge",
              up=_migration_031_cross_project),
]

_ALL_MIGRATIONS: list[Migration] = sorted(
    C01_MIGRATIONS + C02_MIGRATIONS + C04_MIGRATIONS + C31_MIGRATIONS,
    key=lambda m: m.version,
)


# ════════════════════════════════════════════════════════════════════════════
# 4. DEFAULT REUSE POLICY
# ════════════════════════════════════════════════════════════════════════════
def _rule_verified_or_high(rec: KnowledgeRecord, ctx: dict[str, Any]):
    c = rec.confidence
    if c in (Confidence.VERIFIED, Confidence.HIGH):
        return (ReuseDecision.ALLOW,
                f"confidence={c.value} meets threshold", {})
    if c is Confidence.MEDIUM:
        return (ReuseDecision.CAUTION,
                "confidence=medium; consider more evidence before promoting",
                {})
    return (ReuseDecision.DENY,
            f"confidence={c.value} below promotion threshold", {})


def _rule_no_sensitive_tags(rec: KnowledgeRecord, ctx: dict[str, Any]):
    bad = sorted(set(t.lower() for t in rec.tags) & _SENSITIVE_TAG_HINTS)
    if bad:
        return (ReuseDecision.DENY,
                f"sensitive tags present: {bad}", {"tags": bad})
    return (ReuseDecision.ALLOW, "no sensitive tags", {})


def _rule_no_project_identifiers(rec: KnowledgeRecord, ctx: dict[str, Any]):
    pid = rec.owner_project_id or ctx.get("project_id", "")
    if not pid:
        return (ReuseDecision.NOT_APPLICABLE,
                "no owner project id to scan for", {})
    blob = json.dumps(rec.content, default=str) + " " + \
        " ".join(rec.tags) + " " + rec.key + " " + rec.rationale
    if pid in blob:
        return (ReuseDecision.DENY,
                f"payload contains source project identifier '{pid}'",
                {"project_id": pid})
    # any project:<x> tag from another project
    tag_project_ids = [
        t.split(":", 1)[1] for t in rec.tags
        if t.startswith("project:")
    ]
    foreign = [x for x in tag_project_ids if x and x != pid]
    if foreign:
        return (ReuseDecision.DENY,
                f"foreign project tags present: {foreign}",
                {"foreign_tags": foreign})
    return (ReuseDecision.ALLOW, "no project identifiers found", {})


def _rule_fresh_enough(rec: KnowledgeRecord, ctx: dict[str, Any]):
    max_age = int(ctx.get("max_age_days", 90))
    try:
        when = datetime.fromisoformat(rec.updated_at)
    except Exception:
        return (ReuseDecision.NOT_APPLICABLE,
                "could not parse updated_at", {})
    age = (datetime.now(timezone.utc) - when).days
    if age <= max_age:
        return (ReuseDecision.ALLOW,
                f"age {age}d <= {max_age}d", {"age_days": age})
    return (ReuseDecision.CAUTION,
            f"age {age}d > {max_age}d; verify freshness",
            {"age_days": age})


def _rule_nonempty_content(rec: KnowledgeRecord, ctx: dict[str, Any]):
    if not rec.content:
        return (ReuseDecision.DENY, "content is empty", {})
    return (ReuseDecision.ALLOW, "content present", {})


DEFAULT_REUSE_RULES: list[ReuseRule] = [
    ReuseRule("nonempty_content",
              "Record must carry non-empty content",
              _rule_nonempty_content),
    ReuseRule("verified_or_high",
              "Only VERIFIED/HIGH confidence promotes (MEDIUM cautions)",
              _rule_verified_or_high),
    ReuseRule("no_sensitive_tags",
              "Reject if PII/secret/internal tags present",
              _rule_no_sensitive_tags),
    ReuseRule("no_project_identifiers",
              "Reject if payload embeds the source project id",
              _rule_no_project_identifiers),
    ReuseRule("fresh_enough",
              "Caution if older than max_age_days (default 90)",
              _rule_fresh_enough),
]


class ReusePolicy:
    def __init__(self, rules: Sequence[ReuseRule] | None = None) -> None:
        self.rules = list(rules if rules is not None else DEFAULT_REUSE_RULES)

    def evaluate(
        self, rec: KnowledgeRecord, *, context: dict[str, Any] | None = None,
    ) -> ReuseEvaluation:
        ctx = dict(context or {})
        evaluation = ReuseEvaluation(
            key=rec.key, owner_project_id=rec.owner_project_id,
        )
        any_deny = False
        any_caution = False
        for rule in self.rules:
            try:
                decision, reason, detail = rule.check(rec, ctx)
            except Exception as exc:
                decision = ReuseDecision.DENY
                reason = f"rule raised: {type(exc).__name__}: {exc}"
                detail = {}
            evaluation.rules.append({
                "name": rule.name,
                "decision": decision.value,
                "reason": reason,
                "detail": detail,
            })
            if decision is ReuseDecision.DENY:
                any_deny = True
            elif decision is ReuseDecision.CAUTION:
                any_caution = True
        if any_deny:
            evaluation.overall = PromotionAction.REJECT
        elif any_caution:
            evaluation.overall = PromotionAction.HOLD
        else:
            evaluation.overall = PromotionAction.PROMOTE
        evaluation.rationale = (
            f"policy: {len(evaluation.rules)} rule(s); "
            f"overall={evaluation.overall.value}"
        )
        return evaluation


# ════════════════════════════════════════════════════════════════════════════
# 5. VERSION COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════════
_VER_CONSTRAINT_RE = re.compile(
    r"^\s*(>=|<=|==|!=|>|<)\s*([0-9]+(?:\.[0-9]+)*)\s*$"
)


def _parse_version(v: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", v or "")
    return tuple(int(p) for p in parts) if parts else (0,)


def _satisfies_constraint(actual: str, constraint: str) -> bool:
    m = _VER_CONSTRAINT_RE.match(constraint or "")
    if not m:
        return False
    op, target = m.group(1), m.group(2)
    a = _parse_version(actual)
    t = _parse_version(target)
    # pad shorter tuple
    n = max(len(a), len(t))
    a = a + (0,) * (n - len(a))
    t = t + (0,) * (n - len(t))
    if op == "==":
        return a == t
    if op == "!=":
        return a != t
    if op == ">=":
        return a >= t
    if op == "<=":
        return a <= t
    if op == ">":
        return a > t
    if op == "<":
        return a < t
    return False


def check_version_compat(
    rec: KnowledgeRecord, env: dict[str, str],
) -> tuple[bool, list[str]]:
    """Return (compatible, reasons). Empty version_req → compatible."""
    if not rec.version_req:
        return (True, [])
    reasons: list[str] = []
    for k, constraint in rec.version_req.items():
        actual = env.get(k)
        if actual is None:
            reasons.append(f"env missing '{k}' for constraint '{constraint}'")
            continue
        if not _satisfies_constraint(str(actual), constraint):
            reasons.append(
                f"{k}={actual} does not satisfy '{constraint}'"
            )
    return (len(reasons) == 0, reasons)


# ════════════════════════════════════════════════════════════════════════════
# 6. LEAK CHECKER
# ════════════════════════════════════════════════════════════════════════════
class LeakChecker:
    def check(
        self, content: dict[str, Any],
        *, source_project_id: str, target_project_id: str,
        tags: Sequence[str] = (),
    ) -> tuple[LeakVerdict, list[str]]:
        reasons: list[str] = []
        if not source_project_id:
            return (LeakVerdict.SAFE, ["missing source project id; nothing to scan"])
        blob = json.dumps(content, default=str) + " " + " ".join(tags)
        # 1. source project id appears in payload
        if source_project_id in blob:
            reasons.append(
                f"payload contains source project id '{source_project_id}'"
            )
        # 2. foreign project tags
        foreign_tags = [
            t for t in tags
            if t.startswith("project:")
            and t.split(":", 1)[1] not in (source_project_id,
                                            target_project_id, "")
        ]
        if foreign_tags:
            reasons.append(f"foreign project tags: {foreign_tags}")
        # 3. target project id appearing when we did NOT intend that
        # (often indicates copy-paste)
        if target_project_id and source_project_id != target_project_id and \
                target_project_id in blob:
            reasons.append(
                f"payload mentions target project id '{target_project_id}' — "
                f"verify this is intentional"
            )
        if not reasons:
            return (LeakVerdict.SAFE, [])
        # decide verdict: project-id present → LEAK; else SUSPICIOUS
        if any("source project id" in r for r in reasons):
            return (LeakVerdict.LEAK, reasons)
        return (LeakVerdict.SUSPICIOUS, reasons)


# ════════════════════════════════════════════════════════════════════════════
# 7. STORE (facade)
# ════════════════════════════════════════════════════════════════════════════
class CrossProjectKnowledgeStore:
    """Tenant-isolated knowledge store with explicit promotion/demotion."""

    def __init__(
        self,
        storage: SQLiteStorage,
        *,
        policy: ReusePolicy | None = None,
        leak_checker: LeakChecker | None = None,
        max_rows_per_scope: int = 5000,
        max_results_per_query: int = 500,
        auto_migrate: bool = True,
    ) -> None:
        if max_rows_per_scope < 1:
            raise ValidationError("max_rows_per_scope must be >= 1")
        if max_results_per_query < 1:
            raise ValidationError("max_results_per_query must be >= 1")
        self.storage = storage
        self.policy = policy or ReusePolicy()
        self.leak = leak_checker or LeakChecker()
        self.max_rows_per_scope = max_rows_per_scope
        self.max_results_per_query = max_results_per_query
        self._initialized = False
        if auto_migrate:
            self.initialize()

    # ---- lifecycle ----
    def initialize(self) -> None:
        if self._initialized:
            return
        MigrationRunner(self.storage, migrations=_ALL_MIGRATIONS).run()
        self._initialized = True

    # ---- audit ----
    def _audit(
        self, *, project_id: str, action: str, scope: str = "",
        key: str = "", result_count: int = 0, detail: str = "",
    ) -> None:
        self.storage.execute(
            "INSERT INTO c31_audit(id, ts, project_id, action, scope, key, "
            "result_count, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            (_new_id(), now_iso(), project_id, action, scope, key,
             int(result_count), detail),
        )

    def audit_log(
        self, *, project_id: str | None = None, limit: int = 100,
    ) -> list[AuditEntry]:
        sql = "SELECT * FROM c31_audit"
        params: list[Any] = []
        if project_id is not None:
            sql += " WHERE project_id=?"
            params.append(project_id)
        sql += " ORDER BY ts DESC LIMIT ?;"
        params.append(int(limit))
        rows = self.storage.query(sql, params)
        return [AuditEntry(
            id=r["id"], ts=r["ts"], project_id=r["project_id"],
            action=r["action"], scope=r["scope"], key=r["key"],
            result_count=int(r["result_count"]), detail=r["detail"],
        ) for r in rows]

    # ---- put ----
    def put_project(
        self, project_id: str, key: str,
        content: dict[str, Any],
        *,
        tags: Iterable[str] = (),
        version_req: dict[str, str] | None = None,
        confidence: Confidence = Confidence.UNKNOWN,
        provenance: Provenance | None = None,
        rationale: str = "",
    ) -> KnowledgeRecord:
        if not project_id:
            raise ValidationError("project_id required for project-scoped put")
        return self._put(
            scope=KnowledgeScope.PROJECT, owner=project_id, key=key,
            content=content, tags=tags, version_req=version_req,
            confidence=confidence, provenance=provenance, rationale=rationale,
        )

    def put_shared(
        self, key: str, content: dict[str, Any],
        *,
        tags: Iterable[str] = (),
        version_req: dict[str, str] | None = None,
        confidence: Confidence = Confidence.UNKNOWN,
        provenance: Provenance | None = None,
        rationale: str = "",
    ) -> KnowledgeRecord:
        return self._put(
            scope=KnowledgeScope.SHARED, owner="", key=key,
            content=content, tags=tags, version_req=version_req,
            confidence=confidence, provenance=provenance, rationale=rationale,
        )

    def _put(
        self, *, scope: KnowledgeScope, owner: str, key: str,
        content: dict[str, Any], tags: Iterable[str],
        version_req: dict[str, str] | None, confidence: Confidence,
        provenance: Provenance | None, rationale: str,
    ) -> KnowledgeRecord:
        if not key:
            raise ValidationError("key required")
        if not isinstance(content, dict):
            raise ValidationError("content must be a dict")
        existing = self.storage.query_one(
            "SELECT id FROM c31_knowledge WHERE scope=? AND "
            "owner_project_id=? AND key=?;",
            (scope.value, owner, key),
        )
        if existing is not None:
            raise ValidationError(
                f"duplicate key '{key}' in scope={scope.value} "
                f"owner='{owner or '<shared>'}'"
            )
        # enforce per-scope bound
        count_row = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM c31_knowledge WHERE scope=?;",
            (scope.value,),
        )
        if count_row and int(count_row["c"]) >= self.max_rows_per_scope:
            raise ValidationError(
                f"scope '{scope.value}' full "
                f"(>= {self.max_rows_per_scope})"
            )
        prov = provenance or Provenance(
            source="cross_project_store",
            source_type=ProvenanceType.SYSTEM,
            confidence=confidence,
        )
        rec = KnowledgeRecord(
            id=_new_id(), scope=scope, owner_project_id=owner,
            key=key, content=dict(content),
            tags=sorted({str(t) for t in tags if str(t).strip()}),
            version_req=dict(version_req or {}),
            confidence=confidence, provenance=prov, rationale=rationale,
        )
        # Knowledge creation and its audit entry must commit together.
        # Otherwise an audit failure can leave an un-audited knowledge row.
        with self.storage.transaction():
            self.storage.execute(
                "INSERT INTO c31_knowledge(id, scope, owner_project_id, key, "
                "content, tags, version_req, confidence, provenance_json, "
                "rationale, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (
                    rec.id, rec.scope.value, rec.owner_project_id, rec.key,
                    json.dumps(rec.content, default=str),
                    json.dumps(rec.tags),
                    json.dumps(rec.version_req),
                    rec.confidence.value,
                    json.dumps(rec.provenance.to_dict(), default=str),
                    rec.rationale, rec.status,
                    rec.created_at, rec.updated_at,
                ),
            )
            self._audit(
                project_id=owner, action="put", scope=scope.value,
                key=key, result_count=1,
                detail=f"confidence={rec.confidence.value}",
            )

        return rec

    # ---- row helpers ----
    def _row_to_rec(self, row: dict[str, Any]) -> KnowledgeRecord:
        return KnowledgeRecord(
            id=row["id"],
            scope=KnowledgeScope(row["scope"]),
            owner_project_id=row["owner_project_id"],
            key=row["key"],
            content=json.loads(row["content"] or "{}"),
            tags=json.loads(row["tags"] or "[]"),
            version_req=json.loads(row["version_req"] or "{}"),
            confidence=Confidence(row["confidence"]),
            provenance=_prov_from_json(row["provenance_json"]),
            rationale=row["rationale"],
            status=row["status"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ---- resolve (isolation-critical) ----
    def resolve(self, project_id: str, key: str) -> KnowledgeRecord | None:
        """Return the project-local record if present, else shared.

        ISOLATION GUARANTEE: this NEVER returns another project's row.
        """
        row = self.storage.query_one(
            "SELECT * FROM c31_knowledge WHERE key=? AND status='active' "
            "AND (scope='shared' OR "
            "     (scope='project' AND owner_project_id=?)) "
            "ORDER BY CASE scope WHEN 'project' THEN 0 ELSE 1 END LIMIT 1;",
            (key, project_id),
        )
        self._audit(
            project_id=project_id, action="resolve", key=key,
            result_count=1 if row else 0,
        )
        return self._row_to_rec(row) if row else None

    # ---- search (isolation-critical) ----
    def search(
        self, project_id: str,
        *,
        tag: str | None = None,
        key_prefix: str | None = None,
        min_confidence: Confidence | None = None,
        include_shared: bool = True,
        include_project_local: bool = True,
        limit: int | None = None,
    ) -> list[KnowledgeRecord]:
        """Search visible knowledge. Never returns other projects' rows."""
        sql = ["SELECT * FROM c31_knowledge WHERE status='active'"]
        params: list[Any] = []
        clauses: list[str] = []
        if include_project_local:
            clauses.append("(scope='project' AND owner_project_id=?)")
            params.append(project_id)
        if include_shared:
            clauses.append("(scope='shared')")
        if not clauses:
            return []
        sql.append(" AND (" + " OR ".join(clauses) + ")")
        if tag is not None:
            sql.append(" AND tags LIKE ?")
            params.append(f"%\"{tag}\"%")
        if key_prefix is not None:
            sql.append(" AND key LIKE ?")
            params.append(f"{key_prefix}%")
        if min_confidence is not None:
            order = ["unknown", "assumption", "low", "medium", "high",
                     "verified"]
            allowed = [c for c in order
                       if order.index(c) >= order.index(min_confidence.value)]
            placeholders = ",".join("?" * len(allowed))
            sql.append(f" AND confidence IN ({placeholders})")
            params.extend(allowed)
        sql.append(" ORDER BY updated_at DESC LIMIT ?;")
        cap = min(
            int(limit) if limit is not None else self.max_results_per_query,
            self.max_results_per_query,
        )
        params.append(cap)
        rows = self.storage.query("".join(sql), params)
        out = [self._row_to_rec(r) for r in rows]
        self._audit(
            project_id=project_id, action="search",
            result_count=len(out),
            detail=f"tag={tag} key_prefix={key_prefix}",
        )
        return out

    def list_project(self, project_id: str) -> list[KnowledgeRecord]:
        rows = self.storage.query(
            "SELECT * FROM c31_knowledge WHERE scope='project' AND "
            "owner_project_id=? AND status='active' ORDER BY key;",
            (project_id,),
        )
        return [self._row_to_rec(r) for r in rows]

    def list_shared(self) -> list[KnowledgeRecord]:
        rows = self.storage.query(
            "SELECT * FROM c31_knowledge WHERE scope='shared' AND "
            "status='active' ORDER BY key;"
        )
        return [self._row_to_rec(r) for r in rows]

    # ---- version-aware resolve ----
    def resolve_compatible(
        self, project_id: str, key: str, env: dict[str, str],
    ) -> tuple[KnowledgeRecord | None, bool, list[str]]:
        rec = self.resolve(project_id, key)
        if rec is None:
            return (None, True, [])
        ok, reasons = check_version_compat(rec, env)
        return (rec, ok, reasons)

    # ---- leak check ----
    def leak_check(
        self, content: dict[str, Any], *,
        source_project_id: str, target_project_id: str,
        tags: Sequence[str] = (),
    ) -> tuple[LeakVerdict, list[str]]:
        return self.leak.check(
            content, source_project_id=source_project_id,
            target_project_id=target_project_id, tags=tags,
        )

    # ---- promotion ----
    def promote(
        self, project_id: str, key: str, *,
        actor: str, rationale: str = "",
        evidence: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> tuple[PromotionRecord, ReuseEvaluation | None]:
        """Attempt to promote a project-private record to shared."""
        row = self.storage.query_one(
            "SELECT * FROM c31_knowledge WHERE scope='project' AND "
            "owner_project_id=? AND key=? AND status='active';",
            (project_id, key),
        )
        if row is None:
            raise ValidationError(
                f"no active project record '{key}' for '{project_id}'"
            )
        rec = self._row_to_rec(row)
        evaluation = self.policy.evaluate(rec, context={"project_id": project_id})

        # Leak check as an additional hard gate
        verdict, reasons = self.leak.check(
            rec.content, source_project_id=project_id,
            target_project_id="", tags=rec.tags,
        )
        if verdict is LeakVerdict.LEAK:
            evaluation.overall = PromotionAction.REJECT
            evaluation.rationale += f"; leak_check={verdict.value}: {reasons}"

        transition = PromotionRecord(
            action=evaluation.overall,
            from_scope=KnowledgeScope.PROJECT,
            to_scope=KnowledgeScope.SHARED,
            owner_project_id=project_id, key=key,
            actor=actor, rationale=rationale,
            policy_verdict=evaluation.overall.value,
            evidence={
                "policy_evaluation": evaluation.to_dict(),
                "leak_verdict": verdict.value,
                "leak_reasons": reasons,
                **(evidence or {}),
            },
        )
        if dry_run:
            return transition, evaluation

        with self.storage.transaction():
            # Persist the transition regardless of verdict (audit)
            self._persist_transition(transition)
    
            if evaluation.overall is PromotionAction.PROMOTE:
                # create a shared copy; keep original
                existing = self.storage.query_one(
                    "SELECT id FROM c31_knowledge WHERE scope='shared' AND "
                    "owner_project_id='' AND key=?;",
                    (key,),
                )
                if existing is not None:
                    # Already shared → mark HOLD
                    transition.action = PromotionAction.HOLD
                    transition.policy_verdict = "hold"
                    transition.rationale += (
                        "; shared record with same key already exists"
                    )
                    self._update_transition_action(transition)
                    return transition, evaluation
                new_rec = KnowledgeRecord(
                    id=_new_id(), scope=KnowledgeScope.SHARED, owner_project_id="",
                    key=key, content=rec.content, tags=rec.tags,
                    version_req=rec.version_req, confidence=rec.confidence,
                    provenance=Provenance(
                        source=f"promoted_from:{project_id}",
                        source_type=ProvenanceType.AGENT,
                        reference=rec.id,
                        confidence=rec.confidence,
                        notes=f"promoted by {actor}",
                    ),
                    rationale=(
                        f"promoted from project '{project_id}': "
                        f"{rationale or rec.rationale}"
                    ),
                )
                self.storage.execute(
                    "INSERT INTO c31_knowledge(id, scope, owner_project_id, key, "
                    "content, tags, version_req, confidence, provenance_json, "
                    "rationale, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (
                        new_rec.id, new_rec.scope.value,
                        new_rec.owner_project_id, new_rec.key,
                        json.dumps(new_rec.content, default=str),
                        json.dumps(new_rec.tags),
                        json.dumps(new_rec.version_req),
                        new_rec.confidence.value,
                        json.dumps(new_rec.provenance.to_dict(), default=str),
                        new_rec.rationale, new_rec.status,
                        new_rec.created_at, new_rec.updated_at,
                    ),
                )
                self._audit(
                    project_id=project_id, action="promote",
                    scope="shared", key=key, result_count=1,
                    detail=f"actor={actor}",
                )
                return transition, evaluation
    
            # HOLD or REJECT: no row created
            self._audit(
                project_id=project_id,
                action=f"promote_{evaluation.overall.value}",
                scope="project", key=key, result_count=0,
                detail=f"actor={actor}",
            )
            return transition, evaluation
    
    def _persist_transition(self, t: PromotionRecord) -> None:
        self.storage.execute(
            "INSERT INTO c31_transitions(id, ts, action, from_scope, "
            "to_scope, owner_project_id, key, actor, rationale, "
            "policy_verdict, evidence_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
            (
                t.id, t.ts, t.action.value,
                t.from_scope.value, t.to_scope.value,
                t.owner_project_id, t.key, t.actor, t.rationale,
                t.policy_verdict,
                json.dumps(t.evidence, default=str),
            ),
        )
    def _update_transition_action(self, t: PromotionRecord) -> None:
        self.storage.execute(
            "UPDATE c31_transitions SET action=?, policy_verdict=?, "
            "rationale=?, evidence_json=? WHERE id=?;",
            (
                t.action.value, t.policy_verdict, t.rationale,
                json.dumps(t.evidence, default=str), t.id,
            ),
        )

    # ---- demotion ----
    def demote_to_project(
        self, key: str, *, target_project_id: str,
        actor: str, rationale: str = "",
    ) -> PromotionRecord:
        """Move a shared record into a specific project (archive shared)."""
        if not target_project_id:
            raise ValidationError("target_project_id required")
        row = self.storage.query_one(
            "SELECT * FROM c31_knowledge WHERE scope='shared' AND "
            "owner_project_id='' AND key=? AND status='active';",
            (key,),
        )
        if row is None:
            raise ValidationError(f"no active shared record '{key}'")
        rec = self._row_to_rec(row)
        with self.storage.transaction():
            # Copy to project (may collide)
            existing = self.storage.query_one(
                "SELECT id FROM c31_knowledge WHERE scope='project' AND "
                "owner_project_id=? AND key=? AND status='active';",
                (target_project_id, key),
            )
            if existing is None:
                new_rec = KnowledgeRecord(
                    id=_new_id(), scope=KnowledgeScope.PROJECT,
                    owner_project_id=target_project_id,
                    key=key, content=rec.content, tags=rec.tags,
                    version_req=rec.version_req, confidence=rec.confidence,
                    provenance=Provenance(
                        source=f"demoted_from_shared_by:{actor}",
                        source_type=ProvenanceType.AGENT,
                        reference=rec.id,
                        confidence=rec.confidence,
                    ),
                    rationale=(
                        f"demoted from shared: {rationale or 'no reason'}"
                    ),
                )
                self.storage.execute(
                    "INSERT INTO c31_knowledge(id, scope, owner_project_id, key, "
                    "content, tags, version_req, confidence, provenance_json, "
                    "rationale, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    (
                        new_rec.id, new_rec.scope.value,
                        new_rec.owner_project_id, new_rec.key,
                        json.dumps(new_rec.content, default=str),
                        json.dumps(new_rec.tags),
                        json.dumps(new_rec.version_req),
                        new_rec.confidence.value,
                        json.dumps(new_rec.provenance.to_dict(), default=str),
                        new_rec.rationale, new_rec.status,
                        new_rec.created_at, new_rec.updated_at,
                    ),
                )
            else:
                # Archived history occupies the natural key, so reactivation
                # must update that historical row rather than INSERT a second
                # row that violates the uniqueness constraint. This preserves
                # the row's lineage while making the latest state active.
                self.storage.execute(
                    "UPDATE c31_knowledge SET status='active', content=?, tags=?, "
                    "version_req=?, confidence=?, provenance_json=?, rationale=?, "
                    "updated_at=? WHERE id=? AND status='archived';",
                    (
                        json.dumps(rec.content, default=str),
                        json.dumps(rec.tags),
                        json.dumps(rec.version_req),
                        rec.confidence.value,
                        json.dumps(Provenance(
                            source=f'demoted_from_shared_by:{actor}',
                            source_type=ProvenanceType.AGENT,
                            reference=rec.id,
                            confidence=rec.confidence,
                        ).to_dict(), default=str),
                        f"demoted from shared: {rationale or 'no reason'}",
                        now_iso(), existing["id"],
                    ),
                )
            # Archive the shared original
            self.storage.execute(
                "UPDATE c31_knowledge SET status='archived', updated_at=? "
                "WHERE id=?;",
                (now_iso(), rec.id),
            )
            transition = PromotionRecord(
                action=PromotionAction.REJECT,       # not a promotion
                from_scope=KnowledgeScope.SHARED,
                to_scope=KnowledgeScope.PROJECT,
                owner_project_id=target_project_id, key=key,
                actor=actor, rationale=rationale or "demotion",
                policy_verdict="demote_to_project",
            )
            self._persist_transition(transition)
            self._audit(
                project_id=target_project_id, action="demote_to_project",
                scope="shared", key=key, result_count=1,
                detail=f"actor={actor}",
            )
            return transition

    def forget_shared(
        self, key: str, *, actor: str, rationale: str = "",
    ) -> PromotionRecord:
        row = self.storage.query_one(
            "SELECT * FROM c31_knowledge WHERE scope='shared' AND "
            "owner_project_id='' AND key=? AND status='active';",
            (key,),
        )
        if row is None:
            raise ValidationError(f"no active shared record '{key}'")
        if not rationale:
            raise ValidationError("rationale is required to forget shared")
        transition = PromotionRecord(
            action=PromotionAction.REJECT,
            from_scope=KnowledgeScope.SHARED,
            to_scope=KnowledgeScope.SHARED,
            owner_project_id="", key=key,
            actor=actor, rationale=rationale,
            policy_verdict="forget_shared",
        )
        # Archival + audit must be atomic. A failed audit must not silently
        # leave the shared knowledge archived without a corresponding record.
        with self.storage.transaction():
            self.storage.execute(
                "UPDATE c31_knowledge SET status='archived', updated_at=? "
                "WHERE id=?;", (now_iso(), row["id"]),
            )
            self._persist_transition(transition)
            self._audit(project_id="", action="forget_shared",
                         scope="shared", key=key, result_count=1,
                         detail=f"actor={actor}")
        return transition

    # ---- history ----
    def history(self, key: str) -> list[PromotionRecord]:
        rows = self.storage.query(
            "SELECT * FROM c31_transitions WHERE key=? ORDER BY ts;",
            (key,),
        )
        out: list[PromotionRecord] = []
        for r in rows:
            out.append(PromotionRecord(
                id=r["id"], ts=r["ts"],
                action=PromotionAction(r["action"]),
                from_scope=KnowledgeScope(r["from_scope"]),
                to_scope=KnowledgeScope(r["to_scope"]),
                owner_project_id=r["owner_project_id"],
                key=r["key"], actor=r["actor"],
                rationale=r["rationale"],
                policy_verdict=r["policy_verdict"],
                evidence=json.loads(r["evidence_json"] or "{}"),
            ))
        return out

    def history_for_project(
        self, project_id: str, key: str,
    ) -> list[PromotionRecord]:
        """Return only this project's transitions plus global shared transitions.

        Project-owned transition records are isolated by owner_project_id.
        Shared/global transitions use an empty owner and are visible because
        they describe shared-state lifecycle rather than another tenant's
        private transition history.
        """
        if not project_id:
            raise ValidationError("project_id required")
        rows = self.storage.query(
            "SELECT * FROM c31_transitions "
            "WHERE key=? AND (owner_project_id=? OR owner_project_id='') "
            "ORDER BY ts;",
            (key, project_id),
        )
        return [
            PromotionRecord(
                id=r["id"], ts=r["ts"],
                action=PromotionAction(r["action"]),
                from_scope=KnowledgeScope(r["from_scope"]),
                to_scope=KnowledgeScope(r["to_scope"]),
                owner_project_id=r["owner_project_id"],
                key=r["key"], actor=r["actor"],
                rationale=r["rationale"],
                policy_verdict=r["policy_verdict"],
                evidence=json.loads(r["evidence_json"] or "{}"),
            )
            for r in rows
        ]

    # ---- stats ----
    def stats(self) -> dict[str, Any]:
        r1 = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM c31_knowledge WHERE scope='project' "
            "AND status='active';"
        )
        r2 = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM c31_knowledge WHERE scope='shared' "
            "AND status='active';"
        )
        r3 = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM c31_knowledge WHERE status='archived';"
        )
        r4 = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM c31_transitions;"
        )
        r5 = self.storage.query_one(
            "SELECT COUNT(*) AS c FROM c31_audit;"
        )
        return {
            "project_rows": int(r1["c"]) if r1 else 0,
            "shared_rows": int(r2["c"]) if r2 else 0,
            "archived_rows": int(r3["c"]) if r3 else 0,
            "transitions": int(r4["c"]) if r4 else 0,
            "audit_entries": int(r5["c"]) if r5 else 0,
        }


# ---- provenance from JSON ----
def _prov_from_json(blob: str) -> Provenance:
    try:
        d = json.loads(blob or "{}")
    except json.JSONDecodeError:
        d = {}
    if not isinstance(d, dict):
        d = {}
    try:
        st = ProvenanceType(d.get("source_type", "system"))
    except ValueError:
        st = ProvenanceType.SYSTEM
    try:
        cf = Confidence(d.get("confidence", "unknown"))
    except ValueError:
        cf = Confidence.UNKNOWN
    return Provenance(
        source=str(d.get("source", "system")),
        source_type=st,
        reference=d.get("reference"),
        confidence=cf,
        notes=str(d.get("notes", "")),
    )


# ════════════════════════════════════════════════════════════════════════════
# 8. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _fresh_store() -> tuple[CrossProjectKnowledgeStore,
                             tempfile.TemporaryDirectory]:
    td = tempfile.TemporaryDirectory()
    storage = SQLiteStorage(Path(td.name) / "c31.sqlite3")
    storage.initialize()
    return CrossProjectKnowledgeStore(storage), td


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
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C31 self-tests…")

    # ---- put / get / isolation ----
    def t_put_project_isolation() -> None:
        store, td = _fresh_store()
        try:
            rec = store.put_project(
                "proj-a", "db.url",
                {"url": "sqlite:///a.db"},
                confidence=Confidence.HIGH,
            )
            assert rec.scope is KnowledgeScope.PROJECT
            assert rec.owner_project_id == "proj-a"
            # resolve from same project sees it
            got = store.resolve("proj-a", "db.url")
            assert got is not None and got.id == rec.id
            # resolve from another project must NOT see it
            other = store.resolve("proj-b", "db.url")
            assert other is None, "project row leaked!"
        finally:
            td.cleanup()

    def t_put_shared_visible_everywhere() -> None:
        store, td = _fresh_store()
        try:
            store.put_shared(
                "pattern.retry",
                {"strategy": "exponential", "base_ms": 100},
                confidence=Confidence.VERIFIED,
                tags=["pattern"],
            )
            for pid in ("proj-a", "proj-b", "proj-c"):
                got = store.resolve(pid, "pattern.retry")
                assert got is not None
                assert got.is_shared()
        finally:
            td.cleanup()

    def t_project_shadows_shared() -> None:
        store, td = _fresh_store()
        try:
            store.put_shared("k", {"src": "shared"},
                              confidence=Confidence.HIGH)
            store.put_project("proj-a", "k", {"src": "local"},
                               confidence=Confidence.HIGH)
            got = store.resolve("proj-a", "k")
            assert got is not None
            assert got.scope is KnowledgeScope.PROJECT
            assert got.content["src"] == "local"
            # other project sees shared
            other = store.resolve("proj-b", "k")
            assert other is not None
            assert other.content["src"] == "shared"
        finally:
            td.cleanup()

    def t_duplicate_key_rejected() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("p", "k", {"v": 1})
            try:
                store.put_project("p", "k", {"v": 2})
            except ValidationError:
                return
            raise AssertionError("expected duplicate rejection")
        finally:
            td.cleanup()

    check("put: project isolation enforced on read",
          t_put_project_isolation)
    check("put: shared visible from any project",
          t_put_shared_visible_everywhere)
    check("resolve: project shadows shared",
          t_project_shadows_shared)
    check("put: duplicate key in same scope rejected",
          t_duplicate_key_rejected)

    # ---- search isolation ----
    def t_search_isolated() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("a", "k1", {"x": 1}, tags=["t1"])
            store.put_project("a", "k2", {"x": 2}, tags=["t1"])
            store.put_project("b", "k3", {"x": 3}, tags=["t1"])
            store.put_shared("k4", {"x": 4}, tags=["t1"],
                              confidence=Confidence.HIGH)
            # a's search sees k1, k2, k4 only
            res = store.search("a", tag="t1")
            keys = {r.key for r in res}
            assert keys == {"k1", "k2", "k4"}, keys
            # b's search sees k3, k4 only
            res = store.search("b", tag="t1")
            keys = {r.key for r in res}
            assert keys == {"k3", "k4"}, keys
        finally:
            td.cleanup()

    def t_search_exclude_shared() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("a", "k1", {"x": 1})
            store.put_shared("k2", {"x": 2})
            res = store.search("a", include_shared=False)
            assert {r.key for r in res} == {"k1"}
            res = store.search("a", include_project_local=False)
            assert {r.key for r in res} == {"k2"}
        finally:
            td.cleanup()

    def t_search_min_confidence() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("a", "low", {},
                               confidence=Confidence.LOW)
            store.put_project("a", "high", {},
                               confidence=Confidence.HIGH)
            store.put_project("a", "ver", {},
                               confidence=Confidence.VERIFIED)
            res = store.search("a", min_confidence=Confidence.HIGH)
            assert {r.key for r in res} == {"high", "ver"}
        finally:
            td.cleanup()

    check("search: project isolation enforced",
          t_search_isolated)
    check("search: include/exclude filters", t_search_exclude_shared)
    check("search: min_confidence filters", t_search_min_confidence)

    # ---- version compatibility ----
    def t_version_constraint_parser() -> None:
        assert _satisfies_constraint("3.11.5", ">=3.11")
        assert _satisfies_constraint("3.11", ">=3.11")
        assert not _satisfies_constraint("3.10", ">=3.11")
        assert _satisfies_constraint("2.0.1", "==2.0.1")
        assert _satisfies_constraint("2.5", ">2.0")
        assert _satisfies_constraint("1.0", "<2.0")

    def t_version_compat_check() -> None:
        rec = KnowledgeRecord(
            id="x", scope=KnowledgeScope.SHARED, owner_project_id="",
            key="k", content={}, version_req={"python": ">=3.11"},
        )
        ok, reasons = check_version_compat(rec, {"python": "3.11.5"})
        assert ok and reasons == []
        ok, reasons = check_version_compat(rec, {"python": "3.10"})
        assert not ok and reasons
        ok, reasons = check_version_compat(rec, {})
        assert not ok and reasons

    def t_version_empty_req_ok() -> None:
        rec = KnowledgeRecord(
            id="x", scope=KnowledgeScope.SHARED, owner_project_id="",
            key="k", content={},
        )
        ok, _ = check_version_compat(rec, {})
        assert ok

    def t_resolve_compatible_gate() -> None:
        store, td = _fresh_store()
        try:
            store.put_project(
                "a", "k", {"v": 1},
                version_req={"python": ">=3.11"},
                confidence=Confidence.HIGH,
            )
            rec, ok, _ = store.resolve_compatible("a", "k",
                                                    {"python": "3.11"})
            assert rec is not None and ok
            rec, ok, reasons = store.resolve_compatible(
                "a", "k", {"python": "3.8"},
            )
            assert rec is not None and not ok and reasons
        finally:
            td.cleanup()

    check("version: constraint parser", t_version_constraint_parser)
    check("version: compat check returns reasons",
          t_version_compat_check)
    check("version: empty req always compatible",
          t_version_empty_req_ok)
    check("version: resolve_compatible gates by env",
          t_resolve_compatible_gate)

    # ---- leak checker ----
    def t_leak_safe() -> None:
        lc = LeakChecker()
        v, _ = lc.check({"x": 1}, source_project_id="a",
                         target_project_id="b")
        assert v is LeakVerdict.SAFE

    def t_leak_project_id_in_payload() -> None:
        lc = LeakChecker()
        v, reasons = lc.check(
            {"note": "internal to proj-a"},
            source_project_id="proj-a", target_project_id="proj-b",
        )
        assert v is LeakVerdict.LEAK
        assert any("source project id" in r for r in reasons)

    def t_leak_foreign_tag() -> None:
        lc = LeakChecker()
        v, reasons = lc.check(
            {"x": 1}, source_project_id="a", target_project_id="b",
            tags=["project:c"],
        )
        assert v is LeakVerdict.SUSPICIOUS or v is LeakVerdict.LEAK

    def t_leak_target_mention_suspicious() -> None:
        lc = LeakChecker()
        # source_project_id must not accidentally appear as a substring
        # anywhere in the payload/target — a single letter like "a" would
        # (e.g. it's inside "filename"), which flips this into a LEAK
        # verdict via the *source*-id reason instead of testing the
        # target-mention (SUSPICIOUS) path this test is actually for.
        v, reasons = lc.check(
            {"ref": "proj-b/filename"},
            source_project_id="proj-a", target_project_id="proj-b",
        )
        assert v is LeakVerdict.SUSPICIOUS

    check("leak: clean → SAFE", t_leak_safe)
    check("leak: source project id → LEAK",
          t_leak_project_id_in_payload)
    check("leak: foreign tag flagged", t_leak_foreign_tag)
    check("leak: target mention → SUSPICIOUS",
          t_leak_target_mention_suspicious)

    # ---- promotion ----
    def t_promote_success() -> None:
        store, td = _fresh_store()
        try:
            store.put_project(
                "proj-a", "pattern.retry",
                {"strategy": "exponential"},
                tags=["pattern"],
                confidence=Confidence.VERIFIED,
            )
            tr, ev = store.promote(
                "proj-a", "pattern.retry",
                actor="reviewer", rationale="proven twice",
            )
            assert tr.action is PromotionAction.PROMOTE
            assert ev is not None and ev.overall is PromotionAction.PROMOTE
            # shared copy exists; project original preserved
            shared = store.resolve("proj-b", "pattern.retry")
            assert shared is not None and shared.is_shared()
            local = store.list_project("proj-a")
            assert any(r.key == "pattern.retry" for r in local)
        finally:
            td.cleanup()

    def t_promote_rejected_by_sensitive_tag() -> None:
        store, td = _fresh_store()
        try:
            store.put_project(
                "proj-a", "secret.key",
                {"value": "xyz"}, tags=["secret", "internal"],
                confidence=Confidence.VERIFIED,
            )
            tr, ev = store.promote(
                "proj-a", "secret.key",
                actor="reviewer", rationale="test",
            )
            assert ev is not None
            assert ev.overall is PromotionAction.REJECT
            assert tr.action is PromotionAction.REJECT
            # no shared row created
            assert store.resolve("proj-b", "secret.key") is None
        finally:
            td.cleanup()

    def t_promote_rejected_by_low_confidence() -> None:
        store, td = _fresh_store()
        try:
            store.put_project(
                "proj-a", "maybe", {"v": 1},
                confidence=Confidence.LOW,
            )
            tr, ev = store.promote("proj-a", "maybe",
                                    actor="r", rationale="try")
            assert ev is not None
            assert ev.overall is PromotionAction.REJECT
        finally:
            td.cleanup()

    def t_promote_hold_on_medium() -> None:
        store, td = _fresh_store()
        try:
            store.put_project(
                "proj-a", "med", {"v": 1},
                confidence=Confidence.MEDIUM,
            )
            tr, ev = store.promote("proj-a", "med",
                                    actor="r", rationale="try")
            assert ev is not None and ev.overall is PromotionAction.HOLD
            assert store.resolve("proj-b", "med") is None
        finally:
            td.cleanup()

    def t_promote_rejected_on_leak() -> None:
        store, td = _fresh_store()
        try:
            # Content embeds source project id
            store.put_project(
                "proj-a", "thing",
                {"note": "internal-only proj-a secret recipe"},
                confidence=Confidence.VERIFIED,
            )
            tr, ev = store.promote(
                "proj-a", "thing", actor="r", rationale="try",
            )
            assert ev is not None and ev.overall is PromotionAction.REJECT
            assert store.resolve("proj-b", "thing") is None
        finally:
            td.cleanup()

    def t_promote_dry_run_no_writes() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("a", "k", {"v": 1},
                               confidence=Confidence.VERIFIED)
            tr, ev = store.promote("a", "k", actor="r",
                                    rationale="dry", dry_run=True)
            assert ev is not None and ev.overall is PromotionAction.PROMOTE
            # nothing actually created
            assert store.resolve("b", "k") is None
            # no transition logged either
            hist = store.history("k")
            assert hist == []
        finally:
            td.cleanup()

    def t_promote_missing_row() -> None:
        store, td = _fresh_store()
        try:
            try:
                store.promote("a", "nope", actor="r")
            except ValidationError:
                return
            raise AssertionError("expected error")
        finally:
            td.cleanup()

    check("promote: success creates shared copy",
          t_promote_success)
    check("promote: sensitive tags → REJECT",
          t_promote_rejected_by_sensitive_tag)
    check("promote: low confidence → REJECT",
          t_promote_rejected_by_low_confidence)
    check("promote: medium confidence → HOLD",
          t_promote_hold_on_medium)
    check("promote: leak detected → REJECT",
          t_promote_rejected_on_leak)
    check("promote: dry-run makes no writes",
          t_promote_dry_run_no_writes)
    check("promote: missing project row → error",
          t_promote_missing_row)

    # ---- demotion ----
    def t_demote_to_project() -> None:
        store, td = _fresh_store()
        try:
            store.put_shared("k", {"v": 1},
                              confidence=Confidence.HIGH)
            tr = store.demote_to_project(
                "k", target_project_id="proj-a",
                actor="r", rationale="only relevant to A",
            )
            # shared archived
            assert store.resolve("proj-b", "k") is None
            # project A now has it
            assert store.resolve("proj-a", "k") is not None
        finally:
            td.cleanup()

    def t_forget_shared_requires_rationale() -> None:
        store, td = _fresh_store()
        try:
            store.put_shared("k", {"v": 1})
            try:
                store.forget_shared("k", actor="r", rationale="")
            except ValidationError:
                return
            raise AssertionError("expected error")
        finally:
            td.cleanup()

    def t_forget_shared_ok() -> None:
        store, td = _fresh_store()
        try:
            store.put_shared("k", {"v": 1})
            store.forget_shared("k", actor="r",
                                 rationale="obsolete")
            assert store.resolve("a", "k") is None
        finally:
            td.cleanup()

    check("demote: shared → project, shared archived",
          t_demote_to_project)
    check("forget: rationale required",
          t_forget_shared_requires_rationale)
    check("forget: archives shared", t_forget_shared_ok)

    # ---- history ----
    def t_history_records_transitions() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("p", "k", {"v": 1},
                               confidence=Confidence.VERIFIED)
            store.promote("p", "k", actor="a", rationale="r")
            hist = store.history("k")
            assert any(t.action is PromotionAction.PROMOTE for t in hist)
            assert hist[-1].actor == "a"
        finally:
            td.cleanup()

    check("history: transition log per key", t_history_records_transitions)

    # ---- audit ----
    def t_audit_log_entries() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("p", "k", {"v": 1})
            store.resolve("p", "k")
            store.search("p")
            entries = store.audit_log(project_id="p")
            actions = {e.action for e in entries}
            assert "put" in actions
            assert "resolve" in actions
            assert "search" in actions
        finally:
            td.cleanup()

    check("audit: every operation logged", t_audit_log_entries)

    # ---- stats ----
    def t_stats() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("a", "k1", {})
            store.put_project("a", "k2", {})
            store.put_shared("s1", {},
                              confidence=Confidence.HIGH)
            st = store.stats()
            assert st["project_rows"] == 2
            assert st["shared_rows"] == 1
        finally:
            td.cleanup()

    check("stats: counts returned", t_stats)

    # ---- to_dict ----
    def t_to_dict() -> None:
        store, td = _fresh_store()
        try:
            rec = store.put_project("p", "k", {"v": 1},
                                     tags=["t"], version_req={"python": ">=3.11"},
                                     confidence=Confidence.HIGH)
            d = rec.to_dict()
            assert d["scope"] == "project"
            assert d["owner_project_id"] == "p"
            assert d["version_req"]["python"] == ">=3.11"
        finally:
            td.cleanup()

    check("to_dict: record serialisable", t_to_dict)

    # ---- determinism ----
    def t_deterministic_search() -> None:
        store, td = _fresh_store()
        try:
            store.put_project("a", "k1", {"x": 1}, tags=["x"])
            store.put_project("a", "k2", {"x": 2}, tags=["x"])
            r1 = [r.key for r in store.search("a", tag="x")]
            r2 = [r.key for r in store.search("a", tag="x")]
            assert r1 == r2
        finally:
            td.cleanup()

    check("deterministic: same search returns same result",
          t_deterministic_search)

    # ---- E2E ----
    def t_e2e_workflow() -> None:
        store, td = _fresh_store()
        try:
            # Project A discovers a reusable pattern
            store.put_project(
                "proj-a", "pattern.cache_warming",
                {"strategy": "warm cache on cold start"},
                tags=["pattern", "performance"],
                confidence=Confidence.VERIFIED,
                rationale="fixed stampedes across services",
            )
            # Project B tries to use it — nothing shared yet
            assert store.resolve("proj-b", "pattern.cache_warming") is None

            # Promote from A → shared
            tr, ev = store.promote(
                "proj-a", "pattern.cache_warming",
                actor="architect", rationale="cross-team pattern",
            )
            assert tr.action is PromotionAction.PROMOTE

            # Now B can use it; A still has its local copy
            assert store.resolve("proj-b", "pattern.cache_warming") is not None
            assert store.resolve("proj-a", "pattern.cache_warming") is not None

            # B later demotes it because it discovered a project-specific
            # variant; shared version archived
            store.demote_to_project(
                "pattern.cache_warming", target_project_id="proj-b",
                actor="b-lead", rationale="B has custom warmup",
            )
            assert store.resolve("proj-c", "pattern.cache_warming") is None
            assert store.resolve("proj-b", "pattern.cache_warming") is not None
            assert store.resolve("proj-a", "pattern.cache_warming") is not None

            # History preserved
            hist = store.history("pattern.cache_warming")
            kinds = [t.policy_verdict for t in hist]
            assert "promote" in kinds
            assert "demote_to_project" in kinds
        finally:
            td.cleanup()

    check("e2e: put → promote → use → demote → history",
          t_e2e_workflow)

    # ---- isolation invariant stress ----
    def t_isolation_stress_100_rows() -> None:
        store, td = _fresh_store()
        try:
            # 20 projects, 5 keys each, all with the same key names
            for i in range(20):
                pid = f"proj-{i}"
                for j in range(5):
                    store.put_project(pid, f"key-{j}",
                                       {"owner": pid, "j": j})
            # Each project's search must only see its own 5 keys
            for i in range(20):
                pid = f"proj-{i}"
                res = store.search(pid)
                assert len(res) == 5, (pid, len(res))
                for r in res:
                    assert r.owner_project_id == pid
        finally:
            td.cleanup()

    check("isolation: 20 projects × 5 keys never leak",
          t_isolation_stress_100_rows)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 9. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C31 — Cross-Project Knowledge Engine")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31_demo.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)

        print("\n[1] Put project-private knowledge (proj-alpha):")
        store.put_project(
            "proj-alpha", "secret.db.dsn",
            {"dsn": "postgres://user:pass@internal-alpha-db:5432/x"},
            tags=["internal"],
            confidence=Confidence.HIGH,
        )
        print("    ✓ stored (owner=proj-alpha)")
        # Try to read from another project → blocked
        got = store.resolve("proj-beta", "secret.db.dsn")
        print(f"    resolve from proj-beta: {got}")
        got = store.resolve("proj-alpha", "secret.db.dsn")
        print(f"    resolve from proj-alpha: {got.key if got else None}")

        print("\n[2] Put shared pattern knowledge:")
        store.put_shared(
            "pattern.retry.exponential",
            {"strategy": "exponential backoff", "base_ms": 100,
             "max_ms": 10_000},
            tags=["pattern", "reliability"],
            confidence=Confidence.VERIFIED,
        )
        for pid in ("proj-alpha", "proj-beta"):
            r = store.resolve(pid, "pattern.retry.exponential")
            print(f"    {pid} sees: {r.key if r else None}")

        print("\n[3] Promotion attempt: project → shared")
        # A safe candidate
        store.put_project(
            "proj-alpha", "pattern.cache_warming",
            {"strategy": "warm cache on cold start"},
            tags=["pattern", "performance"],
            confidence=Confidence.VERIFIED,
            rationale="fixed stampedes",
        )
        tr, ev = store.promote(
            "proj-alpha", "pattern.cache_warming",
            actor="architect", rationale="reusable across teams",
        )
        print(f"    verdict: {ev.overall.value if ev else '?'}")
        print(f"    action: {tr.action.value}")
        if ev:
            for r in ev.rules:
                print(f"      [{r['decision']:14s}] {r['name']}: "
                      f"{_short(r['reason'], 70)}")

        print("\n[4] Promotion blocked by leak/sensitive tags:")
        store.put_project(
            "proj-alpha", "secret.api_key",
            {"value": "sk-abc123-def456-ghi789-jkl012"},
            tags=["secret", "pii"],
            confidence=Confidence.VERIFIED,
        )
        tr2, ev2 = store.promote(
            "proj-alpha", "secret.api_key",
            actor="architect", rationale="try to promote",
        )
        print(f"    verdict: {ev2.overall.value if ev2 else '?'}")
        if ev2:
            for r in ev2.rules:
                if r["decision"] != "allow":
                    print(f"      [{r['decision']:14s}] {r['name']}: "
                          f"{_short(r['reason'], 80)}")

        print("\n[5] Version compatibility:")
        store.put_project(
            "proj-alpha", "pattern.asyncio_gather",
            {"recipe": "use asyncio.gather for parallelism"},
            version_req={"python": ">=3.11"},
            confidence=Confidence.HIGH,
        )
        rec, ok, reasons = store.resolve_compatible(
            "proj-alpha", "pattern.asyncio_gather", {"python": "3.11.5"},
        )
        print(f"    env python=3.11.5 → compatible={ok}")
        rec, ok, reasons = store.resolve_compatible(
            "proj-alpha", "pattern.asyncio_gather", {"python": "3.8"},
        )
        print(f"    env python=3.8    → compatible={ok}; reasons={reasons}")

        print("\n[6] Leak check between two projects:")
        v, reasons = store.leak_check(
            {"note": "this uses proj-alpha internal config"},
            source_project_id="proj-alpha",
            target_project_id="proj-beta",
        )
        print(f"    verdict: {v.value}; reasons={reasons}")
        v, reasons = store.leak_check(
            {"note": "generic pattern"},
            source_project_id="proj-alpha",
            target_project_id="proj-beta",
        )
        print(f"    clean payload verdict: {v.value}")

        print("\n[7] Demotion (shared → specific project):")
        store.demote_to_project(
            "pattern.cache_warming",
            target_project_id="proj-beta",
            actor="beta-lead",
            rationale="beta has custom warmup",
        )
        print(f"    proj-gamma sees: "
              f"{store.resolve('proj-gamma', 'pattern.cache_warming')}")
        print(f"    proj-beta sees : "
              f"{store.resolve('proj-beta', 'pattern.cache_warming').owner_project_id}")

        print("\n[8] History for 'pattern.cache_warming':")
        for t in store.history("pattern.cache_warming"):
            print(f"    {t.ts[:19]}  {t.policy_verdict:20s}  "
                  f"{t.from_scope.value} → {t.to_scope.value}  "
                  f"actor={t.actor}")

        print("\n[9] Stats and audit tail:")
        for k, v in store.stats().items():
            print(f"    {k}: {v}")
        print("    last audit entries:")
        for e in store.audit_log(limit=5):
            print(f"      {e.ts[:19]}  project={e.project_id:12s}  "
                  f"action={e.action:20s}  count={e.result_count}")

    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
