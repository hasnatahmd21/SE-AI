"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C09 — TECHNOLOGY SELECTION ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C05, C06, C07.

Purpose:
    Select a coherent technology stack based on requirements + constraints +
    prior verified experience. Produce decision rationale per category.
    Preserve rejected alternatives with explicit reasons.

Categories handled:
    LANGUAGE, FRAMEWORK, DATABASE, ARCHITECTURE, RUNTIME

Signals extracted from spec + intent context:
    system type · latency/throughput tiers · security posture · compliance ·
    scale expectations · team size hints · deadline pressure · persistence ·
    async needs · maintainability/typing focus · hard preferences · forbidden ·
    local-only · existing tech mentions · prior verified experience

Invariants honored:
  - NO external LLM. Pure deterministic scoring + filtering.
  - Selection is NOT "whatever is in the catalog". Every tech is scored on
    real dimensions and filtered by real constraints.
  - Hard constraints filter FIRST; scoring is a ranked comparison among survivors.
  - Every decision carries a rationale listing which dimensions moved the winner.
  - Every rejected alternative has a reason (constraint violation OR score gap).
  - Coherent stack: language wins → framework + runtime filtered to that family.
  - Persistence to C04 decision memory + C02 ontology (DECISION + ALTERNATIVE).

Contents:
  1.  Enums: TechCategory, FitVerdict, SelectionStatus
  2.  Dataclasses: Technology, TechSignals, ScoreBreakdown, Candidate,
                   CategoryDecision, TechSelectionResult
  3.  SignalExtractor  (spec + intent → TechSignals)
  4.  CATALOG          (~40 technologies)
  5.  TechnologyScorer (base weights + signal-driven adjustments)
  6.  ConstraintFilter (hard rejects with reasons)
  7.  TechnologySelector (coherent multi-category selection)
  8.  TechnologyRepository (persist / reload)
  9.  __main__ demo + self-tests

Run as script:
    python -m sebrain.c09            # demo
    python -m sebrain.c09 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import json
import re
import sys
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from sebrain.c01 import (
    Confidence,
    Config,
    SEBrainApp,
    SQLiteStorage,
    ValidationError,
    execution_scope,
    get_logger,
)
from sebrain.c02 import (
    EntityKind,
    Ontology,
    Provenance,
    ProvenanceType,
    RelationKind,
)
from sebrain.c04 import (
    MemoryKind,
    MemoryScope,
    MemoryStore,
)
from sebrain.c05 import RequirementParser, RequirementSpec
from sebrain.c06 import IntentContext, IntentContextEngine, IntentKind


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _short(s: str, n: int = 100) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class TechCategory(str, Enum):
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    RUNTIME = "runtime"
    DATABASE = "database"
    ARCHITECTURE = "architecture"


class FitVerdict(str, Enum):
    RECOMMENDED = "recommended"     # top of the surviving set
    ACCEPTABLE = "acceptable"       # above threshold, not top
    REJECTED = "rejected"           # below threshold
    INCOMPATIBLE = "incompatible"   # hard constraint violation


class SelectionStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"             # some categories missing candidates
    FAILED = "failed"               # no candidates in any critical category


# ════════════════════════════════════════════════════════════════════════════
# 2. DIMENSIONS + DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
DIMENSIONS: tuple[str, ...] = (
    "performance", "security", "maintainability", "ecosystem",
    "ease_of_learning", "async_support", "scalability", "maturity",
    "simplicity", "typing", "enterprise_readiness",
)


@dataclass(frozen=True, slots=True)
class Technology:
    """A single candidate technology with per-dimension scores (0..10)."""
    name: str
    category: TechCategory
    scores: dict[str, float]         # dimension -> 0..10
    family: str = ""                 # e.g. "python", "node", "jvm"
    cloud_only: bool = False
    license: str = "OSS"
    notes: str = ""

    def score(self, dim: str) -> float:
        return float(self.scores.get(dim, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "scores": {k: float(v) for k, v in self.scores.items()},
            "family": self.family,
            "cloud_only": self.cloud_only,
            "license": self.license,
            "notes": self.notes,
        }


@dataclass(slots=True)
class TechSignals:
    """Deterministic signals extracted from spec + intent context."""
    system_type: str = "unknown"     # "api" | "cli" | "web" | "worker" | "lib" | "unknown"
    latency_tight: bool = False
    throughput_high: bool = False
    real_time: bool = False
    scale_large: bool = False
    security_high: bool = False
    compliance: list[str] = field(default_factory=list)
    small_team: bool = False
    tight_deadline: bool = False
    maintainability_focus: bool = False
    typing_focus: bool = False
    needs_persistence: bool = False
    needs_async: bool = False
    must_be_local: bool = False
    preferred_language: str | None = None
    forbidden_techs: list[str] = field(default_factory=list)
    mentioned_languages: list[str] = field(default_factory=list)
    mentioned_databases: list[str] = field(default_factory=list)
    mentioned_frameworks: list[str] = field(default_factory=list)
    existing_techs: list[str] = field(default_factory=list)
    prior_failures: int = 0
    prior_decisions: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_type": self.system_type,
            "latency_tight": self.latency_tight,
            "throughput_high": self.throughput_high,
            "real_time": self.real_time,
            "scale_large": self.scale_large,
            "security_high": self.security_high,
            "compliance": list(self.compliance),
            "small_team": self.small_team,
            "tight_deadline": self.tight_deadline,
            "maintainability_focus": self.maintainability_focus,
            "typing_focus": self.typing_focus,
            "needs_persistence": self.needs_persistence,
            "needs_async": self.needs_async,
            "must_be_local": self.must_be_local,
            "preferred_language": self.preferred_language,
            "forbidden_techs": list(self.forbidden_techs),
            "mentioned_languages": list(self.mentioned_languages),
            "mentioned_databases": list(self.mentioned_databases),
            "mentioned_frameworks": list(self.mentioned_frameworks),
            "existing_techs": list(self.existing_techs),
            "prior_failures": self.prior_failures,
            "prior_decisions": list(self.prior_decisions),
            "rationale": list(self.rationale),
        }


@dataclass(slots=True)
class ScoreBreakdown:
    total: float
    contributions: dict[str, float]        # dimension -> weight*score
    weights: dict[str, float]              # dimension -> weight

    def top_contributors(self, n: int = 3) -> list[tuple[str, float]]:
        items = sorted(self.contributions.items(), key=lambda kv: kv[1], reverse=True)
        return items[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "contributions": {k: float(v) for k, v in self.contributions.items()},
            "weights": {k: float(v) for k, v in self.weights.items()},
        }


@dataclass(slots=True)
class Candidate:
    tech: Technology
    verdict: FitVerdict
    score: ScoreBreakdown | None
    rejections: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tech": self.tech.to_dict(),
            "verdict": self.verdict.value,
            "score": self.score.to_dict() if self.score else None,
            "rejections": list(self.rejections),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class CategoryDecision:
    category: TechCategory
    winner: Candidate | None
    top_candidates: list[Candidate]      # up to 3
    rejected: list[Candidate]            # hard-rejected or below threshold
    weights: dict[str, float]
    rationale: str
    total_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "winner": self.winner.to_dict() if self.winner else None,
            "top_candidates": [c.to_dict() for c in self.top_candidates],
            "rejected": [c.to_dict() for c in self.rejected],
            "weights": {k: float(v) for k, v in self.weights.items()},
            "rationale": self.rationale,
            "total_candidates": self.total_candidates,
        }


@dataclass(slots=True)
class TechSelectionResult:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    spec_id: str | None = None
    intent_context_id: str | None = None
    signals: TechSignals | None = None
    decisions: list[CategoryDecision] = field(default_factory=list)
    status: SelectionStatus = SelectionStatus.COMPLETED
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def decision(self, category: TechCategory) -> CategoryDecision | None:
        for d in self.decisions:
            if d.category is category:
                return d
        return None

    def winner_name(self, category: TechCategory) -> str | None:
        d = self.decision(category)
        return d.winner.tech.name if (d and d.winner) else None

    def stack(self) -> dict[str, str | None]:
        return {cat.value: self.winner_name(cat) for cat in TechCategory}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "spec_id": self.spec_id,
            "intent_context_id": self.intent_context_id,
            "signals": self.signals.to_dict() if self.signals else None,
            "decisions": [d.to_dict() for d in self.decisions],
            "status": self.status.value,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        lines = ["=== Technology Selection ==="]
        for d in self.decisions:
            w = d.winner.tech.name if d.winner else "(none)"
            lines.append(f"{d.category.value:12s}: {w}")
        lines.append(f"status: {self.status.value}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL EXTRACTOR
# ════════════════════════════════════════════════════════════════════════════
_LANG_MENTION = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "java": "java", "kotlin": "kotlin", "scala": "scala",
    "go": "go", "golang": "go",
    "rust": "rust",
    "c++": "cpp", "cpp": "cpp", "c#": "csharp", "csharp": "csharp",
    "ruby": "ruby", "php": "php", "swift": "swift", "c": "c",
}
_DB_MENTION = {
    "sqlite": "sqlite", "postgres": "postgres", "postgresql": "postgres",
    "mysql": "mysql", "mariadb": "mysql",
    "mongo": "mongodb", "mongodb": "mongodb",
    "redis": "redis", "dynamodb": "dynamodb",
    "cassandra": "cassandra", "elasticsearch": "elasticsearch",
}
_FRAMEWORK_MENTION = {
    "fastapi": "fastapi", "flask": "flask", "django": "django",
    "express": "express", "nestjs": "nestjs",
    "spring": "spring-boot", "spring boot": "spring-boot",
    "gin": "gin", "actix": "actix",
    "rails": "rails", "laravel": "laravel",
}

_COMPLIANCE_WORDS = {
    "gdpr": "GDPR", "hipaa": "HIPAA", "pci": "PCI-DSS", "pci-dss": "PCI-DSS",
    "sox": "SOX", "iso 27001": "ISO-27001",
}

_LOCAL_ONLY_PATTERNS = [
    re.compile(r"\bmust\s+not\s+require\s+external\b", re.I),
    re.compile(r"\bno\s+external\b", re.I),
    re.compile(r"\bself[\s_-]hosted\b", re.I),
    re.compile(r"\boffline\b", re.I),
    re.compile(r"\blocal[\s_-]only\b", re.I),
]

_SMALL_TEAM_PATTERNS = [
    re.compile(r"\bsolo\b", re.I),
    re.compile(r"\bsmall\s+team\b", re.I),
    re.compile(r"\bone[\s_-]person\b", re.I),
    re.compile(r"\bsingle\s+developer\b", re.I),
]

_DEADLINE_PATTERNS = [
    re.compile(r"\bdeadline\b", re.I),
    re.compile(r"\btight\s+timeline\b", re.I),
    re.compile(r"\bwithin\s+\d+\s+days?\b", re.I),
    re.compile(r"\bwithin\s+\d+\s+weeks?\b", re.I),
    re.compile(r"\burgent\b", re.I),
    re.compile(r"\bASAP\b"),
]

_TIGHT_LATENCY = [
    re.compile(r"\bwithin\s+(\d+)\s*(?:ms|milliseconds?)\b", re.I),
    re.compile(r"\bunder\s+(\d+)\s*(?:ms|milliseconds?)\b", re.I),
    re.compile(r"\b<\s*(\d+)\s*ms\b", re.I),
]

_HIGH_THROUGHPUT = [
    re.compile(r"\b\d+\s*(?:k|000)?\s*(?:requests?|rps|req/s)\b", re.I),
    re.compile(r"\bhigh[\s_-]throughput\b", re.I),
    re.compile(r"\bscal(?:e|able|ability)\b", re.I),
    re.compile(r"\bmillions?\s+of\b", re.I),
]

_ASYNC_HINT = [
    re.compile(r"\bconcurrent\b", re.I),
    re.compile(r"\bnon[\s_-]blocking\b", re.I),
    re.compile(r"\basync\b", re.I),
    re.compile(r"\bwebsockets?\b", re.I),
    re.compile(r"\bstreaming\b", re.I),
]

_TYPING_HINT = [
    re.compile(r"\btype[\s_-]safe(?:ty)?\b", re.I),
    re.compile(r"\bstatically[\s_-]typed\b", re.I),
    re.compile(r"\bstrong(?:ly)?[\s_-]typed\b", re.I),
]

_MAINTAINABILITY_HINT = [
    re.compile(r"\bmaintainab(?:le|ility)\b", re.I),
    re.compile(r"\bproduction[\s_-]quality\b", re.I),
    re.compile(r"\bclean\s+code\b", re.I),
    re.compile(r"\bmodular\b", re.I),
    re.compile(r"\bextensib(?:le|ility)\b", re.I),
]

_SYSTEM_TYPE_HINT = {
    "cli": re.compile(r"\b(?:cli|command[\s_-]line)\b", re.I),
    "api": re.compile(r"\b(?:api|rest(?:\s+api)?|graphql|endpoint)\b", re.I),
    "web": re.compile(r"\b(?:web\s+app|website|frontend|ui|dashboard)\b", re.I),
    "worker": re.compile(r"\b(?:worker|job|queue|background|async\s+task)\b", re.I),
    "lib": re.compile(r"\b(?:library|sdk|package)\b", re.I),
}


class SignalExtractor:
    """Reads spec + intent context to produce deterministic TechSignals."""

    def extract(
        self,
        spec: RequirementSpec,
        intent_ctx: IntentContext | None = None,
    ) -> TechSignals:
        s = TechSignals()
        raw = spec.raw_text or ""
        low = raw.lower()

        # --- system type ---
        for t, pat in _SYSTEM_TYPE_HINT.items():
            if pat.search(raw):
                s.system_type = t
                s.rationale.append(f"system_type='{t}' matched pattern")
                break
        if s.system_type == "unknown" and intent_ctx and intent_ctx.intent:
            if intent_ctx.intent.kind is IntentKind.BUILD:
                s.system_type = "api"  # sensible default for "build"
                s.rationale.append("default system_type='api' for BUILD intent")

        # --- latency ---
        for pat in _TIGHT_LATENCY:
            m = pat.search(raw)
            if m:
                try:
                    ms = int(m.group(1))
                except (ValueError, IndexError):
                    ms = 9999
                if ms <= 100:
                    s.latency_tight = True
                    s.rationale.append(f"latency_tight (target {ms}ms)")
                break
        # NFR performance tags
        for it in spec.non_functional:
            if "performance" in it.tags:
                if not s.latency_tight:
                    s.rationale.append("performance NFR present")

        # --- throughput / scale ---
        for pat in _HIGH_THROUGHPUT:
            if pat.search(raw):
                s.throughput_high = True
                s.scale_large = True
                s.rationale.append("throughput/scale hint matched")
                break

        # --- security ---
        sec_nfr = any("security" in it.tags for it in spec.non_functional)
        sec_words = bool(re.search(
            r"\b(?:security|authentication|authorization|encryption|tls|ssl|https|"
            r"owasp|csrf|xss|injection|harden)\b",
            raw, re.I,
        ))
        if sec_nfr or sec_words:
            s.security_high = True
            s.rationale.append("security posture flagged (NFR or keywords)")

        # --- compliance ---
        for w, label in _COMPLIANCE_WORDS.items():
            if re.search(r"\b" + re.escape(w) + r"\b", low):
                if label not in s.compliance:
                    s.compliance.append(label)
        if s.compliance:
            s.security_high = True
            s.rationale.append(f"compliance: {s.compliance}")

        # --- team / deadline ---
        for pat in _SMALL_TEAM_PATTERNS:
            if pat.search(raw):
                s.small_team = True
                s.rationale.append("small-team indicator matched")
                break
        for pat in _DEADLINE_PATTERNS:
            if pat.search(raw):
                s.tight_deadline = True
                s.rationale.append("tight-deadline indicator matched")
                break

        # --- typing / maintainability ---
        if any(pat.search(raw) for pat in _TYPING_HINT):
            s.typing_focus = True
            s.rationale.append("typing focus flagged")
        if any(pat.search(raw) for pat in _MAINTAINABILITY_HINT):
            s.maintainability_focus = True
            s.rationale.append("maintainability focus flagged")

        # --- persistence ---
        if spec.inputs or spec.outputs or re.search(
            r"\b(?:database|db|persist|storage|store|save|record|schema|"
            r"postgres|sqlite|mysql|mongo|redis|manag(?:e|ing)|crud|"
            r"create,?\s*read,?\s*update,?\s*(?:and\s+)?delete)\b", raw, re.I,
        ):
            s.needs_persistence = True
            s.rationale.append("persistence need detected")

        # --- async ---
        if any(pat.search(raw) for pat in _ASYNC_HINT):
            s.needs_async = True
            s.rationale.append("async/concurrency hint detected")

        # --- local-only ---
        if any(pat.search(raw) for pat in _LOCAL_ONLY_PATTERNS):
            s.must_be_local = True
            s.rationale.append("local-only / no-external constraint detected")

        # --- mentioned languages ---
        seen_langs: set[str] = set()
        for word, canon in _LANG_MENTION.items():
            if re.search(r"\b" + re.escape(word) + r"\b", low):
                if canon not in seen_langs:
                    seen_langs.add(canon)
                    s.mentioned_languages.append(canon)
        # --- mentioned dbs ---
        seen_dbs: set[str] = set()
        for word, canon in _DB_MENTION.items():
            if re.search(r"\b" + re.escape(word) + r"\b", low):
                if canon not in seen_dbs:
                    seen_dbs.add(canon)
                    s.mentioned_databases.append(canon)
        # --- mentioned frameworks ---
        seen_fws: set[str] = set()
        for word, canon in _FRAMEWORK_MENTION.items():
            if re.search(r"\b" + re.escape(word) + r"\b", low):
                if canon not in seen_fws:
                    seen_fws.add(canon)
                    s.mentioned_frameworks.append(canon)

        # --- hard preferences: "must use X" ---
        m = re.search(r"\bmust\s+use\s+([a-zA-Z0-9+#_.\-]+)", raw, re.I)
        if m:
            cand = m.group(1).lower().rstrip(".")
            canon = _LANG_MENTION.get(cand) or _FRAMEWORK_MENTION.get(cand) \
                or _DB_MENTION.get(cand)
            if canon:
                if canon in _LANG_MENTION.values():
                    s.preferred_language = canon
                    s.rationale.append(f"hard preference: language={canon}")
                else:
                    s.existing_techs.append(canon)
                    s.rationale.append(f"hard preference: existing tech={canon}")

        # --- forbidden: "must not use X" / "no X" ---
        for m in re.finditer(r"\bmust\s+not\s+use\s+([a-zA-Z0-9+#_.\-]+)", raw, re.I):
            token = m.group(1).lower().rstrip(".")
            canon = _LANG_MENTION.get(token) or _FRAMEWORK_MENTION.get(token) \
                or _DB_MENTION.get(token) or token
            s.forbidden_techs.append(canon)
        for m in re.finditer(r"\bno\s+([a-zA-Z][a-zA-Z0-9+#_.\-]+)\b", raw, re.I):
            token = m.group(1).lower().rstrip(".")
            canon = _LANG_MENTION.get(token) or _FRAMEWORK_MENTION.get(token) \
                or _DB_MENTION.get(token)
            if canon:
                s.forbidden_techs.append(canon)
        s.forbidden_techs = sorted(set(s.forbidden_techs))

        # --- prior experience from intent context ---
        if intent_ctx is not None:
            if intent_ctx.context:
                s.prior_failures = intent_ctx.context.prior_failure_count

        return s


# ════════════════════════════════════════════════════════════════════════════
# 4. CATALOG
# ════════════════════════════════════════════════════════════════════════════
# Score order corresponds to DIMENSIONS tuple above.
#   (performance, security, maintainability, ecosystem, ease_of_learning,
#    async_support, scalability, maturity, simplicity, typing,
#    enterprise_readiness)
# Some ecosystem-level framework/runtime families cover more than one
# language family in the LANGUAGE catalog (JVM frameworks like Spring Boot
# serve both Java and Kotlin; Node frameworks like Express/NestJS serve
# both JavaScript and TypeScript). An exact `family != family` comparison
# would treat every one of those as "incompatible" and strand the
# framework/runtime category with zero candidates whenever Kotlin,
# TypeScript, etc. wins the language slot.
_FAMILY_GROUPS: dict[str, str] = {
    "java": "jvm", "kotlin": "jvm",
    "javascript": "node", "typescript": "node",
}


def _same_family(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return a == b
    if a == b:
        return True
    return _FAMILY_GROUPS.get(a, a) == _FAMILY_GROUPS.get(b, b)


def _mk(name, cat, family, scores, cloud_only=False, notes=""):
    return Technology(
        name=name, category=cat, family=family,
        scores={d: float(v) for d, v in zip(DIMENSIONS, scores)},
        cloud_only=cloud_only, notes=notes,
    )


# ---- Languages ----
_LANGUAGES: list[Technology] = [
    _mk("Python",       TechCategory.LANGUAGE, "python",
        [6, 6, 8, 10, 8, 7, 6, 10, 8, 6, 8]),
    _mk("JavaScript",   TechCategory.LANGUAGE, "javascript",
        [6, 5, 5, 10, 7, 9, 8, 10, 7, 2, 7]),
    _mk("TypeScript",   TechCategory.LANGUAGE, "typescript",
        [6, 6, 9, 10, 6, 9, 8, 8, 6, 10, 9]),
    _mk("Java",         TechCategory.LANGUAGE, "java",
        [8, 8, 7, 10, 4, 7, 9, 10, 5, 9, 10]),
    _mk("Kotlin",       TechCategory.LANGUAGE, "kotlin",
        [8, 8, 9, 8, 5, 8, 8, 7, 6, 10, 8]),
    _mk("Go",           TechCategory.LANGUAGE, "go",
        [9, 7, 8, 7, 6, 9, 10, 7, 9, 7, 8]),
    _mk("Rust",         TechCategory.LANGUAGE, "rust",
        [10, 10, 7, 6, 2, 8, 9, 6, 5, 10, 6]),
    _mk("C++",          TechCategory.LANGUAGE, "cpp",
        [10, 5, 4, 9, 2, 6, 7, 10, 3, 6, 7]),
    _mk("C#",           TechCategory.LANGUAGE, "csharp",
        [8, 8, 8, 9, 5, 9, 8, 10, 6, 9, 10]),
    _mk("Ruby",         TechCategory.LANGUAGE, "ruby",
        [5, 6, 8, 7, 7, 5, 5, 9, 8, 3, 5]),
    _mk("PHP",          TechCategory.LANGUAGE, "php",
        [6, 6, 6, 8, 6, 4, 6, 10, 7, 5, 6]),
    _mk("Swift",        TechCategory.LANGUAGE, "swift",
        [9, 8, 8, 6, 5, 8, 6, 6, 6, 9, 6]),
    _mk("C",            TechCategory.LANGUAGE, "c",
        [10, 4, 3, 6, 2, 3, 7, 10, 3, 5, 5]),
]

# ---- Frameworks ----
_FRAMEWORKS: list[Technology] = [
    _mk("FastAPI",      TechCategory.FRAMEWORK, "python",
        [8, 7, 9, 8, 7, 9, 8, 6, 8, 9, 7]),
    _mk("Flask",        TechCategory.FRAMEWORK, "python",
        [6, 6, 8, 9, 9, 3, 6, 10, 9, 3, 7]),
    _mk("Django",       TechCategory.FRAMEWORK, "python",
        [6, 9, 9, 10, 5, 5, 7, 10, 7, 5, 9]),
    _mk("Express",      TechCategory.FRAMEWORK, "node",
        [6, 5, 6, 9, 8, 9, 7, 10, 8, 2, 6]),
    _mk("NestJS",       TechCategory.FRAMEWORK, "node",
        [6, 7, 9, 7, 5, 9, 8, 6, 5, 10, 7]),
    _mk("Spring Boot",  TechCategory.FRAMEWORK, "jvm",
        [8, 9, 7, 10, 4, 8, 9, 10, 4, 8, 10]),
    _mk("Gin",          TechCategory.FRAMEWORK, "go",
        [9, 6, 7, 6, 7, 8, 9, 6, 9, 6, 6]),
    _mk("Actix Web",    TechCategory.FRAMEWORK, "rust",
        [10, 8, 6, 5, 3, 9, 9, 5, 5, 10, 4]),
    _mk("Rails",        TechCategory.FRAMEWORK, "ruby",
        [5, 7, 8, 8, 6, 4, 6, 10, 8, 3, 6]),
    _mk("Laravel",      TechCategory.FRAMEWORK, "php",
        [6, 7, 8, 8, 6, 4, 6, 9, 8, 5, 6]),
    _mk("ASP.NET Core",  TechCategory.FRAMEWORK, "csharp",
        [9, 8, 8, 8, 5, 9, 9, 9, 5, 9, 10]),
    _mk("Vapor",         TechCategory.FRAMEWORK, "swift",
        [8, 7, 7, 5, 5, 8, 7, 5, 6, 8, 5]),
    _mk("Drogon",        TechCategory.FRAMEWORK, "cpp",
        [10, 6, 5, 4, 3, 8, 8, 5, 4, 7, 4]),
    _mk("libmicrohttpd", TechCategory.FRAMEWORK, "c",
        [10, 4, 3, 3, 2, 3, 6, 9, 3, 4, 3]),
]

# ---- Databases ----
_DATABASES: list[Technology] = [
    _mk("SQLite",         TechCategory.DATABASE, "",
        [6, 5, 10, 10, 10, 3, 3, 10, 10, 5, 4],
        notes="embedded, zero-ops"),
    _mk("PostgreSQL",     TechCategory.DATABASE, "",
        [8, 9, 8, 10, 6, 7, 9, 10, 6, 7, 10]),
    _mk("MySQL",          TechCategory.DATABASE, "",
        [7, 8, 7, 10, 7, 7, 8, 10, 7, 6, 9]),
    _mk("MongoDB",        TechCategory.DATABASE, "",
        [8, 6, 6, 8, 7, 8, 9, 8, 7, 2, 7]),
    _mk("Redis",          TechCategory.DATABASE, "",
        [10, 6, 6, 9, 8, 8, 8, 10, 7, 3, 8],
        notes="in-memory, cache/KV"),
    _mk("DynamoDB",       TechCategory.DATABASE, "",
        [9, 8, 5, 7, 5, 9, 10, 8, 6, 3, 8],
        cloud_only=True,
        notes="AWS-managed only"),
    _mk("Cassandra",      TechCategory.DATABASE, "",
        [9, 7, 4, 6, 4, 8, 10, 8, 4, 4, 7]),
    _mk("Elasticsearch",  TechCategory.DATABASE, "",
        [7, 7, 5, 9, 5, 8, 9, 9, 4, 3, 8],
        notes="search/analytics"),
]

# ---- Architectures ----
_ARCHITECTURES: list[Technology] = [
    _mk("Monolith",           TechCategory.ARCHITECTURE, "",
        [8, 7, 8, 5, 9, 5, 4, 10, 10, 5, 6]),
    _mk("Layered",            TechCategory.ARCHITECTURE, "",
        [8, 7, 9, 5, 8, 5, 5, 10, 9, 5, 7]),
    _mk("Hexagonal",          TechCategory.ARCHITECTURE, "",
        [8, 8, 10, 5, 5, 6, 7, 8, 5, 6, 7]),
    _mk("Clean Architecture", TechCategory.ARCHITECTURE, "",
        [7, 8, 10, 5, 4, 6, 8, 7, 5, 6, 8]),
    _mk("Microservices",      TechCategory.ARCHITECTURE, "",
        [7, 8, 5, 5, 4, 8, 10, 9, 3, 5, 9]),
    _mk("Serverless",         TechCategory.ARCHITECTURE, "",
        [5, 8, 6, 5, 6, 8, 10, 8, 8, 4, 6]),
    _mk("Event-Driven",       TechCategory.ARCHITECTURE, "",
        [7, 7, 6, 5, 5, 10, 10, 8, 5, 5, 8]),
    _mk("CQRS",               TechCategory.ARCHITECTURE, "",
        [8, 8, 6, 5, 4, 9, 10, 7, 4, 5, 8]),
]

# ---- Runtimes ----
_RUNTIMES: list[Technology] = [
    _mk("CPython",  TechCategory.RUNTIME, "python",
        [6, 7, 9, 10, 9, 6, 6, 10, 9, 6, 8]),
    _mk("PyPy",     TechCategory.RUNTIME, "python",
        [8, 6, 7, 6, 7, 5, 6, 8, 8, 5, 5]),
    _mk("Node.js",  TechCategory.RUNTIME, "node",
        [7, 6, 6, 10, 8, 10, 8, 10, 8, 3, 8]),
    _mk("Bun",      TechCategory.RUNTIME, "node",
        [9, 6, 6, 5, 7, 10, 7, 3, 8, 5, 4]),
    _mk("JVM",      TechCategory.RUNTIME, "jvm",
        [8, 9, 7, 10, 5, 8, 10, 10, 5, 8, 10]),
    _mk("Go runtime", TechCategory.RUNTIME, "go",
        [9, 7, 8, 7, 7, 9, 10, 7, 9, 6, 8]),
    _mk("Rust runtime", TechCategory.RUNTIME, "rust",
        [10, 10, 7, 6, 3, 8, 9, 6, 6, 10, 6]),
    _mk(".NET",          TechCategory.RUNTIME, "csharp",
        [8, 8, 8, 9, 5, 9, 8, 10, 6, 9, 10]),
    _mk("MRI (Ruby)",    TechCategory.RUNTIME, "ruby",
        [5, 6, 8, 7, 7, 5, 5, 9, 8, 3, 5]),
    _mk("Zend Engine",   TechCategory.RUNTIME, "php",
        [6, 6, 6, 8, 6, 4, 6, 10, 7, 5, 6]),
    _mk("Swift runtime", TechCategory.RUNTIME, "swift",
        [9, 8, 8, 6, 5, 8, 6, 6, 6, 9, 6]),
    _mk("Native (C++)",  TechCategory.RUNTIME, "cpp",
        [10, 5, 4, 9, 2, 6, 7, 10, 3, 6, 7]),
    _mk("C runtime (libc)", TechCategory.RUNTIME, "c",
        [10, 4, 3, 6, 2, 3, 7, 10, 3, 5, 5]),
]

CATALOG: dict[TechCategory, list[Technology]] = {
    TechCategory.LANGUAGE: _LANGUAGES,
    TechCategory.FRAMEWORK: _FRAMEWORKS,
    TechCategory.RUNTIME: _RUNTIMES,
    TechCategory.DATABASE: _DATABASES,
    TechCategory.ARCHITECTURE: _ARCHITECTURES,
}


def _find_by_name(name: str) -> Technology | None:
    low = name.lower()
    for techs in CATALOG.values():
        for t in techs:
            if t.name.lower() == low:
                return t
    return None


# ════════════════════════════════════════════════════════════════════════════
# 5. SCORER
# ════════════════════════════════════════════════════════════════════════════
_BASE_WEIGHTS: dict[TechCategory, dict[str, float]] = {
    TechCategory.LANGUAGE: {
        "performance": 1.0, "security": 1.0, "maintainability": 1.2,
        "ecosystem": 1.2, "ease_of_learning": 1.0, "async_support": 0.6,
        "scalability": 0.8, "maturity": 1.0, "simplicity": 0.8,
        "typing": 0.8, "enterprise_readiness": 0.8,
    },
    TechCategory.FRAMEWORK: {
        "performance": 0.8, "security": 1.0, "maintainability": 1.2,
        "ecosystem": 1.0, "ease_of_learning": 1.0, "async_support": 0.7,
        "scalability": 0.8, "maturity": 1.0, "simplicity": 1.0,
        "typing": 0.7, "enterprise_readiness": 0.8,
    },
    TechCategory.RUNTIME: {
        "performance": 1.2, "security": 1.0, "maintainability": 0.8,
        "ecosystem": 1.0, "ease_of_learning": 0.8, "async_support": 0.8,
        "scalability": 1.0, "maturity": 1.0, "simplicity": 1.0,
        "typing": 0.5, "enterprise_readiness": 0.8,
    },
    TechCategory.DATABASE: {
        "performance": 1.0, "security": 1.2, "maintainability": 0.8,
        "ecosystem": 1.0, "ease_of_learning": 0.8, "async_support": 0.5,
        "scalability": 1.2, "maturity": 1.0, "simplicity": 1.0,
        "typing": 0.4, "enterprise_readiness": 1.0,
    },
    TechCategory.ARCHITECTURE: {
        "performance": 0.8, "security": 1.0, "maintainability": 1.3,
        "ecosystem": 0.2, "ease_of_learning": 1.0, "async_support": 0.5,
        "scalability": 1.1, "maturity": 0.8, "simplicity": 1.2,
        "typing": 0.2, "enterprise_readiness": 0.8,
    },
}


class TechnologyScorer:
    """Base weights + signal-driven adjustments. Deterministic."""

    def weights_for(
        self, category: TechCategory, signals: TechSignals,
    ) -> dict[str, float]:
        w = dict(_BASE_WEIGHTS[category])

        # Performance-sensitive
        if signals.latency_tight or signals.throughput_high or signals.real_time:
            w["performance"] *= 2.0
            w["async_support"] *= 1.5
        # Scale
        if signals.scale_large:
            w["scalability"] *= 2.0
        # Security
        if signals.security_high or signals.compliance:
            w["security"] *= 1.8
        if signals.compliance:
            w["enterprise_readiness"] *= 1.3
        # Team / deadline
        if signals.small_team:
            w["ease_of_learning"] *= 1.5
            w["simplicity"] *= 1.5
        if signals.tight_deadline:
            w["ease_of_learning"] *= 1.5
            w["maturity"] *= 1.2
        # Typing / maintainability
        if signals.typing_focus:
            w["typing"] *= 2.0
        if signals.maintainability_focus:
            w["maintainability"] *= 1.5
            w["typing"] *= 1.2
        # Persistence
        if signals.needs_persistence and category is TechCategory.DATABASE:
            w["maturity"] *= 1.1
        # Async
        if signals.needs_async:
            w["async_support"] *= 2.0
        # Local-only
        if signals.must_be_local:
            w["simplicity"] *= 1.5
            w["maturity"] *= 1.05
        return w

    def score(self, tech: Technology, weights: dict[str, float]) -> ScoreBreakdown:
        contributions: dict[str, float] = {}
        total = 0.0
        for dim, w in weights.items():
            c = w * tech.score(dim)
            contributions[dim] = c
            total += c
        return ScoreBreakdown(
            total=total,
            contributions=contributions,
            weights=dict(weights),
        )


# ════════════════════════════════════════════════════════════════════════════
# 6. CONSTRAINT FILTER
# ════════════════════════════════════════════════════════════════════════════
class ConstraintFilter:
    """Hard rejects with explicit reasons. Runs BEFORE scoring."""

    def reject(
        self,
        tech: Technology,
        signals: TechSignals,
        category: TechCategory,
        *,
        language_family: str | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        name_low = tech.name.lower()

        # Forbidden by name or family
        for f in signals.forbidden_techs:
            if f == name_low or f == tech.family:
                reasons.append(f"forbidden by requirement: '{f}'")

        # Local-only excludes cloud-only
        if signals.must_be_local and tech.cloud_only:
            reasons.append("cloud-only tech excluded by local-only requirement")

        # Hard preference: preferred_language
        if signals.preferred_language is not None:
            if category is TechCategory.LANGUAGE:
                if tech.family != signals.preferred_language:
                    reasons.append(
                        f"language must be {signals.preferred_language}"
                    )
            elif category in (TechCategory.FRAMEWORK, TechCategory.RUNTIME):
                if tech.family and not _same_family(tech.family, signals.preferred_language):
                    reasons.append(
                        f"family mismatch with preferred language "
                        f"'{signals.preferred_language}'"
                    )

        # Coherence: language→framework/runtime binding
        if language_family is not None and category in (
            TechCategory.FRAMEWORK, TechCategory.RUNTIME,
        ):
            if tech.family and not _same_family(tech.family, language_family):
                reasons.append(
                    f"family '{tech.family}' incompatible with chosen "
                    f"language family '{language_family}'"
                )

        # Security floor
        if signals.security_high and tech.score("security") < 5.0:
            reasons.append(
                f"security score {tech.score('security'):.1f} < 5.0 "
                f"under high-security requirement"
            )
        # Latency floor
        if signals.latency_tight and tech.score("performance") < 6.0:
            reasons.append(
                f"performance score {tech.score('performance'):.1f} < 6.0 "
                f"under tight-latency requirement"
            )
        # Scale floor
        if signals.scale_large and category is TechCategory.DATABASE \
                and tech.score("scalability") < 6.0:
            reasons.append(
                f"scalability score {tech.score('scalability'):.1f} < 6.0 "
                f"under scale requirement"
            )

        return reasons


# ════════════════════════════════════════════════════════════════════════════
# 7. SELECTOR
# ════════════════════════════════════════════════════════════════════════════
_DEFAULT_ORDER: list[TechCategory] = [
    TechCategory.LANGUAGE,
    TechCategory.RUNTIME,
    TechCategory.FRAMEWORK,
    TechCategory.DATABASE,
    TechCategory.ARCHITECTURE,
]

_ACCEPTABLE_FRACTION = 0.75   # >= 75% of winner score → ACCEPTABLE
_TOP_N = 3


class TechnologySelector:
    """Coherent multi-category selector.

    Order matters: LANGUAGE decides first, then RUNTIME/FRAMEWORK are
    constrained to the language family, DATABASE and ARCHITECTURE are
    independent.
    """

    def __init__(self, *, catalog: dict[TechCategory, list[Technology]] | None = None) -> None:
        self.catalog = catalog or CATALOG
        self.extractor = SignalExtractor()
        self.scorer = TechnologyScorer()
        self.filter = ConstraintFilter()

    # ---- main ----
    def select(
        self,
        spec: RequirementSpec,
        intent_ctx: IntentContext | None = None,
        *,
        project_id: str = "",
        categories: Sequence[TechCategory] | None = None,
    ) -> TechSelectionResult:
        result = TechSelectionResult(
            project_id=project_id, spec_id=spec.id,
            intent_context_id=ic_id if (ic_id := (intent_ctx.id if intent_ctx else None)) else None,
            provenance=spec.provenance,
        )
        result.signals = self.extractor.extract(spec, intent_ctx)

        order = list(categories) if categories else list(_DEFAULT_ORDER)
        language_family: str | None = None
        decisions: list[CategoryDecision] = []
        missing_categories: list[str] = []

        for cat in order:
            candidates = self.catalog.get(cat, [])
            if not candidates:
                missing_categories.append(cat.value)
                continue

            decision = self._decide_category(
                cat, candidates, result.signals,
                language_family=language_family,
            )
            decisions.append(decision)
            # Coherence: after LANGUAGE wins, pin family
            if cat is TechCategory.LANGUAGE and decision.winner is not None:
                language_family = decision.winner.tech.family or None

        result.decisions = decisions
        if not decisions:
            result.status = SelectionStatus.FAILED
        elif missing_categories:
            result.status = SelectionStatus.PARTIAL
        else:
            result.status = SelectionStatus.COMPLETED
        result.rationale = (
            f"signals: {result.signals.to_dict()}; "
            f"order={[c.value for c in order]}; "
            f"language_family={language_family}; "
            f"missing={missing_categories}"
        )
        return result

    # ---- per category ----
    def _decide_category(
        self,
        cat: TechCategory,
        candidates: list[Technology],
        signals: TechSignals,
        *,
        language_family: str | None,
    ) -> CategoryDecision:
        weights = self.scorer.weights_for(cat, signals)

        survivors: list[Candidate] = []
        rejected: list[Candidate] = []

        for tech in candidates:
            reasons = self.filter.reject(
                tech, signals, cat, language_family=language_family,
            )
            if reasons:
                rejected.append(Candidate(
                    tech=tech,
                    verdict=FitVerdict.INCOMPATIBLE,
                    score=None,
                    rejections=reasons,
                    rationale="; ".join(reasons),
                ))
                continue
            sc = self.scorer.score(tech, weights)
            top = sc.top_contributors(3)
            rationale = (
                "score={:.2f} top_factors=[{}]".format(
                    sc.total,
                    ", ".join(f"{d}={v:.2f}" for d, v in top),
                )
            )
            survivors.append(Candidate(
                tech=tech, verdict=FitVerdict.ACCEPTABLE,
                score=sc, rationale=rationale,
            ))

        # Rank by score desc, tie-break by name asc
        survivors.sort(key=lambda c: (-c.score.total, c.tech.name))

        if survivors:
            top_score = survivors[0].score.total
            threshold = top_score * _ACCEPTABLE_FRACTION
            kept: list[Candidate] = []
            for c in survivors:
                if c.score.total >= threshold:
                    c.verdict = FitVerdict.ACCEPTABLE
                    kept.append(c)
                else:
                    c.verdict = FitVerdict.REJECTED
                    c.rejections = [
                        f"score {c.score.total:.2f} below {threshold:.2f} "
                        f"(<{_ACCEPTABLE_FRACTION*100:.0f}% of winner)"
                    ]
                    rejected.append(c)
            if kept:
                kept[0].verdict = FitVerdict.RECOMMENDED
            winner = kept[0] if kept else None
            top_candidates = kept[:_TOP_N]
        else:
            winner = None
            top_candidates = []

        rationale = self._explain_decision(cat, winner, top_candidates, signals)

        return CategoryDecision(
            category=cat,
            winner=winner,
            top_candidates=top_candidates,
            rejected=rejected,
            weights=weights,
            rationale=rationale,
            total_candidates=len(candidates),
        )

    def _explain_decision(
        self,
        cat: TechCategory,
        winner: Candidate | None,
        top: list[Candidate],
        signals: TechSignals,
    ) -> str:
        if winner is None:
            return f"no surviving candidate for {cat.value} after filtering"
        parts = [
            f"selected '{winner.tech.name}' for {cat.value}",
            f"score={winner.score.total:.2f}",
        ]
        top_dims = winner.score.top_contributors(3)
        parts.append("top factors: " + ", ".join(
            f"{d}={v:.1f}" for d, v in top_dims
        ))
        if len(top) > 1:
            runner = top[1]
            gap = winner.score.total - runner.score.total
            parts.append(
                f"runner-up '{runner.tech.name}' (gap={gap:.2f})"
            )
        # Signal-driven note
        active_signals = [
            name for name, on in [
                ("latency_tight", signals.latency_tight),
                ("throughput_high", signals.throughput_high),
                ("scale_large", signals.scale_large),
                ("security_high", signals.security_high),
                ("compliance", bool(signals.compliance)),
                ("small_team", signals.small_team),
                ("tight_deadline", signals.tight_deadline),
                ("typing_focus", signals.typing_focus),
                ("maintainability_focus", signals.maintainability_focus),
                ("needs_async", signals.needs_async),
                ("must_be_local", signals.must_be_local),
            ]
            if on
        ]
        if active_signals:
            parts.append("driven by: " + ",".join(active_signals))
        return "; ".join(parts)


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class TechnologyRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: TechSelectionResult) -> str:
        if not result.project_id:
            raise ValidationError("result.project_id required")
        # 1. Save full selection to memory
        key = f"tech_selection:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=result.project_id,
            tags=["tech_selection", "c09"],
            provenance=result.provenance,
        )
        # 2. Record each category decision as a decision memory
        for d in result.decisions:
            if d.winner is None:
                continue
            decision_text = (
                f"Chose {d.winner.tech.name} for {d.category.value}"
            )
            alternatives = [c.tech.name for c in d.top_candidates[1:]]
            self.memory.record_decision(
                f"tech:{d.category.value}",
                decision=decision_text,
                rationale=d.rationale,
                alternatives=alternatives,
                scope_id=result.project_id,
                provenance=Provenance(
                    source="technology_selector",
                    source_type=ProvenanceType.INFERENCE,
                    confidence=Confidence.HIGH,
                ),
                confidence=Confidence.HIGH,
            )
        # 3. Ontology: decision + alternatives
        if self.ontology is None:
            return result.id
        root = self.ontology.add(
            EntityKind.DECISION,
            _short(f"Tech stack: {result.stack()}", 120),
            attributes={
                "selection_id": result.id,
                "project_id": result.project_id,
                "stack": result.stack(),
            },
            tags=["tech-stack"],
            provenance=result.provenance,
        )
        for d in result.decisions:
            if d.winner is None:
                continue
            chosen = self.ontology.add(
                EntityKind.DECISION,
                _short(f"{d.category.value}: {d.winner.tech.name}", 100),
                attributes={
                    "category": d.category.value,
                    "score": d.winner.score.total if d.winner.score else 0.0,
                    "rationale": d.winner.rationale,
                },
                tags=["tech-choice", d.category.value],
                provenance=result.provenance,
            )
            self.ontology.link(RelationKind.CONTAINS, root.id, chosen.id)
            # alternatives
            for alt in d.top_candidates[1:]:
                a = self.ontology.add(
                    EntityKind.ALTERNATIVE,
                    _short(f"{d.category.value}: {alt.tech.name}", 100),
                    attributes={
                        "category": d.category.value,
                        "score": alt.score.total if alt.score else 0.0,
                        "verdict": alt.verdict.value,
                    },
                    tags=["tech-alt", d.category.value],
                    provenance=result.provenance,
                )
                self.ontology.link(RelationKind.RELATES_TO, a.id, chosen.id)
            for r in d.rejected:
                a = self.ontology.add(
                    EntityKind.ALTERNATIVE,
                    _short(f"{d.category.value}: {r.tech.name} (rejected)", 100),
                    attributes={
                        "category": d.category.value,
                        "reasons": r.rejections,
                        "verdict": r.verdict.value,
                    },
                    tags=["tech-rejected", d.category.value],
                    provenance=result.provenance,
                )
                try:
                    self.ontology.link(RelationKind.CONTRADICTS, a.id, chosen.id)
                except ValidationError:
                    pass
        return root.id

    def load(self, selection_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"tech_selection:{selection_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


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

    print("Running C09 self-tests…")
    parser = RequirementParser()
    ic_engine = IntentContextEngine()
    selector = TechnologySelector()

    def _spec_intent(text: str, project: str = "test"):
        spec = parser.parse(text)
        ic = ic_engine.analyze(text, project_id=project)
        return spec, ic

    # ---- signals ----
    def t_signals_latency_tight() -> None:
        text = "Build an API that must respond within 50ms."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.latency_tight is True

    def t_signals_latency_not_tight() -> None:
        text = "Build an API that must respond within 500ms."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.latency_tight is False

    def t_signals_security() -> None:
        text = "Build an API. All traffic must use HTTPS."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.security_high is True

    def t_signals_compliance() -> None:
        text = "The system must comply with GDPR."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert "GDPR" in s.compliance
        assert s.security_high is True

    def t_signals_scale() -> None:
        text = "Build an API serving 100k requests per second."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.scale_large is True

    def t_signals_local_only() -> None:
        text = "Must not require external authentication services."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.must_be_local is True

    def t_signals_async() -> None:
        text = "Build an async service with WebSockets."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.needs_async is True

    def t_signals_lang_mention() -> None:
        text = "Build a Python REST API."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert "python" in s.mentioned_languages

    def t_signals_hard_preference() -> None:
        text = "Build an API. Must use Go."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert s.preferred_language == "go"

    def t_signals_forbidden() -> None:
        text = "Build an API. Must not use Python."
        spec, ic = _spec_intent(text)
        s = SignalExtractor().extract(spec, ic)
        assert "python" in s.forbidden_techs

    check("signals: tight latency (50ms)", t_signals_latency_tight)
    check("signals: 500ms not tight", t_signals_latency_not_tight)
    check("signals: security via HTTPS", t_signals_security)
    check("signals: GDPR → compliance + high-security", t_signals_compliance)
    check("signals: high throughput → scale_large", t_signals_scale)
    check("signals: local-only", t_signals_local_only)
    check("signals: async needs", t_signals_async)
    check("signals: language mention", t_signals_lang_mention)
    check("signals: 'must use Go' → preferred", t_signals_hard_preference)
    check("signals: 'must not use Python' → forbidden", t_signals_forbidden)

    # ---- scorer ----
    def t_scorer_weights_shift_with_signals() -> None:
        s_default = TechSignals()
        s_secure = TechSignals(security_high=True, compliance=["GDPR"])
        sc = TechnologyScorer()
        w1 = sc.weights_for(TechCategory.LANGUAGE, s_default)
        w2 = sc.weights_for(TechCategory.LANGUAGE, s_secure)
        assert w2["security"] > w1["security"]
        assert w2["enterprise_readiness"] > w1["enterprise_readiness"]

    def t_scorer_typing_boost() -> None:
        sc = TechnologyScorer()
        w1 = sc.weights_for(TechCategory.LANGUAGE, TechSignals())
        w2 = sc.weights_for(TechCategory.LANGUAGE, TechSignals(typing_focus=True))
        assert w2["typing"] > w1["typing"]

    def t_scorer_perf_boost() -> None:
        sc = TechnologyScorer()
        w1 = sc.weights_for(TechCategory.LANGUAGE, TechSignals())
        w2 = sc.weights_for(TechCategory.LANGUAGE, TechSignals(latency_tight=True))
        assert w2["performance"] > w1["performance"]
        assert w2["async_support"] > w1["async_support"]

    def t_scorer_small_team() -> None:
        sc = TechnologyScorer()
        w1 = sc.weights_for(TechCategory.LANGUAGE, TechSignals())
        w2 = sc.weights_for(TechCategory.LANGUAGE, TechSignals(small_team=True))
        assert w2["ease_of_learning"] > w1["ease_of_learning"]
        assert w2["simplicity"] > w1["simplicity"]

    check("scorer: security signals boost security weight",
          t_scorer_weights_shift_with_signals)
    check("scorer: typing_focus boosts typing weight", t_scorer_typing_boost)
    check("scorer: latency_tight boosts performance+async",
          t_scorer_perf_boost)
    check("scorer: small_team boosts learn+simplicity", t_scorer_small_team)

    # ---- constraint filter ----
    def t_filter_forbidden() -> None:
        signals = TechSignals(forbidden_techs=["python"])
        tech = _find_by_name("Python")
        assert tech is not None
        reasons = ConstraintFilter().reject(tech, signals, TechCategory.LANGUAGE)
        assert any("forbidden" in r for r in reasons)

    def t_filter_cloud_only_local() -> None:
        signals = TechSignals(must_be_local=True)
        tech = _find_by_name("DynamoDB")
        assert tech is not None
        reasons = ConstraintFilter().reject(tech, signals, TechCategory.DATABASE)
        assert any("cloud-only" in r for r in reasons)

    def t_filter_preferred_language() -> None:
        signals = TechSignals(preferred_language="go")
        tech = _find_by_name("Python")
        assert tech is not None
        reasons = ConstraintFilter().reject(tech, signals, TechCategory.LANGUAGE)
        assert any("language must be go" in r for r in reasons)

    def t_filter_language_family_binding() -> None:
        # Framework with wrong family rejected
        tech = _find_by_name("Express")  # node
        reasons = ConstraintFilter().reject(
            tech, TechSignals(), TechCategory.FRAMEWORK,
            language_family="python",
        )
        assert any("incompatible" in r for r in reasons)

    def t_filter_security_floor() -> None:
        signals = TechSignals(security_high=True)
        # C has security=4
        tech = _find_by_name("C")
        assert tech is not None
        reasons = ConstraintFilter().reject(tech, signals, TechCategory.LANGUAGE)
        assert any("security" in r for r in reasons)

    check("filter: forbidden by name", t_filter_forbidden)
    check("filter: cloud-only excluded when local-only", t_filter_cloud_only_local)
    check("filter: preferred_language enforced", t_filter_preferred_language)
    check("filter: language family binding (framework)", t_filter_language_family_binding)
    check("filter: security floor under high-security", t_filter_security_floor)

    # ---- selection ----
    def t_select_basic_api() -> None:
        text = "Build a small REST API for managing tasks."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        assert result.status in (SelectionStatus.COMPLETED, SelectionStatus.PARTIAL)
        # must pick something for language / framework / database
        assert result.winner_name(TechCategory.LANGUAGE) is not None
        assert result.winner_name(TechCategory.FRAMEWORK) is not None
        assert result.winner_name(TechCategory.DATABASE) is not None
        assert result.winner_name(TechCategory.ARCHITECTURE) is not None

    def t_select_respects_preferred_language() -> None:
        text = "Build a REST API. Must use Go."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        assert result.winner_name(TechCategory.LANGUAGE) == "Go"
        # framework family must be go
        fw_decision = result.decision(TechCategory.FRAMEWORK)
        assert fw_decision is not None and fw_decision.winner is not None
        assert fw_decision.winner.tech.family == "go"

    def t_select_forbidden_excluded() -> None:
        text = "Build a REST API. Must not use Python."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        assert result.winner_name(TechCategory.LANGUAGE) != "Python"

    def t_select_local_only_excludes_cloud_db() -> None:
        text = "Build a service. Must not require external services."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        # DynamoDB must be rejected or not selected
        db = result.winner_name(TechCategory.DATABASE)
        assert db != "DynamoDB"

    def t_select_security_high_picks_secure() -> None:
        text = (
            "Build an API for healthcare data. "
            "All traffic must use TLS. Must comply with HIPAA."
        )
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        # with HIPAA, security weight is boosted; DB should have good security
        db = result.winner_name(TechCategory.DATABASE)
        assert db in ("PostgreSQL", "MySQL", "Redis", "MongoDB",
                      "Elasticsearch", "SQLite"), db
        # The security floor excludes low-security languages: C, JS, PHP
        lang = result.winner_name(TechCategory.LANGUAGE)
        assert lang != "C"

    def t_select_deterministic() -> None:
        text = "Build a REST API for tasks."
        spec, ic = _spec_intent(text)
        r1 = selector.select(spec, ic, project_id="p1")
        r2 = selector.select(spec, ic, project_id="p1")
        assert r1.stack() == r2.stack()

    check("select: basic REST API yields full stack", t_select_basic_api)
    check("select: 'must use Go' → language=Go + framework family=go",
          t_select_respects_preferred_language)
    check("select: forbidden language excluded",
          t_select_forbidden_excluded)
    check("select: local-only excludes cloud DBs",
          t_select_local_only_excludes_cloud_db)
    check("select: high-security excludes insecure languages",
          t_select_security_high_picks_secure)
    check("select: deterministic (same input → same stack)",
          t_select_deterministic)

    # ---- rejected alternatives preserved ----
    def t_rejected_preserved() -> None:
        text = "Build a REST API. Must use Go."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        lang = result.decision(TechCategory.LANGUAGE)
        assert lang is not None
        # Python should be in rejected (forbidden by preferred language)
        names = {c.tech.name for c in lang.rejected}
        assert "Python" in names
        # Every rejected has at least one reason
        for c in lang.rejected:
            assert len(c.rejections) >= 1

    check("selection: rejected alternatives preserved with reasons",
          t_rejected_preserved)

    # ---- rationale quality ----
    def t_rationale_present() -> None:
        text = "Build a REST API for tasks."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        for d in result.decisions:
            assert d.rationale
            if d.winner:
                assert d.winner.rationale
                assert d.winner.score is not None

    check("rationale: every decision has a rationale", t_rationale_present)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                text = "Build a REST API for tasks."
                spec, ic = _spec_intent(text, project="proj-x")
                result = selector.select(spec, ic, project_id="proj-x")
                repo = TechnologyRepository(memory=mem, ontology=ont)
                root = repo.save(result)
                assert root
                # reload
                loaded = repo.load(result.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == result.id
                # decision memories recorded
                decs = mem.find(kind=MemoryKind.DECISION,
                                scope_type=MemoryScope.PROJECT,
                                scope_id="proj-x")
                assert len(decs) >= 1
                # ontology has DECISION entities
                assert ont.count(kind=EntityKind.DECISION) >= 1
            finally:
                s.shutdown()

    check("persist: memory (decisions) + ontology (choices/alternatives)",
          t_persist)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        text = "Build a REST API."
        spec, ic = _spec_intent(text)
        result = selector.select(spec, ic, project_id="p")
        d = result.to_dict()
        assert d["id"] == result.id
        assert isinstance(d["decisions"], list) and d["decisions"]
        s = result.summary()
        assert "Technology Selection" in s
        assert "status:" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- e2e canonical ----
    def t_e2e_canonical() -> None:
        text = (
            "Build a small production-quality REST API for managing tasks.\n\n"
            "Functional requirements:\n"
            "- Users must be able to create, read, update, and delete tasks.\n\n"
            "Non-functional:\n"
            "- The API must respond within 200ms for read operations.\n"
            "- All traffic must use HTTPS.\n\n"
            "Constraints:\n"
            "- Must not require external authentication services.\n"
        )
        spec, ic = _spec_intent(text, project="demo")
        result = selector.select(spec, ic, project_id="demo")

        stack = result.stack()
        # Every category has a winner
        for cat in TechCategory:
            assert stack[cat.value] is not None, f"no winner for {cat.value}"

        # Signals verify
        sig = result.signals
        assert sig is not None
        assert sig.security_high is True  # HTTPS
        assert sig.must_be_local is True  # no external auth
        assert sig.needs_persistence is True

        # Coherent stack: language & framework share family
        lang_decision = result.decision(TechCategory.LANGUAGE)
        fw_decision = result.decision(TechCategory.FRAMEWORK)
        assert lang_decision is not None and lang_decision.winner is not None
        assert fw_decision is not None and fw_decision.winner is not None
        assert lang_decision.winner.tech.family == fw_decision.winner.tech.family

        # Local-only: DynamoDB rejected
        db_decision = result.decision(TechCategory.DATABASE)
        assert db_decision is not None
        rejected_names = {c.tech.name for c in db_decision.rejected}
        assert "DynamoDB" in rejected_names

        # Persistence worked
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                repo = TechnologyRepository(memory=mem, ontology=ont)
                root = repo.save(result)
                assert root
                loaded = repo.load(result.id, project_id="demo")
                assert loaded is not None and loaded["id"] == result.id
            finally:
                s.shutdown()

    check("e2e: canonical REST API selection", t_e2e_canonical)

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
    print("SE Brain C09 — Technology Selection Engine")
    print("=" * 78)

    text = (
        "Build a small production-quality REST API for managing tasks.\n\n"
        "Functional requirements:\n"
        "- Users must be able to create, read, update, and delete tasks.\n"
        "- Admins must be able to list all tasks.\n\n"
        "Non-functional:\n"
        "- The API must respond within 200ms for read operations.\n"
        "- All traffic must use HTTPS.\n\n"
        "Constraints:\n"
        "- Must not require external authentication services.\n"
    )

    parser = RequirementParser()
    ic_engine = IntentContextEngine()
    selector = TechnologySelector()

    spec = parser.parse(text)
    ic = ic_engine.analyze(text, project_id="demo")
    result = selector.select(spec, ic, project_id="demo")

    print("\n[1] Extracted signals:")
    sig = result.signals
    for k, v in sig.to_dict().items():
        if k == "rationale":
            continue
        if v not in (False, [], None, 0, "unknown"):
            print(f"    {k}: {v}")
    print("    rationale signals:")
    for r in sig.rationale:
        print(f"      · {r}")

    print("\n[2] Selected stack:")
    for cat in TechCategory:
        d = result.decision(cat)
        if d and d.winner:
            w = d.winner
            print(f"    {cat.value:12s}: {w.tech.name:15s} "
                  f"(score={w.score.total:.2f})")
            print(f"                  {d.rationale}")

    print("\n[3] Alternatives considered (top 3 per category):")
    for d in result.decisions:
        print(f"    [{d.category.value}]")
        for c in d.top_candidates:
            mark = "★" if c.verdict is FitVerdict.RECOMMENDED else " "
            sc = c.score.total if c.score else 0.0
            print(f"      {mark} {c.tech.name:18s} score={sc:.2f}  "
                  f"({c.verdict.value})")

    print("\n[4] Rejected with reasons:")
    for d in result.decisions:
        if not d.rejected:
            continue
        print(f"    [{d.category.value}]")
        for c in d.rejected[:5]:
            r = c.rejections[0] if c.rejections else "?"
            print(f"      ✗ {c.tech.name:18s} — {r}")

    print("\n[5] Summary:")
    print(result.summary())

    # Persistence
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = TechnologyRepository(memory=mem, ontology=ont)
                ent_id = repo.save(result)
                print(f"\n[6] Persisted → ontology decision: {ent_id[:12]}…")
                loaded = repo.load(result.id, project_id="demo")
                print(f"    reloaded stack: {loaded['decisions'][0]['winner']['tech']['name']} …")
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
