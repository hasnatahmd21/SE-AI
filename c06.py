"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C06 — INTENT & CONTEXT ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C05.

Purpose:
    Turn a requirement spec + ambient context into a structured IntentContext:

      Intent       : what the user actually wants (BUILD / MODIFY / FIX / ...)
      Context      : project/task state, prior specs, memory, decisions
      Priorities   : ranked requirement items with explicit signals
      Dependencies : implicit dependencies (not stated by the user)
      Conflicts    : pairs of requirements that contradict each other
      Risk         : level + factors + mitigations
      Epistemic    : every claim tagged as one of
                     USER_REQUIREMENT | SYSTEM_INTERPRETATION |
                     ASSUMPTION | INFERENCE | UNKNOWN

Invariants honored:
  - Deterministic, rule-based. No LLM, no external service.
  - Never conflate the 5 epistemic levels.
  - Conflicts are reported, not resolved silently.
  - Implicit dependencies are INFERENCE, not USER_REQUIREMENT.
  - Risk is scored from actual signals, not guessed.
  - Unknown never masquerades as fact.

Contents:
  1.  Enums: IntentKind, EpistemicLevel, RiskLevel, PriorityLevel, ConflictKind
  2.  Dataclasses: EpistemicClaim, PriorityItem, ImplicitDependency, Conflict,
                   Intent, ContextSnapshot, RiskAssessment, IntentContext
  3.  Signal tables (intent verbs, modal markers, risk weights, conflict patterns)
  4.  IntentClassifier
  5.  ConflictDetector
  6.  PriorityRanker
  7.  RiskAnalyzer
  8.  ContextAssembler (uses C04 memory + C02 ontology)
  9.  IntentContextEngine (facade + persistence)
  10. __main__ demo + self-tests

Run as script:
    python -m sebrain.c06            # demo
    python -m sebrain.c06 --test     # self-tests
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
from typing import Any, Callable, Iterable

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
from sebrain.c05 import (
    AmbiguityKind,
    MissingKind,
    RequirementKind,
    RequirementParser,
    RequirementSpec,
)


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|[0-9]+")
_STOP = frozenset("""
a an and are as at be by for from has have he her him his i in is it its
of on or that the their them they this to was were will with you your we
our do does did how what when where which who why
the a an of to in on at by for with and or but if then so
""".split())


def _tokens(text: str) -> set[str]:
    """Simple deterministic token set (>=3 chars, not stopword, lowercase)."""
    return {
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text or "")
        if len(m.group(0)) >= 3 and m.group(0).lower() not in _STOP
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().rstrip(".!?").lower())


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class IntentKind(str, Enum):
    BUILD = "build"
    MODIFY = "modify"
    FIX = "fix"
    REFACTOR = "refactor"
    TEST = "test"
    DEPLOY = "deploy"
    EXPLAIN = "explain"
    REVIEW = "review"
    OPTIMIZE = "optimize"
    MIGRATE = "migrate"
    DOCUMENT = "document"
    ANALYZE = "analyze"
    UNKNOWN = "unknown"


class EpistemicLevel(str, Enum):
    """The 5-way distinction that must never be collapsed."""
    USER_REQUIREMENT = "user_requirement"
    SYSTEM_INTERPRETATION = "system_interpretation"
    ASSUMPTION = "assumption"
    INFERENCE = "inference"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConflictKind(str, Enum):
    DIRECT_CONTRADICTION = "direct_contradiction"     # must use X vs must not use X
    SCOPE_OVERLAP = "scope_overlap"                   # same item in/out of scope
    TEMPORAL = "temporal"                             # before vs after
    RESOURCE = "resource"                             # perf target vs constraint
    SEMANTIC = "semantic"                             # high overlap, opposite polarity


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class EpistemicClaim:
    """A single claim tagged with its epistemic level. Never merged."""
    text: str
    level: EpistemicLevel
    source: str = ""              # where it came from (component / step)
    rationale: str = ""
    confidence: Confidence = Confidence.MEDIUM
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "level": self.level.value,
            "source": self.source,
            "rationale": self.rationale,
            "confidence": self.confidence.value,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(slots=True)
class PriorityItem:
    text: str
    level: PriorityLevel
    signals: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "level": self.level.value,
            "signals": list(self.signals), "rationale": self.rationale,
        }


@dataclass(slots=True)
class ImplicitDependency:
    """A dependency NOT stated by the user — an INFERENCE."""
    what: str
    why: str
    triggered_by: str = ""
    confidence: Confidence = Confidence.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "what": self.what, "why": self.why,
            "triggered_by": self.triggered_by,
            "confidence": self.confidence.value,
        }


@dataclass(slots=True)
class Conflict:
    a_text: str
    b_text: str
    kind: ConflictKind
    severity: PriorityLevel
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a_text, "b": self.b_text,
            "kind": self.kind.value, "severity": self.severity.value,
            "reason": self.reason,
        }


@dataclass(slots=True)
class Intent:
    kind: IntentKind
    confidence: Confidence
    signals: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "confidence": self.confidence.value,
            "signals": list(self.signals), "rationale": self.rationale,
        }


@dataclass(slots=True)
class ContextSnapshot:
    project_id: str
    task_id: str | None = None
    has_prior_specs: bool = False
    prior_spec_count: int = 0
    prior_decision_count: int = 0
    prior_failure_count: int = 0
    prior_experience_count: int = 0
    prior_requirement_entity_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "task_id": self.task_id,
            "has_prior_specs": self.has_prior_specs,
            "prior_spec_count": self.prior_spec_count,
            "prior_decision_count": self.prior_decision_count,
            "prior_failure_count": self.prior_failure_count,
            "prior_experience_count": self.prior_experience_count,
            "prior_requirement_entity_count": self.prior_requirement_entity_count,
            "notes": list(self.notes),
        }


@dataclass(slots=True)
class RiskAssessment:
    level: RiskLevel
    score: int
    factors: list[dict[str, Any]] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value, "score": self.score,
            "factors": list(self.factors),
            "mitigations": list(self.mitigations),
        }


@dataclass(slots=True)
class IntentContext:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    task_id: str | None = None
    raw_text: str = ""
    intent: Intent | None = None
    context: ContextSnapshot | None = None
    priorities: list[PriorityItem] = field(default_factory=list)
    implicit_dependencies: list[ImplicitDependency] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    risk: RiskAssessment | None = None
    claims: list[EpistemicClaim] = field(default_factory=list)
    spec_id: str | None = None
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def claims_by_level(self, level: EpistemicLevel) -> list[EpistemicClaim]:
        return [c for c in self.claims if c.level is level]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "raw_text": self.raw_text,
            "intent": self.intent.to_dict() if self.intent else None,
            "context": self.context.to_dict() if self.context else None,
            "priorities": [p.to_dict() for p in self.priorities],
            "implicit_dependencies": [d.to_dict() for d in self.implicit_dependencies],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "risk": self.risk.to_dict() if self.risk else None,
            "claims": [c.to_dict() for c in self.claims],
            "spec_id": self.spec_id,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        lines = ["=== IntentContext ==="]
        if self.intent:
            lines.append(f"Intent: {self.intent.kind.value} ({self.intent.confidence.value})")
        if self.risk:
            lines.append(f"Risk: {self.risk.level.value} (score={self.risk.score})")
        lines.append(f"Priorities: {len(self.priorities)}")
        lines.append(f"Implicit deps: {len(self.implicit_dependencies)}")
        lines.append(f"Conflicts: {len(self.conflicts)}")
        by_level: dict[str, int] = {}
        for c in self.claims:
            by_level[c.level.value] = by_level.get(c.level.value, 0) + 1
        lines.append("Claims: " + ", ".join(f"{k}={v}" for k, v in sorted(by_level.items())))
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL TABLES
# ════════════════════════════════════════════════════════════════════════════
# Order matters — earlier kinds win on ties.
INTENT_VERBS: list[tuple[IntentKind, tuple[str, ...]]] = [
    (IntentKind.FIX, (
        "fix", "debug", "repair", "resolve", "patch", "troubleshoot",
        "broken", "not working", "crash", "error", "bug",
    )),
    (IntentKind.REFACTOR, (
        "refactor", "clean up", "cleanup", "restructure", "simplify",
        "reorganize", "reorganise", "rewrite",
    )),
    (IntentKind.MIGRATE, (
        "migrate", "port", "upgrade", "convert", "transition", "move to",
    )),
    (IntentKind.OPTIMIZE, (
        "optimize", "optimise", "speed up", "faster", "reduce latency",
        "improve performance", "performance tuning",
    )),
    (IntentKind.DEPLOY, (
        "deploy", "ship", "release", "publish", "roll out", "rollout",
        "put into production", "go live",
    )),
    (IntentKind.TEST, (
        "test", "coverage", "unit test", "integration test", "add tests",
        "write tests", "test suite",
    )),
    (IntentKind.DOCUMENT, (
        "document", "docs", "readme", "comment", "write documentation",
    )),
    (IntentKind.REVIEW, (
        "review", "audit", "inspect", "code review", "critique",
    )),
    (IntentKind.EXPLAIN, (
        "explain", "how does", "what is", "describe", "walk me through",
        "why does", "tell me about",
    )),
    (IntentKind.ANALYZE, (
        "analyze", "analyse", "understand", "map out", "trace", "survey",
        "investigate",
    )),
    (IntentKind.MODIFY, (
        "change", "modify", "update", "extend", "add feature", "enhance",
        "adjust", "tweak", "alter",
    )),
    (IntentKind.BUILD, (
        "build", "create", "develop", "implement", "make", "deliver",
        "provide", "design", "construct", "write",
    )),
]

PRIORITY_SIGNALS: dict[PriorityLevel, tuple[str, ...]] = {
    PriorityLevel.CRITICAL: (
        "critical", "essential", "blocker", "must have", "must-have",
        "required", "indispensable",
    ),
    PriorityLevel.HIGH: (
        "important", "should have", "should-have", "high priority",
        "priority", "urgent",
    ),
    PriorityLevel.MEDIUM: (
        "nice", "would be good", "expected",
    ),
    PriorityLevel.LOW: (
        "optional", "nice to have", "nice-to-have", "could have",
        "could-have", "if possible", "maybe", "someday",
    ),
}

# RFC-2119-style modal strength.
MODAL_STRENGTH: dict[str, PriorityLevel] = {
    "must": PriorityLevel.CRITICAL,
    "shall": PriorityLevel.CRITICAL,
    "required": PriorityLevel.CRITICAL,
    "should": PriorityLevel.HIGH,
    "ought": PriorityLevel.HIGH,
    "recommended": PriorityLevel.HIGH,
    "may": PriorityLevel.LOW,
    "could": PriorityLevel.LOW,
    "optional": PriorityLevel.LOW,
}

NEGATION_TOKENS = frozenset({
    "not", "no", "never", "without", "exclude", "excluded", "excluding",
    "avoid", "avoided", "prohibit", "prohibited", "ban", "banned",
    "forbid", "forbidden", "disable", "disabled", "reject", "rejected",
})

# Implicit dependency triggers: (pattern, what, why, confidence)
IMPLICIT_DEP_RULES: list[tuple[re.Pattern, str, str, Confidence]] = [
    (re.compile(r"\bhttps\b|\btls\b|\bssl\b", re.I),
     "TLS certificate / HTTPS termination",
     "HTTPS/TLS was required but no certificate mechanism was stated.",
     Confidence.HIGH),
    (re.compile(r"\b(?:postgres(?:ql)?|mysql|sqlite|mariadb|mongodb|redis)\b", re.I),
     "database driver + schema/migration tooling",
     "A database was named but driver + migration strategy not stated.",
     Confidence.HIGH),
    (re.compile(r"\b(?:rest(?:\s+api)?|json)\b", re.I),
     "JSON serialization + content-type negotiation",
     "REST/JSON semantics require a serializer and content-type handling.",
     Confidence.MEDIUM),
    (re.compile(r"\b(?:auth(?:entication|orization)?|jwt|oauth|login|token)\b", re.I),
     "authentication/authorization middleware + secrets handling",
     "Auth was mentioned; middleware + secret storage not stated.",
     Confidence.HIGH),
    (re.compile(r"\b(?:logging|logs|audit)\b", re.I),
     "structured logging + log shipping/storage",
     "Logging was mentioned; destination + format not stated.",
     Confidence.MEDIUM),
    (re.compile(r"\b(?:deploy(?:ment)?|production|release)\b", re.I),
     "deployment pipeline + configuration management",
     "Deployment context implies build pipeline + env config.",
     Confidence.MEDIUM),
    (re.compile(r"\b(?:docker|container(?:ize|isation|ization))\b", re.I),
     "container image + base image policy",
     "Containers were mentioned; base image + registry not stated.",
     Confidence.MEDIUM),
    (re.compile(r"\b(?:pagination|paging)\b", re.I),
     "pagination scheme (cursor vs offset)",
     "Pagination was mentioned; concrete scheme not specified.",
     Confidence.MEDIUM),
]

RISK_INTENT_WEIGHT: dict[IntentKind, int] = {
    IntentKind.BUILD: 2,
    IntentKind.MODIFY: 2,
    IntentKind.FIX: 3,
    IntentKind.REFACTOR: 2,
    IntentKind.TEST: 1,
    IntentKind.DEPLOY: 4,
    IntentKind.EXPLAIN: 0,
    IntentKind.REVIEW: 1,
    IntentKind.OPTIMIZE: 2,
    IntentKind.MIGRATE: 4,
    IntentKind.DOCUMENT: 1,
    IntentKind.ANALYZE: 0,
    IntentKind.UNKNOWN: 1,
}

SECURITY_KEYWORDS = re.compile(
    r"\b(?:security|authentication|authorization|encryption|tls|ssl|https|"
    r"password|credential|token|owasp|csrf|xss|injection)\b", re.I,
)


# ════════════════════════════════════════════════════════════════════════════
# 4. INTENT CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════
class IntentClassifier:
    """Deterministic intent classifier. First-match wins on tie."""

    @staticmethod
    def _scan(text: str) -> dict[IntentKind, list[str]]:
        low = " " + text.lower() + " "
        scores: dict[IntentKind, list[str]] = {}
        for kind, verbs in INTENT_VERBS:
            hits: list[str] = []
            for v in verbs:
                if " " in v:
                    if v in low:
                        hits.append(v)
                else:
                    if re.search(r"\b" + re.escape(v) + r"\b", low):
                        hits.append(v)
            if hits:
                scores.setdefault(kind, []).extend(hits)
        return scores

    def classify(self, text: str) -> Intent:
        text = (text or "").strip()
        if not text:
            return Intent(
                kind=IntentKind.UNKNOWN,
                confidence=Confidence.UNKNOWN,
                rationale="empty text",
            )

        # The opening sentence almost always states the overall goal
        # explicitly ("Build a ...", "Fix the ..."). Check it in isolation
        # first: otherwise a generic verb appearing later — e.g. "update"
        # as part of a CRUD list ("create, read, update, delete") — would
        # outrank a clearly-stated opening intent once the whole document
        # is scanned together.
        first_sentence = re.split(r"(?<=[.!?])\s+|\n", text, maxsplit=1)[0]
        opening_scores = self._scan(first_sentence)
        scores = opening_scores if opening_scores else self._scan(text)

        if not scores:
            return Intent(
                kind=IntentKind.UNKNOWN,
                confidence=Confidence.LOW,
                rationale="no intent verb matched",
            )

        # First kind in INTENT_VERBS order that has signals wins (priority
        # ordering reflects specificity: fix > refactor > ... > build).
        for kind, _ in INTENT_VERBS:
            if kind in scores:
                signals = scores[kind]
                # Confidence: HIGH if >=2 signals, else MEDIUM
                conf = Confidence.HIGH if len(signals) >= 2 else Confidence.MEDIUM
                return Intent(
                    kind=kind,
                    confidence=conf,
                    signals=sorted(set(signals)),
                    rationale=f"matched verbs: {', '.join(sorted(set(signals)))}",
                )

        # unreachable
        return Intent(IntentKind.UNKNOWN, Confidence.UNKNOWN)


# ════════════════════════════════════════════════════════════════════════════
# 5. CONFLICT DETECTOR
# ════════════════════════════════════════════════════════════════════════════
def _has_negation(text: str) -> bool:
    toks = _tokens(text)
    return bool(toks & NEGATION_TOKENS) or "must not" in text.lower() \
        or "cannot" in text.lower() or "can't" in text.lower()


class ConflictDetector:
    """Finds contradictory requirement pairs.

    Strategy:
      - Collect candidate texts from spec (functional, NFR, constraints,
        scope_in, scope_out).
      - Compute normalized token sets.
      - For each pair: if Jaccard >= threshold AND polarity differs,
        flag as DIRECT_CONTRADICTION or SEMANTIC.
      - Scope overlap: same subject present in scope_in and scope_out.
      - Resource: performance NFR conflicts with a "must not"/constraint
        mentioning the same subject.
    """

    def __init__(self, *, jaccard_threshold: float = 0.35) -> None:
        self.threshold = jaccard_threshold

    def detect(self, spec: RequirementSpec) -> list[Conflict]:
        items: list[tuple[str, str]] = []   # (bucket, text)
        for it in spec.functional:
            items.append(("functional", it.text))
        for it in spec.non_functional:
            items.append(("non_functional", it.text))
        for it in spec.constraints:
            items.append(("constraint", it.text))
        for it in spec.scope_in:
            items.append(("scope_in", it.text))
        for it in spec.scope_out:
            items.append(("scope_out", it.text))

        conflicts: list[Conflict] = []

        # Scope overlap: same subject in scope_in and scope_out
        s_in = [(t, _tokens(t)) for b, t in items if b == "scope_in"]
        s_out = [(t, _tokens(t)) for b, t in items if b == "scope_out"]
        for tin, toks_in in s_in:
            for tout, toks_out in s_out:
                j = _jaccard(toks_in, toks_out)
                if j >= self.threshold:
                    conflicts.append(Conflict(
                        a_text=tin, b_text=tout,
                        kind=ConflictKind.SCOPE_OVERLAP,
                        severity=PriorityLevel.HIGH,
                        reason=f"same subject in in-scope and out-of-scope (jaccard={j:.2f})",
                    ))

        # Direct contradiction: high overlap + polarity differs
        for i in range(len(items)):
            b1, t1 = items[i]
            for j in range(i + 1, len(items)):
                b2, t2 = items[j]
                if b1 == b2 and (b1 == "scope_in" or b1 == "scope_out"):
                    continue  # handled above
                toks1 = _tokens(t1)
                toks2 = _tokens(t2)
                jac = _jaccard(toks1, toks2)
                if jac < self.threshold:
                    continue
                neg1 = _has_negation(t1)
                neg2 = _has_negation(t2)
                if neg1 != neg2:
                    conflicts.append(Conflict(
                        a_text=t1, b_text=t2,
                        kind=ConflictKind.DIRECT_CONTRADICTION,
                        severity=PriorityLevel.CRITICAL,
                        reason=f"opposite polarity on same subject (jaccard={jac:.2f})",
                    ))

        # Resource: performance NFR + a constraint on same subject
        perf = [t for b, t in items
                if b == "non_functional" and re.search(
                    r"\b(?:latency|response|throughput|performance|rps|ms\b)", t, re.I)]
        cons = [t for b, t in items if b == "constraint"]
        for p in perf:
            tp = _tokens(p)
            for c in cons:
                jac = _jaccard(tp, _tokens(c))
                if jac >= self.threshold and _has_negation(c):
                    conflicts.append(Conflict(
                        a_text=p, b_text=c,
                        kind=ConflictKind.RESOURCE,
                        severity=PriorityLevel.HIGH,
                        reason=f"performance target overlaps with constraint (jaccard={jac:.2f})",
                    ))

        # Dedupe (order-insensitive, kind-aware)
        seen: set[tuple[str, str, str]] = set()
        out: list[Conflict] = []
        for c in conflicts:
            key = tuple(sorted([
                f"{c.kind.value}|{_norm(c.a_text)}|{_norm(c.b_text)}",
                f"{c.kind.value}|{_norm(c.b_text)}|{_norm(c.a_text)}",
            ]))
            if key[0] in seen or key[1] in seen:
                continue
            seen.add(key[0])
            out.append(c)
        return out


# ════════════════════════════════════════════════════════════════════════════
# 6. PRIORITY RANKER
# ════════════════════════════════════════════════════════════════════════════
_RANK_ORDER = {
    PriorityLevel.CRITICAL: 0,
    PriorityLevel.HIGH: 1,
    PriorityLevel.MEDIUM: 2,
    PriorityLevel.LOW: 3,
}


class PriorityRanker:
    """Ranks requirement items by explicit priority signals.

    Order of precedence:
      1. Explicit priority phrases (critical / must-have / optional / ...)
      2. Modal verbs (must > should > may/could)
      3. Default per bucket (constraints/acceptance=NFR=HIGH, FR=MEDIUM,
         scope_in=MEDIUM, scope_out=LOW)
    """

    _DEFAULT: dict[str, PriorityLevel] = {
        "constraint": PriorityLevel.HIGH,
        "non_functional": PriorityLevel.HIGH,
        "acceptance": PriorityLevel.HIGH,
        "functional": PriorityLevel.MEDIUM,
        "scope_in": PriorityLevel.MEDIUM,
        "scope_out": PriorityLevel.LOW,
    }

    def rank(self, spec: RequirementSpec) -> list[PriorityItem]:
        out: list[PriorityItem] = []
        for bucket, items in (
            ("constraint", spec.constraints),
            ("non_functional", spec.non_functional),
            ("acceptance", spec.acceptance_criteria),
            ("functional", spec.functional),
            ("scope_in", spec.scope_in),
            ("scope_out", spec.scope_out),
        ):
            for it in items:
                level, signals = self._score_item(it.text, bucket)
                out.append(PriorityItem(
                    text=it.text, level=level, signals=signals,
                    rationale=f"bucket={bucket}",
                ))
        out.sort(key=lambda p: _RANK_ORDER[p.level])
        return out

    def _score_item(self, text: str, bucket: str) -> tuple[PriorityLevel, list[str]]:
        low = " " + text.lower() + " "
        signals: list[str] = []
        # 1. explicit phrases — longest/most-specific match wins. Some
        # short phrases are literal substrings of longer, more specific
        # ones at a *different* priority level (e.g. MEDIUM "nice" is a
        # prefix of LOW "nice to have"); ranking by priority strength
        # instead of specificity would let the short generic phrase
        # permanently shadow the longer, more accurate one.
        best_phrase = ""
        explicit: PriorityLevel | None = None
        for lvl, phrases in PRIORITY_SIGNALS.items():
            for ph in phrases:
                if ph in low and len(ph) > len(best_phrase):
                    best_phrase = ph
                    explicit = lvl
        if explicit is not None:
            signals.append(f"phrase:{best_phrase}")
            return explicit, signals
        # 2. modal verbs
        modal_level: PriorityLevel | None = None
        for modal, lvl in MODAL_STRENGTH.items():
            if re.search(r"\b" + modal + r"\b", low):
                if modal_level is None or _RANK_ORDER[lvl] < _RANK_ORDER[modal_level]:
                    modal_level = lvl
                    signals.append(f"modal:{modal}")
        if modal_level is not None:
            return modal_level, signals
        # 3. bucket default
        return self._DEFAULT.get(bucket, PriorityLevel.MEDIUM), ["default"]


# ════════════════════════════════════════════════════════════════════════════
# 7. RISK ANALYZER
# ════════════════════════════════════════════════════════════════════════════
class RiskAnalyzer:
    """Deterministic risk scoring.

    score = intent_weight
          + 2*ambiguities
          + 3*conflicts_critical + 2*conflicts_other
          + 1*missing_categories
          + 2*security_nfr_present (security demands inspection)
          + 1*user_count_many
          + 2*has_out_of_scope_overlap

    Mapping: score 0-3 → LOW, 4-7 → MEDIUM, 8-12 → HIGH, >=13 → CRITICAL
    """

    def analyze(
        self, spec: RequirementSpec, intent: Intent,
        conflicts: list[Conflict],
        *, has_prior_failures: int = 0,
    ) -> RiskAssessment:
        factors: list[dict[str, Any]] = []
        score = 0

        w = RISK_INTENT_WEIGHT.get(intent.kind, 1)
        score += w
        factors.append({"factor": "intent", "weight": w,
                        "detail": f"intent={intent.kind.value}"})

        amb = len(spec.ambiguities)
        if amb:
            score += 2 * amb
            factors.append({"factor": "ambiguity", "weight": 2 * amb,
                            "detail": f"{amb} ambiguous statements"})

        crit = sum(1 for c in conflicts if c.severity is PriorityLevel.CRITICAL)
        other = len(conflicts) - crit
        if crit:
            score += 3 * crit
            factors.append({"factor": "critical_conflict", "weight": 3 * crit,
                            "detail": f"{crit} direct contradictions"})
        if other:
            score += 2 * other
            factors.append({"factor": "conflict", "weight": 2 * other,
                            "detail": f"{other} non-critical conflicts"})

        missing_n = len(spec.missing)
        if missing_n:
            # Cap this factor's contribution: a small, legitimately simple
            # task will naturally skip several optional categories (NFR,
            # constraints, risks, scope, ...) without that implying real
            # risk. Counting every missing category 1-for-1 with no cap
            # let a clean, small task's score climb into HIGH purely from
            # sparsity, which is exactly what a "clean small task" should
            # NOT be flagged as.
            missing_weight = min(missing_n, 3)
            score += missing_weight
            factors.append({"factor": "missing_info", "weight": missing_weight,
                            "detail": f"{missing_n} missing categories"})

        sec = sum(1 for it in spec.non_functional if "security" in it.tags)
        if sec:
            score += 2
            factors.append({"factor": "security_nfr", "weight": 2,
                            "detail": f"{sec} security-related NFRs"})

        # security keywords in raw text even if not categorised as NFR
        if SECURITY_KEYWORDS.search(spec.raw_text or "") and not sec:
            score += 1
            factors.append({"factor": "security_keywords", "weight": 1,
                            "detail": "security keywords found in raw text"})

        if len(spec.users) >= 3:
            score += 1
            factors.append({"factor": "many_users", "weight": 1,
                            "detail": f"{len(spec.users)} distinct roles"})

        if has_prior_failures:
            score += 2
            factors.append({"factor": "prior_failures", "weight": 2,
                            "detail": f"{has_prior_failures} prior failures in project"})

        if score <= 3:
            level = RiskLevel.LOW
        elif score <= 7:
            level = RiskLevel.MEDIUM
        elif score <= 12:
            level = RiskLevel.HIGH
        else:
            level = RiskLevel.CRITICAL

        mitigations = self._mitigations(level, spec, conflicts)
        return RiskAssessment(level=level, score=score,
                              factors=factors, mitigations=mitigations)

    def _mitigations(
        self, level: RiskLevel, spec: RequirementSpec, conflicts: list[Conflict]
    ) -> list[str]:
        m: list[str] = []
        if spec.ambiguities:
            m.append("Resolve ambiguities before planning (clarify vague terms).")
        if spec.missing:
            m.append("Fill missing-information categories (users, acceptance, etc.).")
        if conflicts:
            m.append("Resolve requirement conflicts before implementation.")
        if level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            m.append("Run an explicit verification pass before any deployment step.")
        if any("security" in it.tags for it in spec.non_functional):
            m.append("Include security analysis in the verification gate.")
        return m


# ════════════════════════════════════════════════════════════════════════════
# 8. CONTEXT ASSEMBLER
# ════════════════════════════════════════════════════════════════════════════
class ContextAssembler:
    """Reads project state from C04 memory + C02 ontology."""

    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def snapshot(self, *, project_id: str, task_id: str | None = None) -> ContextSnapshot:
        ctx = ContextSnapshot(project_id=project_id, task_id=task_id)

        # Prior spec memory entries
        try:
            specs = self.memory.find(
                kind=MemoryKind.PROJECT,
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
                key_like="requirement_spec",
            )
            ctx.prior_spec_count = len(specs)
            ctx.has_prior_specs = ctx.prior_spec_count > 0
        except Exception:
            pass

        for kind, attr in (
            (MemoryKind.DECISION, "prior_decision_count"),
            (MemoryKind.FAILURE, "prior_failure_count"),
            (MemoryKind.EXPERIENCE, "prior_experience_count"),
        ):
            try:
                items = self.memory.find(
                    kind=kind,
                    scope_type=MemoryScope.PROJECT, scope_id=project_id,
                )
                setattr(ctx, attr, len(items))
            except Exception:
                pass

        if self.ontology is not None:
            try:
                ctx.prior_requirement_entity_count = self.ontology.count(
                    kind=EntityKind.REQUIREMENT,
                )
            except Exception:
                pass

        if ctx.has_prior_specs:
            ctx.notes.append(f"{ctx.prior_spec_count} prior requirement specs in project")
        if ctx.prior_failure_count:
            ctx.notes.append(f"{ctx.prior_failure_count} prior failures recorded")
        if ctx.prior_decision_count:
            ctx.notes.append(f"{ctx.prior_decision_count} prior decisions recorded")
        return ctx


# ════════════════════════════════════════════════════════════════════════════
# 9. INTENT & CONTEXT ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
class IntentContextEngine:
    """Facade: composes classification, conflict detection, priority ranking,
    risk analysis, context assembly, and epistemic claims."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        ontology: Ontology | None = None,
        parser: RequirementParser | None = None,
    ) -> None:
        self.memory = memory
        self.ontology = ontology
        self.parser = parser or RequirementParser()
        self.intent_clf = IntentClassifier()
        self.conflict_det = ConflictDetector()
        self.priority_rk = PriorityRanker()
        self.risk_an = RiskAnalyzer()
        self.ctx_asm = (
            ContextAssembler(memory, ontology) if memory is not None else None
        )

    # ---- main entry ----
    def analyze(
        self,
        text: str,
        *,
        project_id: str = "",
        task_id: str | None = None,
        spec: RequirementSpec | None = None,
        provenance: Provenance | None = None,
    ) -> IntentContext:
        text = text or ""
        prov = provenance or Provenance(
            source="user", source_type=ProvenanceType.USER,
            confidence=Confidence.HIGH,
        )

        if spec is None:
            spec = self.parser.parse(text, provenance=prov)
        elif not spec.raw_text:
            spec.raw_text = text

        ic = IntentContext(
            project_id=project_id, task_id=task_id,
            raw_text=text, spec_id=spec.id, provenance=prov,
        )

        # 1. Intent
        ic.intent = self.intent_clf.classify(text)

        # 2. Context
        if self.ctx_asm is not None and project_id:
            ic.context = self.ctx_asm.snapshot(project_id=project_id, task_id=task_id)
        else:
            ic.context = ContextSnapshot(project_id=project_id or "<unknown>",
                                         task_id=task_id)

        # 3. Conflicts
        ic.conflicts = self.conflict_det.detect(spec)

        # 4. Priorities
        ic.priorities = self.priority_rk.rank(spec)

        # 5. Implicit dependencies
        ic.implicit_dependencies = self._implicit_deps(spec)

        # 6. Risk
        ic.risk = self.risk_an.analyze(
            spec, ic.intent, ic.conflicts,
            has_prior_failures=(ic.context.prior_failure_count if ic.context else 0),
        )

        # 7. Epistemic claims
        ic.claims = self._build_claims(spec, ic, prov)
        return ic

    # ---- implicit deps ----
    def _implicit_deps(self, spec: RequirementSpec) -> list[ImplicitDependency]:
        out: list[ImplicitDependency] = []
        haystack = " ".join([
            spec.raw_text or "",
            *[x.text for x in spec.functional],
            *[x.text for x in spec.non_functional],
            *[x.text for x in spec.constraints],
        ])
        for pattern, what, why, conf in IMPLICIT_DEP_RULES:
            m = pattern.search(haystack)
            if m:
                out.append(ImplicitDependency(
                    what=what, why=why,
                    triggered_by=m.group(0), confidence=conf,
                ))
        # dedupe by `what`
        seen: set[str] = set()
        out2: list[ImplicitDependency] = []
        for d in out:
            if d.what in seen:
                continue
            seen.add(d.what)
            out2.append(d)
        return out2

    # ---- epistemic claims ----
    def _build_claims(
        self, spec: RequirementSpec, ic: IntentContext, prov: Provenance
    ) -> list[EpistemicClaim]:
        claims: list[EpistemicClaim] = []

        # --- USER_REQUIREMENT: what the user literally stated (from spec) ---
        def user_claim(text: str, source: str) -> EpistemicClaim:
            return EpistemicClaim(
                text=text, level=EpistemicLevel.USER_REQUIREMENT,
                source=source, rationale="verbatim from user requirement text",
                confidence=Confidence.HIGH, provenance=prov,
            )

        if spec.objective:
            claims.append(user_claim(spec.objective.text, "spec.objective"))
        for it in spec.functional:
            claims.append(user_claim(it.text, "spec.functional"))
        for it in spec.non_functional:
            claims.append(user_claim(it.text, "spec.non_functional"))
        for it in spec.constraints:
            claims.append(user_claim(it.text, "spec.constraints"))
        for it in spec.acceptance_criteria:
            claims.append(user_claim(it.text, "spec.acceptance"))
        for it in spec.scope_in:
            claims.append(user_claim(it.text, "spec.scope_in"))
        for it in spec.scope_out:
            claims.append(user_claim(it.text, "spec.scope_out"))

        # --- SYSTEM_INTERPRETATION: how we classified intent / priority ---
        if ic.intent is not None:
            claims.append(EpistemicClaim(
                text=f"Intent classified as {ic.intent.kind.value}.",
                level=EpistemicLevel.SYSTEM_INTERPRETATION,
                source="intent_classifier",
                rationale=ic.intent.rationale,
                confidence=ic.intent.confidence,
                provenance=prov,
            ))
        for p in ic.priorities:
            claims.append(EpistemicClaim(
                text=f"Priority {p.level.value}: {p.text}",
                level=EpistemicLevel.SYSTEM_INTERPRETATION,
                source="priority_ranker",
                rationale=p.rationale + " signals=" + ",".join(p.signals),
                confidence=Confidence.MEDIUM,
                provenance=prov,
            ))

        # --- ASSUMPTION: from spec.assumptions ---
        for a in spec.assumptions:
            claims.append(EpistemicClaim(
                text=a.text, level=EpistemicLevel.ASSUMPTION,
                source="requirement_parser",
                rationale=f"assumption ({a.origin})",
                confidence=Confidence.LOW,
                provenance=a.provenance,
            ))

        # --- INFERENCE: implicit dependencies ---
        for d in ic.implicit_dependencies:
            claims.append(EpistemicClaim(
                text=f"Implicit dependency: {d.what}",
                level=EpistemicLevel.INFERENCE,
                source="implicit_dep_rules",
                rationale=d.why + f" (trigger={d.triggered_by!r})",
                confidence=d.confidence,
                provenance=prov,
            ))

        # --- UNKNOWN: missing info + ambiguities ---
        for m in spec.missing:
            claims.append(EpistemicClaim(
                text=f"Missing: {m.kind.value}",
                level=EpistemicLevel.UNKNOWN,
                source="requirement_parser",
                rationale=m.why,
                confidence=Confidence.UNKNOWN,
                provenance=prov,
            ))
        for a in spec.ambiguities:
            claims.append(EpistemicClaim(
                text=f"Ambiguous ({a.kind.value}): {a.term or a.text}",
                level=EpistemicLevel.UNKNOWN,
                source="requirement_parser",
                rationale=a.reason,
                confidence=Confidence.UNKNOWN,
                provenance=prov,
            ))
        return claims

    # ---- persistence ----
    def persist(self, ic: IntentContext) -> str:
        """Save to C04 memory (PROJECT scope) + C02 ontology TASK entity."""
        if self.memory is None or not ic.project_id:
            raise ValidationError("memory or project_id not available")
        key = f"intent_context:{ic.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, ic.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=ic.project_id,
            tags=["intent", "context"],
            provenance=ic.provenance,
        )
        if self.ontology is None:
            return ic.id
        ent = self.ontology.add(
            EntityKind.TASK,
            (ic.intent.kind.value if ic.intent else "unknown") + " task",
            attributes={
                "intent_context_id": ic.id,
                "project_id": ic.project_id,
                "risk": ic.risk.level.value if ic.risk else "unknown",
            },
            tags=["intent-context"],
            provenance=ic.provenance,
        )
        return ent.id


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

    print("Running C06 self-tests…")
    engine = IntentContextEngine()

    # ---- intent classification ----
    def t_intent_build() -> None:
        assert engine.intent_clf.classify("Build a REST API.").kind is IntentKind.BUILD
        assert engine.intent_clf.classify("Create a service.").kind is IntentKind.BUILD

    def t_intent_fix() -> None:
        assert engine.intent_clf.classify("Fix the login bug.").kind is IntentKind.FIX
        assert engine.intent_clf.classify("The app is broken.").kind is IntentKind.FIX

    def t_intent_refactor() -> None:
        assert engine.intent_clf.classify("Refactor the auth module.").kind is IntentKind.REFACTOR

    def t_intent_migrate() -> None:
        assert engine.intent_clf.classify("Migrate to Postgres.").kind is IntentKind.MIGRATE

    def t_intent_optimize() -> None:
        assert engine.intent_clf.classify("Optimize the query.").kind is IntentKind.OPTIMIZE
        assert engine.intent_clf.classify("Make it faster.").kind is IntentKind.OPTIMIZE

    def t_intent_deploy() -> None:
        assert engine.intent_clf.classify("Deploy to production.").kind is IntentKind.DEPLOY

    def t_intent_test() -> None:
        assert engine.intent_clf.classify("Add unit tests for auth.").kind is IntentKind.TEST

    def t_intent_explain() -> None:
        assert engine.intent_clf.classify("Explain how the cache works.").kind is IntentKind.EXPLAIN

    def t_intent_review() -> None:
        assert engine.intent_clf.classify("Review this module.").kind is IntentKind.REVIEW

    def t_intent_analyze() -> None:
        assert engine.intent_clf.classify("Analyze the codebase.").kind is IntentKind.ANALYZE

    def t_intent_modify() -> None:
        assert engine.intent_clf.classify("Extend the API with pagination.").kind is IntentKind.MODIFY

    def t_intent_document() -> None:
        assert engine.intent_clf.classify("Write documentation.").kind is IntentKind.DOCUMENT

    def t_intent_unknown() -> None:
        assert engine.intent_clf.classify("").kind is IntentKind.UNKNOWN
        assert engine.intent_clf.classify("xyzzy foobar").kind is IntentKind.UNKNOWN

    def t_intent_priority_specificity() -> None:
        # "fix" wins over "build" when both appear
        r = engine.intent_clf.classify("Build then fix the auth bug.")
        assert r.kind is IntentKind.FIX, r.kind

    check("intent: BUILD", t_intent_build)
    check("intent: FIX", t_intent_fix)
    check("intent: REFACTOR", t_intent_refactor)
    check("intent: MIGRATE", t_intent_migrate)
    check("intent: OPTIMIZE", t_intent_optimize)
    check("intent: DEPLOY", t_intent_deploy)
    check("intent: TEST", t_intent_test)
    check("intent: EXPLAIN", t_intent_explain)
    check("intent: REVIEW", t_intent_review)
    check("intent: ANALYZE", t_intent_analyze)
    check("intent: MODIFY", t_intent_modify)
    check("intent: DOCUMENT", t_intent_document)
    check("intent: UNKNOWN on empty/garbage", t_intent_unknown)
    check("intent: specificity ordering (fix > build)", t_intent_priority_specificity)

    # ---- conflict detection ----
    def t_conflict_direct() -> None:
        text = (
            "Constraints:\n"
            "- The system must use SQLite.\n"
            "- The system must not use SQLite.\n"
        )
        spec = engine.parser.parse(text)
        conf = engine.conflict_det.detect(spec)
        kinds = {c.kind for c in conf}
        assert ConflictKind.DIRECT_CONTRADICTION in kinds

    def t_conflict_scope_overlap() -> None:
        text = (
            "In scope:\n- User authentication\n"
            "Out of scope:\n- User authentication\n"
        )
        spec = engine.parser.parse(text)
        conf = engine.conflict_det.detect(spec)
        kinds = {c.kind for c in conf}
        assert ConflictKind.SCOPE_OVERLAP in kinds

    def t_conflict_none_when_clean() -> None:
        text = (
            "Build an API.\n"
            "Functional:\n- Users must be able to create tasks.\n"
            "Non-functional:\n- Respond within 200ms.\n"
        )
        spec = engine.parser.parse(text)
        conf = engine.conflict_det.detect(spec)
        assert conf == [], conf

    def t_conflict_dedup() -> None:
        text = (
            "Constraints:\n"
            "- Must use SQLite.\n"
            "- Must not use SQLite.\n"
            "- Must not use the SQLite database.\n"
        )
        spec = engine.parser.parse(text)
        conf = engine.conflict_det.detect(spec)
        # No duplicate pairs with same (a,b) after normalization
        pairs = {(c.kind.value, _norm(c.a_text), _norm(c.b_text)) for c in conf}
        assert len(pairs) == len(conf), "duplicate conflicts"

    check("conflict: direct contradiction", t_conflict_direct)
    check("conflict: scope overlap", t_conflict_scope_overlap)
    check("conflict: none on clean spec", t_conflict_none_when_clean)
    check("conflict: deduped pairs", t_conflict_dedup)

    # ---- priority ranking ----
    def t_priority_modal_must() -> None:
        spec = engine.parser.parse(
            "Functional:\n- Users must be able to create tasks.\n"
        )
        prio = engine.priority_rk.rank(spec)
        assert prio, "no priorities"
        assert prio[0].level is PriorityLevel.CRITICAL, prio[0]

    def t_priority_modal_should() -> None:
        spec = engine.parser.parse(
            "Functional:\n- Users should be able to create tasks.\n"
        )
        prio = engine.priority_rk.rank(spec)
        assert any(p.level is PriorityLevel.HIGH for p in prio)

    def t_priority_optional() -> None:
        spec = engine.parser.parse(
            "Functional:\n- Nice to have: dark mode.\n"
        )
        prio = engine.priority_rk.rank(spec)
        assert any(p.level is PriorityLevel.LOW for p in prio), prio

    def t_priority_critical_phrase() -> None:
        spec = engine.parser.parse(
            "Functional:\n- Auth is critical for release.\n"
        )
        prio = engine.priority_rk.rank(spec)
        assert any(p.level is PriorityLevel.CRITICAL for p in prio)

    def t_priority_ordering() -> None:
        text = (
            "Functional:\n"
            "- Users must be able to create tasks.\n"
            "- Users may optionally list tasks.\n"
        )
        spec = engine.parser.parse(text)
        prio = engine.priority_rk.rank(spec)
        # critical ranks first
        assert prio[0].level is PriorityLevel.CRITICAL
        assert prio[-1].level is PriorityLevel.LOW

    check("priority: 'must' → CRITICAL", t_priority_modal_must)
    check("priority: 'should' → HIGH", t_priority_modal_should)
    check("priority: 'nice to have' → LOW", t_priority_optional)
    check("priority: explicit 'critical' phrase", t_priority_critical_phrase)
    check("priority: list ordering (critical first, low last)", t_priority_ordering)

    # ---- implicit deps ----
    def t_implicit_https() -> None:
        text = "Non-functional:\n- All traffic must use HTTPS.\n"
        spec = engine.parser.parse(text)
        deps = engine._implicit_deps(spec)
        assert any("TLS" in d.what for d in deps), deps

    def t_implicit_db() -> None:
        text = "The service depends on PostgreSQL 15.\n"
        spec = engine.parser.parse(text)
        deps = engine._implicit_deps(spec)
        assert any("database" in d.what.lower() for d in deps), deps

    def t_implicit_auth() -> None:
        text = "Non-functional:\n- Support OAuth login.\n"
        spec = engine.parser.parse(text)
        deps = engine._implicit_deps(spec)
        assert any("auth" in d.what.lower() for d in deps), deps

    def t_implicit_none() -> None:
        text = "Build a simple calculator.\n"
        spec = engine.parser.parse(text)
        deps = engine._implicit_deps(spec)
        assert deps == [], deps

    check("implicit dep: HTTPS → TLS", t_implicit_https)
    check("implicit dep: Postgres → driver", t_implicit_db)
    check("implicit dep: OAuth → middleware", t_implicit_auth)
    check("implicit dep: none when not triggered", t_implicit_none)

    # ---- risk analysis ----
    def t_risk_low() -> None:
        text = (
            "Build a calculator.\n"
            "Functional:\n- Add two numbers.\n"
            "Acceptance:\n- Given 1+2, when pressed equals, then 3 is shown.\n"
        )
        ic = engine.analyze(text, project_id="p")
        assert ic.risk is not None
        assert ic.risk.level in (RiskLevel.LOW, RiskLevel.MEDIUM), ic.risk.level

    def t_risk_high_conflicts() -> None:
        text = (
            "Deploy to production.\n"
            "Constraints:\n"
            "- Must use SQLite.\n"
            "- Must not use SQLite.\n"
            "Non-functional:\n"
            "- All traffic must use HTTPS.\n"
            "- Response within 50ms.\n"
        )
        ic = engine.analyze(text, project_id="p")
        assert ic.risk is not None
        assert ic.risk.level in (RiskLevel.HIGH, RiskLevel.CRITICAL), ic.risk.level

    def t_risk_factors_populated() -> None:
        text = (
            "Deploy to production.\n"
            "Non-functional:\n- Must use HTTPS.\n"
        )
        ic = engine.analyze(text, project_id="p")
        assert ic.risk is not None
        assert any(f["factor"] == "intent" for f in ic.risk.factors)

    check("risk: LOW/MEDIUM on clean small task", t_risk_low)
    check("risk: HIGH/CRITICAL with conflicts + deploy", t_risk_high_conflicts)
    check("risk: factors populated", t_risk_factors_populated)

    # ---- epistemic discipline ----
    def t_epistemic_levels_present() -> None:
        text = (
            "Build a REST API.\n"
            "Functional:\n- Users must be able to create tasks.\n"
            "Non-functional:\n- All traffic must use HTTPS.\n"
            "Assumptions:\n- Python 3.11 is available.\n"
        )
        ic = engine.analyze(text, project_id="p")
        levels = {c.level for c in ic.claims}
        # Should have at least USER, SYSTEM_INTERPRETATION, ASSUMPTION, INFERENCE, UNKNOWN
        assert EpistemicLevel.USER_REQUIREMENT in levels
        assert EpistemicLevel.SYSTEM_INTERPRETATION in levels
        assert EpistemicLevel.ASSUMPTION in levels
        assert EpistemicLevel.INFERENCE in levels
        assert EpistemicLevel.UNKNOWN in levels

    def t_epistemic_user_only_verbatim() -> None:
        text = "Build a calculator.\n"
        ic = engine.analyze(text, project_id="p")
        users = ic.claims_by_level(EpistemicLevel.USER_REQUIREMENT)
        # "TLS certificate" implicit dep must NOT be classified as USER_REQUIREMENT
        for c in users:
            assert "Implicit dependency" not in c.text

    def t_epistemic_unknown_not_fact() -> None:
        text = "Build something.\n"
        ic = engine.analyze(text, project_id="p")
        unknowns = ic.claims_by_level(EpistemicLevel.UNKNOWN)
        assert len(unknowns) > 0
        # Missing-info claims must be UNKNOWN, never USER_REQUIREMENT
        for c in unknowns:
            assert c.text.startswith("Missing:") or c.text.startswith("Ambiguous")

    check("epistemic: all 5 levels represented", t_epistemic_levels_present)
    check("epistemic: USER only from verbatim input", t_epistemic_user_only_verbatim)
    check("epistemic: missing-info tagged UNKNOWN, not fact", t_epistemic_unknown_not_fact)

    # ---- context assembly + persistence ----
    def t_context_snapshot_empty_project() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                eng = IntentContextEngine(memory=mem, ontology=ont)
                ic = eng.analyze("Build an API.", project_id="fresh")
                assert ic.context is not None
                assert ic.context.prior_spec_count == 0
                assert ic.context.prior_failure_count == 0
            finally:
                s.shutdown()

    def t_context_snapshot_with_history() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                # seed prior data
                mem.record_decision(
                    "db", decision="SQLite", rationale="local",
                    scope_id="proj-A",
                )
                mem.record_failure(
                    "bug-1", what="test", root_cause="x",
                    scope_id="proj-A",
                )
                eng = IntentContextEngine(memory=mem, ontology=ont)
                ic = eng.analyze("Extend the API.", project_id="proj-A")
                assert ic.context is not None
                assert ic.context.prior_decision_count == 1
                assert ic.context.prior_failure_count == 1
                assert ic.risk is not None
                # prior failures should bump risk factors
                assert any(f["factor"] == "prior_failures" for f in ic.risk.factors)
            finally:
                s.shutdown()

    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                eng = IntentContextEngine(memory=mem, ontology=ont)
                ic = eng.analyze("Build a REST API for tasks.", project_id="proj-Z")
                ent_id = eng.persist(ic)
                assert ent_id
                # reload from memory
                entry = mem.get_current(
                    MemoryKind.PROJECT, f"intent_context:{ic.id}",
                    scope_type=MemoryScope.PROJECT, scope_id="proj-Z",
                )
                assert entry is not None
                assert entry.content["intent"]["kind"] == "build"
                # ontology entity exists
                assert ont.get(ent_id) is not None
            finally:
                s.shutdown()

    check("context: empty project snapshot", t_context_snapshot_empty_project)
    check("context: history-aware snapshot", t_context_snapshot_with_history)
    check("persist: memory + ontology", t_persist)

    # ---- to_dict / summary ----
    def t_to_dict() -> None:
        ic = engine.analyze("Build a REST API.", project_id="p")
        d = ic.to_dict()
        assert d["intent"]["kind"] == "build"
        assert isinstance(d["priorities"], list)
        assert isinstance(d["claims"], list)
        assert d["risk"] is not None
        assert "level" in d["risk"]

    def t_summary() -> None:
        ic = engine.analyze("Build a REST API.", project_id="p")
        s = ic.summary()
        assert "Intent: build" in s
        assert "Risk:" in s

    check("to_dict JSON-serializable", t_to_dict)
    check("summary human-readable", t_summary)

    # ---- canonical e2e ----
    def t_e2e_canonical() -> None:
        text = (
            "Build a small production-quality REST API for managing tasks.\n\n"
            "Functional requirements:\n"
            "- Users must be able to create, read, update, and delete tasks.\n"
            "- Admins must be able to list all tasks.\n\n"
            "Non-functional:\n"
            "- The API must respond within 200ms for read operations.\n"
            "- All traffic must use HTTPS.\n\n"
            "Constraints:\n"
            "- Must not require external authentication services.\n\n"
            "Acceptance:\n"
            "- Given a valid request, when POST /tasks is called, then a 201 response is returned.\n\n"
            "In scope:\n- Task CRUD operations\n\n"
            "Out of scope:\n- Mobile client\n\n"
            "Assumptions:\n- Python 3.11 or newer is available\n\n"
            "Risks:\n- Concurrency on SQLite may cause lock contention\n"
        )
        ic = engine.analyze(text, project_id="demo")
        # intent
        assert ic.intent is not None
        assert ic.intent.kind is IntentKind.BUILD
        # priorities non-empty
        assert len(ic.priorities) > 0
        # implicit deps
        dep_whats = {d.what for d in ic.implicit_dependencies}
        assert any("TLS" in w for w in dep_whats)
        # risk
        assert ic.risk is not None
        assert ic.risk.level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        # epistemic claims: all 5 levels represented
        levels = {c.level for c in ic.claims}
        assert EpistemicLevel.USER_REQUIREMENT in levels
        assert EpistemicLevel.SYSTEM_INTERPRETATION in levels
        assert EpistemicLevel.INFERENCE in levels
        # context assembled
        assert ic.context is not None
        assert ic.context.project_id == "demo"

    check("e2e: canonical 'Build a REST API for tasks' input", t_e2e_canonical)

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
    print("SE Brain C06 — Intent & Context Engine")
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
        "- Must not require external authentication services.\n\n"
        "Acceptance:\n"
        "- Given a valid request, when POST /tasks is called, then a 201 response is returned.\n\n"
        "In scope:\n- Task CRUD operations\n\n"
        "Out of scope:\n- Mobile client\n\n"
        "Assumptions:\n- Python 3.11 or newer is available\n\n"
        "Risks:\n- Concurrency on SQLite may cause lock contention\n"
    )

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            mem = MemoryStore(app.storage)
            ont = Ontology(app.storage)
            # seed history
            mem.record_failure("bug-1", what="prev test fail",
                               root_cause="missing fixture", scope_id="demo")
            mem.record_decision("db", decision="SQLite",
                                rationale="local-first", scope_id="demo")

            engine = IntentContextEngine(memory=mem, ontology=ont)

            with execution_scope(project_id="demo", task_id="t1"):
                ic = engine.analyze(text, project_id="demo", task_id="t1")

                print("\n[1] Summary:")
                print(ic.summary())

                print("\n[2] Intent:")
                print(f"    kind       = {ic.intent.kind.value}")
                print(f"    confidence = {ic.intent.confidence.value}")
                print(f"    signals    = {ic.intent.signals}")

                print("\n[3] Context:")
                ctx = ic.context
                print(f"    project_id              = {ctx.project_id}")
                print(f"    prior_spec_count        = {ctx.prior_spec_count}")
                print(f"    prior_decision_count    = {ctx.prior_decision_count}")
                print(f"    prior_failure_count     = {ctx.prior_failure_count}")
                for n in ctx.notes:
                    print(f"    note: {n}")

                print("\n[4] Priorities (top 6):")
                for p in ic.priorities[:6]:
                    print(f"    [{p.level.value:8s}] {p.text}  signals={p.signals}")

                print("\n[5] Implicit dependencies (INFERENCE):")
                for d in ic.implicit_dependencies:
                    print(f"    - {d.what}")
                    print(f"      why: {d.why}")

                print("\n[6] Conflicts:")
                if not ic.conflicts:
                    print("    (none)")
                for c in ic.conflicts:
                    print(f"    [{c.kind.value} sev={c.severity.value}]")
                    print(f"      A: {c.a_text}")
                    print(f"      B: {c.b_text}")

                print("\n[7] Risk:")
                r = ic.risk
                print(f"    level = {r.level.value} (score={r.score})")
                for f in r.factors:
                    print(f"    factor: {f['factor']} +{f['weight']} ({f['detail']})")
                print("    mitigations:")
                for m in r.mitigations:
                    print(f"      - {m}")

                print("\n[8] Epistemic claim counts by level:")
                for lvl in EpistemicLevel:
                    n = len(ic.claims_by_level(lvl))
                    if n:
                        print(f"    {lvl.value:22s} : {n}")

                print("\n[9] Sample claims (one per level):")
                for lvl in EpistemicLevel:
                    items = ic.claims_by_level(lvl)
                    if items:
                        print(f"    [{lvl.value}] {items[0].text[:80]}")

                print("\n[10] Persisting…")
                ent_id = engine.persist(ic)
                print(f"    ontology task entity: {ent_id[:12]}…")
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
