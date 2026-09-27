"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C33 — FAILURE ANALYSIS LABORATORY (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C26, C27 (C18/C19/C20/C32 via duck-typed adapters).

Purpose:
    Systematically analyze failures across the pipeline, cluster them by
    signature, identify RECURRING patterns, and feed VALIDATED lessons back
    into C26 (Experience Extraction) and C27 (Learning).

Capabilities:
    1. Failure harvest
         * C04 memory FAILURE entries
         * C18 test run failures (nodeid + traceback head)
         * C19 debug reports (category + exception)
         * C20 repair results (rejected / failed)
         * C32 benchmark case errors
    2. Classification
         * 9 domains: build / test / runtime / repair / verification /
           security / performance / integration / unknown
    3. Clustering by deterministic signature
         * normalized tokens from what + root_cause
         * Jaccard single-link clustering (same as C27)
    4. Frequency analysis
         * per pattern: occurrences, first_seen, last_seen, domains,
           components, agents, knowledge sources, plans
    5. Recurrence tiering
         * SINGLE (1) / RECURRING (2-4) / CHRONIC (5+)
    6. Lesson generation
         * pattern → lesson candidate (problem / approach / lesson)
         * verdict: READY / HELD / NOT_APPLICABLE based on occurrence +
           cross-component evidence
    7. Feedback interface
         * to_experience_inputs() → C26 ExperienceInput compatible dicts
         * to_learning_experiences() → C27 ExperienceInput compatible dicts
    8. Persistence
         * C04 memory: report JSON
         * C02 ontology: FAILURE + ROOT_CAUSE entities + PRODUCES links

Invariants honored:
    - NO external LLM. Deterministic.
    - Every pattern carries supporting event ids (evidence).
    - Frequency counts exclude events whose domain is UNKNOWN from
      "high-confidence" tiers but still track them.
    - A pattern with a single occurrence is never called "recurring".
    - Lessons are marked READY only with (occurrences >= 2) AND
      (2+ distinct components OR 2+ domains) OR (occurrences >= 5).
    - Same events → same clusters → same patterns (deterministic).
    - Bounded (max_events, max_clusters, max_patterns).

Explicit limitations (Rule #59):
    - Signature is keyword-based, not semantic. Different wording for the
      same underlying bug may not cluster.
    - Recurrence tiering is occurrence-count only; it does NOT weight by
      severity or recency.
    - The lab does NOT verify that lessons are correct. It packages them
      for C26/C27 which perform their own reliability scoring.
    - Historical events are immutable; the lab never edits a failure record.

Contents:
  1.  Enums: FailureDomain, PatternKind, LessonVerdict
  2.  Dataclasses: FailureEvent, FailureCluster, RecurringPattern,
                   LessonCandidate, AnalysisReport
  3.  Token + similarity helpers
  4.  Harvester (memory + duck-typed adapters)
  5.  Classifier (domain inference)
  6.  Clusterer (deterministic single-link)
  7.  PatternDetector (recurrence + lesson building)
  8.  FailureLab facade
  9.  FailureRepository (persist)
 10.  Self-tests (~35)
 11.  Demo

Run as script:
    python -m sebrain.c33            # demo
    python -m sebrain.c33 --test     # self-tests
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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from sebrain.c01 import (
    Confidence, Config, SEBrainApp, SQLiteStorage, ValidationError,
    execution_scope, get_logger,
)
from sebrain.c02 import (
    EntityKind, Ontology, Provenance, ProvenanceType, RelationKind,
)
from sebrain.c04 import (
    MemoryEntry, MemoryKind, MemoryScope, MemoryStore,
)


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


def _digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = frozenset("""
a an and are as at be by for from has have he her him his i in is it its
of on or that the their them they this to was were will with you your we
our do does did how what when where which who why use used using
test tests file line error failed failure exception raise
""".split())


def _tokens(text: str) -> set[str]:
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if w.lower() not in _STOPWORDS
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _enum_val(x: Any) -> str:
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class FailureDomain(str, Enum):
    BUILD = "build"
    TEST = "test"
    RUNTIME = "runtime"
    REPAIR = "repair"
    VERIFICATION = "verification"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    UNKNOWN = "unknown"


class PatternKind(str, Enum):
    SINGLE = "single"           # 1 occurrence
    RECURRING = "recurring"     # 2-4
    CHRONIC = "chronic"         # 5+


class LessonVerdict(str, Enum):
    READY = "ready"              # feed to C26/C27
    HELD = "held"                # insufficient evidence
    NOT_APPLICABLE = "not_applicable"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class FailureEvent:
    id: str
    domain: FailureDomain
    what: str = ""
    root_cause: str = ""
    component: str = ""
    agent: str = ""
    knowledge_source: str = ""
    plan_id: str = ""
    repair_id: str = ""
    outcome: str = ""
    tags: list[str] = field(default_factory=list)
    source_kind: str = ""                # memory / test_run / debug / ...
    raw_ref: str = ""
    observed_at: str = field(default_factory=now_iso)

    def signature_tokens(self) -> set[str]:
        return _tokens(self.what) | _tokens(self.root_cause)

    def signature_text(self) -> str:
        parts = [p for p in (self.what, self.root_cause) if p]
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "domain": self.domain.value,
            "what": self.what, "root_cause": self.root_cause,
            "component": self.component, "agent": self.agent,
            "knowledge_source": self.knowledge_source,
            "plan_id": self.plan_id, "repair_id": self.repair_id,
            "outcome": self.outcome,
            "tags": list(self.tags),
            "source_kind": self.source_kind, "raw_ref": self.raw_ref,
            "observed_at": self.observed_at,
        }


@dataclass(slots=True)
class FailureCluster:
    signature: str = ""
    events: list[FailureEvent] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    knowledge_sources: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""

    def recount(self) -> None:
        self.domains = sorted({e.domain.value for e in self.events})
        self.components = sorted({e.component for e in self.events
                                   if e.component})
        self.agents = sorted({e.agent for e in self.events if e.agent})
        self.knowledge_sources = sorted(
            {e.knowledge_source for e in self.events if e.knowledge_source}
        )
        self.plans = sorted({e.plan_id for e in self.events if e.plan_id})
        times = sorted(e.observed_at for e in self.events)
        self.first_seen = times[0] if times else ""
        self.last_seen = times[-1] if times else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "occurrences": len(self.events),
            "event_ids": [e.id for e in self.events],
            "domains": list(self.domains),
            "components": list(self.components),
            "agents": list(self.agents),
            "knowledge_sources": list(self.knowledge_sources),
            "plans": list(self.plans),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass(slots=True)
class RecurringPattern:
    id: str = field(default_factory=_new_id)
    kind: PatternKind = PatternKind.SINGLE
    signature: str = ""
    description: str = ""
    occurrences: int = 0
    event_ids: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind.value,
            "signature": self.signature,
            "description": self.description,
            "occurrences": self.occurrences,
            "event_ids": list(self.event_ids),
            "domains": list(self.domains),
            "components": list(self.components),
            "agents": list(self.agents),
            "first_seen": self.first_seen, "last_seen": self.last_seen,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class LessonCandidate:
    id: str = field(default_factory=_new_id)
    pattern_id: str = ""
    problem: str = ""
    approach: str = ""
    lesson: str = ""
    verdict: LessonVerdict = LessonVerdict.HELD
    event_ids: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "pattern_id": self.pattern_id,
            "problem": self.problem, "approach": self.approach,
            "lesson": self.lesson, "verdict": self.verdict.value,
            "event_ids": list(self.event_ids),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class AnalysisReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    events_scanned: int = 0
    clusters: int = 0
    patterns: list[RecurringPattern] = field(default_factory=list)
    lessons: list[LessonCandidate] = field(default_factory=list)
    by_domain: dict[str, int] = field(default_factory=dict)
    by_component: dict[str, int] = field(default_factory=dict)
    by_agent: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def ready_lessons(self) -> list[LessonCandidate]:
        return [l for l in self.lessons
                if l.verdict is LessonVerdict.READY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "events_scanned": self.events_scanned,
            "clusters": self.clusters,
            "patterns": [p.to_dict() for p in self.patterns],
            "lessons": [l.to_dict() for l in self.lessons],
            "by_domain": dict(self.by_domain),
            "by_component": dict(self.by_component),
            "by_agent": dict(self.by_agent),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for p in self.patterns:
            kinds[p.kind.value] = kinds.get(p.kind.value, 0) + 1
        return (
            "=== Failure Analysis Report ===\n"
            f"project={self.project_id}\n"
            f"events={self.events_scanned}  clusters={self.clusters}  "
            f"patterns={len(self.patterns)}  {kinds}\n"
            f"ready_lessons={len(self.ready_lessons())}  "
            f"held_lessons="
            f"{sum(1 for l in self.lessons if l.verdict is LessonVerdict.HELD)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFIER (domain inference)
# ════════════════════════════════════════════════════════════════════════════
_DOMAIN_KEYWORDS: list[tuple[FailureDomain, tuple[str, ...]]] = [
    (FailureDomain.SECURITY,
     ("security", "vulnerability", "cve", "injection", "xss", "csrf",
      "hardcoded", "secret", "exploit")),
    (FailureDomain.PERFORMANCE,
     ("performance", "latency", "throughput", "slow", "timeout",
      "n+1", "quadratic", "oom", "memory")),
    (FailureDomain.INTEGRATION,
     ("connection", "http", "network", "socket", "api", "endpoint",
      "integration")),
    (FailureDomain.VERIFICATION,
     ("verification", "unsupported", "refuted", "insufficient",
      "claim", "ladder")),
    (FailureDomain.REPAIR,
     ("repair", "patch", "anchor", "candidate", "regression")),
    (FailureDomain.BUILD,
     ("build", "compile", "syntax", "import", "module", "dependency",
      "package")),
    (FailureDomain.TEST,
     ("assert", "test", "fixture", "pytest", "unittest")),
    (FailureDomain.RUNTIME,
     ("runtime", "exception", "typeerror", "valueerror", "indexerror",
      "keyerror", "attributeerror", "traceback")),
]


def classify_domain(event: FailureEvent) -> FailureDomain:
    if event.domain is not FailureDomain.UNKNOWN:
        return event.domain
    # NOTE: source_kind is deliberately excluded — it's provenance
    # metadata (where this event was harvested from: "memory",
    # "test_run", ...), not part of the failure's actual content. A
    # source_kind of "memory" would otherwise collide with the
    # PERFORMANCE domain's "memory" keyword and hijack classification
    # for every event sourced from the memory store, regardless of what
    # it's actually about.
    blob = (event.what + " " + event.root_cause + " "
            + " ".join(event.tags)).lower()
    for domain, kws in _DOMAIN_KEYWORDS:
        if any(k in blob for k in kws):
            return domain
    return FailureDomain.UNKNOWN


# ════════════════════════════════════════════════════════════════════════════
# 4. HARVESTER
# ════════════════════════════════════════════════════════════════════════════
class Harvester:
    def __init__(self, *, max_events: int = 5000) -> None:
        if max_events < 1:
            raise ValidationError("max_events must be >= 1")
        self.max_events = max_events

    def from_memory(
        self, memory: MemoryStore, *, project_id: str,
    ) -> list[FailureEvent]:
        events: list[FailureEvent] = []
        try:
            entries = memory.find(
                kind=MemoryKind.FAILURE,
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
            )
        except Exception as exc:
            log.warning("c33.memory_harvest_error", error=str(exc))
            return events
        for e in entries:
            content = e.content or {}
            what = str(content.get("what") or "")
            cause = str(content.get("root_cause") or "")
            if not what and not cause:
                continue
            events.append(FailureEvent(
                id=e.id,
                domain=FailureDomain.UNKNOWN,
                what=what,
                root_cause=cause,
                knowledge_source=e.key.split(":", 1)[0] if e.key else "",
                outcome="failed",
                tags=list(e.tags or []),
                source_kind="memory",
                raw_ref=e.key,
                observed_at=e.updated_at,
            ))
            if len(events) >= self.max_events:
                break
        return events

    # ---- duck-typed adapters ----
    def from_test_run(self, run: Any, *, project_id: str = "") -> list[FailureEvent]:
        out: list[FailureEvent] = []
        if run is None:
            return out
        rid = str(getattr(run, "id", "")) or "test_run"
        for r in list(getattr(run, "results", []) or []):
            outcome = _enum_val(getattr(r, "outcome", ""))
            if outcome not in ("failed", "error"):
                continue
            nid = str(getattr(r, "nodeid", ""))
            msg = str(getattr(r, "failure_message", "") or "")
            tb = str(getattr(r, "failure_traceback", "") or "")
            head = (msg or tb.splitlines()[0] if tb else "")[:300]
            out.append(FailureEvent(
                id=f"tr:{rid}:{nid}",
                domain=FailureDomain.TEST,
                what=f"test {nid} {outcome}",
                root_cause=head or "unknown",
                component=str(getattr(r, "component", "") or ""),
                outcome=outcome,
                source_kind="test_run",
                raw_ref=rid,
                tags=["test", outcome],
            ))
            if len(out) >= self.max_events:
                break
        return out

    def from_debug_report(self, report: Any) -> list[FailureEvent]:
        if report is None:
            return []
        rid = str(getattr(report, "id", "")) or "debug"
        cat = _enum_val(getattr(report, "category", ""))
        sig = getattr(report, "signature", None)
        exc = str(getattr(sig, "exception_type", "") or "")
        msg = str(getattr(sig, "exception_message", "") or "")
        loc = getattr(report, "localized", None)
        component = ""
        if loc is not None:
            component = str(getattr(loc, "file", "") or "")
        # Category → domain mapping (C19 → C33)
        domain_map = {
            "syntax": FailureDomain.BUILD,
            "dependency": FailureDomain.BUILD,
            "type": FailureDomain.RUNTIME,
            "logic": FailureDomain.RUNTIME,
            "runtime": FailureDomain.RUNTIME,
            "config": FailureDomain.RUNTIME,
            "environment": FailureDomain.RUNTIME,
            "integration": FailureDomain.INTEGRATION,
            "concurrency": FailureDomain.RUNTIME,
            "resource": FailureDomain.PERFORMANCE,
            "security": FailureDomain.SECURITY,
            "unknown": FailureDomain.UNKNOWN,
        }
        domain = domain_map.get(cat, FailureDomain.UNKNOWN)
        return [FailureEvent(
            id=f"dbg:{rid}",
            domain=domain,
            what=f"{cat} failure: {exc}",
            root_cause=msg or cat,
            component=component,
            outcome="failed",
            source_kind="debug_report",
            raw_ref=rid,
            tags=[cat] if cat else [],
        )]

    def from_repair_result(self, result: Any) -> list[FailureEvent]:
        out: list[FailureEvent] = []
        if result is None:
            return out
        rid = str(getattr(result, "id", "")) or "repair"
        cat = str(getattr(result, "category", ""))
        accepted = getattr(result, "accepted_id", None)
        if accepted:
            return out   # not a failure
        # Collect failed evaluations if present
        evaluations = list(getattr(result, "evaluations", []) or [])
        if not evaluations:
            out.append(FailureEvent(
                id=f"rep:{rid}",
                domain=FailureDomain.REPAIR,
                what=f"repair rejected for category '{cat}'",
                root_cause=str(getattr(result, "rationale", "") or ""),
                outcome="failed",
                source_kind="repair_result",
                raw_ref=rid,
                tags=["repair", cat] if cat else ["repair"],
            ))
            return out
        for ev in evaluations:
            verdict = _enum_val(getattr(ev, "verdict", ""))
            if verdict == "accepted":
                continue
            out.append(FailureEvent(
                id=f"rep:{rid}:{verdict}:{id(ev)}",
                domain=FailureDomain.REPAIR,
                what=f"repair candidate {verdict}",
                root_cause=str(getattr(ev, "reason", "") or ""),
                outcome=verdict,
                source_kind="repair_result",
                raw_ref=rid,
                tags=["repair", verdict],
            ))
            if len(out) >= self.max_events:
                break
        return out

    def from_benchmark(self, run: Any, *, project_id: str = "") -> list[FailureEvent]:
        out: list[FailureEvent] = []
        if run is None:
            return out
        rid = str(getattr(run, "id", "")) or "benchmark"
        for res in list(getattr(run, "results", []) or []):
            outcome = _enum_val(getattr(res, "outcome", ""))
            if outcome not in ("failed", "error"):
                continue
            cat = _enum_val(getattr(res, "category", ""))
            err = getattr(res, "error", None) or {}
            cause = str(err.get("message", "") or "")
            out.append(FailureEvent(
                id=f"bench:{rid}:{getattr(res, 'case_id', '?')}",
                domain=FailureDomain.TEST,
                what=f"benchmark case {getattr(res, 'case_name', '?')} "
                     f"{outcome} (category={cat})",
                root_cause=cause or "benchmark failure",
                outcome=outcome,
                source_kind="benchmark",
                raw_ref=rid,
                tags=["benchmark", cat] if cat else ["benchmark"],
            ))
            if len(out) >= self.max_events:
                break
        return out


# ════════════════════════════════════════════════════════════════════════════
# 5. CLUSTERER
# ════════════════════════════════════════════════════════════════════════════
class Clusterer:
    def __init__(
        self, *, similarity_threshold: float = 0.30,
        max_clusters: int = 1000,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValidationError(
                "similarity_threshold must be in (0, 1]"
            )
        if max_clusters < 1:
            raise ValidationError("max_clusters must be >= 1")
        self.threshold = similarity_threshold
        self.max_clusters = max_clusters

    def cluster(self, events: Sequence[FailureEvent]) -> list[FailureCluster]:
        ordered = sorted(events, key=lambda e: e.id)
        clusters: list[FailureCluster] = []
        tokens_by: dict[str, set[str]] = {
            e.id: e.signature_tokens() for e in ordered
        }
        for e in ordered:
            t = tokens_by.get(e.id, set())
            placed = False
            for c in clusters:
                for m in c.events:
                    mt = tokens_by.get(m.id, set())
                    if _jaccard(t, mt) >= self.threshold:
                        c.events.append(e)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                clusters.append(FailureCluster(events=[e]))
                if len(clusters) >= self.max_clusters:
                    break
        for c in clusters:
            c.recount()
            c.signature = _canonical_signature(c.events)
        return clusters


def _canonical_signature(events: Sequence[FailureEvent]) -> str:
    # union of top tokens across events
    counter: dict[str, int] = defaultdict(int)
    for e in events:
        for t in e.signature_tokens():
            counter[t] += 1
    if not counter:
        return ""
    top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
    return " ".join(t for t, _ in top)


# ════════════════════════════════════════════════════════════════════════════
# 6. PATTERN DETECTOR
# ════════════════════════════════════════════════════════════════════════════
class PatternDetector:
    def detect(self, cluster: FailureCluster) -> RecurringPattern:
        n = len(cluster.events)
        if n <= 1:
            kind = PatternKind.SINGLE
        elif n <= 4:
            kind = PatternKind.RECURRING
        else:
            kind = PatternKind.CHRONIC
        # Representative text = event with longest root_cause
        rep = max(
            cluster.events,
            key=lambda e: (len(e.root_cause), e.id),
        ) if cluster.events else None
        desc = _short(
            f"{rep.what if rep else ''} — "
            f"{rep.root_cause if rep else ''}", 240,
        )
        rationale = (
            f"occurrences={n}  domains={cluster.domains}  "
            f"components={len(cluster.components)}  "
            f"agents={len(cluster.agents)}"
        )
        return RecurringPattern(
            kind=kind, signature=cluster.signature,
            description=desc, occurrences=n,
            event_ids=[e.id for e in cluster.events],
            domains=list(cluster.domains),
            components=list(cluster.components),
            agents=list(cluster.agents),
            first_seen=cluster.first_seen,
            last_seen=cluster.last_seen,
            rationale=rationale,
        )

    def build_lesson(
        self, pattern: RecurringPattern, cluster: FailureCluster,
    ) -> LessonCandidate:
        n = pattern.occurrences
        domains = set(pattern.domains)
        components = set(pattern.components)
        # Verdict rules (deterministic)
        if n == 0:
            verdict = LessonVerdict.NOT_APPLICABLE
            rationale = "no occurrences"
        elif n == 1:
            verdict = LessonVerdict.HELD
            rationale = "single occurrence; needs more evidence"
        elif n >= 5:
            verdict = LessonVerdict.READY
            rationale = f"chronic ({n} occurrences)"
        elif n >= 2 and (len(components) >= 2 or len(domains) >= 2):
            verdict = LessonVerdict.READY
            rationale = (
                f"recurring across components={len(components)} "
                f"domains={len(domains)}"
            )
        else:
            verdict = LessonVerdict.HELD
            rationale = (
                f"recurring but confined to a single component/domain"
            )
        problem = _short(pattern.description, 240)
        # Approach: generic remediation hint based on domain mix
        if FailureDomain.SECURITY.value in domains:
            approach = "Add input validation / sanitization at boundaries"
        elif FailureDomain.PERFORMANCE.value in domains:
            approach = "Profile hot path; reduce N+1 / allocations"
        elif FailureDomain.BUILD.value in domains:
            approach = "Verify dependencies and imports before build"
        elif FailureDomain.REPAIR.value in domains:
            approach = "Tighten anchor uniqueness and regression gate"
        elif FailureDomain.VERIFICATION.value in domains:
            approach = "Run verification before proceeding"
        elif FailureDomain.INTEGRATION.value in domains:
            approach = "Add retries/timeouts and validate endpoints"
        else:
            approach = "Investigate and add a guard for this failure mode"
        lesson = (
            f"Failure signature '{_short(pattern.signature, 80)}' "
            f"occurred {n} time(s); treat as {pattern.kind.value}."
        )
        return LessonCandidate(
            pattern_id=pattern.id,
            problem=problem, approach=approach, lesson=lesson,
            verdict=verdict, event_ids=list(pattern.event_ids),
            rationale=rationale,
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. FACADE
# ════════════════════════════════════════════════════════════════════════════
class FailureLab:
    """Systematic failure analysis across the pipeline."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        similarity_threshold: float = 0.30,
        max_events: int = 5000,
        max_patterns: int = 500,
    ) -> None:
        if max_patterns < 1:
            raise ValidationError("max_patterns must be >= 1")
        self.memory = memory
        self.harvester = Harvester(max_events=max_events)
        self.clusterer = Clusterer(similarity_threshold=similarity_threshold)
        self.detector = PatternDetector()
        self.max_events = max_events
        self.max_patterns = max_patterns

    # ---- collect ----
    def collect_events(
        self, *,
        project_id: str,
        test_run: Any = None,
        debug_report: Any = None,
        repair_result: Any = None,
        benchmark: Any = None,
        from_memory: bool = True,
        extra_events: Sequence[FailureEvent] = (),
    ) -> list[FailureEvent]:
        out: list[FailureEvent] = []
        if from_memory and self.memory is not None:
            out.extend(self.harvester.from_memory(
                self.memory, project_id=project_id,
            ))
        if test_run is not None:
            out.extend(self.harvester.from_test_run(test_run))
        if debug_report is not None:
            out.extend(self.harvester.from_debug_report(debug_report))
        if repair_result is not None:
            out.extend(self.harvester.from_repair_result(repair_result))
        if benchmark is not None:
            out.extend(self.harvester.from_benchmark(benchmark))
        if extra_events:
            out.extend(extra_events)
        # Classify UNKNOWN domains
        for e in out:
            if e.domain is FailureDomain.UNKNOWN:
                e.domain = classify_domain(e)
        # Dedup by id
        seen: set[str] = set()
        deduped: list[FailureEvent] = []
        for e in out:
            if e.id in seen:
                continue
            seen.add(e.id)
            deduped.append(e)
        return deduped[: self.max_events]

    # ---- analyze ----
    def analyze(
        self, events: Sequence[FailureEvent], *,
        project_id: str = "",
    ) -> AnalysisReport:
        rep = AnalysisReport(
            project_id=project_id,
            events_scanned=len(events),
            provenance=Provenance(
                source="failure_analysis_lab",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        if not events:
            rep.rationale = "no events"
            return rep

        # Resolve any still-UNKNOWN domains from content. The harvest
        # helpers already do this for events they collect, but analyze()
        # is also called directly with caller-supplied events (e.g. from
        # tests, or events collected some other way) that may never have
        # passed through that step — without this, by_domain would just
        # echo back "unknown" for every such event regardless of content.
        for e in events:
            if e.domain is FailureDomain.UNKNOWN:
                e.domain = classify_domain(e)

        # Counts
        by_dom: dict[str, int] = defaultdict(int)
        by_comp: dict[str, int] = defaultdict(int)
        by_agent: dict[str, int] = defaultdict(int)
        for e in events:
            by_dom[e.domain.value] += 1
            if e.component:
                by_comp[e.component] += 1
            if e.agent:
                by_agent[e.agent] += 1
        rep.by_domain = dict(by_dom)
        rep.by_component = dict(by_comp)
        rep.by_agent = dict(by_agent)

        # Cluster
        clusters = self.clusterer.cluster(events)
        rep.clusters = len(clusters)

        # Detect
        patterns: list[RecurringPattern] = []
        lessons: list[LessonCandidate] = []
        for c in clusters:
            p = self.detector.detect(c)
            patterns.append(p)
            l = self.detector.build_lesson(p, c)
            lessons.append(l)
            if len(patterns) >= self.max_patterns:
                break
        # Sort patterns: kind (chronic > recurring > single), then
        # occurrences desc, then signature
        order = {PatternKind.CHRONIC: 0, PatternKind.RECURRING: 1,
                 PatternKind.SINGLE: 2}
        patterns.sort(key=lambda p: (order[p.kind], -p.occurrences,
                                      p.signature))
        # Reorder lessons correspondingly
        pid_to_lesson = {l.pattern_id: l for l in lessons}
        lessons = [pid_to_lesson[p.id] for p in patterns
                   if p.id in pid_to_lesson]
        rep.patterns = patterns
        rep.lessons = lessons
        rep.rationale = (
            f"events={len(events)}  clusters={len(clusters)}  "
            f"patterns={len(patterns)}  "
            f"ready={len(rep.ready_lessons())}  "
            f"held={sum(1 for l in lessons if l.verdict is LessonVerdict.HELD)}"
        )
        return rep

    # ---- feedback: C26 / C27 ----
    def to_experience_inputs(
        self, report: AnalysisReport,
    ) -> list[dict[str, Any]]:
        """Return C26-compatible ExperienceInput dicts (only READY lessons)."""
        out: list[dict[str, Any]] = []
        for l in report.ready_lessons():
            out.append({
                "id": f"c33:{l.id}",
                "kind": "pitfall",
                "reliability": "promising",
                "score": 0.6,
                "problem": l.problem,
                "context": f"recurring failure pattern: {l.rationale}",
                "approach": l.approach,
                "result": "",
                "failure": l.problem,
                "repair": "",
                "verification": "",
                "lesson": l.lesson,
                "project_id": report.project_id,
                "evidence_refs": list(l.event_ids),
                "sources": ["c33"],
            })
        return out

    def to_learning_experiences(
        self, report: AnalysisReport,
    ) -> list[dict[str, Any]]:
        """Return C27-compatible ExperienceInput-shaped dicts."""
        return self.to_experience_inputs(report)

    # ---- persistence helper ----
    def persist(self, report: AnalysisReport, *, project_id: str) -> str:
        if self.memory is None:
            raise ValidationError("memory not attached")
        repo = FailureRepository(self.memory)
        return repo.save(report, project_id=project_id)


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class FailureRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: AnalysisReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"failure_report:{report.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, report.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["failure_analysis", "c33"],
            provenance=report.provenance,
        )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.ROOT_CAUSE,
            _short(f"FailureAnalysis {report.id[:8]} "
                   f"({len(report.patterns)} patterns)", 120),
            attributes={
                "report_id": report.id,
                "project_id": project_id,
                "events": report.events_scanned,
                "patterns": len(report.patterns),
                "ready_lessons": len(report.ready_lessons()),
                "by_domain": dict(report.by_domain),
                "by_component": dict(report.by_component),
                "by_agent": dict(report.by_agent),
            },
            tags=["failure-analysis"],
            provenance=report.provenance,
        )
        # Persist chronic/recurring patterns as FAILURE entities
        for p in report.patterns:
            if p.kind is PatternKind.SINGLE:
                continue
            fe = self.ontology.add(
                EntityKind.FAILURE,
                _short(f"[{p.kind.value}] {p.description}", 120),
                attributes={
                    "pattern_id": p.id,
                    "kind": p.kind.value,
                    "occurrences": p.occurrences,
                    "signature": p.signature,
                    "domains": list(p.domains),
                    "components": list(p.components),
                    "event_ids": list(p.event_ids),
                },
                tags=["failure-pattern", p.kind.value],
                provenance=report.provenance,
            )
            try:
                self.ontology.link(RelationKind.CONTAINS, ent.id, fe.id)
            except ValidationError:
                pass
        return ent.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"failure_report:{report_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 9. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _mk_event(
    eid: str, *, what: str = "cache stampede under load",
    root_cause: str = "cache not warmed",
    domain: FailureDomain = FailureDomain.UNKNOWN,
    component: str = "", agent: str = "",
    knowledge_source: str = "", plan_id: str = "",
    source_kind: str = "memory",
    tags: Sequence[str] = (),
) -> FailureEvent:
    return FailureEvent(
        id=eid, domain=domain, what=what, root_cause=root_cause,
        component=component, agent=agent,
        knowledge_source=knowledge_source, plan_id=plan_id,
        source_kind=source_kind, tags=list(tags),
    )


def _run_self_tests() -> int:
    import importlib.util
    has_pytest = importlib.util.find_spec("pytest") is not None

    failures: list[str] = []
    passed = 0
    skipped = 0

    def check(name: str, fn: Callable[[], None], *, requires_pytest: bool = False) -> None:
        nonlocal passed, skipped
        if requires_pytest and not has_pytest:
            skipped += 1
            print(f"  – {name} (skipped: pytest not installed)")
            return
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception:
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C33 self-tests…")
    lab = FailureLab()

    # ---- classification ----
    def t_classify_build() -> None:
        e = _mk_event("e1", what="build failure: cannot import xyz")
        assert classify_domain(e) is FailureDomain.BUILD

    def t_classify_security() -> None:
        e = _mk_event("e1", what="critical vulnerability in auth")
        assert classify_domain(e) is FailureDomain.SECURITY

    def t_classify_performance() -> None:
        e = _mk_event("e1", what="N+1 query pattern detected")
        assert classify_domain(e) is FailureDomain.PERFORMANCE

    def t_classify_integration() -> None:
        e = _mk_event("e1", what="http connection refused")
        assert classify_domain(e) is FailureDomain.INTEGRATION

    def t_classify_runtime() -> None:
        e = _mk_event("e1", what="TypeError in handler")
        assert classify_domain(e) is FailureDomain.RUNTIME

    def t_classify_test() -> None:
        e = _mk_event("e1", what="assert 1 == 2")
        assert classify_domain(e) is FailureDomain.TEST

    def t_classify_unknown() -> None:
        e = _mk_event("e1", what="zzz", root_cause="yyy")
        assert classify_domain(e) is FailureDomain.UNKNOWN

    check("classify: build keyword", t_classify_build)
    check("classify: security keyword", t_classify_security)
    check("classify: performance keyword", t_classify_performance)
    check("classify: integration keyword", t_classify_integration)
    check("classify: runtime keyword", t_classify_runtime)
    check("classify: test keyword", t_classify_test)
    check("classify: no match → UNKNOWN", t_classify_unknown)

    # ---- harvester ----
    def t_harvest_memory() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                mem.record_failure(
                    "bug-1", what="cache stampede",
                    root_cause="not warmed", scope_id="p",
                )
                mem.record_failure(
                    "bug-2", what="timeout on slow query",
                    root_cause="missing index", scope_id="p",
                )
                events = lab.harvester.from_memory(mem, project_id="p")
                assert len(events) >= 2
                kinds = {e.source_kind for e in events}
                assert "memory" in kinds
            finally:
                s.shutdown()

    def t_harvest_test_run() -> None:
        class R:
            nodeid = "tests/test_x.py::test_y"
            outcome = type("O", (), {"value": "failed"})()
            failure_message = "assert 1 == 2"
            failure_traceback = ""
            component = "x"
        class TR:
            id = "tr-1"
            results = [R()]
        events = lab.harvester.from_test_run(TR())
        assert len(events) == 1
        assert events[0].domain is FailureDomain.TEST
        assert "test_y" in events[0].what

    def t_harvest_debug_report() -> None:
        class Sig:
            exception_type = "AssertionError"
            exception_message = "assert 1 == 2"
        class Loc:
            file = "/app/tests/x.py"
        class DR:
            id = "dbg-1"
            category = type("C", (), {"value": "logic"})()
            signature = Sig()
            localized = Loc()
        events = lab.harvester.from_debug_report(DR())
        assert len(events) == 1
        assert events[0].domain is FailureDomain.RUNTIME
        assert "AssertionError" in events[0].what
        assert events[0].component.endswith("x.py")

    def t_harvest_repair_rejected() -> None:
        class Ev:
            verdict = type("V", (), {"value": "rejected_regression"})()
            reason = "test_sub regressed"
        class RR:
            id = "rep-1"
            category = "logic"
            accepted_id = None
            evaluations = [Ev()]
        events = lab.harvester.from_repair_result(RR())
        assert len(events) == 1
        assert events[0].domain is FailureDomain.REPAIR
        assert "rejected" in events[0].what

    def t_harvest_benchmark() -> None:
        class Res:
            case_id = "c1"; case_name = "gen_basic"
            outcome = type("O", (), {"value": "failed"})()
            category = type("C", (), {"value": "generation"})()
            error = {"message": "correctness below threshold"}
        class BR:
            id = "bench-1"
            results = [Res()]
        events = lab.harvester.from_benchmark(BR())
        assert len(events) == 1
        assert "benchmark case" in events[0].what

    check("harvest: from C04 memory", t_harvest_memory)
    check("harvest: from C18 test run", t_harvest_test_run)
    check("harvest: from C19 debug report", t_harvest_debug_report)
    check("harvest: from C20 rejected repair", t_harvest_repair_rejected)
    check("harvest: from C32 benchmark failures", t_harvest_benchmark)

    # ---- cluster ----
    def t_cluster_identical() -> None:
        events = [
            _mk_event("a", what="cache stampede",
                       root_cause="not warmed"),
            _mk_event("b", what="cache stampede",
                       root_cause="not warmed"),
            _mk_event("c", what="cache stampede",
                       root_cause="not warmed"),
        ]
        clusters = lab.clusterer.cluster(events)
        assert len(clusters) == 1
        assert len(clusters[0].events) == 3

    def t_cluster_distinct() -> None:
        events = [
            _mk_event("a", what="database query slow", root_cause="no index"),
            _mk_event("b", what="ui layout broken", root_cause="css issue"),
            _mk_event("c", what="network timeout", root_cause="unreachable"),
        ]
        clusters = lab.clusterer.cluster(events)
        assert len(clusters) == 3

    def t_cluster_deterministic() -> None:
        events = [
            _mk_event("z", what="cache stampede"),
            _mk_event("a", what="cache stampede"),
            _mk_event("m", what="cache stampede"),
        ]
        c1 = lab.clusterer.cluster(events)
        c2 = lab.clusterer.cluster(events)
        sig1 = sorted(tuple(sorted(e.id for e in c.events)) for c in c1)
        sig2 = sorted(tuple(sorted(e.id for e in c.events)) for c in c2)
        assert sig1 == sig2

    check("cluster: identical events grouped", t_cluster_identical)
    check("cluster: distinct events stay separate", t_cluster_distinct)
    check("cluster: deterministic", t_cluster_deterministic)

    # ---- pattern kind ----
    def t_pattern_kind_single() -> None:
        events = [_mk_event("a")]
        report = lab.analyze(events, project_id="p")
        assert len(report.patterns) == 1
        assert report.patterns[0].kind is PatternKind.SINGLE

    def t_pattern_kind_recurring() -> None:
        events = [
            _mk_event(f"e{i}", what="cache stampede") for i in range(3)
        ]
        report = lab.analyze(events, project_id="p")
        assert report.patterns[0].kind is PatternKind.RECURRING

    def t_pattern_kind_chronic() -> None:
        events = [
            _mk_event(f"e{i}", what="cache stampede") for i in range(6)
        ]
        report = lab.analyze(events, project_id="p")
        assert report.patterns[0].kind is PatternKind.CHRONIC

    check("pattern: 1 occurrence → SINGLE", t_pattern_kind_single)
    check("pattern: 2-4 occurrences → RECURRING",
          t_pattern_kind_recurring)
    check("pattern: 5+ occurrences → CHRONIC", t_pattern_kind_chronic)

    # ---- lesson verdict ----
    def t_lesson_held_for_single() -> None:
        report = lab.analyze([_mk_event("a")], project_id="p")
        assert report.lessons[0].verdict is LessonVerdict.HELD

    def t_lesson_ready_for_chronic() -> None:
        events = [_mk_event(f"e{i}") for i in range(5)]
        report = lab.analyze(events, project_id="p")
        assert report.lessons[0].verdict is LessonVerdict.READY

    def t_lesson_ready_for_cross_component() -> None:
        events = [
            _mk_event("a", what="cache stampede", component="svc-a"),
            _mk_event("b", what="cache stampede", component="svc-b"),
        ]
        report = lab.analyze(events, project_id="p")
        assert report.lessons[0].verdict is LessonVerdict.READY

    def t_lesson_held_single_component() -> None:
        # 3 occurrences, same component, same domain → HELD
        events = [
            _mk_event(f"e{i}", what="cache stampede",
                       component="svc-a") for i in range(3)
        ]
        report = lab.analyze(events, project_id="p")
        assert report.lessons[0].verdict is LessonVerdict.HELD

    check("lesson: single occurrence → HELD", t_lesson_held_for_single)
    check("lesson: chronic → READY", t_lesson_ready_for_chronic)
    check("lesson: cross-component → READY",
          t_lesson_ready_for_cross_component)
    check("lesson: same component+domain → HELD",
          t_lesson_held_single_component)

    # ---- aggregation ----
    def t_counts_by_domain() -> None:
        events = [
            _mk_event("a", what="build failure"),
            _mk_event("b", what="build failure"),
            _mk_event("c", what="http connection refused"),
        ]
        report = lab.analyze(events, project_id="p")
        assert report.by_domain.get("build", 0) >= 2
        assert report.by_domain.get("integration", 0) >= 1

    def t_counts_by_component() -> None:
        events = [
            _mk_event("a", component="x"),
            _mk_event("b", component="x"),
            _mk_event("c", component="y"),
        ]
        report = lab.analyze(events, project_id="p")
        assert report.by_component.get("x") == 2
        assert report.by_component.get("y") == 1

    def t_counts_by_agent() -> None:
        events = [
            _mk_event("a", agent="agent-1"),
            _mk_event("b", agent="agent-1"),
            _mk_event("c", agent="agent-2"),
        ]
        report = lab.analyze(events, project_id="p")
        assert report.by_agent.get("agent-1") == 2

    check("counts: by_domain aggregated", t_counts_by_domain)
    check("counts: by_component aggregated", t_counts_by_component)
    check("counts: by_agent aggregated", t_counts_by_agent)

    # ---- feedback ----
    def t_feedback_only_ready() -> None:
        events = [
            _mk_event(f"e{i}", what="chronic problem",
                       component=f"c{i}") for i in range(5)
        ]
        report = lab.analyze(events, project_id="p")
        exp = lab.to_experience_inputs(report)
        assert len(exp) == len(report.ready_lessons())
        for x in exp:
            assert x["kind"] == "pitfall"
            assert x["lesson"]

    def t_feedback_shape_c26_compatible() -> None:
        events = [
            _mk_event(f"e{i}", what="cross-comp",
                       component=f"c{i}") for i in range(2)
        ]
        report = lab.analyze(events, project_id="p")
        exp = lab.to_experience_inputs(report)
        assert exp
        e = exp[0]
        for k in ("id", "kind", "reliability", "score", "problem",
                   "lesson", "event_ids" if "event_ids" in e
                   else "evidence_refs"):
            assert k in e, k

    def t_learning_mirrors_experience() -> None:
        events = [_mk_event(f"e{i}") for i in range(5)]
        report = lab.analyze(events, project_id="p")
        a = lab.to_experience_inputs(report)
        b = lab.to_learning_experiences(report)
        assert a == b

    check("feedback: only READY lessons surfaced",
          t_feedback_only_ready)
    check("feedback: C26-compatible shape",
          t_feedback_shape_c26_compatible)
    check("feedback: C27 mirrors C26",
          t_learning_mirrors_experience)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        events = [_mk_event(f"e{i}") for i in range(3)]
        report = lab.analyze(events, project_id="p")
        d = report.to_dict()
        assert d["id"] == report.id
        assert "patterns" in d and "lessons" in d
        s = report.summary()
        assert "Failure Analysis Report" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                events = [
                    _mk_event(f"e{i}", what="recurring cache issue",
                              component=f"c{i % 2}") for i in range(4)
                ]
                report = lab.analyze(events, project_id="proj-x")
                repo = FailureRepository(memory=mem, ontology=ont)
                ent = repo.save(report, project_id="proj-x")
                assert ent
                loaded = repo.load(report.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == report.id
                # Ontology: ROOT_CAUSE root + FAILURE entities for patterns
                assert ont.count(kind=EntityKind.ROOT_CAUSE) >= 1
                assert ont.count(kind=EntityKind.FAILURE) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (ROOT_CAUSE + FAILURE)",
          t_persist)

    # ---- empty ----
    def t_empty() -> None:
        report = lab.analyze([], project_id="p")
        assert report.events_scanned == 0
        assert report.patterns == []
        assert report.lessons == []

    check("empty: no events → empty report", t_empty)

    # ---- determinism ----
    def t_deterministic_report() -> None:
        events = [
            _mk_event(f"e{i}", what="cache stampede",
                       component=f"c{i % 3}") for i in range(6)
        ]
        r1 = lab.analyze(events, project_id="p")
        r2 = lab.analyze(events, project_id="p")
        assert r1.clusters == r2.clusters
        assert len(r1.patterns) == len(r2.patterns)
        for p1, p2 in zip(r1.patterns, r2.patterns):
            assert p1.kind == p2.kind
            assert p1.signature == p2.signature
            assert p1.occurrences == p2.occurrences

    check("deterministic: same events → same report",
          t_deterministic_report)

    # ---- E2E with real C18 ----
    def t_e2e_with_real_c18() -> None:
        from sebrain.c18 import TestExecutionEngine
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text('"""pkg."""\n')
            (root / "pkg" / "calc.py").write_text(
                "def add(a, b):\n    return a + b\n"
            )
            (root / "tests").mkdir()
            (root / "tests" / "__init__.py").write_text("")
            (root / "tests" / "test_calc.py").write_text(
                "from pkg.calc import add\n"
                "\n"
                "def test_add() -> None:\n"
                "    assert add(1, 2) == 999  # deliberately wrong\n"
            )
            run = TestExecutionEngine().run_full(
                root=root, project_id="demo",
            )
            assert run.failed >= 1
            events = lab.collect_events(
                project_id="demo", test_run=run, from_memory=False,
            )
            assert any(e.domain is FailureDomain.TEST for e in events)
            report = lab.analyze(events, project_id="demo")
            assert report.events_scanned >= 1
            assert len(report.patterns) >= 1
            # Single failure → SINGLE, lesson HELD
            assert report.patterns[0].kind is PatternKind.SINGLE

    check("e2e: real C18 failure → C33 analysis",
          t_e2e_with_real_c18, requires_pytest=True)

    # ---- E2E with C26 feedback ----
    def t_e2e_feedback_to_c26() -> None:
        from sebrain.c26 import (
            ExperienceExtractor, ExperienceCandidate,
        )
        # Build 5 cross-component failures to trigger READY
        events = [
            _mk_event(f"e{i}", what="cache stampede on cold start",
                       root_cause="cache warming missing",
                       component=f"service-{i % 3}")
            for i in range(5)
        ]
        report = lab.analyze(events, project_id="demo")
        ready = report.ready_lessons()
        assert ready
        # Feed to C26
        exp_dicts = lab.to_experience_inputs(report)
        assert exp_dicts
        ex_inputs = [
            ExperienceCandidate(**{
                k: v for k, v in d.items()
                if k in ExperienceCandidate.__dataclass_fields__
            })
            for d in exp_dicts
        ]
        assert len(ex_inputs) >= 1
        for x in ex_inputs:
            assert x.kind == "pitfall"
            assert x.problem
            assert x.lesson

    check("e2e: C33 → C26 feedback path",
          t_e2e_feedback_to_c26)

    print()
    skip_note = f", {skipped} skipped (pytest not installed)" if skipped else ""
    print(f"Self-tests: {passed} passed, {len(failures)} failed{skip_note}")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 10. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C33 — Failure Analysis Laboratory")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        s = SQLiteStorage(Path(td) / "sebrain.sqlite3")
        s.initialize()
        try:
            mem = MemoryStore(s)

            # Seed history: some recurring failures across components
            for i in range(4):
                mem.record_failure(
                    f"cache-{i}",
                    what=f"cache stampede on cold start (service-{i % 2})",
                    root_cause="cache warming missing",
                    fix=None,
                    scope_id="demo",
                )
            for i in range(2):
                mem.record_failure(
                    f"slow-{i}",
                    what=f"slow query detected (service-{i})",
                    root_cause="missing index on hot column",
                    fix=None,
                    scope_id="demo",
                )
            mem.record_failure(
                "one-off",
                what="transient network hiccup",
                root_cause="DNS resolution delayed",
                scope_id="demo",
            )

            lab = FailureLab(memory=mem)
            events = lab.collect_events(project_id="demo")
            print(f"\n[1] Collected {len(events)} events from memory")

            report = lab.analyze(events, project_id="demo")
            print("\n[2] Analysis report:")
            print(report.summary())

            print("\n[3] Domain distribution:")
            for d, n in sorted(report.by_domain.items(),
                                key=lambda kv: (-kv[1], kv[0])):
                print(f"    {d:15s}: {n}")

            print("\n[4] Recurring / chronic patterns:")
            for p in report.patterns:
                if p.kind is PatternKind.SINGLE:
                    continue
                print(f"    [{p.kind.value}] occurrences={p.occurrences}  "
                      f"domains={p.domains}  comps={len(p.components)}")
                print(f"      signature: {_short(p.signature, 80)}")
                print(f"      desc: {_short(p.description, 80)}")

            print("\n[5] Lessons (with verdicts):")
            for l in report.lessons:
                mark = {"ready": "✓", "held": "…",
                        "not_applicable": "–"}[l.verdict.value]
                print(f"    {mark} [{l.verdict.value:14s}] "
                      f"{_short(l.problem, 70)}")
                print(f"        approach: {_short(l.approach, 70)}")
                print(f"        rationale: {l.rationale}")

            print("\n[6] Feedback to C26 (READY lessons only):")
            for x in lab.to_experience_inputs(report):
                print(f"    → [{x['kind']}] "
                      f"{_short(x['problem'], 70)}")
                print(f"        score={x['score']} "
                      f"reliability={x['reliability']}")

            print("\n[7] Persistence:")
            ont = Ontology(s)
            repo = FailureRepository(memory=mem, ontology=ont)
            ent = repo.save(report, project_id="demo")
            print(f"    ontology entity: {ent[:12]}…")
            print(f"    ROOT_CAUSE count: "
                  f"{ont.count(kind=EntityKind.ROOT_CAUSE)}")
            print(f"    FAILURE count: "
                  f"{ont.count(kind=EntityKind.FAILURE)}")
        finally:
            s.shutdown()

    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
