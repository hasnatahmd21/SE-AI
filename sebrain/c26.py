"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C26 — EXPERIENCE EXTRACTION (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04 (C18/C19/C22 via duck-typed integration).

Purpose:
    After work is completed, extract REUSABLE experience from the evidence
    trail — but NEVER promote every outcome to permanent knowledge.

    Lifecycle:
        Outcome (raw signals)
            → ExperienceCandidate (structured, still speculative)
            → ReliabilityScoring (evidence-based)
            → PromotionGate (accept / hold / reject)
            → ExperienceRecord (only if promoted)

Captured fields (per the specification):
    problem · context · approach · decision · implementation ·
    result · failure · repair · verification · lesson

Reliability model:
    RELIABLE         ← multi-source, verified, no unresolved failures
    PROMISING        ← verified or 2+ sources, low residual risk
    WEAK             ← single source or inferred, no contradiction
    UNRELIABLE       ← contradicted by failures OR unverified-only
    UNKNOWN          ← insufficient signal to judge

Promotion rules (deterministic):
    RELIABLE   → PROMOTED
    PROMISING  → PROMOTED (with `promotion_notes`)
    WEAK       → HELD (kept as candidate, not promoted)
    UNRELIABLE → REJECTED
    UNKNOWN    → HELD

    Additional hard rule: any UNRESOLVED failure tied to the same
    problem → UNRELIABLE (cannot promote).

Invariants honored:
    - NO external LLM. Deterministic scoring.
    - Never auto-promote on a single unverified outcome.
    - Never silent-mutate an experience: promotion is an explicit state.
    - Historical candidates are preserved (even rejected ones).
    - Same inputs + same timestamps → same decisions (deterministic).
    - Provenance on every record (source subsystem, evidence ids).
    - Bounded: max_candidates, max_sources_per_candidate.

Explicit limitations (Rule #59):
    - "Reusable" is heuristic — signal keyword overlap is a proxy, not
      semantic similarity. Cross-domain generalization requires C27.
    - Verification levels depend on C22's honest ladder; if no verification
      evidence is present, candidates degrade to WEAK/UNKNOWN.
    - No automatic retraction: once promoted, an experience stays until a
      later phase explicitly demotes it (C27 handles demotion).
    - Time-window queries rely on C04's updated_at; clock skew across
      machines is not handled here.

Contents:
  1.  Enums: ExperienceKind, Reliability, PromotionStatus, SignalSource
  2.  Dataclasses: Signal, OutcomeBundle, ExperienceCandidate,
                   PromotionDecision, ExperienceRecord, ExtractionReport
  3.  Harvester (reads from C04 memory; accepts duck-typed C18/C19/C22)
  4.  Extractor (bundles signals → candidates)
  5.  ReliabilityScorer (evidence-based)
  6.  PromotionGate (candidate + reliability → decision)
  7.  ExperienceExtractor facade
  8.  ExperienceRepository (persist to C04 + C02)
  9.  Self-tests (~30)
 10.  Demo

Run as script:
    python -m sebrain.c26            # demo
    python -m sebrain.c26 --test     # self-tests
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
""".split())


def _tokens(text: str) -> set[str]:
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if w.lower() not in _STOPWORDS
    }


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class ExperienceKind(str, Enum):
    """Coarse classification for reuse."""
    BUILD = "build"              # successfully built something new
    FIX = "fix"                  # repaired a diagnosed failure
    REFACTOR = "refactor"        # behavior-preserving restructure
    OPTIMIZATION = "optimization"  # performance improvement
    PATTERN = "pattern"          # a repeated shape worth naming
    PITFALL = "pitfall"          # a failure to avoid repeating
    WORKFLOW = "workflow"        # a sequence of steps
    UNKNOWN = "unknown"


class Reliability(str, Enum):
    """Evidence-based reliability of an experience candidate."""
    RELIABLE = "reliable"
    PROMISING = "promising"
    WEAK = "weak"
    UNRELIABLE = "unreliable"
    UNKNOWN = "unknown"


_RELIABILITY_RANK = {
    Reliability.UNRELIABLE: 0,
    Reliability.UNKNOWN: 1,
    Reliability.WEAK: 2,
    Reliability.PROMISING: 3,
    Reliability.RELIABLE: 4,
}


class PromotionStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    HELD = "held"
    REJECTED = "rejected"


class SignalSource(str, Enum):
    """Where a signal came from. Multi-source strengthens reliability."""
    DECISION_MEMORY = "decision_memory"
    FAILURE_MEMORY = "failure_memory"
    EXPERIENCE_MEMORY = "experience_memory"
    TEST_RUN = "test_run"
    DEBUG_REPORT = "debug_report"
    REPAIR_RESULT = "repair_result"
    VERIFICATION_BUNDLE = "verification_bundle"
    ARCHITECTURE = "architecture"
    SECURITY_REPORT = "security_report"
    PERF_REPORT = "perf_report"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Signal:
    """A single raw observation contributing to an experience candidate."""
    source: SignalSource
    ref: str                    # id / key of the originating artifact
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "ref": self.ref,
            "text": self.text,
            "metadata": dict(self.metadata),
            "observed_at": self.observed_at,
        }


@dataclass(slots=True)
class OutcomeBundle:
    """A grouped set of signals presumed to describe the same episode."""
    signals: list[Signal] = field(default_factory=list)
    project_id: str = ""
    task_id: str = ""

    def sources(self) -> set[SignalSource]:
        return {s.source for s in self.signals}

    def by_source(self, src: SignalSource) -> list[Signal]:
        return [s for s in self.signals if s.source is src]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "task_id": self.task_id,
            "signals": [s.to_dict() for s in self.signals],
        }


@dataclass(slots=True)
class ExperienceCandidate:
    id: str = field(default_factory=_new_id)
    kind: ExperienceKind = ExperienceKind.UNKNOWN
    problem: str = ""
    context: str = ""
    approach: str = ""
    decision: str = ""
    implementation: str = ""
    result: str = ""
    failure: str = ""
    repair: str = ""
    verification: str = ""
    lesson: str = ""
    sources: list[SignalSource] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "problem": self.problem,
            "context": self.context,
            "approach": self.approach,
            "decision": self.decision,
            "implementation": self.implementation,
            "result": self.result,
            "failure": self.failure,
            "repair": self.repair,
            "verification": self.verification,
            "lesson": self.lesson,
            "sources": [s.value for s in self.sources],
            "evidence_refs": list(self.evidence_refs),
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def has_content(self) -> bool:
        return any((self.problem, self.approach, self.lesson,
                    self.failure, self.repair, self.context,
                    self.decision, self.result))


@dataclass(slots=True)
class PromotionDecision:
    candidate_id: str
    status: PromotionStatus
    reliability: Reliability
    score: float
    rationale: str = ""
    promotion_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "reliability": self.reliability.value,
            "score": float(self.score),
            "rationale": self.rationale,
            "promotion_notes": self.promotion_notes,
        }


@dataclass(slots=True)
class ExperienceRecord:
    id: str = field(default_factory=_new_id)
    candidate: ExperienceCandidate = field(default_factory=ExperienceCandidate)
    decision: PromotionDecision = field(
        default_factory=lambda: PromotionDecision(
            candidate_id="", status=PromotionStatus.CANDIDATE,
            reliability=Reliability.UNKNOWN, score=0.0,
        )
    )
    supersedes: str | None = None
    status: str = "active"          # active | superseded | archived
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "candidate": self.candidate.to_dict(),
            "decision": self.decision.to_dict(),
            "supersedes": self.supersedes,
            "status": self.status,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ExtractionReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    bundles_scanned: int = 0
    candidates: list[ExperienceCandidate] = field(default_factory=list)
    decisions: list[PromotionDecision] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)   # candidate ids
    held: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "bundles_scanned": self.bundles_scanned,
            "candidates": [c.to_dict() for c in self.candidates],
            "decisions": [d.to_dict() for d in self.decisions],
            "promoted": list(self.promoted),
            "held": list(self.held),
            "rejected": list(self.rejected),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Extraction Report ===\n"
            f"project={self.project_id}\n"
            f"bundles={self.bundles_scanned}  "
            f"candidates={len(self.candidates)}\n"
            f"promoted={len(self.promoted)}  "
            f"held={len(self.held)}  "
            f"rejected={len(self.rejected)}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. HARVESTER — read from C04 memory + duck-typed artifacts
# ════════════════════════════════════════════════════════════════════════════
class Harvester:
    """Reads outcomes from memory and external artifacts.

    Everything returns `Signal` objects — nothing is interpreted here.
    """

    def __init__(self, *, max_signals: int = 500) -> None:
        if max_signals < 1:
            raise ValidationError("max_signals must be >= 1")
        self.max_signals = max_signals

    # ---- memory-based ----
    def from_memory(
        self, memory: MemoryStore, *, project_id: str,
    ) -> list[Signal]:
        signals: list[Signal] = []
        try:
            decisions = memory.find(
                kind=MemoryKind.DECISION,
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
            )
            for d in decisions:
                signals.append(Signal(
                    source=SignalSource.DECISION_MEMORY,
                    ref=d.id,
                    text=_short(
                        f"{d.content.get('decision','')} | "
                        f"{d.content.get('rationale','')}",
                        400,
                    ),
                    metadata={
                        "key": d.key,
                        "alternatives": d.content.get("alternatives", []),
                    },
                    observed_at=d.updated_at,
                ))
            failures = memory.find(
                kind=MemoryKind.FAILURE,
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
            )
            for f in failures:
                signals.append(Signal(
                    source=SignalSource.FAILURE_MEMORY,
                    ref=f.id,
                    text=_short(
                        f"{f.content.get('what','')} | "
                        f"cause={f.content.get('root_cause','')} | "
                        f"fix={f.content.get('fix','')}",
                        400,
                    ),
                    metadata={"key": f.key},
                    observed_at=f.updated_at,
                ))
            exps = memory.find(
                kind=MemoryKind.EXPERIENCE,
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
            )
            for e in exps:
                signals.append(Signal(
                    source=SignalSource.EXPERIENCE_MEMORY,
                    ref=e.id,
                    text=_short(
                        f"{e.content.get('problem','')} | "
                        f"approach={e.content.get('approach','')} | "
                        f"lesson={e.content.get('lesson','')}",
                        400,
                    ),
                    metadata={"key": e.key},
                    observed_at=e.updated_at,
                ))
        except Exception as exc:
            log.warning("c26.memory_harvest_error", error=str(exc))
        return signals[: self.max_signals]

    # ---- duck-typed adapters ----
    def from_test_run(self, run: Any) -> list[Signal]:
        out: list[Signal] = []
        if run is None:
            return out
        rid = str(getattr(run, "id", "")) or "test_run"
        status = getattr(getattr(run, "status", None), "value", "unknown")
        passed = int(getattr(run, "passed", 0) or 0)
        failed = int(getattr(run, "failed", 0) or 0)
        errors = int(getattr(run, "errors", 0) or 0)
        out.append(Signal(
            source=SignalSource.TEST_RUN,
            ref=rid,
            text=(f"test_run status={status} passed={passed} "
                  f"failed={failed} errors={errors}"),
            metadata={
                "status": status, "passed": passed,
                "failed": failed, "errors": errors,
            },
        ))
        return out

    def from_debug_report(self, report: Any) -> list[Signal]:
        out: list[Signal] = []
        if report is None:
            return out
        rid = str(getattr(report, "id", "")) or "debug"
        cat = getattr(getattr(report, "category", None), "value", "unknown")
        sig = getattr(report, "signature", None)
        exc = getattr(sig, "exception_type", "") if sig else ""
        msg = getattr(sig, "exception_message", "") if sig else ""
        out.append(Signal(
            source=SignalSource.DEBUG_REPORT,
            ref=rid,
            text=_short(f"failure {cat} {exc}: {msg}", 300),
            metadata={"category": cat, "exception": exc},
        ))
        return out

    def from_repair_result(self, result: Any) -> list[Signal]:
        out: list[Signal] = []
        if result is None:
            return out
        rid = str(getattr(result, "id", "")) or "repair"
        accepted = getattr(result, "accepted_id", None)
        cat = str(getattr(result, "category", ""))
        out.append(Signal(
            source=SignalSource.REPAIR_RESULT,
            ref=rid,
            text=_short(
                f"repair {cat} accepted={bool(accepted)}", 300,
            ),
            metadata={
                "category": cat,
                "accepted": bool(accepted),
                "accepted_id": accepted,
            },
        ))
        return out

    def from_verification_bundle(self, bundle: Any) -> list[Signal]:
        out: list[Signal] = []
        if bundle is None:
            return out
        bid = str(getattr(bundle, "id", "")) or "verify"
        results = list(getattr(bundle, "results", []) or [])
        supported = sum(
            1 for r in results
            if getattr(getattr(r, "result", None), "value", "") == "supported"
        )
        refuted = sum(
            1 for r in results
            if getattr(getattr(r, "result", None), "value", "") == "refuted"
        )
        out.append(Signal(
            source=SignalSource.VERIFICATION_BUNDLE,
            ref=bid,
            text=(f"verification supported={supported} refuted={refuted} "
                  f"total={len(results)}"),
            metadata={
                "supported": supported, "refuted": refuted,
                "total": len(results),
            },
        ))
        return out


# ════════════════════════════════════════════════════════════════════════════
# 4. EXTRACTOR — bundle signals → ExperienceCandidate
# ════════════════════════════════════════════════════════════════════════════
class Extractor:
    """Deterministic transformation from signals to candidates.

    Heuristic rules (documented):
      - Signal `text` fields are joined per bucket (problem/approach/...).
      - Kind is inferred from the mix of sources:
          * REPAIR_RESULT accepted + DEBUG_REPORT → FIX
          * DEBUG_REPORT without accepted repair  → PITFALL
          * VERIFICATION_BUNDLE with refuted>0    → PITFALL
          * TEST_RUN with 0 failed/errors + DECISION/EXPERIENCE → BUILD
          * TEST_RUN with failures only            → PITFALL
          * Otherwise                              → UNKNOWN
    """

    def candidate_from(
        self, bundle: OutcomeBundle,
    ) -> ExperienceCandidate | None:
        if not bundle.signals:
            return None
        cand = ExperienceCandidate()
        cand.provenance = Provenance(
            source="experience_extractor",
            source_type=ProvenanceType.INFERENCE,
            confidence=Confidence.MEDIUM,
        )
        src_set = bundle.sources()
        cand.sources = sorted(src_set, key=lambda s: s.value)

        # ---- Problem: first failure/debug signal text, or fallback ----
        for s in bundle.signals:
            if s.source in (SignalSource.DEBUG_REPORT,
                            SignalSource.FAILURE_MEMORY):
                cand.problem = s.text
                break
        if not cand.problem:
            for s in bundle.signals:
                if s.source is SignalSource.EXPERIENCE_MEMORY:
                    cand.problem = s.text
                    break

        # ---- Context: test run summary (if any) ----
        for s in bundle.signals:
            if s.source is SignalSource.TEST_RUN:
                cand.context = s.text
                break

        # ---- Decision: first decision signal ----
        for s in bundle.signals:
            if s.source is SignalSource.DECISION_MEMORY:
                cand.decision = s.text
                break

        # ---- Implementation / Approach: experience memory text ----
        for s in bundle.signals:
            if s.source is SignalSource.EXPERIENCE_MEMORY:
                cand.approach = s.text
                break

        # ---- Repair: accepted repair signal ----
        for s in bundle.signals:
            if s.source is SignalSource.REPAIR_RESULT:
                if s.metadata.get("accepted"):
                    cand.repair = s.text
                break

        # ---- Result: verification summary + test outcome ----
        parts: list[str] = []
        for s in bundle.signals:
            if s.source is SignalSource.VERIFICATION_BUNDLE:
                parts.append(s.text)
            elif s.source is SignalSource.TEST_RUN:
                parts.append(s.text)
        cand.result = " | ".join(parts) if parts else ""

        # ---- Verification: verbatim from verification bundle ----
        for s in bundle.signals:
            if s.source is SignalSource.VERIFICATION_BUNDLE:
                cand.verification = s.text
                break

        # ---- Failure: first failure signal text (already used for
        # problem; keep a copy for the record) ----
        for s in bundle.signals:
            if s.source is SignalSource.FAILURE_MEMORY:
                cand.failure = s.text
                break

        # ---- Lesson (heuristic) ----
        cand.lesson = self._derive_lesson(cand)

        # ---- Kind ----
        cand.kind = self._infer_kind(bundle)

        # ---- Evidence refs ----
        cand.evidence_refs = [f"{s.source.value}:{s.ref}"
                              for s in bundle.signals]

        return cand if cand.has_content() else None

    def _infer_kind(self, bundle: OutcomeBundle) -> ExperienceKind:
        src = bundle.sources()
        has_accepted_repair = any(
            s.source is SignalSource.REPAIR_RESULT
            and s.metadata.get("accepted")
            for s in bundle.signals
        )
        has_debug = SignalSource.DEBUG_REPORT in src
        has_verify = SignalSource.VERIFICATION_BUNDLE in src
        def _refuted_count(s: "Signal") -> int:
            if "refuted" in s.metadata:
                return int(s.metadata.get("refuted", 0))
            # Fall back to parsing "refuted=N" out of the signal text —
            # tests (and some real producers) only put the count there,
            # not in metadata, so reading metadata alone silently treats
            # every refutation as zero.
            m = re.search(r"refuted=(\d+)", s.text)
            return int(m.group(1)) if m else 0

        refuted = sum(
            _refuted_count(s)
            for s in bundle.signals
            if s.source is SignalSource.VERIFICATION_BUNDLE
        )
        has_test = SignalSource.TEST_RUN in src
        test_failed = any(
            int(s.metadata.get("failed", 0)) > 0
            or int(s.metadata.get("errors", 0)) > 0
            for s in bundle.signals
            if s.source is SignalSource.TEST_RUN
        )
        has_decision = SignalSource.DECISION_MEMORY in src
        has_exp = SignalSource.EXPERIENCE_MEMORY in src

        if has_accepted_repair and has_debug:
            return ExperienceKind.FIX
        if has_debug and not has_accepted_repair:
            return ExperienceKind.PITFALL
        if has_verify and refuted > 0:
            return ExperienceKind.PITFALL
        if has_test and not test_failed and (has_decision or has_exp):
            return ExperienceKind.BUILD
        if has_test and test_failed:
            return ExperienceKind.PITFALL
        if has_decision and not has_debug:
            return ExperienceKind.PATTERN
        return ExperienceKind.UNKNOWN

    def _derive_lesson(self, cand: ExperienceCandidate) -> str:
        # Deterministic template — never fabricates specifics
        bits: list[str] = []
        if cand.repair:
            bits.append(f"Repair pattern: {_short(cand.repair, 160)}")
        if cand.verification and "refuted=0" in cand.verification:
            bits.append("Verification passed with no refutations.")
        elif cand.verification and "refuted=" in cand.verification:
            # extract count
            m = re.search(r"refuted=(\d+)", cand.verification)
            if m and int(m.group(1)) > 0:
                bits.append(
                    f"Verification had {m.group(1)} refuted claim(s); "
                    f"do not treat this as reliable."
                )
        if cand.problem and not cand.repair:
            bits.append(
                f"Recurring problem signature: {_short(cand.problem, 160)}"
            )
        if not bits and cand.decision:
            # BUILD-kind candidates (clean test run + a decision made, no
            # problem/repair/verification involved) otherwise get no
            # lesson text at all even though there's real content to
            # summarize.
            bits.append(f"Decision that shipped cleanly: {_short(cand.decision, 160)}")
        return " ".join(bits)


# ════════════════════════════════════════════════════════════════════════════
# 5. RELIABILITY SCORER
# ════════════════════════════════════════════════════════════════════════════
class ReliabilityScorer:
    """Scores a candidate with evidence-based rules.

    Score components (0.0 .. 1.0):
        source_diversity   : #distinct sources / 5, capped at 1.0
        verification_ok    : +0.30 if verified, +0.10 if partial, else 0
        no_residual_failure: +0.20 if no unresolved failure, else -0.40
        has_lesson         : +0.15 if lesson non-empty
        has_repair         : +0.10 if accepted repair
        prompt_clarity     : +0.10 if problem is non-empty and >40 chars

    Then bucketed into Reliability labels (documented thresholds).
    """

    def score(
        self, cand: ExperienceCandidate,
        *,
        has_unresolved_failure: bool,
    ) -> tuple[Reliability, float, str]:
        if not cand.has_content():
            return Reliability.UNKNOWN, 0.0, "empty candidate"

        # 1. source diversity
        s_div = min(1.0, len(cand.sources) / 5.0)

        # 2. verification
        ver_ok = False
        ver_partial = False
        if cand.verification:
            if "refuted=0" in cand.verification:
                ver_ok = True
            elif "refuted=" in cand.verification:
                ver_partial = True

        # 3. residual failure
        # failure_memory signals unrelated to a repair → unresolved
        residual = has_unresolved_failure

        score = s_div * 0.25
        if ver_ok:
            score += 0.30
        elif ver_partial:
            score += 0.10
        if residual:
            score -= 0.40
        else:
            score += 0.20
        if cand.lesson:
            score += 0.15
        if cand.repair:
            score += 0.10
        if len(cand.problem) >= 40:
            score += 0.10

        score = max(0.0, min(1.0, score))

        # Bucket
        if residual and score < 0.5:
            label = Reliability.UNRELIABLE
        elif score >= 0.75 and ver_ok:
            label = Reliability.RELIABLE
        elif score >= 0.55 and (ver_ok or ver_partial or len(cand.sources) >= 2):
            label = Reliability.PROMISING
        elif score >= 0.30:
            label = Reliability.WEAK
        elif score > 0:
            label = Reliability.WEAK
        else:
            label = Reliability.UNKNOWN

        rationale = (
            f"score={score:.2f} diversity={s_div:.2f} "
            f"ver_ok={ver_ok} ver_partial={ver_partial} "
            f"residual_failure={residual} "
            f"lesson={'y' if cand.lesson else 'n'} "
            f"repair={'y' if cand.repair else 'n'}"
        )
        return label, score, rationale


# ════════════════════════════════════════════════════════════════════════════
# 6. PROMOTION GATE
# ════════════════════════════════════════════════════════════════════════════
class PromotionGate:
    """Turn (candidate, reliability, score) → explicit decision."""

    def decide(
        self, cand: ExperienceCandidate, reliability: Reliability, score: float,
        *, has_unresolved_failure: bool,
    ) -> PromotionDecision:
        if not cand.has_content():
            return PromotionDecision(
                candidate_id=cand.id, status=PromotionStatus.REJECTED,
                reliability=Reliability.UNKNOWN, score=score,
                rationale="empty candidate — nothing to promote",
            )

        if has_unresolved_failure and reliability is Reliability.UNRELIABLE:
            return PromotionDecision(
                candidate_id=cand.id, status=PromotionStatus.REJECTED,
                reliability=reliability, score=score,
                rationale=(
                    "unresolved failure tied to this episode; "
                    "cannot promote"
                ),
            )

        if reliability is Reliability.RELIABLE:
            return PromotionDecision(
                candidate_id=cand.id, status=PromotionStatus.PROMOTED,
                reliability=reliability, score=score,
                rationale="multi-source, verified, no unresolved failure",
            )
        if reliability is Reliability.PROMISING:
            return PromotionDecision(
                candidate_id=cand.id, status=PromotionStatus.PROMOTED,
                reliability=reliability, score=score,
                rationale="promising evidence; promoted with notes",
                promotion_notes=(
                    "consider re-verification before applying in a "
                    "different project (C27 will handle promotion to "
                    "global knowledge)"
                ),
            )
        if reliability is Reliability.UNRELIABLE:
            return PromotionDecision(
                candidate_id=cand.id, status=PromotionStatus.REJECTED,
                reliability=reliability, score=score,
                rationale="contradicted or unresolved; rejected",
            )
        # WEAK / UNKNOWN
        return PromotionDecision(
            candidate_id=cand.id, status=PromotionStatus.HELD,
            reliability=reliability, score=score,
            rationale="insufficient signal to promote; held as candidate",
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. FACADE
# ════════════════════════════════════════════════════════════════════════════
class ExperienceExtractor:
    """Compose harvester + extractor + scorer + gate."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        max_candidates: int = 100,
    ) -> None:
        if max_candidates < 1:
            raise ValidationError("max_candidates must be >= 1")
        self.memory = memory
        self.harvester = Harvester()
        self.extractor = Extractor()
        self.scorer = ReliabilityScorer()
        self.gate = PromotionGate()
        self.max_candidates = max_candidates

    # ---- bundle construction ----
    def build_bundles(
        self,
        *,
        project_id: str,
        task_id: str = "",
        test_run: Any = None,
        debug_report: Any = None,
        repair_result: Any = None,
        verification_bundle: Any = None,
        from_memory: bool = True,
    ) -> list[OutcomeBundle]:
        """Group signals into bundles.

        Rule: if any of {debug, repair, verify} is provided, form a single
        bundle containing all of them + test_run + (memory signals
        related by keyword overlap). Otherwise, form a single memory-only
        bundle plus one bundle per external artifact.
        """
        sigs: list[Signal] = []
        if from_memory and self.memory is not None:
            sigs.extend(self.harvester.from_memory(
                self.memory, project_id=project_id,
            ))
        if test_run is not None:
            sigs.extend(self.harvester.from_test_run(test_run))
        if debug_report is not None:
            sigs.extend(self.harvester.from_debug_report(debug_report))
        if repair_result is not None:
            sigs.extend(self.harvester.from_repair_result(repair_result))
        if verification_bundle is not None:
            sigs.extend(self.harvester.from_verification_bundle(
                verification_bundle,
            ))

        if not sigs:
            return []

        # --- Single-episode heuristic: if we have any "episode artifact"
        # (debug/repair/verify), bundle everything into one bundle for
        # that episode. Otherwise split memory signals into a single
        # bundle plus one per standalone artifact.
        episode = any(x is not None for x in
                      (debug_report, repair_result, verification_bundle))
        if episode:
            return [OutcomeBundle(
                signals=sigs, project_id=project_id, task_id=task_id,
            )]

        # No explicit episode → split by source category
        bundles: list[OutcomeBundle] = []
        memory_sources = {
            SignalSource.DECISION_MEMORY, SignalSource.FAILURE_MEMORY,
            SignalSource.EXPERIENCE_MEMORY,
        }
        mem = [s for s in sigs if s.source in memory_sources]
        others = [s for s in sigs if s.source not in memory_sources]
        if mem:
            bundles.append(OutcomeBundle(
                signals=mem, project_id=project_id, task_id=task_id,
            ))
        if others:
            # group by source
            by_src: dict[SignalSource, list[Signal]] = {}
            for s in others:
                by_src.setdefault(s.source, []).append(s)
            for src, group in by_src.items():
                bundles.append(OutcomeBundle(
                    signals=group, project_id=project_id, task_id=task_id,
                ))
        return bundles

    # ---- scoring helpers ----
    def _has_unresolved_failure(
        self, cand: ExperienceCandidate, bundle: OutcomeBundle,
    ) -> bool:
        """Return True if there is a failure signal that is not paired
        with an accepted repair in the same bundle.
        """
        failure_sigs = [
            s for s in bundle.signals
            if s.source in (SignalSource.FAILURE_MEMORY,
                            SignalSource.DEBUG_REPORT)
        ]
        accepted_repair = any(
            s.source is SignalSource.REPAIR_RESULT
            and s.metadata.get("accepted")
            for s in bundle.signals
        )
        refuted = any(
            "refuted=" in s.text and "refuted=0" not in s.text
            for s in bundle.signals
            if s.source is SignalSource.VERIFICATION_BUNDLE
        )
        if refuted:
            return True
        if failure_sigs and not accepted_repair:
            # A failure with no accepted repair inside the episode
            return True
        return False

    # ---- main ----
    def extract(
        self,
        *,
        project_id: str,
        task_id: str = "",
        test_run: Any = None,
        debug_report: Any = None,
        repair_result: Any = None,
        verification_bundle: Any = None,
        from_memory: bool = True,
    ) -> ExtractionReport:
        report = ExtractionReport(
            project_id=project_id,
            provenance=Provenance(
                source="experience_extractor",
                source_type=ProvenanceType.INFERENCE,
                confidence=Confidence.MEDIUM,
            ),
        )
        bundles = self.build_bundles(
            project_id=project_id, task_id=task_id,
            test_run=test_run, debug_report=debug_report,
            repair_result=repair_result,
            verification_bundle=verification_bundle,
            from_memory=from_memory,
        )
        report.bundles_scanned = len(bundles)

        for b in bundles[: self.max_candidates]:
            cand = self.extractor.candidate_from(b)
            if cand is None:
                continue
            unresolved = self._has_unresolved_failure(cand, b)
            reliability, score, rationale = self.scorer.score(
                cand, has_unresolved_failure=unresolved,
            )
            decision = self.gate.decide(
                cand, reliability, score,
                has_unresolved_failure=unresolved,
            )
            decision.rationale = f"{decision.rationale} | {rationale}"
            report.candidates.append(cand)
            report.decisions.append(decision)
            if decision.status is PromotionStatus.PROMOTED:
                report.promoted.append(cand.id)
            elif decision.status is PromotionStatus.HELD:
                report.held.append(cand.id)
            else:
                report.rejected.append(cand.id)

        report.rationale = (
            f"bundles={report.bundles_scanned} "
            f"candidates={len(report.candidates)} "
            f"promoted={len(report.promoted)} "
            f"held={len(report.held)} rejected={len(report.rejected)}"
        )
        return report

    # ---- promote (persist) ----
    def promote_to_memory(
        self, record: ExperienceRecord, *, project_id: str,
    ) -> str:
        """Persist a promoted experience to C04 experience memory."""
        if self.memory is None:
            raise ValidationError("memory not attached")
        cand = record.candidate
        key = f"experience:{_digest(cand.id)[7:23]}"
        self.memory.upsert(
            MemoryKind.EXPERIENCE, key,
            {
                "problem": cand.problem,
                "approach": cand.approach,
                "lesson": cand.lesson,
                "result": cand.result,
                "context": cand.context,
                "kind": cand.kind.value,
                "reliability": record.decision.reliability.value,
                "score": record.decision.score,
                "evidence_refs": list(cand.evidence_refs),
                "sources": [s.value for s in cand.sources],
            },
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["c26", "experience", cand.kind.value,
                  record.decision.reliability.value],
            provenance=record.provenance,
        )
        return key


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY (persist extraction reports)
# ════════════════════════════════════════════════════════════════════════════
class ExperienceRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(
        self, report: ExtractionReport, *, project_id: str,
    ) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"extraction_report:{report.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, report.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["extraction", "c26"],
            provenance=report.provenance,
        )
        if self.ontology is None:
            return key
        root = self.ontology.add(
            EntityKind.EXPERIENCE,
            _short(
                f"Extraction {report.id[:8]} "
                f"(promoted={len(report.promoted)})", 120,
            ),
            attributes={
                "report_id": report.id,
                "project_id": project_id,
                "bundles_scanned": report.bundles_scanned,
                "candidates": len(report.candidates),
                "promoted": len(report.promoted),
                "held": len(report.held),
                "rejected": len(report.rejected),
            },
            tags=["extraction-report"],
            provenance=report.provenance,
        )
        # Record each candidate as an EXPERIENCE entity
        for cand, dec in zip(report.candidates, report.decisions):
            ce = self.ontology.add(
                EntityKind.EXPERIENCE,
                _short(f"{cand.kind.value}: {_short(cand.problem, 60)}", 120),
                attributes={
                    "candidate_id": cand.id,
                    "kind": cand.kind.value,
                    "reliability": dec.reliability.value,
                    "status": dec.status.value,
                    "score": dec.score,
                },
                tags=["experience-candidate", cand.kind.value,
                      dec.status.value],
                provenance=cand.provenance,
            )
            try:
                self.ontology.link(RelationKind.CONTAINS, root.id, ce.id)
            except ValidationError:
                pass
        return root.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"extraction_report:{report_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 9. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
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

    print("Running C26 self-tests…")

    # ---- fake artifacts ----
    class FakeTestRun:
        def __init__(self, passed=0, failed=0, errors=0, status="succeeded"):
            self.id = "tr-1"
            self.status = type("S", (), {"value": status})()
            self.passed = passed
            self.failed = failed
            self.errors = errors
            self.results = []

    class FakeDebugReport:
        def __init__(self, category="logic", exc="AssertionError"):
            self.id = "dbg-1"
            self.category = type("C", (), {"value": category})()
            self.signature = type("G", (), {
                "exception_type": exc,
                "exception_message": "assert 1 == 2",
            })()

    class FakeRepairResult:
        def __init__(self, accepted=True, category="logic"):
            self.id = "rep-1"
            self.accepted_id = "cand-x" if accepted else None
            self.category = category

    class FakeVerificationBundle:
        def __init__(self, supported=1, refuted=0):
            self.id = "vb-1"
            self.results = (
                [type("R", (), {
                    "result": type("O", (), {"value": "supported"})(),
                })()] * supported
                + [type("R", (), {
                    "result": type("O", (), {"value": "refuted"})(),
                })()] * refuted
            )

    # ---- harvester ----
    def t_harvest_test_run() -> None:
        h = Harvester()
        s = h.from_test_run(FakeTestRun(passed=3, failed=0, errors=0))
        assert len(s) == 1
        assert s[0].source is SignalSource.TEST_RUN
        assert "passed=3" in s[0].text

    def t_harvest_debug_report() -> None:
        h = Harvester()
        s = h.from_debug_report(FakeDebugReport())
        assert len(s) == 1
        assert s[0].source is SignalSource.DEBUG_REPORT
        assert "AssertionError" in s[0].text

    def t_harvest_repair_result() -> None:
        h = Harvester()
        s = h.from_repair_result(FakeRepairResult(accepted=True))
        assert s[0].source is SignalSource.REPAIR_RESULT
        assert s[0].metadata["accepted"] is True

    def t_harvest_verification_bundle() -> None:
        h = Harvester()
        s = h.from_verification_bundle(
            FakeVerificationBundle(supported=2, refuted=0),
        )
        assert s[0].source is SignalSource.VERIFICATION_BUNDLE
        assert "refuted=0" in s[0].text

    def t_harvest_from_memory() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                mem.record_decision(
                    "db", decision="Use SQLite",
                    rationale="local-first", scope_id="proj-x",
                )
                mem.record_failure(
                    "bug-1", what="test fail",
                    root_cause="missing fixture", scope_id="proj-x",
                )
                h = Harvester()
                sigs = h.from_memory(mem, project_id="proj-x")
                srcs = {sig.source for sig in sigs}
                assert SignalSource.DECISION_MEMORY in srcs
                assert SignalSource.FAILURE_MEMORY in srcs
            finally:
                s.shutdown()

    check("harvest: from test run", t_harvest_test_run)
    check("harvest: from debug report", t_harvest_debug_report)
    check("harvest: from repair result", t_harvest_repair_result)
    check("harvest: from verification bundle",
          t_harvest_verification_bundle)
    check("harvest: from C04 memory", t_harvest_from_memory)

    # ---- extractor ----
    def t_extractor_kind_fix() -> None:
        b = OutcomeBundle(signals=[
            Signal(SignalSource.DEBUG_REPORT, "d1",
                   text="failure logic AssertionError"),
            Signal(SignalSource.REPAIR_RESULT, "r1",
                   text="repair accepted=True",
                   metadata={"accepted": True, "category": "logic"}),
        ])
        cand = Extractor().candidate_from(b)
        assert cand is not None
        assert cand.kind is ExperienceKind.FIX
        assert cand.repair

    def t_extractor_kind_pitfall() -> None:
        b = OutcomeBundle(signals=[
            Signal(SignalSource.DEBUG_REPORT, "d1",
                   text="failure logic AssertionError"),
        ])
        cand = Extractor().candidate_from(b)
        assert cand is not None
        assert cand.kind is ExperienceKind.PITFALL

    def t_extractor_kind_build() -> None:
        b = OutcomeBundle(signals=[
            Signal(SignalSource.TEST_RUN, "tr1",
                   text="test_run passed=3 failed=0 errors=0",
                   metadata={"passed": 3, "failed": 0, "errors": 0}),
            Signal(SignalSource.DECISION_MEMORY, "dec1",
                   text="Use SQLite | local-first"),
        ])
        cand = Extractor().candidate_from(b)
        assert cand is not None
        assert cand.kind is ExperienceKind.BUILD

    def t_extractor_kind_pitfall_refuted() -> None:
        b = OutcomeBundle(signals=[
            Signal(SignalSource.VERIFICATION_BUNDLE, "vb1",
                   text="verification supported=1 refuted=2 total=3"),
        ])
        cand = Extractor().candidate_from(b)
        assert cand is not None
        assert cand.kind is ExperienceKind.PITFALL

    def t_extractor_empty_returns_none() -> None:
        b = OutcomeBundle(signals=[])
        assert Extractor().candidate_from(b) is None

    def t_extractor_lesson_not_fabricated() -> None:
        # No repair, no verification → lesson should reflect problem only
        b = OutcomeBundle(signals=[
            Signal(SignalSource.DEBUG_REPORT, "d1",
                   text="failure logic AssertionError at foo"),
        ])
        cand = Extractor().candidate_from(b)
        assert cand is not None
        assert "Recurring problem" in cand.lesson
        # Should NOT claim anything about repair/verification
        assert "Repair pattern" not in cand.lesson

    check("extractor: FIX kind (debug + accepted repair)",
          t_extractor_kind_fix)
    check("extractor: PITFALL kind (debug only)",
          t_extractor_kind_pitfall)
    check("extractor: BUILD kind (clean tests + decision)",
          t_extractor_kind_build)
    check("extractor: PITFALL when verification refuted",
          t_extractor_kind_pitfall_refuted)
    check("extractor: empty bundle → None",
          t_extractor_empty_returns_none)
    check("extractor: lesson does not fabricate content",
          t_extractor_lesson_not_fabricated)

    # ---- reliability scorer ----
    def t_scorer_reliable() -> None:
        cand = ExperienceCandidate(
            problem="P" * 60, approach="A", lesson="L", repair="R",
            verification="verification supported=2 refuted=0 total=2",
            sources=[SignalSource.DEBUG_REPORT,
                     SignalSource.REPAIR_RESULT,
                     SignalSource.VERIFICATION_BUNDLE,
                     SignalSource.TEST_RUN,
                     SignalSource.DECISION_MEMORY],
        )
        label, score, _ = ReliabilityScorer().score(
            cand, has_unresolved_failure=False,
        )
        assert label is Reliability.RELIABLE, (label, score)
        assert score >= 0.75

    def t_scorer_weak_single_source() -> None:
        cand = ExperienceCandidate(
            problem="some problem text that is long enough to earn a point",
            sources=[SignalSource.DEBUG_REPORT],
        )
        label, score, _ = ReliabilityScorer().score(
            cand, has_unresolved_failure=True,
        )
        assert label in (Reliability.WEAK, Reliability.UNRELIABLE)
        assert score < 0.75

    def t_scorer_unreliable_residual() -> None:
        cand = ExperienceCandidate(
            problem="P" * 60,
            sources=[SignalSource.DEBUG_REPORT],
            # No repair, no verification
        )
        label, score, _ = ReliabilityScorer().score(
            cand, has_unresolved_failure=True,
        )
        assert label in (Reliability.UNRELIABLE, Reliability.WEAK), (label, score)

    def t_scorer_promising() -> None:
        cand = ExperienceCandidate(
            problem="P" * 60, lesson="L",
            verification="verification supported=2 refuted=0 total=2",
            sources=[SignalSource.VERIFICATION_BUNDLE,
                     SignalSource.TEST_RUN],
        )
        label, score, _ = ReliabilityScorer().score(
            cand, has_unresolved_failure=False,
        )
        assert label in (Reliability.PROMISING, Reliability.RELIABLE), (label, score)

    check("scorer: multi-source verified → RELIABLE", t_scorer_reliable)
    check("scorer: single source → WEAK/UNRELIABLE",
          t_scorer_weak_single_source)
    check("scorer: residual failure → not promote-worthy",
          t_scorer_unreliable_residual)
    check("scorer: 2 verified sources → PROMISING+",
          t_scorer_promising)

    # ---- promotion gate ----
    def t_gate_promotes_reliable() -> None:
        cand = ExperienceCandidate(id="c1", problem="P" * 60)
        d = PromotionGate().decide(
            cand, Reliability.RELIABLE, 0.9,
            has_unresolved_failure=False,
        )
        assert d.status is PromotionStatus.PROMOTED

    def t_gate_holds_weak() -> None:
        cand = ExperienceCandidate(id="c1", problem="something")
        d = PromotionGate().decide(
            cand, Reliability.WEAK, 0.4,
            has_unresolved_failure=False,
        )
        assert d.status is PromotionStatus.HELD

    def t_gate_rejects_unreliable() -> None:
        cand = ExperienceCandidate(id="c1", problem="something")
        d = PromotionGate().decide(
            cand, Reliability.UNRELIABLE, 0.2,
            has_unresolved_failure=True,
        )
        assert d.status is PromotionStatus.REJECTED

    def t_gate_rejects_empty() -> None:
        cand = ExperienceCandidate()   # empty
        d = PromotionGate().decide(
            cand, Reliability.UNKNOWN, 0.0,
            has_unresolved_failure=False,
        )
        assert d.status is PromotionStatus.REJECTED

    check("gate: RELIABLE → PROMOTED", t_gate_promotes_reliable)
    check("gate: WEAK → HELD", t_gate_holds_weak)
    check("gate: UNRELIABLE → REJECTED", t_gate_rejects_unreliable)
    check("gate: empty candidate → REJECTED", t_gate_rejects_empty)

    # ---- facade (with memory) ----
    def t_extract_fix_episode() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                # Seed a decision + failure
                mem.record_decision(
                    "db", decision="Use SQLite",
                    rationale="local-first", scope_id="proj-x",
                )
                mem.record_failure(
                    "bug-1", what="test auth failed",
                    root_cause="missing fixture", scope_id="proj-x",
                )
                ex = ExperienceExtractor(memory=mem)
                rep = ex.extract(
                    project_id="proj-x",
                    test_run=FakeTestRun(passed=3, failed=0, errors=0),
                    debug_report=FakeDebugReport(),
                    repair_result=FakeRepairResult(accepted=True),
                    verification_bundle=FakeVerificationBundle(
                        supported=3, refuted=0,
                    ),
                )
                # One episode bundle → one candidate
                assert len(rep.candidates) == 1
                cand = rep.candidates[0]
                assert cand.kind is ExperienceKind.FIX
                assert cand.repair
                # Reliability should be at least PROMISING
                dec = rep.decisions[0]
                assert dec.reliability in (
                    Reliability.PROMISING, Reliability.RELIABLE,
                )
                assert dec.status is PromotionStatus.PROMOTED
                assert cand.id in rep.promoted
            finally:
                s.shutdown()

    def t_extract_pitfall_not_promoted() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                mem.record_failure(
                    "bug-1", what="test failed",
                    root_cause="missing fixture", scope_id="proj-x",
                )
                ex = ExperienceExtractor(memory=mem)
                rep = ex.extract(
                    project_id="proj-x",
                    debug_report=FakeDebugReport(),
                    # no accepted repair
                )
                assert len(rep.candidates) == 1
                dec = rep.decisions[0]
                # Without a repair, unresolved failure → not RELIABLE
                assert dec.reliability in (
                    Reliability.UNRELIABLE, Reliability.WEAK,
                )
                assert dec.status in (
                    PromotionStatus.HELD, PromotionStatus.REJECTED,
                )
            finally:
                s.shutdown()

    def t_extract_empty_bundle() -> None:
        ex = ExperienceExtractor()
        rep = ex.extract(project_id="p")
        assert rep.candidates == []
        assert rep.bundles_scanned == 0

    check("facade: fix episode → PROMOTED",
          t_extract_fix_episode)
    check("facade: debug-only (no repair) → not PROMOTED",
          t_extract_pitfall_not_promoted)
    check("facade: empty inputs → empty report", t_extract_empty_bundle)

    # ---- promote_to_memory ----
    def t_promote_to_memory() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ex = ExperienceExtractor(memory=mem)
                cand = ExperienceCandidate(
                    id="cand-1", problem="P" * 60,
                    approach="A", lesson="L",
                    sources=[SignalSource.VERIFICATION_BUNDLE,
                             SignalSource.TEST_RUN],
                )
                dec = PromotionDecision(
                    candidate_id=cand.id, status=PromotionStatus.PROMOTED,
                    reliability=Reliability.PROMISING, score=0.7,
                )
                rec = ExperienceRecord(candidate=cand, decision=dec)
                key = ex.promote_to_memory(rec, project_id="proj-x")
                assert key.startswith("experience:")
                # Verify memory entry exists
                entries = mem.find(
                    kind=MemoryKind.EXPERIENCE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert any(e.key == key for e in entries)
            finally:
                s.shutdown()

    check("promote_to_memory: writes experience to C04",
          t_promote_to_memory)

    # ---- persistence ----
    def t_persist_report() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                ex = ExperienceExtractor(memory=mem)
                rep = ex.extract(
                    project_id="proj-x",
                    test_run=FakeTestRun(passed=5, failed=0, errors=0),
                    debug_report=FakeDebugReport(),
                    repair_result=FakeRepairResult(accepted=True),
                    verification_bundle=FakeVerificationBundle(
                        supported=3, refuted=0,
                    ),
                )
                repo = ExperienceRepository(memory=mem, ontology=ont)
                ent = repo.save(rep, project_id="proj-x")
                assert ent
                loaded = repo.load(rep.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == rep.id
                assert ont.count(kind=EntityKind.EXPERIENCE) >= 2
            finally:
                s.shutdown()

    check("persist: report + candidates to C04/C02",
          t_persist_report)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        ex = ExperienceExtractor()
        rep = ex.extract(
            project_id="p",
            debug_report=FakeDebugReport(),
        )
        d = rep.to_dict()
        assert d["id"] == rep.id
        s = rep.summary()
        assert "Extraction Report" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- determinism ----
    def t_deterministic() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                mem.record_decision("d1", decision="X",
                                     rationale="Y", scope_id="p")
                mem.record_failure("f1", what="W",
                                    root_cause="C", scope_id="p")
                ex1 = ExperienceExtractor(memory=mem)
                ex2 = ExperienceExtractor(memory=mem)
                r1 = ex1.extract(project_id="p")
                r2 = ex2.extract(project_id="p")
                assert len(r1.candidates) == len(r2.candidates)
                if r1.candidates:
                    c1 = r1.candidates[0]
                    c2 = r2.candidates[0]
                    assert c1.kind == c2.kind
                    assert c1.problem == c2.problem
                    assert r1.decisions[0].reliability == \
                           r2.decisions[0].reliability
            finally:
                s.shutdown()

    check("deterministic: same inputs → same decision",
          t_deterministic)

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
                "    assert add(1, 2) == 3\n"
            )
            run = TestExecutionEngine().run_full(
                root=root, project_id="demo",
            )
            assert run.passed == 1
            ex = ExperienceExtractor()
            rep = ex.extract(
                project_id="demo",
                test_run=run,
                from_memory=False,
            )
            # Only a test run → one candidate
            assert len(rep.candidates) == 1
            cand = rep.candidates[0]
            assert cand.context and "test_run" in cand.context

    check("e2e: real C18 run feeds C26 extraction",
          t_e2e_with_real_c18, requires_pytest=True)

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
    print("SE Brain C26 — Experience Extraction")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        s = SQLiteStorage(Path(td) / "sebrain.sqlite3")
        s.initialize()
        try:
            mem = MemoryStore(s)

            # Seed some history
            mem.record_decision(
                "web-framework",
                decision="Use FastAPI for the REST API",
                rationale="Async + type hints + auto docs",
                alternatives=["Flask", "Django"],
                scope_id="demo",
            )
            mem.record_failure(
                "bug-fixture",
                what="test_create_task failed: fixture missing",
                root_cause="pytest fixture not registered in conftest",
                fix="added conftest.py with db_session fixture",
                scope_id="demo",
            )

            ex = ExperienceExtractor(memory=mem)

            class FakeTestRun:
                id = "tr-demo"
                status = type("S", (), {"value": "succeeded"})()
                passed = 12
                failed = 0
                errors = 0
                results = []

            class FakeDebugReport:
                id = "dbg-demo"
                category = type("C", (), {"value": "logic"})()
                signature = type("G", (), {
                    "exception_type": "AssertionError",
                    "exception_message": "assert add(1,2) == 3",
                })()

            class FakeRepairResult:
                id = "rep-demo"
                accepted_id = "cand-demo"
                category = "logic"

            class FakeVerificationBundle:
                id = "vb-demo"
                results = [
                    type("R", (), {
                        "result": type("O", (), {"value": "supported"})(),
                    })(),
                ] * 3

            print("\n[1] Extract from a completed 'FIX' episode:")
            rep = ex.extract(
                project_id="demo",
                test_run=FakeTestRun(),
                debug_report=FakeDebugReport(),
                repair_result=FakeRepairResult(),
                verification_bundle=FakeVerificationBundle(),
            )
            print(rep.summary())
            for cand, dec in zip(rep.candidates, rep.decisions):
                print(f"    [{cand.kind.value}] reliability="
                      f"{dec.reliability.value}  score={dec.score:.2f}")
                print(f"        problem : {_short(cand.problem, 80)}")
                print(f"        lesson  : {_short(cand.lesson, 100)}")
                print(f"        status  : {dec.status.value}")
                print(f"        rationale: {_short(dec.rationale, 120)}")

            print("\n[2] Extract from a 'PITFALL' (unresolved failure):")
            rep2 = ex.extract(
                project_id="demo",
                debug_report=FakeDebugReport(),
                # intentionally no accepted repair
            )
            print(rep2.summary())
            for cand, dec in zip(rep2.candidates, rep2.decisions):
                print(f"    [{cand.kind.value}] reliability="
                      f"{dec.reliability.value}  score={dec.score:.2f}")
                print(f"        status  : {dec.status.value}")
                print(f"        rationale: {_short(dec.rationale, 120)}")

            print("\n[3] Persist extraction report:")
            ont = Ontology(s)
            repo = ExperienceRepository(memory=mem, ontology=ont)
            ent = repo.save(rep, project_id="demo")
            print(f"    ontology entity: {ent[:12]}…")
            print(f"    EXPERIENCE count: "
                  f"{ont.count(kind=EntityKind.EXPERIENCE)}")

            print("\n[4] Promote experience to memory:")
            if rep.promoted:
                # Find the promoted candidate
                idx = next(i for i, c in enumerate(rep.candidates)
                           if c.id in rep.promoted)
                cand = rep.candidates[idx]
                dec = rep.decisions[idx]
                rec = ExperienceRecord(candidate=cand, decision=dec)
                key = ex.promote_to_memory(rec, project_id="demo")
                print(f"    memory key: {key}")

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
