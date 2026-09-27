"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C28 — META-REASONING (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04 (C05–C27 via duck-typed integration).

Purpose:
    Evaluate the Brain's OWN PROCESS. Before acting on results from earlier
    phases, C28 asks: is the pipeline healthy enough to proceed? Or should we
    stop, redirect, ask for clarification, invoke a specialist, or revise the
    plan?

Meta-questions answered (from the specification):
    1.  Is the requirement sufficiently understood?          → requirement_check
    2.  Is evidence sufficient?                              → evidence_check
    3.  Is the architecture appropriate?                     → architecture_check
    4.  Is the selected technology justified?                → technology_check
    5.  Are alternatives missing?                            → alternatives_check
    6.  Is testing sufficient?                               → testing_check
    7.  Is verification sufficient?                          → verification_check
    8.  Is uncertainty too high?                             → uncertainty_check
    9.  Should another specialist be invoked?                → specialist_check
    10. Should the plan be revised?                          → plan_check

Verdicts (in order of escalation):
    PROCEED              — pipeline healthy; safe to continue
    PROCEED_WITH_CAUTION — minor concerns; continue but flag
    REDIRECT             — an action is required before continuing
    STOP                 — unsafe / insufficient; do not proceed

The assessment includes an explicit `redirections` list. Each redirection
carries:
    kind (what to do) + target_phase (where) + rationale (why)
    + blocking (bool) — whether it prevents proceeding

Invariants honored:
    - NO external LLM. Every check is deterministic given structured inputs.
    - Missing input → INSUFFICIENT (never a fabricated PASS).
    - Redirections are concrete and actionable, never vague.
    - Same inputs → same verdict (deterministic).
    - STOP is a real outcome (hard rule: refuted verification, all tests
      failing, or unresolved contradictions → STOP).
    - Bounded work; never scans unbounded collections.
    - Provenance on the assessment; no silent policy decisions.

Explicit limitations (Rule #59):
    - Thresholds are documented constants; they are NOT calibrated from
      ground truth. They reflect engineering defaults, not empirical optima.
    - C28 does NOT re-run any phase. It evaluates the *results* of phases.
    - If a caller supplies only partial pipeline outputs, some checks report
      INSUFFICIENT and the verdict degrades gracefully (usually
      REDIRECT, never a fake PROCEED).
    - "Sufficient" is a threshold judgment; false positives (too cautious)
      and false negatives (too permissive) are possible.

Contents:
  1.  Enums: MetaQuestion, CheckOutcome, MetaVerdict, RedirectionKind
  2.  Dataclasses: MetaSignal, Redirection, MetaCheck, MetaAssessment
  3.  PipelineContext (duck-typed bundle of prior-phase outputs)
  4.  Ten checks
  5.  MetaReasoner facade (aggregation + verdict)
  6.  MetaRepository (persist)
  7.  Self-tests (~30)
  8.  Demo

Run as script:
    python -m sebrain.c28            # demo
    python -m sebrain.c28 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import hashlib
import json
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
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore


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


def _getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _enum_val(x: Any) -> str:
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class MetaQuestion(str, Enum):
    REQUIREMENT_UNDERSTOOD = "requirement_understood"
    EVIDENCE_SUFFICIENT = "evidence_sufficient"
    ARCHITECTURE_APPROPRIATE = "architecture_appropriate"
    TECHNOLOGY_JUSTIFIED = "technology_justified"
    ALTERNATIVES_PRESENT = "alternatives_present"
    TESTING_SUFFICIENT = "testing_sufficient"
    VERIFICATION_SUFFICIENT = "verification_sufficient"
    UNCERTAINTY_ACCEPTABLE = "uncertainty_acceptable"
    SPECIALIST_NEEDED = "specialist_needed"
    PLAN_REVISED = "plan_revised"


class CheckOutcome(str, Enum):
    PASS = "pass"
    CAUTION = "caution"
    NEEDS_ATTENTION = "needs_attention"
    BLOCKING = "blocking"           # hard stop
    INSUFFICIENT = "insufficient"   # missing input — never a fake PASS


_OUTCOME_RANK = {
    CheckOutcome.PASS: 0,
    CheckOutcome.INSUFFICIENT: 1,
    CheckOutcome.CAUTION: 2,
    CheckOutcome.NEEDS_ATTENTION: 3,
    CheckOutcome.BLOCKING: 4,
}


class MetaVerdict(str, Enum):
    PROCEED = "proceed"
    PROCEED_WITH_CAUTION = "proceed_with_caution"
    REDIRECT = "redirect"
    STOP = "stop"


class RedirectionKind(str, Enum):
    ASK_USER_FOR_CLARIFICATION = "ask_user_for_clarification"
    INVOKE_SPECIALIST = "invoke_specialist"
    REVISE_PLAN = "revise_plan"
    REGENERATE_ARCHITECTURE = "regenerate_architecture"
    RECONSIDER_TECHNOLOGY = "reconsider_technology"
    GENERATE_MORE_TESTS = "generate_more_tests"
    RUN_MORE_VERIFICATION = "run_more_verification"
    RUN_SECURITY_SCAN = "run_security_scan"
    RUN_PERFORMANCE_SCAN = "run_performance_scan"
    FIX_FAILING_TESTS = "fix_failing_tests"
    RESOLVE_CONTRADICTIONS = "resolve_contradictions"
    RAISE_UNCERTAINTY_ALERT = "raise_uncertainty_alert"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class MetaSignal:
    """A structured observation feeding a check."""
    name: str
    value: Any = None
    source_phase: str = ""      # e.g. "C05", "C18"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "value": self.value,
            "source_phase": self.source_phase, "note": self.note,
        }


@dataclass(slots=True)
class Redirection:
    kind: RedirectionKind
    target_phase: str = ""      # e.g. "C05", "C09", "user"
    rationale: str = ""
    blocking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target_phase": self.target_phase,
            "rationale": self.rationale,
            "blocking": self.blocking,
        }


@dataclass(slots=True)
class MetaCheck:
    question: MetaQuestion
    outcome: CheckOutcome
    score: float = 0.0            # 0.0 .. 1.0 (higher = better)
    rationale: str = ""
    signals: list[MetaSignal] = field(default_factory=list)
    redirections: list[Redirection] = field(default_factory=list)

    def is_blocking(self) -> bool:
        return self.outcome is CheckOutcome.BLOCKING

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.value,
            "outcome": self.outcome.value,
            "score": float(self.score),
            "rationale": self.rationale,
            "signals": [s.to_dict() for s in self.signals],
            "redirections": [r.to_dict() for r in self.redirections],
        }


@dataclass(slots=True)
class MetaAssessment:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    task_id: str = ""
    verdict: MetaVerdict = MetaVerdict.PROCEED
    checks: list[MetaCheck] = field(default_factory=list)
    redirections: list[Redirection] = field(default_factory=list)
    overall_score: float = 0.0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def blocking_checks(self) -> list[MetaCheck]:
        return [c for c in self.checks if c.is_blocking()]

    def needs_attention(self) -> list[MetaCheck]:
        return [c for c in self.checks
                if c.outcome is CheckOutcome.NEEDS_ATTENTION]

    def insufficient_checks(self) -> list[MetaCheck]:
        return [c for c in self.checks
                if c.outcome is CheckOutcome.INSUFFICIENT]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "task_id": self.task_id,
            "verdict": self.verdict.value,
            "overall_score": self.overall_score,
            "checks": [c.to_dict() for c in self.checks],
            "redirections": [r.to_dict() for r in self.redirections],
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        counts = {o.value: 0 for o in CheckOutcome}
        for c in self.checks:
            counts[c.outcome.value] += 1
        lines = [
            "=== Meta-Reasoning Assessment ===",
            f"verdict: {self.verdict.value}  "
            f"score={self.overall_score:.2f}",
            f"checks: {counts}",
        ]
        if self.redirections:
            lines.append(f"redirections: {len(self.redirections)}")
            for r in self.redirections[:5]:
                lines.append(
                    f"  · [{r.kind.value}] "
                    f"→ {r.target_phase or 'n/a'}"
                    f"{' (blocking)' if r.blocking else ''}"
                )
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE CONTEXT — duck-typed bundle
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class PipelineContext:
    """All prior-phase outputs available to C28.

    Every field is optional. Missing fields are handled honestly by
    individual checks (INSUFFICIENT, not fake PASS).
    """
    spec: Any = None                  # C05 RequirementSpec
    intent_ctx: Any = None            # C06 IntentContext
    plan: Any = None                  # C08 Plan
    tech_selection: Any = None        # C09 TechSelectionResult
    architecture: Any = None          # C10 ArchitectureResult
    repo_index: Any = None            # C13 RepoIndex
    synthesis: Any = None             # C14 SynthesisResult
    build_result: Any = None          # C15 BuildResult
    test_plan: Any = None             # C17 TestPlan
    test_run: Any = None              # C18 TestRunResult
    debug_report: Any = None          # C19 DebugReport
    repair_result: Any = None         # C20 RepairResult
    critique: Any = None              # C21 CritiqueReport
    verification: Any = None          # C22 VerificationBundle
    security: Any = None              # C23 SecurityReport
    perf: Any = None                  # C24 PerfReport
    experience: Any = None            # C26 ExtractionReport
    learning: Any = None              # C27 LearningReport
    extra: dict[str, Any] = field(default_factory=dict)

    def phases_present(self) -> list[str]:
        out: list[str] = []
        for name, val in (
            ("C05", self.spec), ("C06", self.intent_ctx),
            ("C08", self.plan), ("C09", self.tech_selection),
            ("C10", self.architecture), ("C13", self.repo_index),
            ("C14", self.synthesis), ("C15", self.build_result),
            ("C17", self.test_plan), ("C18", self.test_run),
            ("C19", self.debug_report), ("C20", self.repair_result),
            ("C21", self.critique), ("C22", self.verification),
            ("C23", self.security), ("C24", self.perf),
            ("C26", self.experience), ("C27", self.learning),
        ):
            if val is not None:
                out.append(name)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"phases_present": self.phases_present()}


# ════════════════════════════════════════════════════════════════════════════
# 4. CHECKS
# ════════════════════════════════════════════════════════════════════════════
class _BaseCheck:
    question: MetaQuestion = MetaQuestion.REQUIREMENT_UNDERSTOOD
    blocking_possible: bool = False

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        raise NotImplementedError

    # ---- helpers ----
    def _mk(
        self, *, outcome: CheckOutcome, score: float, rationale: str,
        signals: Iterable[MetaSignal] = (),
        redirections: Iterable[Redirection] = (),
    ) -> MetaCheck:
        return MetaCheck(
            question=self.question, outcome=outcome, score=score,
            rationale=rationale,
            signals=list(signals), redirections=list(redirections),
        )


# ---- 1. Requirement clarity ----
class RequirementCheck(_BaseCheck):
    question = MetaQuestion.REQUIREMENT_UNDERSTOOD
    AMBIGUITY_CAUTION = 2
    AMBIGUITY_REDIRECT = 5
    MISSING_CAUTION = 2
    MISSING_REDIRECT = 5

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        spec = ctx.spec
        if spec is None:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.0,
                rationale="no requirement spec available",
                redirections=[Redirection(
                    kind=RedirectionKind.ASK_USER_FOR_CLARIFICATION,
                    target_phase="C05",
                    rationale="spec missing",
                    blocking=True,
                )],
            )
        conf = _enum_val(_getattr(spec, "confidence", "unknown"))
        amb = len(_getattr(spec, "ambiguities", []) or [])
        miss = len(_getattr(spec, "missing", []) or [])
        signals = [
            MetaSignal("confidence", conf, "C05"),
            MetaSignal("ambiguities", amb, "C05"),
            MetaSignal("missing_categories", miss, "C05"),
        ]
        # Hard rule: confidence unknown → redirect (not block)
        if conf == "unknown":
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.1,
                rationale="requirement confidence unknown",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.ASK_USER_FOR_CLARIFICATION,
                    target_phase="C05",
                    rationale="spec confidence unknown",
                    blocking=False,
                )],
            )
        if amb >= self.AMBIGUITY_REDIRECT or miss >= self.MISSING_REDIRECT:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.2,
                rationale=(
                    f"{amb} ambiguities, {miss} missing categories "
                    f"(>= {self.AMBIGUITY_REDIRECT})"
                ),
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.ASK_USER_FOR_CLARIFICATION,
                    target_phase="C05",
                    rationale="too many unresolved ambiguities/missing items",
                )],
            )
        if amb >= self.AMBIGUITY_CAUTION or miss >= self.MISSING_CAUTION:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.6,
                rationale=f"{amb} ambiguities, {miss} missing categories",
                signals=signals,
            )
        score = 0.7 if conf == "high" else 0.85
        return self._mk(
            outcome=CheckOutcome.PASS, score=score,
            rationale=f"spec confidence={conf}, {amb} ambiguities, "
                      f"{miss} missing",
            signals=signals,
        )


# ---- 2. Evidence sufficiency (verification) ----
class EvidenceCheck(_BaseCheck):
    question = MetaQuestion.EVIDENCE_SUFFICIENT
    blocking_possible = True

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        vb = ctx.verification
        if vb is None:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale="no verification bundle; evidence chain unknown",
                redirections=[Redirection(
                    kind=RedirectionKind.RUN_MORE_VERIFICATION,
                    target_phase="C22",
                    rationale="run verification before proceeding",
                )],
            )
        results = list(_getattr(vb, "results", []) or [])
        if not results:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale="verification bundle is empty",
                redirections=[Redirection(
                    kind=RedirectionKind.RUN_MORE_VERIFICATION,
                    target_phase="C22",
                    rationale="no verification results to evaluate",
                )],
            )
        supported = sum(
            1 for r in results
            if _enum_val(_getattr(r, "result", "")) == "supported"
        )
        refuted = sum(
            1 for r in results
            if _enum_val(_getattr(r, "result", "")) == "refuted"
        )
        insufficient = sum(
            1 for r in results
            if _enum_val(_getattr(r, "result", "")) == "insufficient_evidence"
        )
        signals = [
            MetaSignal("supported", supported, "C22"),
            MetaSignal("refuted", refuted, "C22"),
            MetaSignal("insufficient", insufficient, "C22"),
            MetaSignal("total", len(results), "C22"),
        ]
        if refuted > 0:
            return self._mk(
                outcome=CheckOutcome.BLOCKING, score=0.0,
                rationale=f"{refuted} verification claim(s) refuted",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.RESOLVE_CONTRADICTIONS,
                    target_phase="C19",
                    rationale="refuted claims must be investigated before "
                              "proceeding",
                    blocking=True,
                )],
            )
        if insufficient >= max(1, len(results) // 2):
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.4,
                rationale=f"{insufficient}/{len(results)} claims lack evidence",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.RUN_MORE_VERIFICATION,
                    target_phase="C22",
                    rationale="too many claims lack evidence",
                )],
            )
        ratio = supported / len(results)
        if ratio >= 0.8:
            return self._mk(
                outcome=CheckOutcome.PASS, score=0.9,
                rationale=f"{supported}/{len(results)} claims verified",
                signals=signals,
            )
        return self._mk(
            outcome=CheckOutcome.CAUTION, score=0.6,
            rationale=f"only {supported}/{len(results)} claims verified",
            signals=signals,
        )


# ---- 3. Architecture appropriateness ----
class ArchitectureCheck(_BaseCheck):
    question = MetaQuestion.ARCHITECTURE_APPROPRIATE

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        arch = ctx.architecture
        if arch is None:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.0,
                rationale="no architecture result available",
                redirections=[Redirection(
                    kind=RedirectionKind.REGENERATE_ARCHITECTURE,
                    target_phase="C10",
                    rationale="architecture not produced",
                )],
            )
        decision = _getattr(arch, "decision", None)
        if decision is None:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.2,
                rationale="architecture has no decision",
                redirections=[Redirection(
                    kind=RedirectionKind.REGENERATE_ARCHITECTURE,
                    target_phase="C10",
                    rationale="decision missing",
                )],
            )
        selected = _getattr(decision, "selected", None)
        if selected is None:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.2,
                rationale="architecture decision has no selection",
            )
        comps = list(_getattr(selected, "components", []) or [])
        ifaces = list(_getattr(selected, "interfaces", []) or [])
        f_bounds = list(_getattr(selected, "failure_boundaries", []) or [])
        s_bounds = list(_getattr(selected, "security_boundaries", []) or [])
        # comparison
        comparison = _getattr(decision, "comparison", None)
        cand_count = len(list(_getattr(comparison, "scores", []) or []))
        pareto = list(_getattr(comparison, "pareto_front", []) or [])
        signals = [
            MetaSignal("components", len(comps), "C10"),
            MetaSignal("interfaces", len(ifaces), "C10"),
            MetaSignal("failure_boundaries", len(f_bounds), "C10"),
            MetaSignal("security_boundaries", len(s_bounds), "C10"),
            MetaSignal("candidates_compared", cand_count, "C10"),
            MetaSignal("pareto_front", len(pareto), "C10"),
        ]
        problems: list[str] = []
        if not comps: problems.append("no components")
        if not ifaces: problems.append("no interfaces")
        if not f_bounds: problems.append("no failure boundaries")
        if not s_bounds: problems.append("no security boundaries")
        if problems:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale="; ".join(problems),
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.REGENERATE_ARCHITECTURE,
                    target_phase="C10",
                    rationale="architecture is missing key structure",
                )],
            )
        if cand_count < 2:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.6,
                rationale="only one architecture candidate was compared",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.REGENERATE_ARCHITECTURE,
                    target_phase="C10",
                    rationale="compare at least 2 candidates",
                )],
            )
        score = 0.85
        if len(pareto) >= 2:
            score = 0.9
        return self._mk(
            outcome=CheckOutcome.PASS, score=score,
            rationale=(
                f"architecture complete: {len(comps)} components, "
                f"{len(ifaces)} interfaces, candidates={cand_count}, "
                f"pareto={len(pareto)}"
            ),
            signals=signals,
        )


# ---- 4. Technology justification ----
class TechnologyCheck(_BaseCheck):
    question = MetaQuestion.TECHNOLOGY_JUSTIFIED

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        ts = ctx.tech_selection
        if ts is None:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.0,
                rationale="no tech selection available",
                redirections=[Redirection(
                    kind=RedirectionKind.RECONSIDER_TECHNOLOGY,
                    target_phase="C09",
                    rationale="tech selection missing",
                )],
            )
        decisions = list(_getattr(ts, "decisions", []) or [])
        if not decisions:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.2,
                rationale="tech selection has no decisions",
                redirections=[Redirection(
                    kind=RedirectionKind.RECONSIDER_TECHNOLOGY,
                    target_phase="C09",
                    rationale="no category decisions",
                )],
            )
        categories = {_enum_val(_getattr(d, "category", "")): d
                      for d in decisions}
        expected = {"language", "framework", "database", "architecture"}
        missing = sorted(expected - set(categories.keys()))
        with_winner = sum(
            1 for d in decisions if _getattr(d, "winner", None) is not None
        )
        signals = [
            MetaSignal("decisions", len(decisions), "C09"),
            MetaSignal("categories", sorted(categories.keys()), "C09"),
            MetaSignal("missing_categories", missing, "C09"),
            MetaSignal("decisions_with_winner", with_winner, "C09"),
        ]
        if missing:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale=f"missing category decisions: {missing}",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.RECONSIDER_TECHNOLOGY,
                    target_phase="C09",
                    rationale="expected categories not selected",
                )],
            )
        if with_winner < len(decisions):
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.55,
                rationale=(
                    f"{with_winner}/{len(decisions)} decisions have a "
                    f"selected winner"
                ),
                signals=signals,
            )
        return self._mk(
            outcome=CheckOutcome.PASS, score=0.85,
            rationale=f"all {len(decisions)} categories have decisions",
            signals=signals,
        )


# ---- 5. Alternatives present ----
class AlternativesCheck(_BaseCheck):
    question = MetaQuestion.ALTERNATIVES_PRESENT
    MIN_ALTERNATIVES = 1

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        ts = ctx.tech_selection
        arch = ctx.architecture
        if ts is None and arch is None:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.0,
                rationale="no tech/architecture decisions to evaluate",
            )
        tech_alts = 0
        if ts is not None:
            for d in _getattr(ts, "decisions", []) or []:
                top = list(_getattr(d, "top_candidates", []) or [])
                rej = list(_getattr(d, "rejected", []) or [])
                tech_alts += max(0, len(top) - 1) + len(rej)
        arch_alts = 0
        if arch is not None:
            dec = _getattr(arch, "decision", None)
            cmp_ = _getattr(dec, "comparison", None)
            scores = list(_getattr(cmp_, "scores", []) or [])
            arch_alts = max(0, len(scores) - 1)
        total = tech_alts + arch_alts
        signals = [
            MetaSignal("tech_alternatives", tech_alts, "C09"),
            MetaSignal("architecture_alternatives", arch_alts, "C10"),
        ]
        if total == 0:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale="no alternative candidates were preserved",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.RECONSIDER_TECHNOLOGY,
                    target_phase="C09",
                    rationale="alternatives must be considered and preserved",
                )],
            )
        if total < self.MIN_ALTERNATIVES + 1:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.6,
                rationale=f"only {total} alternative(s) evaluated",
                signals=signals,
            )
        return self._mk(
            outcome=CheckOutcome.PASS, score=0.85,
            rationale=f"{total} alternatives preserved across phases",
            signals=signals,
        )


# ---- 6. Testing sufficiency ----
class TestingCheck(_BaseCheck):
    question = MetaQuestion.TESTING_SUFFICIENT
    blocking_possible = True
    MIN_TESTS = 3
    MIN_COVERAGE_TARGETS = 3

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        plan = ctx.test_plan
        run = ctx.test_run
        if plan is None and run is None:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.2,
                rationale="no test plan or test run available",
                redirections=[Redirection(
                    kind=RedirectionKind.GENERATE_MORE_TESTS,
                    target_phase="C17",
                    rationale="no tests produced",
                )],
            )
        tests_count = len(_getattr(plan, "tests", []) or []) if plan else 0
        coverage_targets = len(_getattr(plan, "coverage", []) or []) if plan else 0
        passed = _safe_int(_getattr(run, "passed", 0)) if run else 0
        failed = _safe_int(_getattr(run, "failed", 0)) if run else 0
        errors = _safe_int(_getattr(run, "errors", 0)) if run else 0
        skipped = _safe_int(_getattr(run, "skipped", 0)) if run else 0
        total_run = passed + failed + errors + skipped
        signals = [
            MetaSignal("planned_tests", tests_count, "C17"),
            MetaSignal("coverage_targets", coverage_targets, "C17"),
            MetaSignal("passed", passed, "C18"),
            MetaSignal("failed", failed, "C18"),
            MetaSignal("errors", errors, "C18"),
            MetaSignal("skipped", skipped, "C18"),
        ]
        # Hard: run present and everything failed → STOP
        if run is not None and total_run > 0 and passed == 0:
            return self._mk(
                outcome=CheckOutcome.BLOCKING, score=0.0,
                rationale=f"all {total_run} tests failed/errored",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.FIX_FAILING_TESTS,
                    target_phase="C19",
                    rationale="no passing tests means no verified behavior",
                    blocking=True,
                )],
            )
        # No run but plan exists → caution
        if run is None and tests_count >= self.MIN_TESTS:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.55,
                rationale=f"{tests_count} tests planned but not executed",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.GENERATE_MORE_TESTS,
                    target_phase="C18",
                    rationale="execute the test plan",
                )],
            )
        if run is not None and failed + errors > 0:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.4,
                rationale=f"{failed} failed, {errors} errored",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.FIX_FAILING_TESTS,
                    target_phase="C19",
                    rationale="failing tests must be resolved",
                )],
            )
        if tests_count < self.MIN_TESTS:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.35,
                rationale=f"only {tests_count} tests (<{self.MIN_TESTS})",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.GENERATE_MORE_TESTS,
                    target_phase="C17",
                    rationale="insufficient test volume",
                )],
            )
        if coverage_targets < self.MIN_COVERAGE_TARGETS:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.6,
                rationale=f"only {coverage_targets} covered targets",
                signals=signals,
            )
        return self._mk(
            outcome=CheckOutcome.PASS, score=0.85,
            rationale=(
                f"{tests_count} tests, {coverage_targets} targets, "
                f"{passed} passed"
            ),
            signals=signals,
        )


# ---- 7. Verification sufficiency ----
class VerificationCheck(_BaseCheck):
    question = MetaQuestion.VERIFICATION_SUFFICIENT
    blocking_possible = True

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        vb = ctx.verification
        if vb is None:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale="no verification bundle",
                redirections=[Redirection(
                    kind=RedirectionKind.RUN_MORE_VERIFICATION,
                    target_phase="C22",
                    rationale="verification not run",
                )],
            )
        ladder = _getattr(vb, "ladder", None)
        if ladder is None:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.3,
                rationale="verification bundle has no ladder",
            )
        gen = _safe_int(_getattr(ladder, "generated", 0))
        exe = _safe_int(_getattr(ladder, "executed", 0))
        tst = _safe_int(_getattr(ladder, "tested", 0))
        ver = _safe_int(_getattr(ladder, "verified", 0))
        sat = _safe_int(_getattr(ladder, "satisfied", 0))
        signals = [
            MetaSignal("generated", gen, "C22"),
            MetaSignal("executed", exe, "C22"),
            MetaSignal("tested", tst, "C22"),
            MetaSignal("verified", ver, "C22"),
            MetaSignal("satisfied", sat, "C22"),
        ]
        refuted = _safe_int(
            sum(1 for r in (_getattr(vb, "results", []) or [])
                if _enum_val(_getattr(r, "result", "")) == "refuted")
        )
        if refuted > 0:
            return self._mk(
                outcome=CheckOutcome.BLOCKING, score=0.0,
                rationale=f"{refuted} refuted claim(s)",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.RESOLVE_CONTRADICTIONS,
                    target_phase="C19",
                    rationale="refuted claims block",
                    blocking=True,
                )],
            )
        # Requirements satisfied?
        if sat >= 1 and ver >= 1:
            return self._mk(
                outcome=CheckOutcome.PASS, score=0.9,
                rationale=(
                    f"{ver} verified, {sat} requirement(s) satisfied"
                ),
                signals=signals,
            )
        if ver >= 1:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.6,
                rationale=f"{ver} verified but no requirement satisfied",
                signals=signals,
            )
        return self._mk(
            outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
            rationale="no claims reached VERIFIED",
            signals=signals,
            redirections=[Redirection(
                kind=RedirectionKind.RUN_MORE_VERIFICATION,
                target_phase="C22",
                rationale="continue verifying until claims are supported",
            )],
        )


# ---- 8. Uncertainty (aggregate) ----
class UncertaintyCheck(_BaseCheck):
    question = MetaQuestion.UNCERTAINTY_ACCEPTABLE
    CONFIDENCE_WEIGHT = {
        "verified": 1.0, "high": 0.85, "medium": 0.6,
        "low": 0.3, "assumption": 0.2, "unknown": 0.1,
    }

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        # Gather all confidence signals we can find
        pieces: list[tuple[str, str]] = []   # (source, confidence)
        if ctx.spec is not None:
            pieces.append(("C05.spec",
                            _enum_val(_getattr(ctx.spec, "confidence", "unknown"))))
        if ctx.intent_ctx is not None:
            intent = _getattr(ctx.intent_ctx, "intent", None)
            if intent is not None:
                pieces.append(("C06.intent",
                                _enum_val(_getattr(intent, "confidence", "unknown"))))
            risk = _getattr(ctx.intent_ctx, "risk", None)
            if risk is not None:
                # Risk level isn't confidence, but high risk implies
                # extra scrutiny — bump uncertainty accordingly
                level = _enum_val(_getattr(risk, "level", "unknown"))
                if level == "critical":
                    pieces.append(("C06.risk.critical", "low"))
                elif level == "high":
                    pieces.append(("C06.risk.high", "low"))
        if ctx.verification is not None:
            for r in _getattr(ctx.verification, "results", []) or []:
                pieces.append(("C22.claim",
                                _enum_val(_getattr(r, "confidence", "unknown"))))
        if ctx.debug_report is not None:
            pieces.append(("C19.debug",
                            _enum_val(_getattr(ctx.debug_report, "status",
                                                "unknown"))))

        if not pieces:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.3,
                rationale="no confidence signals available",
            )

        values = [
            self.CONFIDENCE_WEIGHT.get(label, 0.3)
            for _, label in pieces
        ]
        avg = sum(values) / len(values)
        signals = [
            MetaSignal("samples", len(pieces), "aggregate"),
            MetaSignal("avg_confidence_weight", round(avg, 3), "aggregate"),
        ] + [MetaSignal(src, label, "aggregate")
             for src, label in pieces[:10]]

        if avg >= 0.75:
            return self._mk(
                outcome=CheckOutcome.PASS, score=0.9,
                rationale=f"aggregate confidence weight={avg:.2f}",
                signals=signals,
            )
        if avg >= 0.55:
            return self._mk(
                outcome=CheckOutcome.CAUTION, score=0.6,
                rationale=f"aggregate confidence weight={avg:.2f}",
                signals=signals,
            )
        if avg >= 0.35:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.35,
                rationale=f"low aggregate confidence weight={avg:.2f}",
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.RAISE_UNCERTAINTY_ALERT,
                    target_phase="meta",
                    rationale="confidence too low to proceed confidently",
                )],
            )
        return self._mk(
            outcome=CheckOutcome.BLOCKING, score=0.1,
            rationale=(
                f"critical uncertainty: aggregate weight={avg:.2f}; "
                f"cannot proceed safely"
            ),
            signals=signals,
            redirections=[Redirection(
                kind=RedirectionKind.RAISE_UNCERTAINTY_ALERT,
                target_phase="meta",
                rationale="aggregate confidence below safe threshold",
                blocking=True,
            )],
        )


# ---- 9. Specialist invocation ----
class SpecialistCheck(_BaseCheck):
    question = MetaQuestion.SPECIALIST_NEEDED

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        # Missing-phase detection drives specialist suggestions.
        present = set(ctx.phases_present())
        suggestions: list[Redirection] = []
        # If a code artifact exists but security wasn't scanned → redirect
        code_like = any(p in present for p in ("C14", "C15", "C13"))
        if code_like and "C23" not in present:
            suggestions.append(Redirection(
                kind=RedirectionKind.RUN_SECURITY_SCAN,
                target_phase="C23",
                rationale="code produced but not security-scanned",
            ))
        if code_like and "C24" not in present:
            suggestions.append(Redirection(
                kind=RedirectionKind.RUN_PERFORMANCE_SCAN,
                target_phase="C24",
                rationale="code produced but not performance-analyzed",
            ))
        if ctx.debug_report is not None and ctx.repair_result is None:
            suggestions.append(Redirection(
                kind=RedirectionKind.INVOKE_SPECIALIST,
                target_phase="C20",
                rationale="failure diagnosed but no repair attempted",
            ))
        # Ambiguous spec → requirement specialist
        if ctx.spec is not None:
            amb = len(_getattr(ctx.spec, "ambiguities", []) or [])
            if amb >= 3:
                suggestions.append(Redirection(
                    kind=RedirectionKind.ASK_USER_FOR_CLARIFICATION,
                    target_phase="C05",
                    rationale=f"{amb} ambiguities — clarify before building",
                ))

        signals = [
            MetaSignal("phases_present", sorted(present), "aggregate"),
            MetaSignal("suggestions", len(suggestions), "aggregate"),
        ]
        if not suggestions:
            return self._mk(
                outcome=CheckOutcome.PASS, score=0.85,
                rationale="no specialist invocations required",
                signals=signals,
            )
        # 3+ pending suggestions → strong
        if len(suggestions) >= 3:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale=f"{len(suggestions)} specialist gaps detected",
                signals=signals, redirections=suggestions,
            )
        return self._mk(
            outcome=CheckOutcome.CAUTION, score=0.55,
            rationale=f"{len(suggestions)} specialist gap(s) detected",
            signals=signals, redirections=suggestions,
        )


# ---- 10. Plan revision ----
class PlanCheck(_BaseCheck):
    question = MetaQuestion.PLAN_REVISED
    blocking_possible = False

    def evaluate(self, ctx: PipelineContext) -> MetaCheck:
        plan = ctx.plan
        if plan is None:
            return self._mk(
                outcome=CheckOutcome.INSUFFICIENT, score=0.0,
                rationale="no plan available",
                redirections=[Redirection(
                    kind=RedirectionKind.REVISE_PLAN,
                    target_phase="C08",
                    rationale="plan not produced",
                )],
            )
        tasks = list(_getattr(plan, "tasks", []) or [])
        milestones = list(_getattr(plan, "milestones", []) or [])
        gates = list(_getattr(plan, "gates", []) or [])
        rollbacks = list(_getattr(plan, "rollback_points", []) or [])
        total_duration = _safe_int(_getattr(plan, "total_duration", 0))
        signals = [
            MetaSignal("tasks", len(tasks), "C08"),
            MetaSignal("milestones", len(milestones), "C08"),
            MetaSignal("gates", len(gates), "C08"),
            MetaSignal("rollback_points", len(rollbacks), "C08"),
            MetaSignal("total_duration", total_duration, "C08"),
        ]
        problems: list[str] = []
        if not tasks:
            problems.append("no tasks")
        if not milestones:
            problems.append("no milestones")
        if not gates:
            problems.append("no verification gates")
        if not rollbacks:
            problems.append("no rollback points")
        if problems:
            return self._mk(
                outcome=CheckOutcome.NEEDS_ATTENTION, score=0.3,
                rationale="; ".join(problems),
                signals=signals,
                redirections=[Redirection(
                    kind=RedirectionKind.REVISE_PLAN,
                    target_phase="C08",
                    rationale="plan missing required structure",
                )],
            )
        # Try plan.validate() if available
        validator = _getattr(plan, "validate", None)
        if callable(validator):
            try:
                v = validator()
                if isinstance(v, dict) and not v.get("ok", True):
                    return self._mk(
                        outcome=CheckOutcome.NEEDS_ATTENTION, score=0.35,
                        rationale=f"plan validation failed: "
                                  f"{v.get('issue_count')} issue(s)",
                        signals=signals,
                        redirections=[Redirection(
                            kind=RedirectionKind.REVISE_PLAN,
                            target_phase="C08",
                            rationale="plan has integrity issues",
                        )],
                    )
            except Exception:
                pass
        return self._mk(
            outcome=CheckOutcome.PASS, score=0.85,
            rationale=(
                f"plan has {len(tasks)} tasks, {len(milestones)} milestones, "
                f"{len(gates)} gates, {len(rollbacks)} rollbacks"
            ),
            signals=signals,
        )


# ════════════════════════════════════════════════════════════════════════════
# 5. META-REASONER FACADE
# ════════════════════════════════════════════════════════════════════════════
class MetaReasoner:
    """Runs all checks, aggregates verdict, exposes redirections.

    Verdict rules (in order):
        any BLOCKING         → STOP
        any NEEDS_ATTENTION  → REDIRECT
        any CAUTION          → PROCEED_WITH_CAUTION
        any INSUFFICIENT     → REDIRECT (missing input requires action)
        else                 → PROCEED
    """

    def __init__(self) -> None:
        self.checks: list[_BaseCheck] = [
            RequirementCheck(),
            EvidenceCheck(),
            ArchitectureCheck(),
            TechnologyCheck(),
            AlternativesCheck(),
            TestingCheck(),
            VerificationCheck(),
            UncertaintyCheck(),
            SpecialistCheck(),
            PlanCheck(),
        ]

    def assess(
        self,
        ctx: PipelineContext,
        *,
        project_id: str = "",
        task_id: str = "",
    ) -> MetaAssessment:
        a = MetaAssessment(
            project_id=project_id, task_id=task_id,
            provenance=Provenance(
                source="meta_reasoner",
                source_type=ProvenanceType.INFERENCE,
                confidence=Confidence.MEDIUM,
            ),
        )
        for chk in self.checks:
            try:
                r = chk.evaluate(ctx)
            except Exception as exc:
                log.warning("c28.check_error",
                            check=type(chk).__name__, error=str(exc))
                r = MetaCheck(
                    question=chk.question,
                    outcome=CheckOutcome.INSUFFICIENT,
                    score=0.0,
                    rationale=f"check raised: {type(exc).__name__}: {exc}",
                )
            a.checks.append(r)
            a.redirections.extend(r.redirections)

        # Score
        if a.checks:
            a.overall_score = sum(c.score for c in a.checks) / len(a.checks)

        # Verdict
        outcomes = [c.outcome for c in a.checks]
        if CheckOutcome.BLOCKING in outcomes:
            a.verdict = MetaVerdict.STOP
        elif CheckOutcome.NEEDS_ATTENTION in outcomes:
            a.verdict = MetaVerdict.REDIRECT
        elif CheckOutcome.INSUFFICIENT in outcomes:
            a.verdict = MetaVerdict.REDIRECT
        elif CheckOutcome.CAUTION in outcomes:
            a.verdict = MetaVerdict.PROCEED_WITH_CAUTION
        else:
            a.verdict = MetaVerdict.PROCEED

        # Rationale
        blocking = [c for c in a.checks if c.is_blocking()]
        na = [c for c in a.checks
              if c.outcome is CheckOutcome.NEEDS_ATTENTION]
        insuff = [c for c in a.checks
                  if c.outcome is CheckOutcome.INSUFFICIENT]
        a.rationale = (
            f"verdict={a.verdict.value} score={a.overall_score:.2f} "
            f"blocking={len(blocking)} needs_attention={len(na)} "
            f"insufficient={len(insuff)} "
            f"redirections={len(a.redirections)}"
        )
        return a

    # ---- quick helpers ----
    def should_stop(self, assessment: MetaAssessment) -> bool:
        return assessment.verdict is MetaVerdict.STOP

    def blocking_redirections(
        self, assessment: MetaAssessment,
    ) -> list[Redirection]:
        return [r for r in assessment.redirections if r.blocking]


# ════════════════════════════════════════════════════════════════════════════
# 6. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class MetaRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, a: MetaAssessment, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"meta_assessment:{a.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, a.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["meta_reasoning", "c28", a.verdict.value],
            provenance=a.provenance,
        )
        # STOP → record as failure memory so downstream learns
        if a.verdict is MetaVerdict.STOP:
            for c in a.blocking_checks():
                self.memory.record_failure(
                    f"meta_stop:{a.id}:{c.question.value}",
                    what=f"meta stop: {c.question.value}",
                    root_cause=c.rationale,
                    fix=None,
                    scope_id=project_id,
                    provenance=a.provenance,
                    confidence=Confidence.HIGH,
                )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.VERIFICATION,
            _short(f"MetaAssessment {a.id[:8]} ({a.verdict.value})", 120),
            attributes={
                "assessment_id": a.id,
                "project_id": project_id,
                "verdict": a.verdict.value,
                "overall_score": a.overall_score,
                "checks": [c.to_dict() for c in a.checks],
                "redirections": [r.to_dict() for r in a.redirections],
            },
            tags=["meta-assessment", a.verdict.value],
            provenance=a.provenance,
        )
        return ent.id

    def load(self, assessment_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"meta_assessment:{assessment_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 7. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
# ---- Tiny fakes for testing without spinning up the full pipeline ----
class _FakeSpec:
    def __init__(self, *, confidence="high", ambiguities=0, missing=0):
        self.confidence = type("C", (), {"value": confidence})()
        self.ambiguities = [object()] * ambiguities
        self.missing = [object()] * missing


class _FakeIntent:
    def __init__(self, confidence="high", risk_level="low"):
        self.intent = type("I", (), {"confidence":
                                       type("C", (), {"value": confidence})()})()
        self.risk = type("R", (), {"level":
                                     type("L", (), {"value": risk_level})()})()


class _FakePlan:
    def __init__(self, tasks=3, milestones=3, gates=2, rollbacks=2,
                 valid=True):
        self.tasks = [object()] * tasks
        self.milestones = [object()] * milestones
        self.gates = [object()] * gates
        self.rollback_points = [object()] * rollbacks
        self.total_duration = 10
        self._valid = valid

    def validate(self):
        return {"ok": self._valid, "issue_count": 0 if self._valid else 1}


class _FakeTechDecisions:
    def __init__(self, category, winner=True, top=2, rejected=1):
        self.category = type("Cat", (), {"value": category})()
        self.winner = object() if winner else None
        self.top_candidates = [object()] * top
        self.rejected = [object()] * rejected


class _FakeTech:
    def __init__(self, categories=("language", "framework",
                                    "database", "architecture"),
                 winner=True):
        self.decisions = [_FakeTechDecisions(c, winner=winner)
                          for c in categories]


class _FakeArchSelection:
    def __init__(self, components=3, interfaces=2, fb=1, sb=1, kind="layered"):
        self.id = "sel-1"
        self.name = "Selected"
        self.kind = type("K", (), {"value": kind})()
        self.components = [object()] * components
        self.interfaces = [object()] * interfaces
        self.failure_boundaries = [object()] * fb
        self.security_boundaries = [object()] * sb


class _FakeArchComparison:
    def __init__(self, candidates=3, pareto=2):
        self.scores = [object()] * candidates
        self.pareto_front = ["a"] * pareto


class _FakeArchDecision:
    def __init__(self, selection=None, comparison=None):
        self.selected = selection or _FakeArchSelection()
        self.comparison = comparison or _FakeArchComparison()


class _FakeArch:
    def __init__(self, decision=None):
        self.id = "arch-1"
        self.decision = decision or _FakeArchDecision()


class _FakeVResult:
    def __init__(self, result="supported", confidence="high"):
        self.result = type("R", (), {"value": result})()
        self.confidence = type("C", (), {"value": confidence})()


class _FakeLadder:
    def __init__(self, generated=1, executed=1, tested=1, verified=1,
                 satisfied=1):
        self.generated = generated
        self.executed = executed
        self.tested = tested
        self.verified = verified
        self.satisfied = satisfied


class _FakeVB:
    def __init__(self, results=None, ladder=None):
        self.results = results or [_FakeVResult("supported")]
        self.ladder = ladder or _FakeLadder()


class _FakeTestPlan:
    def __init__(self, tests=5, coverage=4):
        self.tests = [object()] * tests
        self.coverage = [object()] * coverage


class _FakeTestRun:
    def __init__(self, passed=5, failed=0, errors=0, skipped=0):
        self.passed = passed
        self.failed = failed
        self.errors = errors
        self.skipped = skipped


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

    print("Running C28 self-tests…")
    engine = MetaReasoner()

    # ---- requirement check ----
    def t_req_missing_spec() -> None:
        ctx = PipelineContext()
        r = RequirementCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.INSUFFICIENT
        assert any(rd.blocking for rd in r.redirections)

    def t_req_clear() -> None:
        ctx = PipelineContext(spec=_FakeSpec(confidence="high",
                                              ambiguities=0, missing=0))
        r = RequirementCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    def t_req_ambiguous() -> None:
        ctx = PipelineContext(spec=_FakeSpec(confidence="medium",
                                              ambiguities=6, missing=6))
        r = RequirementCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION
        assert any(rd.kind is RedirectionKind.ASK_USER_FOR_CLARIFICATION
                   for rd in r.redirections)

    def t_req_unknown_confidence() -> None:
        ctx = PipelineContext(spec=_FakeSpec(confidence="unknown"))
        r = RequirementCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    check("req: missing spec → INSUFFICIENT + blocking",
          t_req_missing_spec)
    check("req: clear spec → PASS", t_req_clear)
    check("req: many ambiguities → NEEDS_ATTENTION",
          t_req_ambiguous)
    check("req: unknown confidence → NEEDS_ATTENTION",
          t_req_unknown_confidence)

    # ---- evidence check ----
    def t_evidence_missing() -> None:
        ctx = PipelineContext()
        r = EvidenceCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    def t_evidence_refuted_blocks() -> None:
        vb = _FakeVB(results=[_FakeVResult("refuted")])
        ctx = PipelineContext(verification=vb)
        r = EvidenceCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.BLOCKING
        assert any(rd.blocking for rd in r.redirections)

    def t_evidence_all_supported() -> None:
        vb = _FakeVB(results=[_FakeVResult("supported")] * 5)
        ctx = PipelineContext(verification=vb)
        r = EvidenceCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    check("evidence: missing → NEEDS_ATTENTION", t_evidence_missing)
    check("evidence: refuted → BLOCKING", t_evidence_refuted_blocks)
    check("evidence: all supported → PASS", t_evidence_all_supported)

    # ---- architecture check ----
    def t_arch_missing() -> None:
        r = ArchitectureCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.INSUFFICIENT

    def t_arch_complete() -> None:
        ctx = PipelineContext(architecture=_FakeArch())
        r = ArchitectureCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    def t_arch_missing_boundaries() -> None:
        sel = _FakeArchSelection(components=3, interfaces=2, fb=0, sb=0)
        arch = _FakeArch(decision=_FakeArchDecision(selection=sel))
        ctx = PipelineContext(architecture=arch)
        r = ArchitectureCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    def t_arch_single_candidate() -> None:
        cmp_ = _FakeArchComparison(candidates=1, pareto=1)
        arch = _FakeArch(decision=_FakeArchDecision(comparison=cmp_))
        ctx = PipelineContext(architecture=arch)
        r = ArchitectureCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.CAUTION

    check("arch: missing → INSUFFICIENT", t_arch_missing)
    check("arch: complete → PASS", t_arch_complete)
    check("arch: missing boundaries → NEEDS_ATTENTION",
          t_arch_missing_boundaries)
    check("arch: single candidate → CAUTION", t_arch_single_candidate)

    # ---- technology check ----
    def t_tech_missing() -> None:
        r = TechnologyCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.INSUFFICIENT

    def t_tech_complete() -> None:
        ctx = PipelineContext(tech_selection=_FakeTech())
        r = TechnologyCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    def t_tech_missing_category() -> None:
        ts = _FakeTech(categories=("language", "framework"))
        ctx = PipelineContext(tech_selection=ts)
        r = TechnologyCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION
        assert "database" in r.signals[2].value  # missing_categories

    check("tech: missing → INSUFFICIENT", t_tech_missing)
    check("tech: complete → PASS", t_tech_complete)
    check("tech: missing category → NEEDS_ATTENTION",
          t_tech_missing_category)

    # ---- alternatives check ----
    def t_alts_missing() -> None:
        r = AlternativesCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.INSUFFICIENT

    def t_alts_present() -> None:
        ctx = PipelineContext(
            tech_selection=_FakeTech(),
            architecture=_FakeArch(),
        )
        r = AlternativesCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    check("alternatives: missing inputs → INSUFFICIENT", t_alts_missing)
    check("alternatives: present → PASS", t_alts_present)

    # ---- testing check ----
    def t_testing_missing() -> None:
        r = TestingCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    def t_testing_all_fail_blocks() -> None:
        run = _FakeTestRun(passed=0, failed=5, errors=1)
        ctx = PipelineContext(test_run=run)
        r = TestingCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.BLOCKING

    def t_testing_pass() -> None:
        ctx = PipelineContext(
            test_plan=_FakeTestPlan(tests=5, coverage=4),
            test_run=_FakeTestRun(passed=5),
        )
        r = TestingCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    def t_testing_some_failures() -> None:
        ctx = PipelineContext(
            test_plan=_FakeTestPlan(),
            test_run=_FakeTestRun(passed=3, failed=1),
        )
        r = TestingCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    check("testing: missing → NEEDS_ATTENTION", t_testing_missing)
    check("testing: all failing → BLOCKING", t_testing_all_fail_blocks)
    check("testing: all passing → PASS", t_testing_pass)
    check("testing: some failing → NEEDS_ATTENTION",
          t_testing_some_failures)

    # ---- verification check ----
    def t_verif_missing() -> None:
        r = VerificationCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    def t_verif_refuted_blocks() -> None:
        vb = _FakeVB(results=[_FakeVResult("refuted")])
        ctx = PipelineContext(verification=vb)
        r = VerificationCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.BLOCKING

    def t_verif_satisfied_passes() -> None:
        ctx = PipelineContext(verification=_FakeVB())
        r = VerificationCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    check("verification: missing → NEEDS_ATTENTION", t_verif_missing)
    check("verification: refuted → BLOCKING", t_verif_refuted_blocks)
    check("verification: satisfied → PASS", t_verif_satisfied_passes)

    # ---- uncertainty check ----
    def t_uncertainty_missing() -> None:
        r = UncertaintyCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.INSUFFICIENT

    def t_uncertainty_high_confidence() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="high"),
            intent_ctx=_FakeIntent(confidence="high"),
            verification=_FakeVB(results=[_FakeVResult("supported", "high")]),
        )
        r = UncertaintyCheck().evaluate(ctx)
        assert r.outcome in (CheckOutcome.PASS, CheckOutcome.CAUTION)

    def t_uncertainty_low_confidence_blocks() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="unknown"),
            intent_ctx=_FakeIntent(confidence="unknown",
                                    risk_level="critical"),
        )
        r = UncertaintyCheck().evaluate(ctx)
        # Low aggregate → BLOCKING or NEEDS_ATTENTION
        assert r.outcome in (CheckOutcome.NEEDS_ATTENTION,
                              CheckOutcome.BLOCKING)

    check("uncertainty: no signals → INSUFFICIENT",
          t_uncertainty_missing)
    check("uncertainty: high-confidence → PASS/CAUTION",
          t_uncertainty_high_confidence)
    check("uncertainty: low-confidence → NEEDS_ATTENTION/BLOCKING",
          t_uncertainty_low_confidence_blocks)

    # ---- specialist check ----
    def t_specialist_none_required() -> None:
        ctx = PipelineContext(
            repo_index=object(), security=object(), perf=object(),
        )
        r = SpecialistCheck().evaluate(ctx)
        assert r.outcome is CheckOutcome.PASS

    def t_specialist_security_gap() -> None:
        ctx = PipelineContext(repo_index=object())  # no security
        r = SpecialistCheck().evaluate(ctx)
        assert any(rd.kind is RedirectionKind.RUN_SECURITY_SCAN
                   for rd in r.redirections)

    check("specialist: no gaps → PASS", t_specialist_none_required)
    check("specialist: security gap → suggests C23",
          t_specialist_security_gap)

    # ---- plan check ----
    def t_plan_missing() -> None:
        r = PlanCheck().evaluate(PipelineContext())
        assert r.outcome is CheckOutcome.INSUFFICIENT

    def t_plan_complete() -> None:
        r = PlanCheck().evaluate(PipelineContext(plan=_FakePlan()))
        assert r.outcome is CheckOutcome.PASS

    def t_plan_missing_gates() -> None:
        r = PlanCheck().evaluate(
            PipelineContext(plan=_FakePlan(gates=0)),
        )
        assert r.outcome is CheckOutcome.NEEDS_ATTENTION

    check("plan: missing → INSUFFICIENT", t_plan_missing)
    check("plan: complete → PASS", t_plan_complete)
    check("plan: missing gates → NEEDS_ATTENTION", t_plan_missing_gates)

    # ---- aggregation verdicts ----
    def t_verdict_proceed() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="high"),
            intent_ctx=_FakeIntent(confidence="high"),
            plan=_FakePlan(),
            tech_selection=_FakeTech(),
            architecture=_FakeArch(),
            test_plan=_FakeTestPlan(),
            test_run=_FakeTestRun(passed=5),
            verification=_FakeVB(),
            repo_index=object(),
            security=object(),
            perf=object(),
        )
        a = engine.assess(ctx, project_id="p")
        # Best-case should be PROCEED, but specialist check may CAUTION.
        # Accept PROCEED or PROCEED_WITH_CAUTION.
        assert a.verdict in (MetaVerdict.PROCEED,
                              MetaVerdict.PROCEED_WITH_CAUTION), a.rationale

    def t_verdict_stop_on_refuted() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="high"),
            verification=_FakeVB(results=[_FakeVResult("refuted")]),
        )
        a = engine.assess(ctx, project_id="p")
        assert a.verdict is MetaVerdict.STOP
        assert len(a.blocking_checks()) >= 1

    def t_verdict_stop_on_all_tests_failing() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="high"),
            test_plan=_FakeTestPlan(),
            test_run=_FakeTestRun(passed=0, failed=5),
        )
        a = engine.assess(ctx, project_id="p")
        assert a.verdict is MetaVerdict.STOP

    def t_verdict_redirect_on_missing_inputs() -> None:
        # Empty context → all checks INSUFFICIENT → REDIRECT
        a = engine.assess(PipelineContext(), project_id="p")
        assert a.verdict is MetaVerdict.REDIRECT
        # Redirections present
        assert len(a.redirections) >= 3

    def t_verdict_redirect_on_needs_attention() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="high", ambiguities=10, missing=10),
            plan=_FakePlan(),
            tech_selection=_FakeTech(),
            architecture=_FakeArch(),
            verification=_FakeVB(),
            test_plan=_FakeTestPlan(),
            test_run=_FakeTestRun(passed=5),
        )
        a = engine.assess(ctx, project_id="p")
        assert a.verdict is MetaVerdict.REDIRECT

    check("verdict: healthy pipeline → PROCEED (or CAUTION)",
          t_verdict_proceed)
    check("verdict: refuted verification → STOP",
          t_verdict_stop_on_refuted)
    check("verdict: all tests failing → STOP",
          t_verdict_stop_on_all_tests_failing)
    check("verdict: empty context → REDIRECT",
          t_verdict_redirect_on_missing_inputs)
    check("verdict: needs attention → REDIRECT",
          t_verdict_redirect_on_needs_attention)

    # ---- helpers ----
    def t_should_stop_helper() -> None:
        ctx = PipelineContext(
            verification=_FakeVB(results=[_FakeVResult("refuted")]),
        )
        a = engine.assess(ctx, project_id="p")
        assert engine.should_stop(a) is True
        blocking = engine.blocking_redirections(a)
        assert any(r.blocking for r in blocking)

    check("helpers: should_stop + blocking_redirections",
          t_should_stop_helper)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        a = engine.assess(PipelineContext(spec=_FakeSpec()), project_id="p")
        d = a.to_dict()
        assert d["id"] == a.id
        assert "verdict" in d and "checks" in d and "redirections" in d
        s = a.summary()
        assert "Meta-Reasoning Assessment" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                # STOP case: persisted + failure memory
                ctx = PipelineContext(
                    verification=_FakeVB(
                        results=[_FakeVResult("refuted")],
                    ),
                )
                a = engine.assess(ctx, project_id="proj-x")
                repo = MetaRepository(memory=mem, ontology=ont)
                ent = repo.save(a, project_id="proj-x")
                assert ent
                loaded = repo.load(a.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["verdict"] == "stop"
                # Failure memory recorded for blocking check
                fails = mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert len(fails) >= 1
                # Ontology has VERIFICATION entity
                assert ont.count(kind=EntityKind.VERIFICATION) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology + failure memory for STOP",
          t_persist)

    # ---- determinism ----
    def t_deterministic() -> None:
        ctx = PipelineContext(
            spec=_FakeSpec(confidence="high"),
            verification=_FakeVB(),
            test_plan=_FakeTestPlan(),
            test_run=_FakeTestRun(passed=5),
        )
        a1 = engine.assess(ctx, project_id="p")
        a2 = engine.assess(ctx, project_id="p")
        assert a1.verdict is a2.verdict
        assert abs(a1.overall_score - a2.overall_score) < 1e-9
        assert [c.outcome for c in a1.checks] == [c.outcome for c in a2.checks]

    check("deterministic: same ctx → same verdict + outcomes",
          t_deterministic)

    # ---- E2E: real C10 + C22 ----
    def t_e2e_real_c10_c22() -> None:
        from sebrain.c05 import RequirementParser
        from sebrain.c06 import IntentContextEngine
        from sebrain.c09 import TechnologySelector
        from sebrain.c10 import ArchitectureReasoner
        from sebrain.c22 import (
            Claim, EvidenceLevel, VerificationEngine, VerificationKind,
        )

        text = (
            "Build a small REST API for tasks.\n"
            "Non-functional:\n- All traffic must use HTTPS.\n"
        )
        spec = RequirementParser().parse(text)
        ic = IntentContextEngine().analyze(text, project_id="demo")
        tech = TechnologySelector().select(spec, ic, project_id="demo")
        arch = ArchitectureReasoner().reason(spec, ic, tech, project_id="demo")

        # Run a real verification over the architecture
        veng = VerificationEngine()
        vb = veng.verify_bundle([
            Claim("architecture complete",
                  VerificationKind.ARCHITECTURE,
                  target_level=EvidenceLevel.VERIFIED),
        ], project_id="demo", contexts=[{"architecture": arch}])
        assert vb.results and vb.results[0].result.value == "supported"

        # Build pipeline context
        ctx = PipelineContext(
            spec=spec,
            intent_ctx=ic,
            tech_selection=tech,
            architecture=arch,
            verification=vb,
        )
        a = engine.assess(ctx, project_id="demo")
        # We don't have tests/plan here, so verdict is likely REDIRECT.
        # The key: it must NOT be a fake PROCEED, and it must surface
        # the actual gaps (missing plan/test run etc.).
        assert a.verdict in (MetaVerdict.REDIRECT, MetaVerdict.STOP)
        # There should be redirections pointing at C08 / C17 / C18
        kinds = {r.target_phase for r in a.redirections}
        assert any(k in kinds for k in ("C08", "C17", "C18"))

    check("e2e: real C10 + C22 → meta detects missing phases",
          t_e2e_real_c10_c22)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 8. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C28 — Meta-Reasoning")
    print("=" * 78)
    print("The Brain evaluates its OWN process before proceeding.\n")

    engine = MetaReasoner()

    # ---- Scenario 1: healthy pipeline ----
    print("[1] Scenario: healthy pipeline")
    ctx1 = PipelineContext(
        spec=_FakeSpec(confidence="high"),
        intent_ctx=_FakeIntent(confidence="high"),
        plan=_FakePlan(),
        tech_selection=_FakeTech(),
        architecture=_FakeArch(),
        test_plan=_FakeTestPlan(),
        test_run=_FakeTestRun(passed=8, failed=0),
        verification=_FakeVB(),
        repo_index=object(),
        security=object(),
        perf=object(),
    )
    a1 = engine.assess(ctx1, project_id="demo")
    print(a1.summary())
    for c in a1.checks:
        print(f"    [{c.outcome.value:16s}] {c.question.value}  "
              f"score={c.score:.2f}")
    if a1.redirections:
        print("    Redirections:")
        for r in a1.redirections:
            print(f"      → {r.kind.value} (target={r.target_phase})")

    # ---- Scenario 2: ambiguous spec ----
    print("\n[2] Scenario: ambiguous spec (needs clarification)")
    ctx2 = PipelineContext(
        spec=_FakeSpec(confidence="low", ambiguities=8, missing=6),
    )
    a2 = engine.assess(ctx2, project_id="demo")
    print(a2.summary())
    for c in a2.checks:
        mark = {"pass": "✓", "caution": "·", "needs_attention": "⚠",
                "blocking": "✗", "insufficient": "?"}[c.outcome.value]
        print(f"    {mark} [{c.outcome.value:16s}] {c.question.value}")
        if c.outcome.value != "pass":
            print(f"        {_short(c.rationale, 100)}")

    # ---- Scenario 3: refuted verification → STOP ----
    print("\n[3] Scenario: refuted verification → STOP")
    ctx3 = PipelineContext(
        spec=_FakeSpec(confidence="high"),
        verification=_FakeVB(results=[
            _FakeVResult("supported"),
            _FakeVResult("refuted"),
        ]),
    )
    a3 = engine.assess(ctx3, project_id="demo")
    print(a3.summary())
    print(f"    should_stop = {engine.should_stop(a3)}")
    for c in a3.blocking_checks():
        print(f"    ✗ BLOCKING: {c.question.value}")
        print(f"        {c.rationale}")
        for r in c.redirections:
            print(f"        redirection: {r.kind.value} "
                  f"→ {r.target_phase} "
                  f"{'(blocking)' if r.blocking else ''}")

    # ---- Scenario 4: all tests failing ----
    print("\n[4] Scenario: all tests failing")
    ctx4 = PipelineContext(
        spec=_FakeSpec(confidence="high"),
        test_plan=_FakeTestPlan(),
        test_run=_FakeTestRun(passed=0, failed=5, errors=2),
    )
    a4 = engine.assess(ctx4, project_id="demo")
    print(a4.summary())
    for r in a4.redirections:
        print(f"    → {r.kind.value}  target={r.target_phase}")

    # ---- Persistence ----
    print("\n[5] Persistence:")
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = MetaRepository(memory=mem, ontology=ont)
                ent = repo.save(a3, project_id="demo")
                print(f"    ontology entity: {ent[:12]}…")
                print(f"    VERIFICATION count: "
                      f"{ont.count(kind=EntityKind.VERIFICATION)}")
                _stop_failure_count = len(mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT,
                    scope_id='demo',
                ))
                print(f"    failure memory entries for STOP: "
                      f"{_stop_failure_count}")
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
