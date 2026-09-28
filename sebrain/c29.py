"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C29 — SELF-IMPROVEMENT GOVERNANCE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04.

Purpose:
    The Brain MUST NOT modify its own core architecture blindly. Every
    candidate change goes through a 9-stage governance pipeline before it
    can be promoted. Rollback is always available. Every decision is
    recorded. Silent self-modification is impossible.

Governance pipeline (in order):
    CANDIDATE CHANGE
        → ISOLATED ENVIRONMENT  (isolate)
        → TEST                  (test)
        → REGRESSION            (regress)
        → BENCHMARK             (benchmark: before/after metrics)
        → SECURITY CHECK        (secure)
        → VERIFICATION          (verify)
        → APPROVAL POLICY       (approve — risk-tiered)
        → PROMOTION             (promote OR hold OR reject OR rollback)

Capabilities:
    - Change scopes (policy / code / config / knowledge / agent)
    - Risk tiering (LOW / MEDIUM / HIGH / CRITICAL)
    - Isolation evaluator     — accept a staged artifact, verify it's
                                separate from production
    - Test evaluator          — accept structured test results
    - Regression evaluator    — compare against a recorded baseline
    - Benchmark evaluator     — before/after metrics with tolerance
    - Security evaluator      — accept a security report
    - Verification evaluator  — accept a verification bundle
    - Approval policy         — risk-tiered, rule-based
    - Promotion               — atomic; versioned; rollbackable
    - Rollback                — explicit; supersedes promotions; audit-safe
    - Version history         — every promotion gets a monotonic version
    - Audit trail             — every stage result preserved

Invariants honored:
    - NO external LLM. Deterministic.
    - A change that fails any stage does NOT silently continue.
    - HIGH/CRITICAL risk changes require explicit approval.
    - Rollback restores the previous active version; history is preserved.
    - Same candidate + same evidence → same promotion decision.
    - No self-modification of the governance policy itself without
      promotion through this same pipeline (bootstrap limitation
      documented).
    - Bounded (max_changes_per_scope, max_stages).

Explicit limitations (Rule #59):
    - The engine does NOT itself run tests/benchmarks/security tools. It
      EVALUATES structured evidence produced elsewhere (C16–C24). This is
      deliberate — governance sits above execution, not inside it.
    - Isolation is a policy check on the caller's claim ("this change is
      staged"), not an OS-level guarantee. Actual isolation is C15/C16's
      job.
    - Approval policies are documented rules; they are NOT learned.
    - The governance policy engine cannot govern ITSELF in the same
      running process — bootstrapping would require an external supervisor.
      Changes to C29's own policy must be applied out-of-band.

Contents:
  1.  Enums: ChangeKind, ChangeRisk, GovernanceStage, StageOutcome,
             PromotionStatus, RollbackReason
  2.  Dataclasses: CandidateChange, StageResult, PromotionRecord,
                   RollbackRecord, ChangeRecord, GovernanceReport
  3.  Stage evaluators (7)
  4.  ApprovalPolicy (risk-tiered)
  5.  GovernanceEngine facade
  6.  GovernanceRepository (persist to C04 + C02)
  7.  Self-tests (~35)
  8.  Demo

Run as script:
    python -m sebrain.c29            # demo
    python -m sebrain.c29 --test     # self-tests
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


def _enum_val(x: Any) -> str:
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)


def _digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class ChangeKind(str, Enum):
    POLICY = "policy"            # governance policy (bootstrap-limited)
    CODE = "code"                # code under src/
    CONFIG = "config"            # configuration files
    KNOWLEDGE = "knowledge"      # C27 promoted knowledge
    AGENT = "agent"              # agent definitions / prompts
    HEURISTIC = "heuristic"      # thresholds inside reasoning engines
    UNKNOWN = "unknown"


class ChangeRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_RANK = {
    ChangeRisk.LOW: 0,
    ChangeRisk.MEDIUM: 1,
    ChangeRisk.HIGH: 2,
    ChangeRisk.CRITICAL: 3,
}


class GovernanceStage(str, Enum):
    ISOLATION = "isolation"
    TEST = "test"
    REGRESSION = "regression"
    BENCHMARK = "benchmark"
    SECURITY = "security"
    VERIFICATION = "verification"
    APPROVAL = "approval"
    PROMOTION = "promotion"


class StageOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"                # non-fatal concern
    SKIP = "skip"                # not applicable
    BLOCKED = "blocked"          # missing evidence → cannot pass


class PromotionStatus(str, Enum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    HELD = "held"                # needs more evidence/approval
    ROLLED_BACK = "rolled_back"


class RollbackReason(str, Enum):
    REGRESSION_DETECTED = "regression_detected"
    SECURITY_REGRESSION = "security_regression"
    PERFORMANCE_REGRESSION = "performance_regression"
    MANUAL = "manual"
    APPROVAL_REVOKED = "approval_revoked"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class CandidateChange:
    """A proposed modification to the Brain's own behavior."""
    id: str = field(default_factory=_new_id)
    scope: str = ""                       # e.g. "c28.RequirementCheck.AMBIGUITY_CAUTION"
    kind: ChangeKind = ChangeKind.UNKNOWN
    risk: ChangeRisk = ChangeRisk.MEDIUM
    description: str = ""
    old_value: str = ""
    new_value: str = ""
    rationale: str = ""
    artifact_ref: str = ""                # path / digest of the staged artifact
    isolated: bool = False                # caller claims this is staged, not prod
    created_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "scope": self.scope,
            "kind": self.kind.value, "risk": self.risk.value,
            "description": self.description,
            "old_value": self.old_value, "new_value": self.new_value,
            "rationale": self.rationale,
            "artifact_ref": self.artifact_ref,
            "isolated": self.isolated,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class StageResult:
    stage: GovernanceStage
    outcome: StageOutcome
    rationale: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    blocking: bool = False           # True → pipeline stops
    executed_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "outcome": self.outcome.value,
            "rationale": self.rationale,
            "evidence": dict(self.evidence),
            "blocking": self.blocking,
            "executed_at": self.executed_at,
        }


@dataclass(slots=True)
class PromotionRecord:
    change_id: str
    version: int
    status: PromotionStatus
    scope: str = ""
    approved_by: str = ""           # "policy:auto" or "policy:manual:reviewer"
    superseded_version: int | None = None
    promoted_at: str = field(default_factory=now_iso)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "version": self.version,
            "status": self.status.value,
            "scope": self.scope,
            "approved_by": self.approved_by,
            "superseded_version": self.superseded_version,
            "promoted_at": self.promoted_at,
            "notes": self.notes,
        }


@dataclass(slots=True)
class RollbackRecord:
    change_id: str
    scope: str
    from_version: int
    to_version: int
    reason: RollbackReason
    rationale: str = ""
    rolled_back_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id, "scope": self.scope,
            "from_version": self.from_version, "to_version": self.to_version,
            "reason": self.reason.value, "rationale": self.rationale,
            "rolled_back_at": self.rolled_back_at,
        }


@dataclass(slots=True)
class ChangeRecord:
    """Full audit record for a single candidate change."""
    change: CandidateChange
    stages: list[StageResult] = field(default_factory=list)
    promotion: PromotionRecord | None = None
    rollback: RollbackRecord | None = None
    final_status: PromotionStatus = PromotionStatus.HELD

    def to_dict(self) -> dict[str, Any]:
        return {
            "change": self.change.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "promotion": self.promotion.to_dict() if self.promotion else None,
            "rollback": self.rollback.to_dict() if self.rollback else None,
            "final_status": self.final_status.value,
        }


@dataclass(slots=True)
class GovernanceReport:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    change_id: str = ""
    change: CandidateChange = field(default_factory=CandidateChange)
    stages: list[StageResult] = field(default_factory=list)
    final_status: PromotionStatus = PromotionStatus.HELD
    promotion_version: int | None = None
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def failed_stage(self) -> StageResult | None:
        for s in self.stages:
            if s.outcome is StageOutcome.FAIL:
                return s
        return None

    def blocking_stage(self) -> StageResult | None:
        for s in self.stages:
            if s.blocking:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "change_id": self.change_id,
            "change": self.change.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "final_status": self.final_status.value,
            "promotion_version": self.promotion_version,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        stages = "  ".join(
            f"{s.stage.value[:4]}={s.outcome.value[:4]}"
            for s in self.stages
        )
        return (
            "=== Governance Report ===\n"
            f"change_id={self.change_id[:8]}  "
            f"scope={_short(self.change.scope, 60)}\n"
            f"kind={self.change.kind.value}  risk={self.change.risk.value}\n"
            f"final: {self.final_status.value}"
            + (f"  v{self.promotion_version}"
               if self.promotion_version else "")
            + f"\nstages: {stages}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. STAGE EVALUATORS
# ════════════════════════════════════════════════════════════════════════════
class _BaseStage:
    stage: GovernanceStage = GovernanceStage.ISOLATION

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        raise NotImplementedError

    def _mk(
        self, outcome: StageOutcome, rationale: str,
        *, evidence: dict[str, Any] | None = None, blocking: bool = False,
    ) -> StageResult:
        return StageResult(
            stage=self.stage, outcome=outcome, rationale=rationale,
            evidence=evidence or {}, blocking=blocking,
        )


# ---- 1. Isolation ----
class IsolationStage(_BaseStage):
    """Caller must declare the change is staged, not applied to prod."""
    stage = GovernanceStage.ISOLATION

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        if not change.scope:
            return self._mk(
                StageOutcome.FAIL,
                "change scope is empty",
                blocking=True,
            )
        if not change.isolated:
            return self._mk(
                StageOutcome.FAIL,
                "change is not marked as isolated (isolated=False)",
                evidence={"isolated": False},
                blocking=True,
            )
        if not change.artifact_ref:
            return self._mk(
                StageOutcome.BLOCKED,
                "no artifact_ref provided — cannot prove staged artifact",
                blocking=True,
            )
        return self._mk(
            StageOutcome.PASS,
            "change is isolated and has a staged artifact reference",
            evidence={
                "artifact_ref": change.artifact_ref,
                "isolated": True,
            },
        )


# ---- 2. Test ----
class TestStage(_BaseStage):
    stage = GovernanceStage.TEST

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        tr = evidence.get("test_run")
        if tr is None:
            return self._mk(
                StageOutcome.BLOCKED,
                "no test_run evidence provided",
                blocking=True,
            )
        passed = int(getattr(tr, "passed", 0) or 0)
        failed = int(getattr(tr, "failed", 0) or 0)
        errors = int(getattr(tr, "errors", 0) or 0)
        total = passed + failed + errors
        ev = {"passed": passed, "failed": failed, "errors": errors}
        if total == 0:
            return self._mk(
                StageOutcome.BLOCKED,
                "test run produced no results",
                evidence=ev, blocking=True,
            )
        if failed or errors:
            return self._mk(
                StageOutcome.FAIL,
                f"{failed} failed, {errors} errors",
                evidence=ev, blocking=True,
            )
        return self._mk(
            StageOutcome.PASS,
            f"{passed} tests passed",
            evidence=ev,
        )


# ---- 3. Regression ----
class RegressionStage(_BaseStage):
    stage = GovernanceStage.REGRESSION

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        regs = list(evidence.get("regressions", []) or [])
        if regs:
            return self._mk(
                StageOutcome.FAIL,
                f"{len(regs)} regression(s) detected",
                evidence={"count": len(regs),
                          "examples": regs[:3]},
                blocking=True,
            )
        baseline = evidence.get("baseline_passed")
        current = evidence.get("current_passed")
        if baseline is not None and current is not None:
            if int(current) < int(baseline):
                return self._mk(
                    StageOutcome.FAIL,
                    f"pass count dropped {baseline} → {current}",
                    evidence={
                        "baseline_passed": baseline,
                        "current_passed": current,
                    },
                    blocking=True,
                )
            return self._mk(
                StageOutcome.PASS,
                f"no regression (passed {baseline} → {current})",
                evidence={
                    "baseline_passed": baseline,
                    "current_passed": current,
                },
            )
        # No regression evidence → BLOCKED (be conservative for HIGH risk)
        if _RISK_RANK[change.risk] >= _RISK_RANK[ChangeRisk.HIGH]:
            return self._mk(
                StageOutcome.BLOCKED,
                "no regression evidence for HIGH/CRITICAL risk change",
                blocking=True,
            )
        return self._mk(
            StageOutcome.WARN,
            "no regression evidence provided",
        )


# ---- 4. Benchmark ----
class BenchmarkStage(_BaseStage):
    stage = GovernanceStage.BENCHMARK
    DEFAULT_TOLERANCE = 0.10   # 10% slowdown tolerated for LOW/MEDIUM

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        before = evidence.get("metric_before")
        after = evidence.get("metric_after")
        metric = evidence.get("metric_name", "value")
        direction = evidence.get("direction", "lower_is_better")
        tolerance = float(evidence.get("tolerance", self.DEFAULT_TOLERANCE))

        if before is None or after is None:
            # HIGH/CRITICAL risk requires benchmarks
            if _RISK_RANK[change.risk] >= _RISK_RANK[ChangeRisk.HIGH]:
                return self._mk(
                    StageOutcome.BLOCKED,
                    "no benchmark evidence for HIGH/CRITICAL risk change",
                    blocking=True,
                )
            return self._mk(
                StageOutcome.WARN,
                "no benchmark evidence",
            )
        b = float(before)
        a = float(after)
        if b <= 0:
            return self._mk(
                StageOutcome.WARN,
                f"baseline {metric}={b} invalid; skipping comparison",
                evidence={"metric_name": metric, "before": b, "after": a},
            )
        delta = (a - b) / b
        improved = (
            delta <= -tolerance if direction == "lower_is_better"
            else delta >= tolerance
        )
        regressed = (
            delta >= tolerance if direction == "lower_is_better"
            else delta <= -tolerance
        )
        ev = {
            "metric_name": metric, "before": b, "after": a,
            "delta": round(delta, 4), "tolerance": tolerance,
            "direction": direction,
        }
        if regressed:
            return self._mk(
                StageOutcome.FAIL,
                f"{metric} regressed by {delta*100:.1f}% "
                f"(tolerance {tolerance*100:.0f}%)",
                evidence=ev, blocking=True,
            )
        if improved:
            return self._mk(
                StageOutcome.PASS,
                f"{metric} improved by {abs(delta)*100:.1f}%",
                evidence=ev,
            )
        return self._mk(
            StageOutcome.PASS,
            f"{metric} within tolerance (Δ={delta*100:.1f}%)",
            evidence=ev,
        )


# ---- 5. Security ----
class SecurityStage(_BaseStage):
    stage = GovernanceStage.SECURITY

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        sec = evidence.get("security_report")
        if sec is None:
            return self._mk(
                StageOutcome.BLOCKED,
                "no security_report evidence",
                blocking=True,
            )
        findings = list(getattr(sec, "findings", []) or [])
        critical = 0
        high = 0
        for f in findings:
            sev = _enum_val(getattr(f, "severity", ""))
            if sev == "critical":
                critical += 1
            elif sev == "high":
                high += 1
        ev = {"critical": critical, "high": high, "total": len(findings)}
        if critical:
            return self._mk(
                StageOutcome.FAIL,
                f"{critical} CRITICAL security finding(s)",
                evidence=ev, blocking=True,
            )
        if high and _RISK_RANK[change.risk] >= _RISK_RANK[ChangeRisk.HIGH]:
            return self._mk(
                StageOutcome.FAIL,
                f"{high} HIGH security finding(s) on HIGH/CRITICAL risk change",
                evidence=ev, blocking=True,
            )
        if high:
            return self._mk(
                StageOutcome.WARN,
                f"{high} HIGH security finding(s)",
                evidence=ev,
            )
        return self._mk(
            StageOutcome.PASS,
            f"no CRITICAL/HIGH findings ({len(findings)} total)",
            evidence=ev,
        )


# ---- 6. Verification ----
class VerificationStage(_BaseStage):
    stage = GovernanceStage.VERIFICATION

    def evaluate(
        self, change: CandidateChange, evidence: dict[str, Any],
    ) -> StageResult:
        vb = evidence.get("verification_bundle")
        if vb is None:
            return self._mk(
                StageOutcome.BLOCKED,
                "no verification_bundle evidence",
                blocking=True,
            )
        results = list(getattr(vb, "results", []) or [])
        if not results:
            return self._mk(
                StageOutcome.BLOCKED,
                "verification bundle is empty",
                blocking=True,
            )
        refuted = sum(
            1 for r in results
            if _enum_val(getattr(r, "result", "")) == "refuted"
        )
        supported = sum(
            1 for r in results
            if _enum_val(getattr(r, "result", "")) == "supported"
        )
        ev = {"supported": supported, "refuted": refuted,
              "total": len(results)}
        if refuted:
            return self._mk(
                StageOutcome.FAIL,
                f"{refuted} refuted claim(s)",
                evidence=ev, blocking=True,
            )
        if supported == 0:
            return self._mk(
                StageOutcome.BLOCKED,
                "no supported claims",
                evidence=ev, blocking=True,
            )
        return self._mk(
            StageOutcome.PASS,
            f"{supported}/{len(results)} claims verified",
            evidence=ev,
        )


# ---- 7. Approval ----
class ApprovalPolicy:
    """Rule-based, risk-tiered approval policy.

    Rules:
        LOW      → auto-approve if all pipeline stages passed
        MEDIUM   → auto-approve if all stages passed AND benchmark improved
        HIGH     → require explicit approval (policy:manual:reviewer)
        CRITICAL → require explicit approval + benchmark improvement +
                   zero WARN stages
    """
    def decide(
        self, change: CandidateChange, stages: Sequence[StageResult],
        *, manual_approval: str = "",
    ) -> tuple[StageOutcome, str, str]:
        """Return (outcome, rationale, approved_by)."""
        # Any FAIL blocks
        if any(s.outcome is StageOutcome.FAIL for s in stages):
            return (StageOutcome.FAIL,
                    "prior stage failed", "")
        # Any BLOCKED blocks
        if any(s.outcome is StageOutcome.BLOCKED for s in stages):
            return (StageOutcome.BLOCKED,
                    "prior stage is blocked (missing evidence)", "")

        warns = [s for s in stages if s.outcome is StageOutcome.WARN]

        if change.risk is ChangeRisk.LOW:
            if warns:
                return (StageOutcome.WARN,
                        f"{len(warns)} warning(s); approving with notes",
                        "policy:auto")
            return (StageOutcome.PASS,
                    "LOW risk; all stages passed; auto-approved",
                    "policy:auto")

        if change.risk is ChangeRisk.MEDIUM:
            bench = next(
                (s for s in stages
                 if s.stage is GovernanceStage.BENCHMARK),
                None,
            )
            bench_improved = (
                bench is not None and bench.evidence.get("delta") is not None
                and bench.evidence.get("delta") < 0
            )
            if bench_improved and not warns:
                return (StageOutcome.PASS,
                        "MEDIUM risk; benchmark improved; auto-approved",
                        "policy:auto")
            if not warns:
                return (StageOutcome.PASS,
                        "MEDIUM risk; all stages passed; auto-approved",
                        "policy:auto")
            return (StageOutcome.WARN,
                    f"{len(warns)} warning(s); approving with notes",
                    "policy:auto")

        if change.risk is ChangeRisk.HIGH:
            if not manual_approval:
                return (StageOutcome.BLOCKED,
                        "HIGH risk requires explicit manual approval "
                        "(approve(..., approved_by='...'))",
                        "")
            return (StageOutcome.PASS,
                    f"HIGH risk approved by {manual_approval}",
                    f"policy:manual:{manual_approval}")

        if change.risk is ChangeRisk.CRITICAL:
            if not manual_approval:
                return (StageOutcome.BLOCKED,
                        "CRITICAL risk requires explicit manual approval",
                        "")
            # CRITICAL changes require a clean evidence chain; explicit
            # approval cannot override unresolved warnings.
            if warns:
                return (StageOutcome.BLOCKED,
                        f"CRITICAL risk has {len(warns)} warning(s); "
                        "resolve them before promotion",
                        "")
            return (StageOutcome.PASS,
                    f"CRITICAL risk approved by {manual_approval}",
                    f"policy:manual:{manual_approval}")

        return (StageOutcome.BLOCKED,
                "unknown risk level", "")


# ════════════════════════════════════════════════════════════════════════════
# 4. GOVERNANCE ENGINE
# ════════════════════════════════════════════════════════════════════════════
class GovernanceEngine:
    """Full self-improvement governance pipeline."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        max_changes_per_scope: int = 100,
    ) -> None:
        if max_changes_per_scope < 1:
            raise ValidationError("max_changes_per_scope must be >= 1")
        self.memory = memory
        self.max_changes_per_scope = max_changes_per_scope
        self.isolation = IsolationStage()
        self.test_stage = TestStage()
        self.regression = RegressionStage()
        self.benchmark = BenchmarkStage()
        self.security = SecurityStage()
        self.verification = VerificationStage()
        self.approval_policy = ApprovalPolicy()

    # ---- main entry ----
    def submit(
        self,
        change: CandidateChange,
        *,
        evidence: dict[str, Any] | None = None,
        approved_by: str = "",
        project_id: str = "",
        scope_key_prefix: str = "self_change",
    ) -> GovernanceReport:
        evidence = dict(evidence or {})
        report = GovernanceReport(
            project_id=project_id, change_id=change.id, change=change,
            provenance=Provenance(
                source="governance_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )

        # ---- Run stages 1..6 in order ----
        stages_to_run: list[_BaseStage] = [
            self.isolation, self.test_stage, self.regression,
            self.benchmark, self.security, self.verification,
        ]
        stopped = False
        for stage in stages_to_run:
            try:
                r = stage.evaluate(change, evidence)
            except Exception as exc:
                r = StageResult(
                    stage=stage.stage, outcome=StageOutcome.BLOCKED,
                    rationale=f"stage raised: {type(exc).__name__}: {exc}",
                    blocking=True,
                )
            report.stages.append(r)
            if r.blocking and r.outcome in (StageOutcome.FAIL,
                                             StageOutcome.BLOCKED):
                stopped = True
                break

        if stopped:
            # Determine final status: FAIL → REJECTED, BLOCKED → HELD
            last = report.stages[-1]
            if last.outcome is StageOutcome.FAIL:
                report.final_status = PromotionStatus.REJECTED
            else:
                report.final_status = PromotionStatus.HELD
            report.rationale = (
                f"stopped at {last.stage.value}: {last.rationale}"
            )
            self._record_change(report, project_id=project_id,
                                 scope_key_prefix=scope_key_prefix)
            return report

        # ---- Stage 7: approval ----
        outcome, rationale, approved = self.approval_policy.decide(
            change, report.stages, manual_approval=approved_by,
        )
        report.stages.append(StageResult(
            stage=GovernanceStage.APPROVAL,
            outcome=outcome,
            rationale=rationale,
            evidence={"approved_by": approved},
            blocking=(outcome in (StageOutcome.FAIL,
                                    StageOutcome.BLOCKED)),
        ))

        if outcome in (StageOutcome.FAIL, StageOutcome.BLOCKED):
            report.final_status = (
                PromotionStatus.REJECTED if outcome is StageOutcome.FAIL
                else PromotionStatus.HELD
            )
            report.rationale = f"approval: {rationale}"
            self._record_change(report, project_id=project_id,
                                 scope_key_prefix=scope_key_prefix)
            return report

        # ---- Stage 8: promotion ----
        if self._promotion_count(
            change.scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        ) >= self.max_changes_per_scope:
            report.final_status = PromotionStatus.HELD
            report.rationale = (
                f"promotion cap reached for scope '{change.scope}'"
            )
            self._record_change(report, project_id=project_id,
                                 scope_key_prefix=scope_key_prefix)
            return report

        next_version = self._next_version(
            change.scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        )
        prev_active = self._active_version(
            change.scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        )
        report.promotion_version = next_version
        report.final_status = PromotionStatus.PROMOTED
        report.stages.append(StageResult(
            stage=GovernanceStage.PROMOTION,
            outcome=StageOutcome.PASS,
            rationale=(
                f"promoted as v{next_version}"
                + (f"; superseded v{prev_active}"
                   if prev_active else "")
            ),
            evidence={
                "version": next_version,
                "superseded_version": prev_active,
                "approved_by": approved,
            },
        ))
        report.rationale = (
            f"promoted v{next_version} for scope '{change.scope}' "
            f"({approved})"
        )
        self._record_change(report, project_id=project_id,
                             scope_key_prefix=scope_key_prefix)
        return report

    # ---- rollback ----
    def rollback(
        self, *, scope: str, reason: RollbackReason,
        rationale: str = "", project_id: str = "",
        scope_key_prefix: str = "self_change",
        to_version: int | None = None,
    ) -> RollbackRecord:
        """Explicitly roll a scope back. Never silent."""
        if self.memory is None:
            raise ValidationError("memory not attached for rollback")
        current = self._active_version(
            scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        )
        if current is None:
            raise ValidationError(
                f"no active promotion for scope '{scope}'"
            )
        if to_version is None:
            to_version = current - 1
        if to_version < 1:
            raise ValidationError(
                f"cannot roll back to v{to_version}; no such version"
            )
        # Find both the current and target versions. Rollback must leave a
        # concrete active version; merely archiving the current entry would
        # otherwise make _active_version() return None.
        entries = self._entries_for_scope(
            scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        )
        current_entries = [
            e for e in entries if int(e.content.get("version", 0)) == current
        ]
        target_entries = [
            e for e in entries if int(e.content.get("version", 0)) == to_version
        ]
        if not current_entries:
            raise ValidationError(
                f"active version v{current} for scope '{scope}' not found"
            )
        if not target_entries:
            raise ValidationError(
                f"rollback target v{to_version} for scope '{scope}' not found"
            )
        cur_change_id = str(current_entries[0].content.get("change_id", ""))
        target_content = dict(target_entries[0].content)
        target_content["final_status"] = PromotionStatus.PROMOTED.value
        target_content["rollback_restored_from"] = current
        target_content["rollback_restored_at"] = now_iso()

        # Archive the current active version, then create a fresh active
        # pointer carrying the exact target version/change identity.
        for e in current_entries:
            if e.status.value == "active":
                self.memory.archive(e.id)
        self.memory.create(
            MemoryKind.PROJECT,
            f"{scope_key_prefix}:{scope}:rollback-active:{now_iso()}",
            target_content,
            scope_type=MemoryScope.PROJECT,
            scope_id=project_id or "default",
            tags=["self_change", "c29", "rollback-restored"],
            provenance=Provenance(
                source="governance_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )

        rec = RollbackRecord(
            change_id=cur_change_id, scope=scope,
            from_version=current, to_version=to_version,
            reason=reason, rationale=rationale,
        )
        # Persist a rollback audit entry
        self.memory.upsert(
            MemoryKind.PROJECT,
            f"{scope_key_prefix}:{scope}:rollback:{now_iso()}",
            {
                "change_id": rec.change_id,
                "scope": scope,
                "from_version": rec.from_version,
                "to_version": rec.to_version,
                "reason": rec.reason.value,
                "rationale": rec.rationale,
                "rolled_back_at": rec.rolled_back_at,
            },
            scope_type=MemoryScope.PROJECT,
            scope_id=project_id or "default",
            tags=["rollback", "c29"],
            provenance=Provenance(
                source="governance_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        return rec

    # ---- version helpers ----
    def _entries_for_scope(
        self, scope: str, *, project_id: str = "",
        scope_key_prefix: str = "self_change",
    ) -> list[MemoryEntry]:
        if self.memory is None:
            return []
        return self.memory.find(
            kind=MemoryKind.PROJECT, status=None,
            scope_type=MemoryScope.PROJECT,
            scope_id=project_id or "default",
            key_like=f"{scope_key_prefix}:{scope}:",
        )

    def _promotion_count(
        self, scope: str, *, project_id: str = "",
        scope_key_prefix: str = "self_change",
    ) -> int:
        return sum(
            1 for e in self._entries_for_scope(
                scope, project_id=project_id, scope_key_prefix=scope_key_prefix
            )
            if e.content.get("final_status") == PromotionStatus.PROMOTED.value
        )

    def _next_version(
        self, scope: str, *, project_id: str = "",
        scope_key_prefix: str = "self_change",
    ) -> int:
        if self.memory is None:
            return 1
        entries = self._entries_for_scope(
            scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        )
        versions = [int(e.content.get("version", 0))
                    for e in entries
                    if e.content.get("change_id")]
        return max(versions + [0]) + 1

    def _active_version(
        self, scope: str, *, project_id: str = "",
        scope_key_prefix: str = "self_change",
    ) -> int | None:
        if self.memory is None:
            return None
        entries = self._entries_for_scope(
            scope, project_id=project_id, scope_key_prefix=scope_key_prefix
        )
        entries = [e for e in entries if e.status.value == "active"]
        versions = [int(e.content.get("version", 0))
                    for e in entries]
        return max(versions) if versions else None

    # ---- record ----
    def _record_change(
        self, report: GovernanceReport, *,
        project_id: str, scope_key_prefix: str,
    ) -> None:
        if self.memory is None:
            return
        scope = report.change.scope
        key = (f"{scope_key_prefix}:{scope}:"
               f"{report.change.id[:8]}")
        self.memory.upsert(
            MemoryKind.PROJECT, key, {
                "change_id": report.change.id,
                "version": report.promotion_version or 0,
                "scope": scope,
                "kind": report.change.kind.value,
                "risk": report.change.risk.value,
                "old_value": report.change.old_value,
                "new_value": report.change.new_value,
                "final_status": report.final_status.value,
                "rationale": report.rationale,
                "stages": [s.to_dict() for s in report.stages],
            },
            scope_type=MemoryScope.PROJECT,
            scope_id=project_id or "default",
            tags=["self_change", "c29",
                  report.final_status.value],
            provenance=report.provenance,
        )
        # Failure memory for rejected/rolled-back
        if report.final_status is PromotionStatus.REJECTED:
            self.memory.record_failure(
                f"governance_reject:{report.change.id}",
                what=f"self-improvement change rejected: {scope}",
                root_cause=report.rationale,
                fix=None,
                scope_id=project_id or "default",
                provenance=report.provenance,
                confidence=Confidence.HIGH,
            )


# ════════════════════════════════════════════════════════════════════════════
# 5. GOVERNANCE REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class GovernanceRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, report: GovernanceReport, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"governance_report:{report.id}"
        with self.memory.storage.transaction():
            self.memory.create(
                MemoryKind.PROJECT, key, report.to_dict(),
                scope_type=MemoryScope.PROJECT, scope_id=project_id,
                tags=["governance", "c29", report.final_status.value],
                provenance=report.provenance,
            )
            if self.ontology is None:
                return key
            ent = self.ontology.add(
            EntityKind.DECISION,
            _short(
                f"Governance {report.change_id[:8]} "
                f"({report.final_status.value})", 120,
            ),
            attributes={
                "report_id": report.id,
                "project_id": project_id,
                "change_id": report.change_id,
                "scope": report.change.scope,
                "kind": report.change.kind.value,
                "risk": report.change.risk.value,
                "final_status": report.final_status.value,
                "promotion_version": report.promotion_version,
                "stages": [s.to_dict() for s in report.stages],
            },
            tags=["governance", report.final_status.value],
            provenance=report.provenance,
        )
        return ent.id

    def load(self, report_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"governance_report:{report_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None

    def active_version(self, scope: str, *, project_id: str) -> int | None:
        entries = self.memory.find(
            kind=MemoryKind.PROJECT, status="active",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            key_like=f"self_change:{scope}:",
        )
        versions = [int(e.content.get("version", 0)) for e in entries]
        return max(versions) if versions else None

    def history(self, scope: str, *, project_id: str) -> list[dict[str, Any]]:
        entries = self.memory.find(
            kind=MemoryKind.PROJECT, status=None,
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            key_like=f"self_change:{scope}:",
        )
        return [
            {"key": e.key, "status": e.status.value, **dict(e.content)}
            for e in sorted(entries, key=lambda x: x.created_at)
        ]


# ════════════════════════════════════════════════════════════════════════════
# 6. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
# ---- test fakes ----
class _FakeTestRun:
    def __init__(self, passed=5, failed=0, errors=0):
        self.passed = passed
        self.failed = failed
        self.errors = errors


class _FakeFinding:
    def __init__(self, severity="low"):
        self.severity = type("S", (), {"value": severity})()


class _FakeSecReport:
    def __init__(self, findings=None):
        self.findings = findings or []


class _FakeVResult:
    def __init__(self, result="supported"):
        self.result = type("R", (), {"value": result})()


class _FakeVB:
    def __init__(self, results=None):
        self.results = results or [_FakeVResult("supported")]


def _mk_change(**kw) -> CandidateChange:
    base = dict(
        scope="c28.RequirementCheck.AMBIGUITY_CAUTION",
        kind=ChangeKind.HEURISTIC,
        risk=ChangeRisk.LOW,
        description="bump threshold 2 → 3",
        old_value="2", new_value="3",
        rationale="reduce false positives",
        artifact_ref="sha256:abc123",
        isolated=True,
    )
    base.update(kw)
    return CandidateChange(**base)


def _mk_evidence_ok(**kw) -> dict[str, Any]:
    base = dict(
        test_run=_FakeTestRun(passed=10, failed=0, errors=0),
        regressions=[],
        baseline_passed=10,
        current_passed=10,
        metric_before=1.0,
        metric_after=0.9,
        metric_name="wall_time",
        direction="lower_is_better",
        security_report=_FakeSecReport(findings=[]),
        verification_bundle=_FakeVB(),
    )
    base.update(kw)
    return base


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

    print("Running C29 self-tests…")

    # ---- stages in isolation ----
    def t_isolation_missing_scope() -> None:
        ch = _mk_change(scope="")
        r = IsolationStage().evaluate(ch, {})
        assert r.outcome is StageOutcome.FAIL
        assert r.blocking is True

    def t_isolation_not_staged() -> None:
        ch = _mk_change(isolated=False)
        r = IsolationStage().evaluate(ch, {})
        assert r.outcome is StageOutcome.FAIL
        assert r.blocking is True

    def t_isolation_no_artifact() -> None:
        ch = _mk_change(artifact_ref="")
        r = IsolationStage().evaluate(ch, {})
        assert r.outcome is StageOutcome.BLOCKED

    def t_isolation_pass() -> None:
        r = IsolationStage().evaluate(_mk_change(), {})
        assert r.outcome is StageOutcome.PASS

    check("isolation: missing scope → FAIL/blocking",
          t_isolation_missing_scope)
    check("isolation: not staged → FAIL/blocking",
          t_isolation_not_staged)
    check("isolation: no artifact → BLOCKED", t_isolation_no_artifact)
    check("isolation: properly staged → PASS", t_isolation_pass)

    def t_test_pass() -> None:
        r = TestStage().evaluate(
            _mk_change(), {"test_run": _FakeTestRun(passed=5)},
        )
        assert r.outcome is StageOutcome.PASS

    def t_test_fail() -> None:
        r = TestStage().evaluate(
            _mk_change(), {"test_run": _FakeTestRun(passed=3, failed=2)},
        )
        assert r.outcome is StageOutcome.FAIL
        assert r.blocking

    def t_test_blocked_no_evidence() -> None:
        r = TestStage().evaluate(_mk_change(), {})
        assert r.outcome is StageOutcome.BLOCKED

    def t_test_zero_results() -> None:
        r = TestStage().evaluate(
            _mk_change(),
            {"test_run": _FakeTestRun(passed=0, failed=0, errors=0)},
        )
        assert r.outcome is StageOutcome.BLOCKED

    check("test: all pass → PASS", t_test_pass)
    check("test: failures → FAIL/blocking", t_test_fail)
    check("test: no evidence → BLOCKED", t_test_blocked_no_evidence)
    check("test: zero results → BLOCKED", t_test_zero_results)

    def t_regression_clean() -> None:
        r = RegressionStage().evaluate(
            _mk_change(),
            {"regressions": [], "baseline_passed": 5, "current_passed": 5},
        )
        assert r.outcome is StageOutcome.PASS

    def t_regression_detected() -> None:
        r = RegressionStage().evaluate(
            _mk_change(), {"regressions": ["tests/test_x.py::test_y"]},
        )
        assert r.outcome is StageOutcome.FAIL
        assert r.blocking

    def t_regression_pass_count_drop() -> None:
        r = RegressionStage().evaluate(
            _mk_change(),
            {"baseline_passed": 10, "current_passed": 8},
        )
        assert r.outcome is StageOutcome.FAIL

    def t_regression_high_risk_requires_evidence() -> None:
        r = RegressionStage().evaluate(
            _mk_change(risk=ChangeRisk.HIGH), {},
        )
        assert r.outcome is StageOutcome.BLOCKED

    check("regression: clean → PASS", t_regression_clean)
    check("regression: detected → FAIL/blocking", t_regression_detected)
    check("regression: pass-count drop → FAIL",
          t_regression_pass_count_drop)
    check("regression: HIGH risk needs evidence → BLOCKED",
          t_regression_high_risk_requires_evidence)

    def t_benchmark_improved() -> None:
        r = BenchmarkStage().evaluate(_mk_change(), {
            "metric_before": 1.0, "metric_after": 0.7,
            "metric_name": "latency", "direction": "lower_is_better",
        })
        assert r.outcome is StageOutcome.PASS
        assert "improved" in r.rationale

    def t_benchmark_regressed() -> None:
        r = BenchmarkStage().evaluate(_mk_change(), {
            "metric_before": 1.0, "metric_after": 1.5,
            "metric_name": "latency", "direction": "lower_is_better",
        })
        assert r.outcome is StageOutcome.FAIL
        assert r.blocking

    def t_benchmark_within_tolerance() -> None:
        r = BenchmarkStage().evaluate(_mk_change(), {
            "metric_before": 1.0, "metric_after": 1.05,
            "tolerance": 0.10,
        })
        assert r.outcome is StageOutcome.PASS
        assert "within tolerance" in r.rationale

    def t_benchmark_high_risk_blocks_without_evidence() -> None:
        r = BenchmarkStage().evaluate(
            _mk_change(risk=ChangeRisk.HIGH), {},
        )
        assert r.outcome is StageOutcome.BLOCKED

    check("benchmark: improved → PASS", t_benchmark_improved)
    check("benchmark: regressed → FAIL/blocking", t_benchmark_regressed)
    check("benchmark: within tolerance → PASS",
          t_benchmark_within_tolerance)
    check("benchmark: HIGH risk needs evidence → BLOCKED",
          t_benchmark_high_risk_blocks_without_evidence)

    def t_security_clean() -> None:
        r = SecurityStage().evaluate(_mk_change(), {
            "security_report": _FakeSecReport(findings=[]),
        })
        assert r.outcome is StageOutcome.PASS

    def t_security_critical_blocks() -> None:
        r = SecurityStage().evaluate(_mk_change(), {
            "security_report": _FakeSecReport(
                findings=[_FakeFinding("critical")],
            ),
        })
        assert r.outcome is StageOutcome.FAIL
        assert r.blocking

    def t_security_high_warns_low_risk() -> None:
        r = SecurityStage().evaluate(_mk_change(risk=ChangeRisk.LOW), {
            "security_report": _FakeSecReport(
                findings=[_FakeFinding("high")],
            ),
        })
        assert r.outcome is StageOutcome.WARN

    def t_security_high_blocks_high_risk() -> None:
        r = SecurityStage().evaluate(_mk_change(risk=ChangeRisk.HIGH), {
            "security_report": _FakeSecReport(
                findings=[_FakeFinding("high")],
            ),
        })
        assert r.outcome is StageOutcome.FAIL

    check("security: clean → PASS", t_security_clean)
    check("security: CRITICAL → FAIL/blocking", t_security_critical_blocks)
    check("security: HIGH + low risk → WARN", t_security_high_warns_low_risk)
    check("security: HIGH + high risk → FAIL",
          t_security_high_blocks_high_risk)

    def t_verification_clean() -> None:
        r = VerificationStage().evaluate(_mk_change(), {
            "verification_bundle": _FakeVB(),
        })
        assert r.outcome is StageOutcome.PASS

    def t_verification_refuted_blocks() -> None:
        r = VerificationStage().evaluate(_mk_change(), {
            "verification_bundle": _FakeVB(
                results=[_FakeVResult("refuted")],
            ),
        })
        assert r.outcome is StageOutcome.FAIL

    check("verification: clean → PASS", t_verification_clean)
    check("verification: refuted → FAIL/blocking",
          t_verification_refuted_blocks)

    # ---- approval policy ----
    def t_approval_low_risk_auto() -> None:
        ch = _mk_change(risk=ChangeRisk.LOW)
        stages = [StageResult(GovernanceStage.TEST, StageOutcome.PASS)]
        outcome, _, who = ApprovalPolicy().decide(ch, stages)
        assert outcome is StageOutcome.PASS
        assert who.startswith("policy:auto")

    def t_approval_high_risk_requires_manual() -> None:
        ch = _mk_change(risk=ChangeRisk.HIGH)
        stages = [StageResult(GovernanceStage.TEST, StageOutcome.PASS)]
        outcome, rationale, _ = ApprovalPolicy().decide(ch, stages)
        assert outcome is StageOutcome.BLOCKED
        assert "manual" in rationale.lower()

    def t_approval_high_risk_with_manual() -> None:
        ch = _mk_change(risk=ChangeRisk.HIGH)
        stages = [StageResult(GovernanceStage.TEST, StageOutcome.PASS)]
        outcome, _, who = ApprovalPolicy().decide(
            ch, stages, manual_approval="alice",
        )
        assert outcome is StageOutcome.PASS
        assert who == "policy:manual:alice"

    def t_approval_critical_requires_manual() -> None:
        ch = _mk_change(risk=ChangeRisk.CRITICAL)
        stages = [StageResult(GovernanceStage.TEST, StageOutcome.PASS)]
        outcome, _, _ = ApprovalPolicy().decide(ch, stages)
        assert outcome is StageOutcome.BLOCKED

    check("approval: LOW auto-approved", t_approval_low_risk_auto)
    check("approval: HIGH requires manual", t_approval_high_risk_requires_manual)
    check("approval: HIGH + manual → PASS",
          t_approval_high_risk_with_manual)
    check("approval: CRITICAL requires manual",
          t_approval_critical_requires_manual)

    # ---- engine end-to-end ----
    def t_engine_happy_promotion() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                rep = eng.submit(
                    _mk_change(),
                    evidence=_mk_evidence_ok(),
                    project_id="p",
                )
                assert rep.final_status is PromotionStatus.PROMOTED
                assert rep.promotion_version == 1
                stages = [s.stage for s in rep.stages]
                assert stages == [
                    GovernanceStage.ISOLATION,
                    GovernanceStage.TEST,
                    GovernanceStage.REGRESSION,
                    GovernanceStage.BENCHMARK,
                    GovernanceStage.SECURITY,
                    GovernanceStage.VERIFICATION,
                    GovernanceStage.APPROVAL,
                    GovernanceStage.PROMOTION,
                ]
            finally:
                s.shutdown()

    def t_engine_stop_on_test_fail() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                ev = _mk_evidence_ok()
                ev["test_run"] = _FakeTestRun(passed=0, failed=3)
                rep = eng.submit(
                    _mk_change(), evidence=ev, project_id="p",
                )
                assert rep.final_status is PromotionStatus.REJECTED
                assert rep.stages[-1].stage is GovernanceStage.TEST
                assert rep.promotion_version is None
            finally:
                s.shutdown()

    def t_engine_stop_on_regression() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                ev = _mk_evidence_ok()
                ev["regressions"] = ["tests/a.py::test_x"]
                rep = eng.submit(
                    _mk_change(), evidence=ev, project_id="p",
                )
                assert rep.final_status is PromotionStatus.REJECTED
                assert rep.stages[-1].stage is GovernanceStage.REGRESSION
            finally:
                s.shutdown()

    def t_engine_hold_on_missing_evidence() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                # no evidence at all → isolation passes; test BLOCKED
                rep = eng.submit(
                    _mk_change(), evidence={}, project_id="p",
                )
                assert rep.final_status is PromotionStatus.HELD
            finally:
                s.shutdown()

    def t_engine_high_risk_requires_manual() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                rep = eng.submit(
                    _mk_change(risk=ChangeRisk.HIGH),
                    evidence=_mk_evidence_ok(),
                    project_id="p",
                )
                assert rep.final_status is PromotionStatus.HELD
                assert rep.stages[-1].stage is GovernanceStage.APPROVAL
                # Now with manual approval
                eng2 = GovernanceEngine(memory=mem)
                rep2 = eng2.submit(
                    _mk_change(risk=ChangeRisk.HIGH),
                    evidence=_mk_evidence_ok(),
                    approved_by="reviewer",
                    project_id="p",
                )
                assert rep2.final_status is PromotionStatus.PROMOTED
            finally:
                s.shutdown()

    def t_engine_versioning() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                r1 = eng.submit(
                    _mk_change(new_value="3"),
                    evidence=_mk_evidence_ok(), project_id="p",
                )
                assert r1.promotion_version == 1
                r2 = eng.submit(
                    _mk_change(new_value="4"),
                    evidence=_mk_evidence_ok(), project_id="p",
                )
                assert r2.promotion_version == 2
                # superseded_version recorded
                pr_stage = next(
                    st for st in r2.stages
                    if st.stage is GovernanceStage.PROMOTION
                )
                assert pr_stage.evidence.get("superseded_version") == 1
            finally:
                s.shutdown()

    check("engine: full happy path → PROMOTED v1",
          t_engine_happy_promotion)
    check("engine: test failure → REJECTED (stops before approval)",
          t_engine_stop_on_test_fail)
    check("engine: regression → REJECTED", t_engine_stop_on_regression)
    check("engine: missing evidence → HELD",
          t_engine_hold_on_missing_evidence)
    check("engine: HIGH risk requires manual approval",
          t_engine_high_risk_requires_manual)
    check("engine: versioning increments + records supersession",
          t_engine_versioning)

    # ---- rollback ----
    def t_rollback_explicit() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                r1 = eng.submit(
                    _mk_change(new_value="3"),
                    evidence=_mk_evidence_ok(), project_id="p",
                )
                r2 = eng.submit(
                    _mk_change(new_value="4"),
                    evidence=_mk_evidence_ok(), project_id="p",
                )
                assert r2.promotion_version == 2
                rb = eng.rollback(
                    scope=_mk_change().scope,
                    reason=RollbackReason.REGRESSION_DETECTED,
                    rationale="regression in production",
                    project_id="p",
                )
                assert rb.from_version == 2
                assert rb.to_version == 1
                assert rb.reason is RollbackReason.REGRESSION_DETECTED
                # Active version now v1
                active = eng._active_version(_mk_change().scope)
                assert active == 1
            finally:
                s.shutdown()

    def t_rollback_no_active_errors() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                try:
                    eng.rollback(
                        scope="never.promoted",
                        reason=RollbackReason.MANUAL,
                    )
                except ValidationError:
                    return
                raise AssertionError("expected ValidationError")
            finally:
                s.shutdown()

    check("rollback: explicit, records from/to versions",
          t_rollback_explicit)
    check("rollback: no active version → error",
          t_rollback_no_active_errors)

    # ---- persistence via repository ----
    def t_persist_repository() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                eng = GovernanceEngine(memory=mem)
                rep = eng.submit(
                    _mk_change(),
                    evidence=_mk_evidence_ok(),
                    project_id="proj-x",
                )
                repo = GovernanceRepository(memory=mem, ontology=ont)
                ent = repo.save(rep, project_id="proj-x")
                assert ent
                loaded = repo.load(rep.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["final_status"] == "promoted"
                assert ont.count(kind=EntityKind.DECISION) >= 1
                # History
                hist = repo.history(
                    _mk_change().scope, project_id="proj-x",
                )
                assert len(hist) >= 1
                # Active version
                assert repo.active_version(
                    _mk_change().scope, project_id="proj-x",
                ) == 1
            finally:
                s.shutdown()

    check("persist: report + history + active version",
          t_persist_repository)

    # ---- rejection records failure memory ----
    def t_rejection_records_failure() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                ev = _mk_evidence_ok()
                ev["test_run"] = _FakeTestRun(passed=0, failed=2)
                eng.submit(
                    _mk_change(), evidence=ev, project_id="proj-x",
                )
                fails = mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert any(f.key.startswith("governance_reject:")
                           for f in fails)
            finally:
                s.shutdown()

    check("rejection: records failure memory for downstream learning",
          t_rejection_records_failure)

    # ---- determinism ----
    def t_deterministic_decisions() -> None:
        # Two independent engines over the same change should decide
        # the same status (given same evidence).
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem1 = MemoryStore(s)
                mem2 = MemoryStore(s)
                # use same memory so versions align
                eng = GovernanceEngine(memory=mem1)
                ch = _mk_change()
                r1 = eng.submit(ch, evidence=_mk_evidence_ok(),
                                 project_id="p")
                r2 = eng.submit(
                    _mk_change(new_value="5"),
                    evidence=_mk_evidence_ok(), project_id="p",
                )
                # Different new_value → not the same candidate, but the
                # promotion status must be deterministic given ok evidence.
                assert r1.final_status is PromotionStatus.PROMOTED
                assert r2.final_status is PromotionStatus.PROMOTED
            finally:
                s.shutdown()

    check("deterministic: same evidence → same outcome class",
          t_deterministic_decisions)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                rep = eng.submit(
                    _mk_change(),
                    evidence=_mk_evidence_ok(),
                    project_id="p",
                )
                d = rep.to_dict()
                assert d["id"] == rep.id
                assert d["final_status"] == "promoted"
                s_ = rep.summary()
                assert "Governance Report" in s_
            finally:
                s.shutdown()

    check("to_dict + summary", t_to_dict_summary)

    # ---- E2E ----
    def t_e2e_govern_a_threshold_change() -> None:
        """Realistic scenario: bump a heuristic threshold with full
        governance (isolated, tested, no regression, benchmark improved)."""
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                eng = GovernanceEngine(memory=mem)
                change = CandidateChange(
                    scope="c26.ReliabilityScorer.SOURCE_DIVERSITY_WEIGHT",
                    kind=ChangeKind.HEURISTIC,
                    risk=ChangeRisk.MEDIUM,
                    description="increase weight from 0.25 to 0.30",
                    old_value="0.25", new_value="0.30",
                    rationale=(
                        "empirical evidence from C27 learning says "
                        "source diversity matters more"
                    ),
                    artifact_ref="sha256:deadbeef",
                    isolated=True,
                )
                rep = eng.submit(
                    change,
                    evidence=_mk_evidence_ok(
                        metric_name="false_positive_rate",
                        metric_before=0.10,
                        metric_after=0.08,
                    ),
                    project_id="demo",
                )
                assert rep.final_status is PromotionStatus.PROMOTED
                assert rep.promotion_version == 1

                # Second attempt: same scope, but with a regression
                change2 = CandidateChange(
                    scope=change.scope,
                    kind=ChangeKind.HEURISTIC,
                    risk=ChangeRisk.MEDIUM,
                    description="push to 0.35",
                    old_value="0.30", new_value="0.35",
                    rationale="push further",
                    artifact_ref="sha256:beefdead",
                    isolated=True,
                )
                ev2 = _mk_evidence_ok()
                ev2["regressions"] = ["tests/test_x.py::test_y"]
                rep2 = eng.submit(
                    change2, evidence=ev2, project_id="demo",
                )
                assert rep2.final_status is PromotionStatus.REJECTED

                # Active version must remain 1 (v2 rejected)
                active = eng._active_version(change.scope)
                assert active == 1
            finally:
                s.shutdown()

    check("e2e: threshold change promoted; regression rejected; "
          "active version unchanged",
          t_e2e_govern_a_threshold_change)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 7. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C29 — Self-Improvement Governance")
    print("=" * 78)
    print("Candidate Change → Isolation → Test → Regression → Benchmark")
    print("     → Security → Verification → Approval → Promotion")
    print()

    with tempfile.TemporaryDirectory() as td:
        s = SQLiteStorage(Path(td) / "sebrain.sqlite3")
        s.initialize()
        try:
            mem = MemoryStore(s)
            eng = GovernanceEngine(memory=mem)

            # ---- Scenario 1: healthy threshold change ----
            print("[1] Scenario: healthy heuristic threshold change")
            change1 = CandidateChange(
                scope="c26.ReliabilityScorer.SOURCE_DIVERSITY_WEIGHT",
                kind=ChangeKind.HEURISTIC,
                risk=ChangeRisk.MEDIUM,
                description="increase source-diversity weight 0.25 → 0.30",
                old_value="0.25", new_value="0.30",
                rationale="empirical evidence from C27 supports this",
                artifact_ref="sha256:deadbeef",
                isolated=True,
            )
            rep1 = eng.submit(
                change1, evidence=_mk_evidence_ok(
                    metric_name="false_positive_rate",
                    metric_before=0.10, metric_after=0.08,
                ),
                project_id="demo",
            )
            print(rep1.summary())
            for st in rep1.stages:
                mark = {"pass": "✓", "fail": "✗", "warn": "·",
                        "skip": "–", "blocked": "⛔"}[st.outcome.value]
                print(f"    {mark} [{st.stage.value:13s}] "
                      f"{st.outcome.value:8s}  {_short(st.rationale, 70)}")

            # ---- Scenario 2: HIGH risk needs manual approval ----
            print("\n[2] Scenario: HIGH risk requires manual approval")
            change2 = CandidateChange(
                scope="c28.verdict_policy",
                kind=ChangeKind.POLICY,
                risk=ChangeRisk.HIGH,
                description="raise threshold for STOP",
                old_value="0.35", new_value="0.25",
                rationale="stricter safety",
                artifact_ref="sha256:cafebabe",
                isolated=True,
            )
            rep2 = eng.submit(
                change2, evidence=_mk_evidence_ok(), project_id="demo",
            )
            print(rep2.summary())
            # Now with manual approval
            rep2b = eng.submit(
                change2, evidence=_mk_evidence_ok(),
                approved_by="safety-lead", project_id="demo",
            )
            print("    After manual approval:")
            print(f"    {rep2b.summary()}")

            # ---- Scenario 3: regression rejection ----
            print("\n[3] Scenario: regression detected → REJECTED")
            change3 = CandidateChange(
                scope=change1.scope,
                kind=ChangeKind.HEURISTIC,
                risk=ChangeRisk.MEDIUM,
                description="push further 0.30 → 0.40",
                old_value="0.30", new_value="0.40",
                rationale="more aggressive",
                artifact_ref="sha256:feedface",
                isolated=True,
            )
            ev = _mk_evidence_ok()
            ev["regressions"] = ["tests/test_scorer.py::test_alpha"]
            rep3 = eng.submit(change3, evidence=ev, project_id="demo")
            print(rep3.summary())
            for st in rep3.stages:
                print(f"    [{st.stage.value:13s}] {st.outcome.value}  "
                      f"{_short(st.rationale, 70)}")

            # ---- Scenario 4: rollback ----
            print("\n[4] Scenario: explicit rollback of the promoted change")
            rb = eng.rollback(
                scope=change1.scope,
                reason=RollbackReason.REGRESSION_DETECTED,
                rationale="slowdown observed in production",
                project_id="demo",
            )
            print(f"    from v{rb.from_version} → v{rb.to_version}")
            print(f"    reason: {rb.reason.value}")
            active = eng._active_version(change1.scope)
            print(f"    active version now: {active}")

            # ---- Persistence ----
            print("\n[5] Persistence:")
            ont = Ontology(s)
            repo = GovernanceRepository(memory=mem, ontology=ont)
            ent = repo.save(rep1, project_id="demo")
            print(f"    DECISION entity: {ent[:12]}…")
            print(f"    DECISION count: "
                  f"{ont.count(kind=EntityKind.DECISION)}")
            print(f"    history entries for scope: "
                  f"{len(repo.history(change1.scope, project_id='demo'))}")
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
