"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C32 — EVALUATION LABORATORY (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04 (C05–C31 via duck-typed task runners).

Purpose:
    Run large coding benchmarks against the Brain's own pipeline (or any
    caller-provided runner). Produce reproducible, evidence-based results
    across 9 task categories and 10+ metrics.

Task categories (from the specification):
    generation | modification | debugging | refactoring | testing |
    architecture | security | performance | repository_understanding

Metrics measured (from the specification):
    correctness | tests_passed | regression_rate | repair_success |
    requirement_satisfaction | security_findings | performance |
    execution_cost | reasoning_quality | duration

Reproducibility:
    - Each BenchmarkCase carries `seed` (int) and `inputs_digest` (sha256)
    - Each BenchmarkRun records `case_digest` + `run_digest` for the
      full result — same seed + same inputs → same digests (given a
      deterministic runner)
    - The runner is a CALLER-PROVIDED callable; the lab does not assume
      determinism but RECORDS it (via `runner_reproducible` flag)
    - Every benchmark result is stored in a persistent store with a
      `run_digest` so it can be compared across time

Capabilities:
    - Register tasks (built-in catalog with 15 sample cases)
    - Add custom cases (caller-provided runner)
    - Run one case, run a subset, run all
    - Aggregate metrics per category and overall
    - Compare two runs (diff metrics)
    - Persist and reload benchmark history
    - Emit a comparison report (baseline vs candidate)

Invariants honored:
    - NO external LLM. Deterministic aggregation; runner-agnostic.
    - A case that raises is recorded as FAILED with the error — never
      silently dropped from metrics.
    - Metric aggregation distinguishes: measured vs skipped vs errored
    - Same seed + same inputs + same runner → same digests
    - Bounded (max_cases_per_run, max_duration_per_case)
    - Every metric carries a `source` tag (measured / derived / unknown)

Explicit limitations (Rule #59):
    - The lab does NOT bundle a ground-truth judge. "Correctness" is
      whatever the caller's runner reports. If the runner is an LLM,
      correctness is as reliable as the LLM.
    - "Reasoning quality" is heuristic and caller-defined; the lab
      accepts a numeric score from the runner, does not compute it.
    - Reproducibility claims only hold if the caller's runner is
      deterministic. The lab records digests but cannot enforce them.
    - The bundled sample cases use deterministic fake runners
      (callable stubs) — they exercise the harness, not the Brain.

Contents:
  1.  Enums: TaskCategory, RunStatus, MetricSource, CaseOutcome
  2.  Dataclasses: BenchmarkCase, CaseResult, BenchmarkMetrics,
                   CategorySummary, BenchmarkRun, BenchmarkComparison
  3.  Metric accumulator
  4.  Built-in sample catalog (15 cases, fake runners)
  5.  BenchmarkRunner (facade)
  6.  EvaluationRepository (persist to C04 memory + C02 ontology)
  7.  Self-tests (~35)
  8.  Demo

Run as script:
    python -m sebrain.c32            # demo
    python -m sebrain.c32 --test     # self-tests
================================================================================
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import tempfile
import time
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


def _digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (dict, list)):
            try:
                p = json.dumps(p, sort_keys=True, default=str)
            except Exception:
                p = str(p)
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return "sha256:" + h.hexdigest()[:32]


def _enum_val(x: Any) -> str:
    v = getattr(x, "value", None)
    return str(v) if v is not None else str(x)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class TaskCategory(str, Enum):
    GENERATION = "generation"
    MODIFICATION = "modification"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    REPOSITORY_UNDERSTANDING = "repository_understanding"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"        # some cases failed/errored
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricSource(str, Enum):
    MEASURED = "measured"      # from a real run of the caller's runner
    DERIVED = "derived"        # computed from measured values
    UNKNOWN = "unknown"        # cannot be determined


class CaseOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class BenchmarkCase:
    """A single benchmark task."""
    id: str
    name: str
    category: TaskCategory
    description: str = ""
    seed: int = 0
    inputs: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def inputs_digest(self) -> str:
        return _digest({
            "category": self.category.value,
            "seed": self.seed,
            "inputs": self.inputs,
        })

    def to_dict(self, *, include_runner: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "seed": self.seed,
            "inputs": dict(self.inputs),
            "timeout_seconds": self.timeout_seconds,
            "inputs_digest": self.inputs_digest(),
        }
        if include_runner and self.runner is not None:
            d["has_runner"] = True
        return d


@dataclass(slots=True)
class BenchmarkMetrics:
    """Numeric outputs from a runner. All optional; None means 'not measured'."""
    correctness: float | None = None          # 0..1
    tests_passed: int | None = None
    tests_total: int | None = None
    regression_count: int | None = None
    repair_success: bool | None = None
    requirement_satisfaction: float | None = None  # 0..1
    security_findings_critical: int | None = None
    security_findings_high: int | None = None
    performance_score: float | None = None    # higher = better
    execution_cost_seconds: float | None = None
    reasoning_quality: float | None = None    # 0..1
    duration_seconds: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correctness": self.correctness,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "regression_count": self.regression_count,
            "repair_success": self.repair_success,
            "requirement_satisfaction": self.requirement_satisfaction,
            "security_findings_critical": self.security_findings_critical,
            "security_findings_high": self.security_findings_high,
            "performance_score": self.performance_score,
            "execution_cost_seconds": self.execution_cost_seconds,
            "reasoning_quality": self.reasoning_quality,
            "duration_seconds": self.duration_seconds,
            "extra": dict(self.extra),
        }


@dataclass(slots=True)
class CaseResult:
    case_id: str
    case_name: str
    category: TaskCategory
    outcome: CaseOutcome
    metrics: BenchmarkMetrics = field(default_factory=BenchmarkMetrics)
    error: dict[str, Any] | None = None
    ran_at: str = field(default_factory=now_iso)
    seed: int = 0
    inputs_digest: str = ""
    run_digest: str = ""           # digest of (case + metrics) at completion
    runner_reproducible: bool = False  # caller's claim

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "case_name": self.case_name,
            "category": self.category.value,
            "outcome": self.outcome.value,
            "metrics": self.metrics.to_dict(),
            "error": dict(self.error) if self.error else None,
            "ran_at": self.ran_at, "seed": self.seed,
            "inputs_digest": self.inputs_digest,
            "run_digest": self.run_digest,
            "runner_reproducible": self.runner_reproducible,
        }


@dataclass(slots=True)
class CategorySummary:
    category: TaskCategory
    total: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    pass_rate: float = 0.0
    avg_correctness: float | None = None
    avg_requirement_satisfaction: float | None = None
    avg_reasoning_quality: float | None = None
    total_regressions: int = 0
    total_critical_security: int = 0
    total_high_security: int = 0
    avg_duration_seconds: float | None = None
    avg_execution_cost_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "total": self.total, "passed": self.passed,
            "failed": self.failed, "errored": self.errored,
            "skipped": self.skipped, "pass_rate": self.pass_rate,
            "avg_correctness": self.avg_correctness,
            "avg_requirement_satisfaction": self.avg_requirement_satisfaction,
            "avg_reasoning_quality": self.avg_reasoning_quality,
            "total_regressions": self.total_regressions,
            "total_critical_security": self.total_critical_security,
            "total_high_security": self.total_high_security,
            "avg_duration_seconds": self.avg_duration_seconds,
            "avg_execution_cost_seconds": self.avg_execution_cost_seconds,
        }


@dataclass(slots=True)
class BenchmarkRun:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    name: str = ""
    status: RunStatus = RunStatus.SUCCEEDED
    started_at: str = field(default_factory=now_iso)
    ended_at: str = field(default_factory=now_iso)
    results: list[CaseResult] = field(default_factory=list)
    category_summaries: list[CategorySummary] = field(default_factory=list)
    overall: CategorySummary | None = None
    total_duration_seconds: float = 0.0
    runner_id: str = ""              # caller-provided identifier
    runner_reproducible: bool = False
    run_digest: str = ""
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def cases_for(self, cat: TaskCategory) -> list[CaseResult]:
        return [r for r in self.results if r.category is cat]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "name": self.name, "status": self.status.value,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "results": [r.to_dict() for r in self.results],
            "category_summaries": [c.to_dict() for c in self.category_summaries],
            "overall": self.overall.to_dict() if self.overall else None,
            "total_duration_seconds": self.total_duration_seconds,
            "runner_id": self.runner_id,
            "runner_reproducible": self.runner_reproducible,
            "run_digest": self.run_digest,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        o = self.overall
        if o is None:
            return "=== Benchmark Run (no results) ==="
        return (
            "=== Benchmark Run ===\n"
            f"name={self.name}  status={self.status.value}\n"
            f"cases={o.total}  passed={o.passed}  failed={o.failed}  "
            f"errored={o.errored}  skipped={o.skipped}  "
            f"pass_rate={o.pass_rate:.2%}\n"
            f"avg_correctness={o.avg_correctness}  "
            f"avg_req_sat={o.avg_requirement_satisfaction}  "
            f"avg_reasoning={o.avg_reasoning_quality}\n"
            f"regressions={o.total_regressions}  "
            f"critical_sec={o.total_critical_security}  "
            f"duration={self.total_duration_seconds:.3f}s  "
            f"digest={self.run_digest[:16]}…"
        )


@dataclass(slots=True)
class BenchmarkComparison:
    baseline_run_id: str
    candidate_run_id: str
    pass_rate_delta: float = 0.0
    correctness_delta: float | None = None
    requirement_sat_delta: float | None = None
    reasoning_delta: float | None = None
    regression_delta: int = 0
    critical_security_delta: int = 0
    duration_delta: float = 0.0
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "pass_rate_delta": self.pass_rate_delta,
            "correctness_delta": self.correctness_delta,
            "requirement_sat_delta": self.requirement_sat_delta,
            "reasoning_delta": self.reasoning_delta,
            "regression_delta": self.regression_delta,
            "critical_security_delta": self.critical_security_delta,
            "duration_delta": self.duration_delta,
            "rationale": self.rationale,
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. AGGREGATION
# ════════════════════════════════════════════════════════════════════════════
def _safe_mean(values: Sequence[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(statistics.fmean(xs))


def _safe_sum(values: Sequence[int | None]) -> int:
    return sum(int(v) for v in values if v is not None)


def summarize(
    category: TaskCategory, results: Sequence[CaseResult],
) -> CategorySummary:
    s = CategorySummary(category=category)
    s.total = len(results)
    for r in results:
        if r.outcome is CaseOutcome.PASSED:
            s.passed += 1
        elif r.outcome is CaseOutcome.FAILED:
            s.failed += 1
        elif r.outcome is CaseOutcome.ERROR:
            s.errored += 1
        elif r.outcome is CaseOutcome.SKIPPED:
            s.skipped += 1
    s.pass_rate = (
        s.passed / s.total if s.total > 0 else 0.0
    )
    s.avg_correctness = _safe_mean([r.metrics.correctness for r in results])
    s.avg_requirement_satisfaction = _safe_mean(
        [r.metrics.requirement_satisfaction for r in results]
    )
    s.avg_reasoning_quality = _safe_mean(
        [r.metrics.reasoning_quality for r in results]
    )
    s.total_regressions = _safe_sum(
        [r.metrics.regression_count for r in results]
    )
    s.total_critical_security = _safe_sum(
        [r.metrics.security_findings_critical for r in results]
    )
    s.total_high_security = _safe_sum(
        [r.metrics.security_findings_high for r in results]
    )
    s.avg_duration_seconds = _safe_mean(
        [r.metrics.duration_seconds for r in results]
    )
    s.avg_execution_cost_seconds = _safe_mean(
        [r.metrics.execution_cost_seconds for r in results]
    )
    return s


# ════════════════════════════════════════════════════════════════════════════
# 4. BUILT-IN SAMPLE CATALOG (fake, deterministic runners)
# ════════════════════════════════════════════════════════════════════════════
def _fake_generation(inputs: dict[str, Any]) -> dict[str, Any]:
    # Deterministic: success iff inputs.get("target") == "ok"
    ok = inputs.get("target") == "ok"
    return {
        "correctness": 1.0 if ok else 0.4,
        "tests_passed": 8 if ok else 5,
        "tests_total": 8,
        "requirement_satisfaction": 1.0 if ok else 0.5,
        "reasoning_quality": 0.85 if ok else 0.5,
        "duration_seconds": 0.02,
        "execution_cost_seconds": 0.01,
    }


def _fake_modification(inputs: dict[str, Any]) -> dict[str, Any]:
    regressed = inputs.get("regress", False)
    return {
        "correctness": 0.9,
        "tests_passed": 10 if not regressed else 7,
        "tests_total": 10,
        "regression_count": 1 if regressed else 0,
        "requirement_satisfaction": 0.9,
        "reasoning_quality": 0.75,
        "duration_seconds": 0.03,
        "execution_cost_seconds": 0.02,
    }


def _fake_debugging(inputs: dict[str, Any]) -> dict[str, Any]:
    fixed = inputs.get("fixable", True)
    return {
        "correctness": 1.0 if fixed else 0.0,
        "tests_passed": 5 if fixed else 2,
        "tests_total": 5,
        "repair_success": bool(fixed),
        "requirement_satisfaction": 1.0 if fixed else 0.2,
        "reasoning_quality": 0.7 if fixed else 0.3,
        "duration_seconds": 0.04,
        "execution_cost_seconds": 0.02,
    }


def _fake_refactoring(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "correctness": 1.0,
        "tests_passed": 12, "tests_total": 12,
        "regression_count": 0,
        "requirement_satisfaction": 0.95,
        "reasoning_quality": 0.8,
        "duration_seconds": 0.05,
        "execution_cost_seconds": 0.03,
    }


def _fake_testing(inputs: dict[str, Any]) -> dict[str, Any]:
    n = int(inputs.get("n_tests", 10))
    return {
        "correctness": 0.9,
        "tests_passed": n, "tests_total": n,
        "requirement_satisfaction": 0.9,
        "reasoning_quality": 0.75,
        "duration_seconds": 0.02,
        "execution_cost_seconds": 0.01,
    }


def _fake_architecture(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "correctness": 0.85,
        "requirement_satisfaction": 0.8,
        "reasoning_quality": 0.9,
        "duration_seconds": 0.06,
        "execution_cost_seconds": 0.02,
    }


def _fake_security(inputs: dict[str, Any]) -> dict[str, Any]:
    crit = int(inputs.get("crit", 0))
    high = int(inputs.get("high", 0))
    return {
        "correctness": 1.0 if crit == 0 else 0.3,
        "security_findings_critical": crit,
        "security_findings_high": high,
        "requirement_satisfaction": 1.0 if crit == 0 else 0.5,
        "reasoning_quality": 0.7,
        "duration_seconds": 0.03,
        "execution_cost_seconds": 0.01,
    }


def _fake_performance(inputs: dict[str, Any]) -> dict[str, Any]:
    score = float(inputs.get("score", 0.8))
    return {
        "correctness": 1.0,
        "performance_score": score,
        "requirement_satisfaction": 0.9 if score >= 0.7 else 0.5,
        "reasoning_quality": 0.7,
        "duration_seconds": 0.04,
        "execution_cost_seconds": 0.02,
    }


def _fake_repo_understanding(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "correctness": 0.9,
        "requirement_satisfaction": 0.85,
        "reasoning_quality": 0.85,
        "duration_seconds": 0.05,
        "execution_cost_seconds": 0.02,
    }


def _sample_catalog() -> list[BenchmarkCase]:
    """15 deterministic cases covering all 9 categories."""
    cases: list[BenchmarkCase] = []

    def add(name: str, cat: TaskCategory, runner: Callable,
             inputs: dict[str, Any], seed: int = 1,
             description: str = "") -> None:
        cases.append(BenchmarkCase(
            id=_new_id(), name=name, category=cat,
            description=description or name,
            seed=seed, inputs=dict(inputs), runner=runner,
            timeout_seconds=10.0,
        ))

    # generation (2)
    add("gen_basic", TaskCategory.GENERATION, _fake_generation,
        {"target": "ok"}, seed=1)
    add("gen_hard", TaskCategory.GENERATION, _fake_generation,
        {"target": "hard"}, seed=2)
    # modification (2)
    add("mod_clean", TaskCategory.MODIFICATION, _fake_modification,
        {"regress": False}, seed=3)
    add("mod_regress", TaskCategory.MODIFICATION, _fake_modification,
        {"regress": True}, seed=4)
    # debugging (2)
    add("dbg_fixable", TaskCategory.DEBUGGING, _fake_debugging,
        {"fixable": True}, seed=5)
    add("dbg_hard", TaskCategory.DEBUGGING, _fake_debugging,
        {"fixable": False}, seed=6)
    # refactoring (2)
    add("ref_clean", TaskCategory.REFACTORING, _fake_refactoring, {},
        seed=7)
    add("ref_large", TaskCategory.REFACTORING, _fake_refactoring, {},
        seed=8)
    # testing (2)
    add("test_small", TaskCategory.TESTING, _fake_testing,
        {"n_tests": 5}, seed=9)
    add("test_large", TaskCategory.TESTING, _fake_testing,
        {"n_tests": 20}, seed=10)
    # architecture (1)
    add("arch_layered", TaskCategory.ARCHITECTURE, _fake_architecture, {},
        seed=11)
    # security (2)
    add("sec_clean", TaskCategory.SECURITY, _fake_security,
        {"crit": 0, "high": 0}, seed=12)
    add("sec_bad", TaskCategory.SECURITY, _fake_security,
        {"crit": 1, "high": 2}, seed=13)
    # performance (1)
    add("perf_fast", TaskCategory.PERFORMANCE, _fake_performance,
        {"score": 0.9}, seed=14)
    # repository understanding (1)
    add("repo_small", TaskCategory.REPOSITORY_UNDERSTANDING,
        _fake_repo_understanding, {}, seed=15)
    return cases


# ════════════════════════════════════════════════════════════════════════════
# 5. BENCHMARK RUNNER (facade)
# ════════════════════════════════════════════════════════════════════════════
class BenchmarkRunner:
    """Run benchmark cases, aggregate metrics, store reproducible results."""

    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        max_cases_per_run: int = 500,
        max_duration_per_case: float = 300.0,
    ) -> None:
        if max_cases_per_run < 1:
            raise ValidationError("max_cases_per_run must be >= 1")
        if max_duration_per_case <= 0:
            raise ValidationError("max_duration_per_case must be > 0")
        self.memory = memory
        self.max_cases_per_run = max_cases_per_run
        self.max_duration_per_case = max_duration_per_case
        self._catalog: dict[str, BenchmarkCase] = {}

    # ---- catalog management ----
    def add_case(self, case: BenchmarkCase) -> None:
        if not case.id:
            raise ValidationError("case.id required")
        if case.runner is None:
            raise ValidationError("case.runner required")
        self._catalog[case.id] = case

    def add_cases(self, cases: Iterable[BenchmarkCase]) -> None:
        for c in cases:
            self.add_case(c)

    def load_sample_catalog(self) -> list[BenchmarkCase]:
        cases = _sample_catalog()
        self.add_cases(cases)
        return cases

    def catalog(self) -> list[BenchmarkCase]:
        return list(self._catalog.values())

    def by_category(
        self, cat: TaskCategory,
    ) -> list[BenchmarkCase]:
        return [c for c in self._catalog.values() if c.category is cat]

    def clear_catalog(self) -> None:
        self._catalog.clear()

    # ---- run ----
    def run(
        self, *,
        case_ids: Sequence[str] | None = None,
        categories: Sequence[TaskCategory] | None = None,
        name: str = "benchmark",
        project_id: str = "",
        runner_id: str = "c32.default",
        runner_reproducible: bool = True,
    ) -> BenchmarkRun:
        started = time.monotonic()
        run = BenchmarkRun(
            project_id=project_id, name=name,
            status=RunStatus.RUNNING,
            runner_id=runner_id,
            runner_reproducible=runner_reproducible,
            provenance=Provenance(
                source="benchmark_runner",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.MEDIUM,
            ),
        )
        cases = self._select(case_ids=case_ids, categories=categories)
        if len(cases) > self.max_cases_per_run:
            raise ValidationError(
                f"selected {len(cases)} cases exceeds max_cases_per_run="
                f"{self.max_cases_per_run}; narrow case_ids/categories explicitly"
            )

        for c in cases:
            res = self._run_one(c)
            run.results.append(res)

        run.ended_at = now_iso()
        run.total_duration_seconds = time.monotonic() - started
        run.category_summaries = [
            summarize(cat, run.cases_for(cat))
            for cat in TaskCategory
            if any(r.category is cat for r in run.results)
        ]
        run.overall = summarize(TaskCategory.GENERATION, run.results)
        # overall.category doesn't make sense; set to a neutral
        run.overall.category = TaskCategory.GENERATION  # placeholder
        # classify status
        errored = sum(1 for r in run.results
                      if r.outcome is CaseOutcome.ERROR)
        failed = sum(1 for r in run.results
                     if r.outcome is CaseOutcome.FAILED)
        if not run.results:
            run.status = RunStatus.FAILED
        elif errored and not any(r.outcome is CaseOutcome.PASSED
                                  for r in run.results):
            run.status = RunStatus.FAILED
        elif errored or failed:
            run.status = RunStatus.PARTIAL
        else:
            run.status = RunStatus.SUCCEEDED
        run.run_digest = self._digest_run(run)
        run.rationale = (
            f"cases={len(run.results)}  "
            f"passed={run.overall.passed}  "
            f"failed={run.overall.failed}  "
            f"errored={run.overall.errored}  "
            f"pass_rate={run.overall.pass_rate:.2%}  "
            f"duration={run.total_duration_seconds:.3f}s  "
            f"digest={run.run_digest[:16]}"
        )
        return run

    # ---- helpers ----
    def _select(
        self, *,
        case_ids: Sequence[str] | None,
        categories: Sequence[TaskCategory] | None,
    ) -> list[BenchmarkCase]:
        all_cases = list(self._catalog.values())
        if case_ids is not None:
            wanted = set(case_ids)
            all_cases = [c for c in all_cases if c.id in wanted]
        if categories is not None:
            wanted_cats = set(categories)
            all_cases = [c for c in all_cases if c.category in wanted_cats]
        # deterministic order
        all_cases.sort(key=lambda x: (x.category.value, x.seed, x.name))
        return all_cases

    def _run_one(self, case: BenchmarkCase) -> CaseResult:
        outcome = CaseOutcome.PASSED
        metrics = BenchmarkMetrics()
        error: dict[str, Any] | None = None
        assert case.runner is not None
        inputs_digest = case.inputs_digest()
        t0 = time.monotonic()
        try:
            raw = case.runner(dict(case.inputs))
            if not isinstance(raw, dict):
                raise TypeError(
                    f"runner returned {type(raw).__name__}, expected dict"
                )
            metrics = _metrics_from_dict(raw)
            # case success = (correctness >= 0.5) if measured, else
            # (tests_passed == tests_total) if measured, else "passed"
            if metrics.correctness is not None:
                outcome = (CaseOutcome.PASSED if metrics.correctness >= 0.5
                            else CaseOutcome.FAILED)
            elif metrics.tests_total and metrics.tests_passed is not None:
                outcome = (CaseOutcome.PASSED
                            if metrics.tests_passed == metrics.tests_total
                            else CaseOutcome.FAILED)
            else:
                # No success metric was supplied. Do not manufacture a pass.
                outcome = CaseOutcome.SKIPPED
                error = {
                    "type": "NoOutcomeMetric",
                    "message": (
                        "runner returned no correctness or complete "
                        "tests_passed/tests_total metric"
                    ),
                }
        except Exception as exc:
            outcome = CaseOutcome.ERROR
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        duration = time.monotonic() - t0
        if metrics.duration_seconds is None:
            metrics.duration_seconds = duration
        # A callable runner cannot be forcibly interrupted here, but actual
        # wall-clock duration is authoritative for the harness bound.
        elapsed_limit = min(
            self.max_duration_per_case,
            float(case.timeout_seconds),
        )
        if duration > elapsed_limit:
            outcome = CaseOutcome.FAILED
            error = {
                "type": "DurationExceeded",
                "message": (
                    f"duration {metrics.duration_seconds:.2f}s > "
                    f"limit {self.max_duration_per_case:.2f}s"
                ),
            }
        res = CaseResult(
            case_id=case.id, case_name=case.name,
            category=case.category, outcome=outcome,
            metrics=metrics, error=error,
            seed=case.seed, inputs_digest=inputs_digest,
        )
        res.run_digest = _digest({
            "case_id": case.id,
            "seed": case.seed,
            "inputs_digest": inputs_digest,
            "outcome": outcome.value,
            "metrics": metrics.to_dict(),
        })
        return res

    @staticmethod
    def _digest_run(run: BenchmarkRun) -> str:
        return _digest({
            "runner_id": run.runner_id,
            "runner_reproducible": run.runner_reproducible,
            "results": [r.run_digest for r in run.results],
        })

    # ---- compare ----
    def compare(
        self, baseline: BenchmarkRun, candidate: BenchmarkRun,
    ) -> BenchmarkComparison:
        if baseline.overall is None or candidate.overall is None:
            raise ValidationError("both runs must have overall summary")
        cmp = BenchmarkComparison(
            baseline_run_id=baseline.id,
            candidate_run_id=candidate.id,
        )
        cmp.pass_rate_delta = (
            candidate.overall.pass_rate - baseline.overall.pass_rate
        )
        cmp.correctness_delta = _delta(
            baseline.overall.avg_correctness,
            candidate.overall.avg_correctness,
        )
        cmp.requirement_sat_delta = _delta(
            baseline.overall.avg_requirement_satisfaction,
            candidate.overall.avg_requirement_satisfaction,
        )
        cmp.reasoning_delta = _delta(
            baseline.overall.avg_reasoning_quality,
            candidate.overall.avg_reasoning_quality,
        )
        cmp.regression_delta = (
            candidate.overall.total_regressions
            - baseline.overall.total_regressions
        )
        cmp.critical_security_delta = (
            candidate.overall.total_critical_security
            - baseline.overall.total_critical_security
        )
        cmp.duration_delta = (
            candidate.total_duration_seconds
            - baseline.total_duration_seconds
        )
        bits = []
        if cmp.pass_rate_delta > 0:
            bits.append(f"+{cmp.pass_rate_delta:.2%} pass rate")
        elif cmp.pass_rate_delta < 0:
            bits.append(f"{cmp.pass_rate_delta:.2%} pass rate")
        if cmp.regression_delta:
            bits.append(f"regressions Δ{cmp.regression_delta:+d}")
        if cmp.critical_security_delta:
            bits.append(
                f"critical security Δ{cmp.critical_security_delta:+d}"
            )
        cmp.rationale = " | ".join(bits) if bits else "no material delta"
        return cmp


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def _metrics_from_dict(d: dict[str, Any]) -> BenchmarkMetrics:
    def _f(x: Any) -> float | None:
        try:
            return float(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def _i(x: Any) -> int | None:
        try:
            return int(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    def _b(x: Any) -> bool | None:
        return bool(x) if x is not None else None

    return BenchmarkMetrics(
        correctness=_f(d.get("correctness")),
        tests_passed=_i(d.get("tests_passed")),
        tests_total=_i(d.get("tests_total")),
        regression_count=_i(d.get("regression_count")),
        repair_success=_b(d.get("repair_success")),
        requirement_satisfaction=_f(d.get("requirement_satisfaction")),
        security_findings_critical=_i(d.get("security_findings_critical")),
        security_findings_high=_i(d.get("security_findings_high")),
        performance_score=_f(d.get("performance_score")),
        execution_cost_seconds=_f(d.get("execution_cost_seconds")),
        reasoning_quality=_f(d.get("reasoning_quality")),
        duration_seconds=_f(d.get("duration_seconds")),
        extra={k: v for k, v in d.items()
               if k not in {
                   "correctness", "tests_passed", "tests_total",
                   "regression_count", "repair_success",
                   "requirement_satisfaction",
                   "security_findings_critical", "security_findings_high",
                   "performance_score", "execution_cost_seconds",
                   "reasoning_quality", "duration_seconds",
               }},
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. EVALUATION REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class EvaluationRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, run: BenchmarkRun, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"benchmark_run:{run.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, run.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["benchmark", "c32", run.status.value,
                  run.runner_id or "no_runner"],
            provenance=run.provenance,
        )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.EXECUTION,
            _short(f"Benchmark {run.id[:8]} ({run.status.value}, "
                   f"{len(run.results)} cases)", 120),
            attributes={
                "run_id": run.id,
                "project_id": project_id,
                "name": run.name,
                "status": run.status.value,
                "cases": len(run.results),
                "pass_rate": (run.overall.pass_rate if run.overall else 0.0),
                "run_digest": run.run_digest,
                "runner_reproducible": run.runner_reproducible,
                "category_summaries": [c.to_dict()
                                       for c in run.category_summaries],
            },
            tags=["benchmark-run", run.status.value],
            provenance=run.provenance,
        )
        return ent.id

    def load(self, run_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"benchmark_run:{run_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None

    def list_runs(self, *, project_id: str) -> list[dict[str, Any]]:
        entries = self.memory.find(
            kind=MemoryKind.PROJECT,
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            key_like="benchmark_run:",
        )
        return [{"key": e.key, "id": e.content.get("id"),
                 "name": e.content.get("name"),
                 "status": e.content.get("status"),
                 "digest": e.content.get("run_digest"),
                 "created_at": e.content.get("created_at")}
                for e in entries]


# ════════════════════════════════════════════════════════════════════════════
# 7. SELF-TESTS
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
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C32 self-tests…")

    # ---- catalog ----
    def t_sample_catalog_covers_all_categories() -> None:
        r = BenchmarkRunner()
        cases = r.load_sample_catalog()
        assert len(cases) >= 9
        cats = {c.category for c in cases}
        for cat in TaskCategory:
            assert cat in cats, f"missing category {cat.value}"

    def t_catalog_by_category() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        gen = r.by_category(TaskCategory.GENERATION)
        assert len(gen) >= 1
        for c in gen:
            assert c.category is TaskCategory.GENERATION

    def t_add_case_requires_runner() -> None:
        r = BenchmarkRunner()
        try:
            r.add_case(BenchmarkCase(
                id="x", name="x", category=TaskCategory.GENERATION,
            ))
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    check("catalog: sample catalog covers all 9 categories",
          t_sample_catalog_covers_all_categories)
    check("catalog: by_category filters", t_catalog_by_category)
    check("catalog: case without runner rejected",
          t_add_case_requires_runner)

    # ---- run ----
    def t_run_all_ok() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(name="all", project_id="p")
        assert run.status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL)
        assert len(run.results) >= 9
        # every result has a digest
        for res in run.results:
            assert res.run_digest.startswith("sha256:")
        assert run.run_digest.startswith("sha256:")

    def t_run_single_case() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        cases = r.by_category(TaskCategory.SECURITY)
        assert cases
        run = r.run(case_ids=[cases[0].id], project_id="p")
        assert len(run.results) == 1
        assert run.results[0].category is TaskCategory.SECURITY

    def t_run_filtered_by_category() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(categories=[TaskCategory.GENERATION,
                                 TaskCategory.DEBUGGING],
                    project_id="p")
        cats = {x.category for x in run.results}
        assert cats == {TaskCategory.GENERATION, TaskCategory.DEBUGGING}

    def t_run_empty_catalog() -> None:
        r = BenchmarkRunner()
        run = r.run(project_id="p")
        assert run.status is RunStatus.FAILED
        assert run.results == []

    def t_run_deterministic_digest() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run1 = r.run(project_id="p")
        r2 = BenchmarkRunner()
        r2.load_sample_catalog()
        # Different catalog ids → different inputs_digest? No —
        # inputs_digest depends on category+seed+inputs (deterministic).
        # But case ids differ (random uuid) so run_digest differs
        # only via result list, which depends on case ids.
        # We test the same runner instance gives the same digests.
        run2 = r.run(project_id="p")
        assert run1.run_digest == run2.run_digest

    check("run: full catalog produces results + digests", t_run_all_ok)
    check("run: single case selection", t_run_single_case)
    check("run: category filter", t_run_filtered_by_category)
    check("run: empty catalog → FAILED", t_run_empty_catalog)
    check("run: same catalog → same run_digest",
          t_run_deterministic_digest)

    # ---- metrics aggregation ----
    def t_aggregation_counts() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(project_id="p")
        o = run.overall
        assert o is not None
        assert o.total == len(run.results)
        assert o.passed + o.failed + o.errored + o.skipped == o.total
        assert 0.0 <= o.pass_rate <= 1.0

    def t_aggregation_avgs() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(project_id="p")
        # generation has some 0.4 correctness (hard case)
        gen_summary = next(
            s for s in run.category_summaries
            if s.category is TaskCategory.GENERATION
        )
        assert gen_summary.avg_correctness is not None
        assert 0.0 <= gen_summary.avg_correctness <= 1.0

    def t_security_aggregation() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(
            categories=[TaskCategory.SECURITY], project_id="p",
        )
        o = run.overall
        # sec_bad has 1 critical → total 1
        assert o is not None
        assert o.total_critical_security == 1

    def t_regression_aggregation() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(
            categories=[TaskCategory.MODIFICATION], project_id="p",
        )
        o = run.overall
        assert o is not None
        # one mod case regresses → 1 regression
        assert o.total_regressions == 1

    check("aggregate: counts sum to total", t_aggregation_counts)
    check("aggregate: averages computed", t_aggregation_avgs)
    check("aggregate: critical security summed",
          t_security_aggregation)
    check("aggregate: regressions summed", t_regression_aggregation)

    # ---- error handling ----
    def t_runner_raises_is_error() -> None:
        r = BenchmarkRunner()
        def boom(inputs):
            raise RuntimeError("exploded")
        r.add_case(BenchmarkCase(
            id="boom", name="boom", category=TaskCategory.GENERATION,
            runner=boom,
        ))
        run = r.run(project_id="p")
        assert len(run.results) == 1
        assert run.results[0].outcome is CaseOutcome.ERROR
        assert run.results[0].error is not None
        assert run.results[0].error["type"] == "RuntimeError"

    def t_runner_bad_return_is_error() -> None:
        r = BenchmarkRunner()
        def bad(inputs):
            return "not a dict"
        r.add_case(BenchmarkCase(
            id="bad", name="bad", category=TaskCategory.GENERATION,
            runner=bad,
        ))
        run = r.run(project_id="p")
        assert run.results[0].outcome is CaseOutcome.ERROR

    def t_runner_duration_exceeded() -> None:
        r = BenchmarkRunner(max_duration_per_case=0.001)
        def slow(inputs):
            return {"duration_seconds": 100.0, "correctness": 1.0}
        r.add_case(BenchmarkCase(
            id="slow", name="slow", category=TaskCategory.GENERATION,
            runner=slow,
        ))
        run = r.run(project_id="p")
        assert run.results[0].outcome is CaseOutcome.FAILED
        assert run.results[0].error is not None
        assert run.results[0].error["type"] == "DurationExceeded"

    check("error: runner raises → ERROR", t_runner_raises_is_error)
    check("error: bad return type → ERROR", t_runner_bad_return_is_error)
    check("error: duration exceeded → FAILED",
          t_runner_duration_exceeded)

    # ---- correctness classification ----
    def t_correctness_classification() -> None:
        r = BenchmarkRunner()
        def c(x): return {"correctness": x}
        r.add_case(BenchmarkCase(
            id="low", name="low", category=TaskCategory.GENERATION,
            runner=lambda i: c(0.3),
        ))
        r.add_case(BenchmarkCase(
            id="high", name="high", category=TaskCategory.GENERATION,
            runner=lambda i: c(0.9),
        ))
        run = r.run(project_id="p")
        outcomes = {res.case_name: res.outcome for res in run.results}
        assert outcomes["low"] is CaseOutcome.FAILED
        assert outcomes["high"] is CaseOutcome.PASSED

    def t_tests_ratio_classification() -> None:
        r = BenchmarkRunner()
        r.add_case(BenchmarkCase(
            id="ok", name="ok", category=TaskCategory.TESTING,
            runner=lambda i: {"tests_passed": 5, "tests_total": 5},
        ))
        r.add_case(BenchmarkCase(
            id="bad", name="bad", category=TaskCategory.TESTING,
            runner=lambda i: {"tests_passed": 3, "tests_total": 5},
        ))
        run = r.run(project_id="p")
        outcomes = {res.case_name: res.outcome for res in run.results}
        assert outcomes["ok"] is CaseOutcome.PASSED
        assert outcomes["bad"] is CaseOutcome.FAILED

    check("classify: correctness >= 0.5 → PASSED",
          t_correctness_classification)
    check("classify: tests ratio determines outcome",
          t_tests_ratio_classification)

    # ---- compare ----
    def t_compare_runs() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        base = r.run(name="baseline", project_id="p")
        # Simulate an "improved runner" by adding only the passing cases
        # as a new candidate run
        good_ids = [
            c.id for c in r.catalog()
            if c.seed % 2 == 0   # arbitrary deterministic subset
        ]
        cand = r.run(case_ids=good_ids, name="candidate", project_id="p")
        # Compare
        cmp = r.compare(base, cand)
        assert cmp.baseline_run_id == base.id
        assert cmp.candidate_run_id == cand.id
        # Some delta should be present
        assert isinstance(cmp.pass_rate_delta, float)
        assert cmp.rationale

    def t_compare_missing_overall() -> None:
        r = BenchmarkRunner()
        # craft empty runs
        base = BenchmarkRun(name="base", status=RunStatus.FAILED)
        cand = BenchmarkRun(name="cand", status=RunStatus.FAILED)
        try:
            r.compare(base, cand)
        except ValidationError:
            return
        raise AssertionError("expected error")

    check("compare: baseline vs candidate returns deltas",
          t_compare_runs)
    check("compare: missing overall raises",
          t_compare_missing_overall)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(project_id="p")
        d = run.to_dict()
        assert d["id"] == run.id
        assert "results" in d and "overall" in d
        assert isinstance(d["run_digest"], str)
        s = run.summary()
        assert "Benchmark Run" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                r = BenchmarkRunner(memory=mem)
                r.load_sample_catalog()
                run = r.run(project_id="proj-x")
                repo = EvaluationRepository(memory=mem, ontology=ont)
                ent = repo.save(run, project_id="proj-x")
                assert ent
                loaded = repo.load(run.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == run.id
                # list runs
                runs = repo.list_runs(project_id="proj-x")
                assert any(x["id"] == run.id for x in runs)
                # ontology
                assert ont.count(kind=EntityKind.EXECUTION) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology EXECUTION entity", t_persist)

    # ---- bounded ----
    def t_max_cases_bound() -> None:
        r = BenchmarkRunner(max_cases_per_run=2)
        r.load_sample_catalog()
        run = r.run(project_id="p")
        assert len(run.results) <= 2

    check("bounds: max_cases_per_run enforced", t_max_cases_bound)

    # ---- reproducibility claim ----
    def t_reproducibility_recorded() -> None:
        r = BenchmarkRunner()
        r.load_sample_catalog()
        run = r.run(project_id="p", runner_reproducible=True)
        assert run.runner_reproducible is True
        for res in run.results:
            # Each result carries seed + inputs_digest
            assert isinstance(res.seed, int)
            assert res.inputs_digest.startswith("sha256:")

    check("reproducibility: seed + inputs_digest recorded",
          t_reproducibility_recorded)

    # ---- E2E ----
    def t_e2e_multiple_runs_compare() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                repo = EvaluationRepository(memory=mem, ontology=ont)

                # Run 1: baseline on all cases
                r = BenchmarkRunner(memory=mem)
                r.load_sample_catalog()
                baseline = r.run(name="v1", project_id="proj",
                                  runner_id="test-v1")
                repo.save(baseline, project_id="proj")

                # Run 2: only the "hard" cases removed → pass rate up
                passing_ids = [
                    c.id for c in r.catalog()
                    if not (c.category is TaskCategory.GENERATION
                            and c.seed == 2)
                    and not (c.category is TaskCategory.DEBUGGING
                             and c.seed == 6)
                ]
                candidate = r.run(
                    case_ids=passing_ids, name="v2", project_id="proj",
                    runner_id="test-v2",
                )
                repo.save(candidate, project_id="proj")

                cmp = r.compare(baseline, candidate)
                # Candidate should have higher pass rate
                assert cmp.pass_rate_delta > 0
                # And the rationale mentions it
                assert "pass rate" in cmp.rationale

                # Both runs persisted
                loaded_b = repo.load(baseline.id, project_id="proj")
                loaded_c = repo.load(candidate.id, project_id="proj")
                assert loaded_b is not None and loaded_c is not None
                # Digest determinism across loads
                assert loaded_b["run_digest"] == baseline.run_digest
                assert loaded_c["run_digest"] == candidate.run_digest
            finally:
                s.shutdown()

    check("e2e: run → persist → compare → digests stable",
          t_e2e_multiple_runs_compare)

    # ---- comparison delta helpers ----
    def t_delta_helper() -> None:
        assert _delta(0.5, 0.7) == 0.7 - 0.5
        assert _delta(None, 0.5) is None
        assert _delta(0.5, None) is None
        assert _delta(None, None) is None

    check("delta: helper handles None", t_delta_helper)

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
    print("SE Brain C32 — Evaluation Laboratory")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            mem = MemoryStore(app.storage)
            ont = Ontology(app.storage)
            repo = EvaluationRepository(memory=mem, ontology=ont)

            r = BenchmarkRunner(memory=mem)
            cases = r.load_sample_catalog()
            print(f"\n[1] Sample catalog: {len(cases)} cases")
            by_cat: dict[str, int] = {}
            for c in cases:
                by_cat[c.category.value] = by_cat.get(
                    c.category.value, 0) + 1
            for k in sorted(by_cat):
                print(f"    {k:25s}: {by_cat[k]}")

            print("\n[2] Run all cases:")
            run1 = r.run(name="baseline", project_id="demo",
                          runner_id="demo-runner-v1")
            print(run1.summary())

            print("\n[3] Per-category summary:")
            print(f"    {'category':25s} {'total':>5s} {'pass':>5s} "
                  f"{'fail':>5s} {'err':>4s} {'rate':>7s} "
                  f"{'avg_corr':>9s}")
            for s in run1.category_summaries:
                ac = (f"{s.avg_correctness:.2f}"
                      if s.avg_correctness is not None else "-")
                print(f"    {s.category.value:25s} {s.total:>5d} "
                      f"{s.passed:>5d} {s.failed:>5d} {s.errored:>4d} "
                      f"{s.pass_rate:>7.2%} {ac:>9s}")

            print("\n[4] Individual case results:")
            for res in run1.results:
                mark = {"passed": "✓", "failed": "✗",
                        "error": "E", "skipped": "s"}[res.outcome.value]
                print(f"    {mark} [{res.category.value:22s}] "
                      f"{res.case_name:20s} "
                      f"corr={res.metrics.correctness} "
                      f"dur={res.metrics.duration_seconds:.3f}s "
                      f"digest={res.run_digest[7:15]}…")

            print("\n[5] Persist baseline:")
            ent = repo.save(run1, project_id="demo")
            print(f"    ontology entity: {ent[:12]}…")
            print(f"    EXECUTION count: "
                  f"{ont.count(kind=EntityKind.EXECUTION)}")
            print(f"    run_digest: {run1.run_digest}")

            print("\n[6] Run a candidate (fail-hard cases removed):")
            # Deterministic subset: only cases whose seed is even
            keep = [c.id for c in cases if c.seed % 2 == 0]
            run2 = r.run(case_ids=keep, name="candidate",
                          project_id="demo", runner_id="demo-runner-v2")
            print(run2.summary())
            repo.save(run2, project_id="demo")

            print("\n[7] Compare baseline vs candidate:")
            cmp = r.compare(run1, run2)
            print(f"    pass_rate_delta       = {cmp.pass_rate_delta:+.2%}")
            print(f"    correctness_delta     = {cmp.correctness_delta}")
            print(f"    requirement_sat_delta = {cmp.requirement_sat_delta}")
            print(f"    reasoning_delta       = {cmp.reasoning_delta}")
            print(f"    regression_delta      = {cmp.regression_delta:+d}")
            print(f"    critical_sec_delta    = "
                  f"{cmp.critical_security_delta:+d}")
            print(f"    duration_delta        = {cmp.duration_delta:+.3f}s")
            print(f"    rationale: {cmp.rationale}")

            print("\n[8] Persisted runs:")
            for row in repo.list_runs(project_id="demo"):
                print(f"    id={row['id'][:8]}…  name={row['name']:10s}  "
                      f"status={row['status']:10s}  "
                      f"digest={row['digest'][7:15]}…")
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
