"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C11 — AGENT ORCHESTRATOR (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C08 (plans), C10 (architecture).

Purpose:
    Execute a Plan (from C08) by selecting agents, assigning tasks,
    managing dependencies, running sequential or parallel work,
    collecting outputs, evaluating results, retrying safely,
    escalating failures, invoking verification gates, and preserving
    a complete execution history.

Capabilities:
    - Interpret plans (topological order + parallel groups)
    - Select agents by capability match (TaskKind → Agent)
    - Assign tasks, track per-task state
    - Manage dependencies (ready-set expands as tasks succeed)
    - Sequential OR wave-parallel execution (ThreadPoolExecutor)
    - Retry with bounded exponential backoff
    - Circuit-breaker / failure escalation
    - Fail-fast or best-effort modes
    - Verification gate evaluation at milestone boundaries
    - Rollback-point triggering on gate failure
    - Full execution history (per-attempt records)

Invariants honored:
  - Deterministic scheduling (topo order + lexicographic tie-break)
  - No unbounded retries (max_retries + max_wall_time)
  - No unbounded parallelism (max_workers)
  - Every attempt produces an ExecutionRecord
  - Every failure carries an error dict (code + message)
  - Agents are workers, not independent brains — no cross-agent calls
  - Orchestration is fully testable with stub agents
  - Persistence to C04 memory + C02 ontology (EXECUTION entity)

Contents:
  1.  Enums: AssignmentState, OrchestrationStatus, ParallelismMode,
             EscalationReason
  2.  Dataclasses: AgentCapability, AgentResult, TaskAssignment,
                   ExecutionRecord, GateResult, OrchestrationResult
  3.  Agent Protocol + ScriptedAgent (for tests)
  4.  AgentRegistry (capability-matched lookup)
  5.  RetryPolicy (bounded backoff)
  6.  GateEvaluator (4 gate conditions from C08)
  7.  Orchestrator (main loop)
  8.  OrchestrationRepository (persist / reload)
  9.  __main__ demo + self-tests

Run as script:
    python -m sebrain.c11            # demo
    python -m sebrain.c11 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

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
from sebrain.c05 import RequirementParser
from sebrain.c06 import IntentContextEngine
from sebrain.c08 import (
    GateCondition,
    MilestonePhase,
    Plan,
    Planner,
    Task,
    TaskKind,
    VerificationGate,
)
from sebrain.c09 import TechnologySelector
from sebrain.c10 import ArchitectureReasoner


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
class AssignmentState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class OrchestrationStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    ABORTED = "aborted"          # escalation tripped


class ParallelismMode(str, Enum):
    SEQUENTIAL = "sequential"
    WAVE = "wave"                # level-wise parallel via ThreadPoolExecutor


class EscalationReason(str, Enum):
    NONE = "none"
    TASK_FAILED_ALL_RETRIES = "task_failed_all_retries"
    CRITICAL_GATE_FAILED = "critical_gate_failed"
    ROLLBACK_TRIGGERED = "rollback_triggered"
    WALL_TIME_EXCEEDED = "wall_time_exceeded"
    NO_AGENT_FOR_TASK = "no_agent_for_task"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class AgentCapability:
    """What an agent can do. Matched against Task.kind."""
    task_kinds: tuple[TaskKind, ...]
    description: str = ""

    def matches(self, task_kind: TaskKind) -> bool:
        return task_kind in self.task_kinds


@dataclass(slots=True)
class AgentResult:
    """What an agent returns from executing a task."""
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    confidence: Confidence = Confidence.MEDIUM

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("successful AgentResult cannot carry an error")
        if not self.success and self.error is None:
            self.error = {"code": "UNKNOWN", "message": "agent failed"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": dict(self.output),
            "error": dict(self.error) if self.error else None,
            "evidence": list(self.evidence),
            "duration_seconds": self.duration_seconds,
            "confidence": self.confidence.value,
        }


@dataclass(slots=True)
class TaskAssignment:
    task_id: str
    agent_name: str
    state: AssignmentState = AssignmentState.PENDING
    attempts: int = 0
    results: list[AgentResult] = field(default_factory=list)
    last_error: dict[str, Any] | None = None
    started_at: str | None = None
    ended_at: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            AssignmentState.SUCCEEDED, AssignmentState.FAILED,
            AssignmentState.SKIPPED, AssignmentState.CANCELLED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "state": self.state.value,
            "attempts": self.attempts,
            "results": [r.to_dict() for r in self.results],
            "last_error": dict(self.last_error) if self.last_error else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(slots=True)
class ExecutionRecord:
    """Immutable audit row of a single attempt."""
    id: str
    task_id: str
    agent_name: str
    attempt: int
    started_at: str
    ended_at: str
    status: str                  # AssignmentState at end of attempt
    duration_seconds: float
    result: dict[str, Any]
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "task_id": self.task_id,
            "agent_name": self.agent_name, "attempt": self.attempt,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "status": self.status, "duration_seconds": self.duration_seconds,
            "result": dict(self.result),
            "error": dict(self.error) if self.error else None,
        }


@dataclass(slots=True)
class GateResult:
    gate_id: str
    condition: str
    passed: bool
    blocking: bool
    milestone_id: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "condition": self.condition,
            "passed": self.passed,
            "blocking": self.blocking,
            "milestone_id": self.milestone_id,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class OrchestrationResult:
    id: str = field(default_factory=_new_id)
    plan_id: str = ""
    project_id: str = ""
    status: OrchestrationStatus = OrchestrationStatus.COMPLETED
    mode: ParallelismMode = ParallelismMode.SEQUENTIAL
    assignments: dict[str, TaskAssignment] = field(default_factory=dict)
    records: list[ExecutionRecord] = field(default_factory=list)
    gate_results: list[GateResult] = field(default_factory=list)
    rollback_triggered: list[str] = field(default_factory=list)  # milestone ids
    escalation: EscalationReason = EscalationReason.NONE
    wall_time_seconds: float = 0.0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in AssignmentState}
        for a in self.assignments.values():
            out[a.state.value] = out.get(a.state.value, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "plan_id": self.plan_id,
            "project_id": self.project_id,
            "status": self.status.value,
            "mode": self.mode.value,
            "assignments": {k: v.to_dict() for k, v in self.assignments.items()},
            "records": [r.to_dict() for r in self.records],
            "gate_results": [g.to_dict() for g in self.gate_results],
            "rollback_triggered": list(self.rollback_triggered),
            "escalation": self.escalation.value,
            "wall_time_seconds": self.wall_time_seconds,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        c = self.counts()
        return (
            "=== Orchestration Result ===\n"
            f"plan_id={self.plan_id}  status={self.status.value}  "
            f"mode={self.mode.value}\n"
            f"tasks: total={len(self.assignments)}  "
            f"succeeded={c.get('succeeded', 0)}  "
            f"failed={c.get('failed', 0)}  "
            f"skipped={c.get('skipped', 0)}\n"
            f"records={len(self.records)}  "
            f"gates={len(self.gate_results)}  "
            f"rollbacks={len(self.rollback_triggered)}  "
            f"escalation={self.escalation.value}\n"
            f"wall_time={self.wall_time_seconds:.3f}s"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. AGENT PROTOCOL + SCRIPTED AGENT
# ════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class Agent(Protocol):
    """Worker interface. Agents are controlled workers, not independent brains."""
    name: str
    capability: AgentCapability

    def execute(self, task: Task, ctx: dict[str, Any]) -> AgentResult:
        ...


class ScriptedAgent:
    """Deterministic agent for testing/demo.

    Configurable to succeed always, fail always, or succeed after N attempts.
    Records invocations for verification.
    """

    def __init__(
        self,
        name: str,
        capability: AgentCapability,
        *,
        behavior: str = "succeed",         # "succeed" | "fail" | "flaky"
        fail_times: int = 0,               # for "flaky": fail first N attempts
        sleep_seconds: float = 0.0,        # simulate work
    ) -> None:
        if behavior not in ("succeed", "fail", "flaky"):
            raise ValidationError(f"unknown behavior: {behavior}")
        self.name = name
        self.capability = capability
        self.behavior = behavior
        self.fail_times = fail_times
        self.sleep_seconds = sleep_seconds
        self._per_task_attempts: dict[str, int] = {}
        self._lock = threading.Lock()
        self.invocations: list[str] = []    # task ids

    def execute(self, task: Task, ctx: dict[str, Any]) -> AgentResult:
        with self._lock:
            self.invocations.append(task.id)
            n = self._per_task_attempts.get(task.id, 0) + 1
            self._per_task_attempts[task.id] = n

        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        should_fail = (
            self.behavior == "fail"
            or (self.behavior == "flaky" and n <= self.fail_times)
        )
        if should_fail:
            return AgentResult(
                success=False,
                error={
                    "code": "SCRIPTED_FAILURE",
                    "message": f"scripted failure on attempt {n}",
                    "task_id": task.id,
                },
                duration_seconds=self.sleep_seconds,
                confidence=Confidence.LOW,
            )
        return AgentResult(
            success=True,
            output={
                "task_id": task.id, "task_name": task.name,
                "attempt": n,
            },
            evidence=[{"kind": "script", "attempt": n}],
            duration_seconds=self.sleep_seconds,
            confidence=Confidence.MEDIUM,
        )


# ════════════════════════════════════════════════════════════════════════════
# 4. AGENT REGISTRY
# ════════════════════════════════════════════════════════════════════════════
class AgentRegistry:
    """Capability-matched lookup. Deterministic: first registered wins on tie."""

    def __init__(self) -> None:
        self._agents: list[Agent] = []
        self._by_name: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.name in self._by_name:
            raise ValidationError(f"agent already registered: {agent.name}")
        self._agents.append(agent)
        self._by_name[agent.name] = agent

    def get(self, name: str) -> Agent | None:
        return self._by_name.get(name)

    def select_for(self, task_kind: TaskKind) -> Agent | None:
        """First registered agent whose capability matches."""
        for a in self._agents:
            if a.capability.matches(task_kind):
                return a
        return None

    def all(self) -> list[Agent]:
        return list(self._agents)

    def names(self) -> list[str]:
        return [a.name for a in self._agents]


# ════════════════════════════════════════════════════════════════════════════
# 5. RETRY POLICY
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_backoff_seconds: float = 0.0
    max_backoff_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValidationError("max_retries must be >= 0")
        if self.base_backoff_seconds < 0:
            raise ValidationError("base_backoff_seconds must be >= 0")

    def backoff_for(self, attempt: int) -> float:
        """Exponential backoff, capped."""
        if attempt <= 0:
            return 0.0
        b = self.base_backoff_seconds * (2 ** (attempt - 1))
        return min(b, self.max_backoff_seconds)


# ════════════════════════════════════════════════════════════════════════════
# 6. GATE EVALUATOR
# ════════════════════════════════════════════════════════════════════════════
class GateEvaluator:
    """Evaluate verification gates based on assignment states."""

    def evaluate(
        self,
        gate: VerificationGate,
        plan: Plan,
        assignments: dict[str, TaskAssignment],
    ) -> GateResult:
        cond = gate.condition
        milestone = plan.milestone(gate.attached_to_milestone_id)

        if cond is GateCondition.ALL_TASKS_DONE:
            if milestone is None:
                return GateResult(
                    gate_id=gate.id, condition=cond.value,
                    passed=False, blocking=gate.blocking,
                    milestone_id=gate.attached_to_milestone_id,
                    rationale="milestone not found",
                )
            ok = all(
                (tid in assignments and
                 assignments[tid].state is AssignmentState.SUCCEEDED)
                for tid in milestone.task_ids
            )
            return GateResult(
                gate_id=gate.id, condition=cond.value,
                passed=ok, blocking=gate.blocking,
                milestone_id=gate.attached_to_milestone_id,
                rationale=(
                    f"ALL_TASKS_DONE for milestone {milestone.id}: "
                    f"{sum(1 for t in milestone.task_ids if t in assignments and assignments[t].state is AssignmentState.SUCCEEDED)}"
                    f"/{len(milestone.task_ids)} succeeded"
                ),
            )

        if cond is GateCondition.ALL_TESTS_PASS:
            testing_ids = [
                t.id for t in plan.tasks if t.kind is TaskKind.TESTING
            ]
            if not testing_ids:
                return GateResult(
                    gate_id=gate.id, condition=cond.value,
                    passed=True, blocking=gate.blocking,
                    milestone_id=gate.attached_to_milestone_id,
                    rationale="no TESTING tasks in plan → vacuous pass",
                )
            ok = all(
                (tid in assignments and
                 assignments[tid].state is AssignmentState.SUCCEEDED)
                for tid in testing_ids
            )
            return GateResult(
                gate_id=gate.id, condition=cond.value,
                passed=ok, blocking=gate.blocking,
                milestone_id=gate.attached_to_milestone_id,
                rationale=(
                    f"{sum(1 for t in testing_ids if t in assignments and assignments[t].state is AssignmentState.SUCCEEDED)}"
                    f"/{len(testing_ids)} tests succeeded"
                ),
            )

        if cond is GateCondition.NO_CRITICAL_FAILURES:
            failed = [
                tid for tid, a in assignments.items()
                if a.state is AssignmentState.FAILED
            ]
            ok = len(failed) == 0
            return GateResult(
                gate_id=gate.id, condition=cond.value,
                passed=ok, blocking=gate.blocking,
                milestone_id=gate.attached_to_milestone_id,
                rationale=(
                    f"{len(failed)} failed tasks" if failed else "no failed tasks"
                ),
            )

        if cond is GateCondition.VERIFICATION_SIGNED:
            ver_ids = [
                t.id for t in plan.tasks if t.kind is TaskKind.VERIFICATION
            ]
            if not ver_ids:
                return GateResult(
                    gate_id=gate.id, condition=cond.value,
                    passed=True, blocking=gate.blocking,
                    milestone_id=gate.attached_to_milestone_id,
                    rationale="no VERIFICATION tasks → vacuous pass",
                )
            ok = all(
                (tid in assignments and
                 assignments[tid].state is AssignmentState.SUCCEEDED)
                for tid in ver_ids
            )
            return GateResult(
                gate_id=gate.id, condition=cond.value,
                passed=ok, blocking=gate.blocking,
                milestone_id=gate.attached_to_milestone_id,
                rationale=(
                    f"{sum(1 for t in ver_ids if t in assignments and assignments[t].state is AssignmentState.SUCCEEDED)}"
                    f"/{len(ver_ids)} verifications signed"
                ),
            )

        return GateResult(
            gate_id=gate.id, condition=cond.value,
            passed=False, blocking=gate.blocking,
            milestone_id=gate.attached_to_milestone_id,
            rationale=f"unknown gate condition: {cond}",
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════
class Orchestrator:
    """Executes a Plan via registered agents.

    Scheduling:
      - Compute ready set = tasks whose deps are all SUCCEEDED
      - SEQUENTIAL: pop one ready task (deterministic min by id), execute
      - WAVE: group by C08 parallel levels; run each level's ready tasks
              concurrently via ThreadPoolExecutor

    Failures:
      - On task failure: retry up to RetryPolicy.max_retries
      - On exhaustion: mark FAILED, escalate
      - fail_fast=True → abort the whole run
      - fail_fast=False → dependents of FAILED task become SKIPPED
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        retry_policy: RetryPolicy | None = None,
        mode: ParallelismMode = ParallelismMode.SEQUENTIAL,
        max_workers: int = 4,
        fail_fast: bool = True,
        max_wall_time_seconds: float | None = 300.0,
        gate_evaluator: GateEvaluator | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValidationError("max_workers must be >= 1")
        self.registry = registry
        self.retry_policy = retry_policy or RetryPolicy()
        self.mode = mode
        self.max_workers = max_workers
        self.fail_fast = fail_fast
        self.max_wall_time_seconds = max_wall_time_seconds
        self.gate_evaluator = gate_evaluator or GateEvaluator()
        self._lock = threading.Lock()

    # ---- main entry ----
    def run(
        self,
        plan: Plan,
        *,
        project_id: str = "",
    ) -> OrchestrationResult:
        started = time.monotonic()
        result = OrchestrationResult(
            plan_id=plan.id, project_id=project_id,
            mode=self.mode,
            provenance=plan.provenance,
        )
        # Initialise assignments
        for t in plan.tasks:
            agent = self.registry.select_for(t.kind)
            if agent is None:
                result.assignments[t.id] = TaskAssignment(
                    task_id=t.id, agent_name="<none>",
                    state=AssignmentState.SKIPPED,
                    last_error={
                        "code": "NO_AGENT",
                        "message": f"no agent for task kind {t.kind.value}",
                    },
                )
                result.escalation = EscalationReason.NO_AGENT_FOR_TASK
                result.rationale = (
                    f"no registered agent for task kind {t.kind.value} "
                    f"(task {t.id})"
                )
                result.status = OrchestrationStatus.FAILED
                result.wall_time_seconds = time.monotonic() - started
                return result
            result.assignments[t.id] = TaskAssignment(
                task_id=t.id, agent_name=agent.name,
                state=AssignmentState.PENDING,
            )

        # Execute
        if self.mode is ParallelismMode.SEQUENTIAL:
            self._run_sequential(plan, result)
        else:
            self._run_wave(plan, result)

        # Evaluate gates
        self._evaluate_gates(plan, result)

        # Rollback points depend on gate results, so this must run after
        # gate evaluation.
        self._check_rollback_points(plan, result)

        # Determine overall status
        result.wall_time_seconds = time.monotonic() - started
        result.status = self._final_status(result)
        if result.escalation is EscalationReason.NONE:
            if result.status is OrchestrationStatus.FAILED:
                result.escalation = EscalationReason.TASK_FAILED_ALL_RETRIES
            elif result.rollback_triggered:
                result.escalation = EscalationReason.ROLLBACK_TRIGGERED
        if not result.rationale:
            c = result.counts()
            result.rationale = (
                f"mode={self.mode.value}; succeeded={c.get('succeeded', 0)}; "
                f"failed={c.get('failed', 0)}; skipped={c.get('skipped', 0)}; "
                f"gates_failed={sum(1 for g in result.gate_results if not g.passed)}"
            )
        # Wall time cap
        if (self.max_wall_time_seconds is not None and
                result.wall_time_seconds > self.max_wall_time_seconds and
                result.status is OrchestrationStatus.COMPLETED):
            result.status = OrchestrationStatus.PARTIAL
            result.escalation = EscalationReason.WALL_TIME_EXCEEDED
            result.rationale += f"; exceeded max_wall_time_seconds={self.max_wall_time_seconds}"
        return result

    # ---- sequential loop ----
    def _run_sequential(self, plan: Plan, result: OrchestrationResult) -> None:
        # Topo order is deterministic and safe
        order = list(plan.execution_order)
        for tid in order:
            a = result.assignments[tid]
            if a.state is not AssignmentState.PENDING:
                continue
            # check deps
            task = plan.task(tid)
            if task is None:
                continue
            blocked = self._deps_blocked(plan, result, task)
            if blocked is not None:
                # a dep failed → skip
                a.state = AssignmentState.SKIPPED
                a.last_error = {
                    "code": "DEP_FAILED",
                    "message": f"dependency failed or skipped: {blocked}",
                }
                continue
            self._execute_with_retries(plan, result, task)
            if self.fail_fast and a.state is AssignmentState.FAILED:
                # mark all remaining as SKIPPED
                self._skip_remaining(plan, result, reason="fail_fast trip")
                return
        # Rollback points depend on gate results, which aren't evaluated
        # until after this method returns (see run()) — checking here was
        # always a no-op (result.gate_results is still empty at this
        # point). Rollback checking now happens once, in run(), right
        # after _evaluate_gates().

    # ---- wave-parallel loop ----
    def _run_wave(self, plan: Plan, result: OrchestrationResult) -> None:
        # Use C08 parallel_groups (level-wise DAG layers)
        for group in plan.parallel_groups:
            # filter to pending tasks; skip those whose deps failed
            ready: list[str] = []
            for tid in group.task_ids:
                a = result.assignments[tid]
                if a.state is not AssignmentState.PENDING:
                    continue
                task = plan.task(tid)
                if task is None:
                    continue
                blocked = self._deps_blocked(plan, result, task)
                if blocked is not None:
                    a.state = AssignmentState.SKIPPED
                    a.last_error = {
                        "code": "DEP_FAILED",
                        "message": f"dependency failed or skipped: {blocked}",
                    }
                    continue
                ready.append(tid)
            if not ready:
                continue

            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                futures: list[tuple[str, Future]] = []
                for tid in ready:
                    task = plan.task(tid)
                    if task is None:
                        continue
                    fut = pool.submit(self._execute_with_retries, plan, result, task)
                    futures.append((tid, fut))
                for tid, fut in futures:
                    try:
                        fut.result()
                    except Exception as exc:
                        log.error("wave.task.exception", task=tid, error=str(exc))

            # After each level, if fail_fast and any FAILED → skip rest
            if self.fail_fast:
                failed = [
                    tid for tid in group.task_ids
                    if result.assignments[tid].state is AssignmentState.FAILED
                ]
                if failed:
                    self._skip_remaining(plan, result, reason="fail_fast trip")
                    return
        # See the comment in _run_sequential — rollback checking happens
        # once in run(), after gates are evaluated.

    # ---- task execution with retries ----
    def _execute_with_retries(
        self, plan: Plan, result: OrchestrationResult, task: Task,
    ) -> None:
        a = result.assignments[task.id]
        agent = self.registry.get(a.agent_name)
        if agent is None:
            a.state = AssignmentState.FAILED
            a.last_error = {
                "code": "AGENT_MISSING",
                "message": f"registered agent not found: {a.agent_name}",
            }
            return

        # Mark running (thread-safe)
        with self._lock:
            a.state = AssignmentState.RUNNING
            if a.started_at is None:
                a.started_at = now_iso()

        attempts_allowed = 1 + self.retry_policy.max_retries
        attempt = 0
        ctx = {
            "project_id": result.project_id,
            "plan_id": plan.id,
            "orchestration_id": result.id,
        }
        while attempt < attempts_allowed:
            attempt += 1
            with self._lock:
                a.attempts = attempt
                if attempt > 1:
                    a.state = AssignmentState.RETRYING
            started_iso = now_iso()
            t0 = time.monotonic()
            try:
                res = agent.execute(task, ctx)
            except Exception as exc:
                res = AgentResult(
                    success=False,
                    error={
                        "code": "AGENT_EXCEPTION",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                )
            duration = time.monotonic() - t0
            ended_iso = now_iso()

            # Record attempt
            with self._lock:
                result.records.append(ExecutionRecord(
                    id=_new_id(),
                    task_id=task.id,
                    agent_name=agent.name,
                    attempt=attempt,
                    started_at=started_iso,
                    ended_at=ended_iso,
                    status=(
                        AssignmentState.SUCCEEDED.value if res.success
                        else AssignmentState.FAILED.value
                    ),
                    duration_seconds=duration,
                    result=res.output,
                    error=res.error,
                ))
                a.results.append(res)

            if res.success:
                with self._lock:
                    a.state = AssignmentState.SUCCEEDED
                    a.ended_at = ended_iso
                return

            # Failure — retry?
            a.last_error = res.error
            if attempt >= attempts_allowed:
                with self._lock:
                    a.state = AssignmentState.FAILED
                    a.ended_at = ended_iso
                return
            backoff = self.retry_policy.backoff_for(attempt)
            if backoff > 0:
                time.sleep(backoff)

    # ---- dependency helpers ----
    def _deps_blocked(
        self, plan: Plan, result: OrchestrationResult, task: Task,
    ) -> str | None:
        for d in task.depends_on:
            da = result.assignments.get(d)
            if da is None:
                continue
            if da.state in (
                AssignmentState.FAILED, AssignmentState.SKIPPED,
                AssignmentState.CANCELLED,
            ):
                return d
            if da.state is not AssignmentState.SUCCEEDED:
                return d   # not yet done (shouldn't happen in topo/wave order)
        return None

    def _skip_remaining(
        self, plan: Plan, result: OrchestrationResult, *, reason: str,
    ) -> None:
        for tid, a in result.assignments.items():
            if a.is_terminal:
                continue
            a.state = AssignmentState.SKIPPED
            a.last_error = {
                "code": "SKIPPED",
                "message": f"orchestrator skipped task: {reason}",
            }

    # ---- gate evaluation ----
    def _evaluate_gates(self, plan: Plan, result: OrchestrationResult) -> None:
        for gate in plan.gates:
            gr = self.gate_evaluator.evaluate(gate, plan, result.assignments)
            result.gate_results.append(gr)

    # ---- rollback points ----
    def _check_rollback_points(
        self, plan: Plan, result: OrchestrationResult,
    ) -> None:
        # If any blocking gate failed → trigger rollback of the milestone
        failed_milestones = set()
        for gr in result.gate_results:
            if not gr.passed and gr.blocking:
                failed_milestones.add(gr.milestone_id)
        for rp in plan.rollback_points:
            if rp.after_milestone_id in failed_milestones:
                result.rollback_triggered.append(rp.after_milestone_id)

    # ---- final status ----
    def _final_status(self, result: OrchestrationResult) -> OrchestrationStatus:
        c = result.counts()
        succeeded = c.get(AssignmentState.SUCCEEDED.value, 0)
        failed = c.get(AssignmentState.FAILED.value, 0)
        skipped = c.get(AssignmentState.SKIPPED.value, 0)
        total = sum(c.values())
        if total == 0:
            return OrchestrationStatus.COMPLETED
        if failed == 0 and skipped == 0:
            return OrchestrationStatus.COMPLETED
        if failed == 0 and skipped > 0:
            return OrchestrationStatus.PARTIAL
        if succeeded == 0:
            return OrchestrationStatus.FAILED
        return OrchestrationStatus.PARTIAL


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class OrchestrationRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: OrchestrationResult) -> str:
        if not result.project_id:
            raise ValidationError("result.project_id required")
        key = f"orchestration:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=result.project_id,
            tags=["orchestration", "c11"],
            provenance=result.provenance,
        )
        # Record failures for C33 later
        for tid, a in result.assignments.items():
            if a.state is AssignmentState.FAILED and a.last_error:
                self.memory.record_failure(
                    f"exec_fail:{result.id}:{tid}",
                    what=f"task {tid} failed after {a.attempts} attempts",
                    root_cause=a.last_error.get("message", "unknown"),
                    fix=None,
                    scope_id=result.project_id,
                    provenance=Provenance(
                        source="orchestrator",
                        source_type=ProvenanceType.SYSTEM,
                        confidence=Confidence.HIGH,
                    ),
                    confidence=Confidence.HIGH,
                )
        if self.ontology is None:
            return result.id
        ent = self.ontology.add(
            EntityKind.EXECUTION,
            _short(f"Orchestration {result.id[:8]} ({result.status.value})", 120),
            attributes={
                "orchestration_id": result.id,
                "plan_id": result.plan_id,
                "project_id": result.project_id,
                "status": result.status.value,
                "mode": result.mode.value,
                "escalation": result.escalation.value,
                "records": len(result.records),
                "gate_results": [g.to_dict() for g in result.gate_results],
            },
            tags=["execution", result.status.value],
            provenance=result.provenance,
        )
        return ent.id

    def load(self, orchestration_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"orchestration:{orchestration_id}",
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
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C11 self-tests…")

    # ---- helpers ----
    _parser = RequirementParser()
    _ic = IntentContextEngine()
    _tech = TechnologySelector()
    _arch = ArchitectureReasoner()
    _planner = Planner()

    def _make_plan(text: str, project: str = "test"):
        spec = _parser.parse(text)
        ic = _ic.analyze(text, project_id=project)
        tech = _tech.select(spec, ic, project_id=project)
        architecture = _arch.reason(spec, ic, tech, project_id=project)
        plan = _planner.plan(spec, ic, project_id=project)
        return spec, ic, tech, architecture, plan

    def _default_registry(
        behavior: str = "succeed",
        fail_times: int = 0,
        sleep_seconds: float = 0.0,
    ) -> AgentRegistry:
        reg = AgentRegistry()
        # One agent handles all task kinds — scripted to succeed
        for tk in TaskKind:
            reg.register(ScriptedAgent(
                f"agent_{tk.value}",
                AgentCapability(task_kinds=(tk,)),
                behavior=behavior,
                fail_times=fail_times,
                sleep_seconds=sleep_seconds,
            ))
        return reg

    # ---- registry ----
    def t_registry_select() -> None:
        reg = _default_registry()
        a = reg.select_for(TaskKind.IMPLEMENTATION)
        assert a is not None
        assert a.capability.matches(TaskKind.IMPLEMENTATION)

    def t_registry_no_match() -> None:
        reg = AgentRegistry()
        assert reg.select_for(TaskKind.TESTING) is None

    def t_registry_duplicate_name_rejected() -> None:
        reg = AgentRegistry()
        reg.register(ScriptedAgent("x", AgentCapability((TaskKind.ANALYSIS,))))
        try:
            reg.register(ScriptedAgent("x", AgentCapability((TaskKind.DESIGN,))))
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    def t_registry_first_registered_wins() -> None:
        reg = AgentRegistry()
        a1 = ScriptedAgent("a", AgentCapability((TaskKind.TESTING,)))
        a2 = ScriptedAgent("b", AgentCapability((TaskKind.TESTING,)))
        reg.register(a1)
        reg.register(a2)
        assert reg.select_for(TaskKind.TESTING) is a1

    check("registry: capability select", t_registry_select)
    check("registry: no match → None", t_registry_no_match)
    check("registry: duplicate name rejected", t_registry_duplicate_name_rejected)
    check("registry: first-registered wins on tie", t_registry_first_registered_wins)

    # ---- retry policy ----
    def t_retry_policy_backoff() -> None:
        p = RetryPolicy(max_retries=3, base_backoff_seconds=1.0, max_backoff_seconds=5.0)
        assert p.backoff_for(0) == 0.0
        assert p.backoff_for(1) == 1.0
        assert p.backoff_for(2) == 2.0
        assert p.backoff_for(3) == 4.0
        assert p.backoff_for(4) == 5.0  # capped

    def t_retry_policy_invalid() -> None:
        try:
            RetryPolicy(max_retries=-1)
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    check("retry policy: exponential backoff capped",
          t_retry_policy_backoff)
    check("retry policy: rejects negative max_retries",
          t_retry_policy_invalid)

    # ---- sequential orchestration ----
    def t_run_sequential_happy() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API for tasks.")
        reg = _default_registry()
        orch = Orchestrator(reg, mode=ParallelismMode.SEQUENTIAL)
        res = orch.run(plan, project_id="p")
        assert res.status is OrchestrationStatus.COMPLETED, res.status
        assert all(
            a.state is AssignmentState.SUCCEEDED
            for a in res.assignments.values()
        )
        assert len(res.records) == len(plan.tasks)

    def t_run_sequential_no_agent() -> None:
        _, _, _, _, plan = _make_plan("Build a REST API.")
        reg = AgentRegistry()   # empty
        orch = Orchestrator(reg)
        res = orch.run(plan, project_id="p")
        assert res.status is OrchestrationStatus.FAILED
        assert res.escalation is EscalationReason.NO_AGENT_FOR_TASK

    check("run: sequential happy path all SUCCEEDED", t_run_sequential_happy)
    check("run: empty registry → FAILED + NO_AGENT escalation",
          t_run_sequential_no_agent)

    # ---- retry behaviour ----
    def t_run_retry_success_after_flake() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        # All agents fail twice then succeed
        reg = AgentRegistry()
        for tk in TaskKind:
            reg.register(ScriptedAgent(
                f"agent_{tk.value}",
                AgentCapability(task_kinds=(tk,)),
                behavior="flaky",
                fail_times=2,
            ))
        orch = Orchestrator(reg, retry_policy=RetryPolicy(max_retries=3))
        res = orch.run(plan, project_id="p")
        assert res.status is OrchestrationStatus.COMPLETED, res.status
        # Every task should have attempts=3 (2 fails + 1 success)
        for a in res.assignments.values():
            assert a.attempts == 3, (a.task_id, a.attempts)

    def t_run_retry_exhausted() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        # All agents always fail
        reg = _default_registry(behavior="fail")
        orch = Orchestrator(
            reg, retry_policy=RetryPolicy(max_retries=2),
            fail_fast=True,
        )
        res = orch.run(plan, project_id="p")
        assert res.status is OrchestrationStatus.FAILED
        assert res.escalation is EscalationReason.TASK_FAILED_ALL_RETRIES
        # First task attempted 3 times (1 + 2 retries)
        first = plan.execution_order[0]
        assert res.assignments[first].attempts == 3
        assert res.assignments[first].state is AssignmentState.FAILED
        # Remaining tasks skipped (fail_fast)
        skipped = sum(
            1 for a in res.assignments.values()
            if a.state is AssignmentState.SKIPPED
        )
        assert skipped == len(plan.tasks) - 1

    def t_run_retry_records_attempts() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        reg = _default_registry(behavior="fail")
        orch = Orchestrator(reg, retry_policy=RetryPolicy(max_retries=1),
                            fail_fast=True)
        res = orch.run(plan, project_id="p")
        first = plan.execution_order[0]
        records = [r for r in res.records if r.task_id == first]
        assert len(records) == 2
        assert records[0].attempt == 1
        assert records[1].attempt == 2

    def t_run_no_fail_fast_skips_dependents() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        # Only fail IMPLEMENTATION; others succeed
        reg = AgentRegistry()
        for tk in TaskKind:
            reg.register(ScriptedAgent(
                f"agent_{tk.value}",
                AgentCapability(task_kinds=(tk,)),
                behavior="fail" if tk is TaskKind.IMPLEMENTATION else "succeed",
            ))
        orch = Orchestrator(reg, retry_policy=RetryPolicy(max_retries=0),
                            fail_fast=False)
        res = orch.run(plan, project_id="p")
        assert res.status is OrchestrationStatus.PARTIAL
        impl_ids = [t.id for t in plan.tasks if t.kind is TaskKind.IMPLEMENTATION]
        for iid in impl_ids:
            assert res.assignments[iid].state is AssignmentState.FAILED
        # Dependents should be SKIPPED
        impl_skipped_dependents = 0
        for t in plan.tasks:
            if any(d in impl_ids for d in t.depends_on):
                if res.assignments[t.id].state is AssignmentState.SKIPPED:
                    impl_skipped_dependents += 1
        assert impl_skipped_dependents >= 1

    check("retry: flaky agent succeeds after retries",
          t_run_retry_success_after_flake)
    check("retry: exhausted retries → FAILED + escalation",
          t_run_retry_exhausted)
    check("retry: attempts are recorded as separate ExecutionRecords",
          t_run_retry_records_attempts)
    check("no_fail_fast: dependents of failed task → SKIPPED",
          t_run_no_fail_fast_skips_dependents)

    # ---- wave parallel ----
    def t_run_wave_parallel() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API for tasks.")
        reg = _default_registry()
        orch = Orchestrator(reg, mode=ParallelismMode.WAVE, max_workers=4)
        res = orch.run(plan, project_id="p")
        assert res.status is OrchestrationStatus.COMPLETED
        assert all(
            a.state is AssignmentState.SUCCEEDED
            for a in res.assignments.values()
        )

    def t_run_wave_deterministic() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        reg1 = _default_registry()
        reg2 = _default_registry()
        o1 = Orchestrator(reg1, mode=ParallelismMode.WAVE)
        o2 = Orchestrator(reg2, mode=ParallelismMode.WAVE)
        r1 = o1.run(plan, project_id="p")
        r2 = o2.run(plan, project_id="p")
        # Same terminal states across runs
        for tid in plan.execution_order:
            assert r1.assignments[tid].state == r2.assignments[tid].state

    def t_run_wave_faster_than_sequential() -> None:
        # If each task sleeps, wave parallel should be faster than sequential
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        reg = _default_registry(sleep_seconds=0.02)
        seq = Orchestrator(reg, mode=ParallelismMode.SEQUENTIAL)
        res_seq = seq.run(plan, project_id="p")

        reg2 = _default_registry(sleep_seconds=0.02)
        wave = Orchestrator(reg2, mode=ParallelismMode.WAVE, max_workers=4)
        res_wave = wave.run(plan, project_id="p")

        # Wave should be <= sequential. Give some slack for scheduler noise.
        assert res_wave.wall_time_seconds <= res_seq.wall_time_seconds + 0.05, (
            res_wave.wall_time_seconds, res_seq.wall_time_seconds
        )

    check("wave: happy path completes", t_run_wave_parallel)
    check("wave: deterministic across runs (state equality)",
          t_run_wave_deterministic)
    check("wave: parallel wall time <= sequential when I/O-bound",
          t_run_wave_faster_than_sequential)

    # ---- gates ----
    def t_gate_all_tasks_done() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        reg = _default_registry()
        res = Orchestrator(reg).run(plan, project_id="p")
        # ALL_TASKS_DONE gate should pass
        for g in res.gate_results:
            if g.condition == GateCondition.ALL_TASKS_DONE.value:
                assert g.passed is True

    def t_gate_tests_pass_passes_when_tests_succeed() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        reg = _default_registry()
        res = Orchestrator(reg).run(plan, project_id="p")
        for g in res.gate_results:
            if g.condition == GateCondition.ALL_TESTS_PASS.value:
                assert g.passed is True, g.rationale

    def t_gate_tests_fail_detected() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        # Fail TESTING; succeed others; no fail_fast
        reg = AgentRegistry()
        for tk in TaskKind:
            reg.register(ScriptedAgent(
                f"agent_{tk.value}",
                AgentCapability(task_kinds=(tk,)),
                behavior="fail" if tk is TaskKind.TESTING else "succeed",
            ))
        res = Orchestrator(
            reg, retry_policy=RetryPolicy(max_retries=0), fail_fast=False,
        ).run(plan, project_id="p")
        gate_tests = [g for g in res.gate_results
                      if g.condition == GateCondition.ALL_TESTS_PASS.value]
        assert gate_tests
        assert all(not g.passed for g in gate_tests)

    def t_gate_no_critical_failures() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        reg = _default_registry()
        res = Orchestrator(reg).run(plan, project_id="p")
        for g in res.gate_results:
            if g.condition == GateCondition.NO_CRITICAL_FAILURES.value:
                assert g.passed is True

    check("gate: ALL_TASKS_DONE passes on full success",
          t_gate_all_tasks_done)
    check("gate: ALL_TESTS_PASS passes on full success",
          t_gate_tests_pass_passes_when_tests_succeed)
    check("gate: ALL_TESTS_PASS fails when tests fail",
          t_gate_tests_fail_detected)
    check("gate: NO_CRITICAL_FAILURES passes on full success",
          t_gate_no_critical_failures)

    # ---- rollback ----
    def t_rollback_triggered_on_gate_failure() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        # fail TESTING → tests gate fails → triggers rollback
        reg = AgentRegistry()
        for tk in TaskKind:
            reg.register(ScriptedAgent(
                f"agent_{tk.value}",
                AgentCapability(task_kinds=(tk,)),
                behavior="fail" if tk is TaskKind.TESTING else "succeed",
            ))
        res = Orchestrator(
            reg, retry_policy=RetryPolicy(max_retries=0), fail_fast=False,
        ).run(plan, project_id="p")
        assert res.rollback_triggered, "expected at least one rollback triggered"

    check("rollback: gate failure triggers rollback points",
          t_rollback_triggered_on_gate_failure)

    # ---- persistence ----
    def t_persist_reload() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                _, _, _, _, plan = _make_plan("Build a small REST API.",
                                              project="proj-x")
                reg = _default_registry()
                res = Orchestrator(reg).run(plan, project_id="proj-x")
                repo = OrchestrationRepository(memory=mem, ontology=ont)
                ent_id = repo.save(res)
                assert ent_id
                loaded = repo.load(res.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == res.id
                assert loaded["status"] == res.status.value
                # Ontology has EXECUTION entity
                assert ont.count(kind=EntityKind.EXECUTION) >= 1
                # Memory has orchestration + any failures (none here)
                entry = mem.get_current(
                    MemoryKind.PROJECT, f"orchestration:{res.id}",
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert entry is not None
            finally:
                s.shutdown()

    def t_persist_records_failures() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                _, _, _, _, plan = _make_plan("Build a small REST API.",
                                              project="proj-x")
                reg = _default_registry(behavior="fail")
                res = Orchestrator(
                    reg, retry_policy=RetryPolicy(max_retries=0),
                    fail_fast=True,
                ).run(plan, project_id="proj-x")
                repo = OrchestrationRepository(memory=mem, ontology=ont)
                repo.save(res)
                # At least one FAILURE memory entry should exist
                failures = mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert len(failures) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology EXECUTION entity", t_persist_reload)
    check("persist: failures recorded as failure memory",
          t_persist_records_failures)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        _, _, _, _, plan = _make_plan("Build a small REST API.")
        res = Orchestrator(_default_registry()).run(plan, project_id="p")
        d = res.to_dict()
        assert d["id"] == res.id
        assert d["plan_id"] == plan.id
        assert isinstance(d["assignments"], dict)
        assert isinstance(d["records"], list)
        assert isinstance(d["gate_results"], list)
        s = res.summary()
        assert "Orchestration Result" in s
        assert "plan_id=" in s

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
        spec, ic, tech, arch, plan = _make_plan(text, project="demo")
        reg = _default_registry()
        res = Orchestrator(
            reg, mode=ParallelismMode.WAVE, max_workers=4,
        ).run(plan, project_id="demo")

        # Every task assigned
        assert set(res.assignments.keys()) == {t.id for t in plan.tasks}
        # All succeeded
        assert all(
            a.state is AssignmentState.SUCCEEDED
            for a in res.assignments.values()
        )
        # Records = sum of attempts
        total_attempts = sum(a.attempts for a in res.assignments.values())
        assert len(res.records) == total_attempts
        # All gates passed
        assert all(g.passed for g in res.gate_results), [
            (g.gate_id, g.passed) for g in res.gate_results
        ]
        # No rollbacks
        assert res.rollback_triggered == []
        # Status completed
        assert res.status is OrchestrationStatus.COMPLETED
        assert res.escalation is EscalationReason.NONE

        # Persistence roundtrip
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                repo = OrchestrationRepository(memory=mem, ontology=ont)
                ent = repo.save(res)
                assert ent
                loaded = repo.load(res.id, project_id="demo")
                assert loaded is not None
                assert loaded["status"] == "completed"
            finally:
                s.shutdown()

    check("e2e: canonical REST API orchestration (wave, all success)",
          t_e2e_canonical)

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
    print("SE Brain C11 — Agent Orchestrator")
    print("=" * 78)

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

    parser = RequirementParser()
    ic_engine = IntentContextEngine()
    selector = TechnologySelector()
    reasoner = ArchitectureReasoner()
    planner = Planner()

    spec = parser.parse(text)
    ic = ic_engine.analyze(text, project_id="demo")
    tech = selector.select(spec, ic, project_id="demo")
    arch = reasoner.reason(spec, ic, tech, project_id="demo")
    plan = planner.plan(spec, ic, project_id="demo")

    print(f"\n[1] Plan: {len(plan.tasks)} tasks, {len(plan.milestones)} milestones, "
          f"duration={plan.total_duration}")

    # Register agents per task kind
    reg = AgentRegistry()
    for tk in TaskKind:
        reg.register(ScriptedAgent(
            f"agent_{tk.value}",
            AgentCapability(task_kinds=(tk,)),
            behavior="succeed",
            sleep_seconds=0.01,   # visible in wall_time
        ))
    print(f"[2] Registered agents: {reg.names()}")

    print("\n[3] Running (WAVE parallel, max_workers=4)…")
    orch = Orchestrator(reg, mode=ParallelismMode.WAVE, max_workers=4)
    res = orch.run(plan, project_id="demo")

    print("\n[4] Summary:")
    print(res.summary())

    print("\n[5] Per-task state:")
    for tid in plan.execution_order:
        a = res.assignments[tid]
        t = plan.task(tid)
        print(f"    {tid}  [{t.kind.value:15s}]  "
              f"{a.state.value:10s}  attempts={a.attempts}  "
              f"agent={a.agent_name}")

    print("\n[6] Gate results:")
    for g in res.gate_results:
        mark = "✓" if g.passed else "✗"
        print(f"    {mark} {g.gate_id}: {g.rationale}")

    print("\n[7] Rollback points triggered:", res.rollback_triggered or "(none)")

    print("\n[8] Execution records sample:")
    for r in res.records[:5]:
        print(f"    · task={r.task_id}  attempt={r.attempt}  "
              f"status={r.status}  {r.duration_seconds*1000:.1f}ms")
    if len(res.records) > 5:
        print(f"    … {len(res.records) - 5} more")

    # Persistence
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = OrchestrationRepository(memory=mem, ontology=ont)
                ent_id = repo.save(res)
                print(f"\n[9] Persisted → ontology entity: {ent_id[:12]}…")
                loaded = repo.load(res.id, project_id="demo")
                print(f"    reloaded status: {loaded['status']}")
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
