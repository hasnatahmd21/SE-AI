"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C08 — PLANNING ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C05, C06.

Purpose:
    Convert a structured RequirementSpec + IntentContext into an EXECUTABLE
    Plan:

        Requirement → Architecture (implicit) → Milestones → Tasks
                    → Dependencies → Execution Order → Verification Criteria

Capabilities:
    - Task decomposition (spec + intent driven, deterministic)
    - Dependency graph (explicit edges, no cycles)
    - Topological sort (Kahn, deterministic via heap)
    - Critical Path Method (ES/EF/LS/LF/slack → critical path)
    - Parallelization (level-wise DAG grouping)
    - Milestones (ordered phases with rollback points)
    - Verification gates (per milestone, blocking)
    - Rollback points (per milestone boundary)

Invariants honored:
  - No LLM. Pure deterministic algorithms.
  - Every task, milestone, gate has a rationale.
  - No cycles allowed — detected, reported, refused.
  - Critical path computed by real CPM (forward + backward pass).
  - Every plan can be topologically executed by an orchestrator.
  - Persistence to C04 memory + C02 ontology.

Contents:
  1.  Enums: TaskKind, TaskStatus, MilestonePhase, GateCondition
  2.  Dataclasses: Task, Milestone, VerificationGate, RollbackPoint,
                   Plan, CriticalPathResult, ParallelGroup
  3.  Graph algorithms: topo sort, CPM, parallel groups, cycles
  4.  PlanBuilder: decompose(spec, intent) → Plan
  5.  Planner: algorithms facade
  6.  PlanRepository: persist / load
  7.  __main__ demo + self-tests

Run as script:
    python -m sebrain.c08            # demo
    python -m sebrain.c08 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import heapq
import json
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
from sebrain.c05 import (
    RequirementKind,
    RequirementSpec,
)
from sebrain.c06 import (
    IntentContext,
    IntentKind,
)


log = get_logger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _short(s: str, n: int = 60) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class TaskKind(str, Enum):
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    VERIFICATION = "verification"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    PACKAGING = "packaging"
    DEPLOYMENT = "deployment"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class MilestonePhase(str, Enum):
    REQUIREMENTS = "requirements"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    QUALITY = "quality"
    DELIVERY = "delivery"


class GateCondition(str, Enum):
    ALL_TASKS_DONE = "all_tasks_done"
    ALL_TESTS_PASS = "all_tests_pass"
    NO_CRITICAL_FAILURES = "no_critical_failures"
    VERIFICATION_SIGNED = "verification_signed"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Task:
    id: str
    name: str
    kind: TaskKind
    duration: int                         # relative units
    depends_on: list[str] = field(default_factory=list)
    milestone_id: str | None = None
    description: str = ""
    parallelizable: bool = True
    tags: list[str] = field(default_factory=list)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind.value,
            "duration": self.duration,
            "depends_on": list(self.depends_on),
            "milestone_id": self.milestone_id,
            "description": self.description,
            "parallelizable": self.parallelizable,
            "tags": list(self.tags),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
        }


@dataclass(slots=True)
class Milestone:
    id: str
    name: str
    phase: MilestonePhase
    task_ids: list[str] = field(default_factory=list)
    depends_on_milestones: list[str] = field(default_factory=list)
    rollback_point_id: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "phase": self.phase.value,
            "task_ids": list(self.task_ids),
            "depends_on_milestones": list(self.depends_on_milestones),
            "rollback_point_id": self.rollback_point_id,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class VerificationGate:
    id: str
    name: str
    condition: GateCondition
    attached_to_milestone_id: str
    blocking: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "condition": self.condition.value,
            "attached_to_milestone_id": self.attached_to_milestone_id,
            "blocking": self.blocking,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class RollbackPoint:
    id: str
    name: str
    after_milestone_id: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "after_milestone_id": self.after_milestone_id,
            "description": self.description,
        }


@dataclass(slots=True)
class ParallelGroup:
    level: int
    task_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "task_ids": list(self.task_ids)}


@dataclass(slots=True)
class CriticalPathResult:
    ES: dict[str, int] = field(default_factory=dict)      # earliest start
    EF: dict[str, int] = field(default_factory=dict)      # earliest finish
    LS: dict[str, int] = field(default_factory=dict)      # latest start
    LF: dict[str, int] = field(default_factory=dict)      # latest finish
    slack: dict[str, int] = field(default_factory=dict)
    critical_path: list[str] = field(default_factory=list)
    project_duration: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ES": dict(self.ES), "EF": dict(self.EF),
            "LS": dict(self.LS), "LF": dict(self.LF),
            "slack": dict(self.slack),
            "critical_path": list(self.critical_path),
            "project_duration": self.project_duration,
        }


@dataclass(slots=True)
class Plan:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    project_id: str = ""
    spec_id: str | None = None
    intent_context_id: str | None = None
    milestones: list[Milestone] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)
    gates: list[VerificationGate] = field(default_factory=list)
    rollback_points: list[RollbackPoint] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)  # task ids
    parallel_groups: list[ParallelGroup] = field(default_factory=list)
    critical: CriticalPathResult = field(default_factory=CriticalPathResult)
    total_duration: int = 0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    # ---- lookups ----
    def task(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def milestone(self, mid: str) -> Milestone | None:
        for m in self.milestones:
            if m.id == mid:
                return m
        return None

    def tasks_by_kind(self, kind: TaskKind) -> list[Task]:
        return [t for t in self.tasks if t.kind is kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "spec_id": self.spec_id,
            "intent_context_id": self.intent_context_id,
            "milestones": [m.to_dict() for m in self.milestones],
            "tasks": [t.to_dict() for t in self.tasks],
            "gates": [g.to_dict() for g in self.gates],
            "rollback_points": [r.to_dict() for r in self.rollback_points],
            "execution_order": list(self.execution_order),
            "parallel_groups": [pg.to_dict() for pg in self.parallel_groups],
            "critical": self.critical.to_dict(),
            "total_duration": self.total_duration,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        lines = [
            "=== Plan ===",
            f"Tasks: {len(self.tasks)}  Milestones: {len(self.milestones)}  "
            f"Gates: {len(self.gates)}  Rollback points: {len(self.rollback_points)}",
            f"Project duration (critical path): {self.total_duration} units",
            f"Parallel groups: {len(self.parallel_groups)}  "
            f"Critical path length: {len(self.critical.critical_path)}",
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. GRAPH ALGORITHMS
# ════════════════════════════════════════════════════════════════════════════
def _build_successors(tasks: Sequence[Task]) -> dict[str, list[str]]:
    """Build successor map. task -> [tasks that depend on it]."""
    succ: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for d in t.depends_on:
            if d in succ:
                succ[d].append(t.id)
    for k in succ:
        succ[k].sort()
    return succ


def _topological_order(tasks: Sequence[Task]) -> list[str] | None:
    """Kahn's algorithm with lexicographic tie-break via heap.

    Returns None if a cycle exists.
    """
    ids = {t.id for t in tasks}
    indeg: dict[str, int] = {t.id: 0 for t in tasks}
    succ = _build_successors(tasks)
    for t in tasks:
        for d in t.depends_on:
            if d in ids:
                indeg[t.id] += 1
    heap = [tid for tid, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    out: list[str] = []
    while heap:
        tid = heapq.heappop(heap)
        out.append(tid)
        for s in succ.get(tid, []):
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(heap, s)
    if len(out) != len(tasks):
        return None
    return out


def _detect_cycles(tasks: Sequence[Task]) -> list[list[str]]:
    """Return list of cycle paths (each path is [a, b, ..., a])."""
    adj: dict[str, list[str]] = {t.id: list(t.depends_on) for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {tid: WHITE for tid in adj}
    cycles: list[list[str]] = []
    seen_canon: set[tuple[str, ...]] = set()
    stack: list[str] = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in adj.get(u, []):
            if v not in color:
                continue
            if color[v] == GRAY:
                try:
                    idx = stack.index(v)
                except ValueError:
                    idx = 0
                cyc = stack[idx:] + [v]
                canon = tuple(sorted(set(cyc)))
                if canon not in seen_canon:
                    seen_canon.add(canon)
                    cycles.append(cyc)
            elif color[v] == WHITE:
                dfs(v)
        stack.pop()
        color[u] = BLACK

    for tid in sorted(adj.keys()):
        if color[tid] == WHITE:
            dfs(tid)
    return cycles


def _critical_path_method(
    tasks: Sequence[Task], topo_order: Sequence[str],
) -> CriticalPathResult:
    """Classic CPM: forward pass (ES/EF), backward pass (LS/LF), slack."""
    by_id = {t.id: t for t in tasks}
    succ = _build_successors(tasks)

    ES: dict[str, int] = {}
    EF: dict[str, int] = {}
    for tid in topo_order:
        t = by_id[tid]
        es = max((EF[d] for d in t.depends_on if d in EF), default=0)
        ES[tid] = es
        EF[tid] = es + t.duration

    project_duration = max(EF.values()) if EF else 0

    LS: dict[str, int] = {}
    LF: dict[str, int] = {}
    for tid in reversed(list(topo_order)):
        t = by_id[tid]
        lf = min((LS[s] for s in succ.get(tid, []) if s in LS),
                 default=project_duration)
        LF[tid] = lf
        LS[tid] = lf - t.duration

    slack = {tid: LS[tid] - ES[tid] for tid in topo_order}
    crit_set = {tid for tid, s in slack.items() if s == 0}
    # Preserve topo ordering along the critical path
    crit_path = [tid for tid in topo_order if tid in crit_set]

    return CriticalPathResult(
        ES=ES, EF=EF, LS=LS, LF=LF, slack=slack,
        critical_path=crit_path, project_duration=project_duration,
    )


def _parallel_groups(
    tasks: Sequence[Task], topo_order: Sequence[str],
) -> list[ParallelGroup]:
    """Level = longest chain of dependencies ending at this task.

    Tasks in the same level can run concurrently (given their deps are in
    earlier levels).
    """
    by_id = {t.id: t for t in tasks}
    level: dict[str, int] = {}
    for tid in topo_order:
        t = by_id[tid]
        if not t.depends_on:
            level[tid] = 0
        else:
            level[tid] = 1 + max(level[d] for d in t.depends_on if d in level)
    buckets: dict[int, list[str]] = {}
    for tid, lvl in level.items():
        buckets.setdefault(lvl, []).append(tid)
    out: list[ParallelGroup] = []
    for lvl in sorted(buckets):
        out.append(ParallelGroup(level=lvl, task_ids=sorted(buckets[lvl])))
    return out


# ════════════════════════════════════════════════════════════════════════════
# 4. PLAN BUILDER — spec/intent → task graph
# ════════════════════════════════════════════════════════════════════════════
_DEFAULT_DURATIONS: dict[TaskKind, int] = {
    TaskKind.ANALYSIS: 2,
    TaskKind.DESIGN: 3,
    TaskKind.IMPLEMENTATION: 5,
    TaskKind.TESTING: 3,
    TaskKind.VERIFICATION: 2,
    TaskKind.SECURITY: 2,
    TaskKind.PERFORMANCE: 2,
    TaskKind.DOCUMENTATION: 2,
    TaskKind.PACKAGING: 1,
    TaskKind.DEPLOYMENT: 2,
}

# Intent → task skeletons. Each entry is a list of
# (kind, name_suffix, depends_on_suffixes, phase).
_INTENT_TEMPLATES: dict[IntentKind, list[tuple[TaskKind, str, list[str], MilestonePhase]]] = {
    IntentKind.BUILD: [
        (TaskKind.ANALYSIS, "requirements-analysis", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DESIGN, "architecture-design", ["requirements-analysis"], MilestonePhase.DESIGN),
        (TaskKind.IMPLEMENTATION, "core-implementation", ["architecture-design"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.TESTING, "unit-tests", ["core-implementation"], MilestonePhase.QUALITY),
        (TaskKind.TESTING, "integration-tests", ["unit-tests"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-requirements", ["integration-tests"], MilestonePhase.QUALITY),
        (TaskKind.DOCUMENTATION, "documentation", ["verify-requirements"], MilestonePhase.DELIVERY),
        (TaskKind.PACKAGING, "package-deliverable", ["documentation"], MilestonePhase.DELIVERY),
    ],
    IntentKind.MODIFY: [
        (TaskKind.ANALYSIS, "impact-analysis", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DESIGN, "change-design", ["impact-analysis"], MilestonePhase.DESIGN),
        (TaskKind.IMPLEMENTATION, "apply-change", ["change-design"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.TESTING, "regression-tests", ["apply-change"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-change", ["regression-tests"], MilestonePhase.QUALITY),
        (TaskKind.DOCUMENTATION, "update-docs", ["verify-change"], MilestonePhase.DELIVERY),
    ],
    IntentKind.FIX: [
        (TaskKind.ANALYSIS, "reproduce-failure", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.ANALYSIS, "root-cause-analysis", ["reproduce-failure"], MilestonePhase.REQUIREMENTS),
        (TaskKind.IMPLEMENTATION, "apply-fix", ["root-cause-analysis"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.TESTING, "regression-test", ["apply-fix"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-fix", ["regression-test"], MilestonePhase.QUALITY),
    ],
    IntentKind.REFACTOR: [
        (TaskKind.ANALYSIS, "structure-analysis", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DESIGN, "refactor-plan", ["structure-analysis"], MilestonePhase.DESIGN),
        (TaskKind.IMPLEMENTATION, "apply-refactor", ["refactor-plan"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.TESTING, "regression-tests", ["apply-refactor"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-behavior-preserved", ["regression-tests"], MilestonePhase.QUALITY),
    ],
    IntentKind.MIGRATE: [
        (TaskKind.ANALYSIS, "migration-assessment", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DESIGN, "migration-strategy", ["migration-assessment"], MilestonePhase.DESIGN),
        (TaskKind.IMPLEMENTATION, "apply-migration", ["migration-strategy"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.TESTING, "migration-tests", ["apply-migration"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-parity", ["migration-tests"], MilestonePhase.QUALITY),
        (TaskKind.DOCUMENTATION, "migration-docs", ["verify-parity"], MilestonePhase.DELIVERY),
    ],
    IntentKind.OPTIMIZE: [
        (TaskKind.ANALYSIS, "baseline-measurement", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DESIGN, "optimization-plan", ["baseline-measurement"], MilestonePhase.DESIGN),
        (TaskKind.IMPLEMENTATION, "apply-optimization", ["optimization-plan"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.PERFORMANCE, "benchmark-after", ["apply-optimization"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-improvement", ["benchmark-after"], MilestonePhase.QUALITY),
    ],
    IntentKind.TEST: [
        (TaskKind.ANALYSIS, "coverage-analysis", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.TESTING, "write-tests", ["coverage-analysis"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.TESTING, "run-tests", ["write-tests"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "verify-coverage", ["run-tests"], MilestonePhase.QUALITY),
    ],
    IntentKind.DEPLOY: [
        (TaskKind.ANALYSIS, "deploy-readiness", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DESIGN, "deployment-plan", ["deploy-readiness"], MilestonePhase.DESIGN),
        (TaskKind.DEPLOYMENT, "execute-deploy", ["deployment-plan"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.VERIFICATION, "smoke-tests", ["execute-deploy"], MilestonePhase.QUALITY),
    ],
    IntentKind.REVIEW: [
        (TaskKind.ANALYSIS, "review-scope", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.ANALYSIS, "code-inspection", ["review-scope"], MilestonePhase.QUALITY),
        (TaskKind.VERIFICATION, "review-report", ["code-inspection"], MilestonePhase.DELIVERY),
    ],
    IntentKind.ANALYZE: [
        (TaskKind.ANALYSIS, "gather-facts", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.ANALYSIS, "synthesize-findings", ["gather-facts"], MilestonePhase.QUALITY),
        (TaskKind.DOCUMENTATION, "analysis-report", ["synthesize-findings"], MilestonePhase.DELIVERY),
    ],
    IntentKind.DOCUMENT: [
        (TaskKind.ANALYSIS, "doc-scope", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DOCUMENTATION, "write-docs", ["doc-scope"], MilestonePhase.IMPLEMENTATION),
        (TaskKind.VERIFICATION, "review-docs", ["write-docs"], MilestonePhase.QUALITY),
    ],
    IntentKind.EXPLAIN: [
        (TaskKind.ANALYSIS, "locate-subject", [], MilestonePhase.REQUIREMENTS),
        (TaskKind.DOCUMENTATION, "produce-explanation", ["locate-subject"], MilestonePhase.DELIVERY),
    ],
    IntentKind.UNKNOWN: [
        (TaskKind.ANALYSIS, "clarify-requirement", [], MilestonePhase.REQUIREMENTS),
    ],
}


class PlanBuilder:
    """Deterministic decomposition. Uses spec + intent to shape the graph."""

    def __init__(self) -> None:
        pass

    # ---- public ----
    def build(
        self,
        spec: RequirementSpec,
        intent_ctx: IntentContext | None = None,
        *,
        project_id: str = "",
    ) -> Plan:
        plan = Plan(project_id=project_id)
        plan.spec_id = spec.id
        plan.intent_context_id = intent_ctx.id if intent_ctx else None
        plan.provenance = spec.provenance

        intent_kind = intent_ctx.intent.kind if (intent_ctx and intent_ctx.intent) \
            else IntentKind.BUILD
        template = _INTENT_TEMPLATES.get(intent_kind, _INTENT_TEMPLATES[IntentKind.BUILD])

        # 1. Build base tasks from template
        name_to_id: dict[str, str] = {}
        counter = 0
        tasks: list[Task] = []
        milestones: dict[MilestonePhase, Milestone] = {}

        for kind, name_suffix, deps_suffixes, phase in template:
            counter += 1
            tid = f"t{counter:03d}"
            deps_ids = [name_to_id[d] for d in deps_suffixes if d in name_to_id]
            task = Task(
                id=tid,
                name=name_suffix,
                kind=kind,
                duration=_DEFAULT_DURATIONS[kind],
                depends_on=deps_ids,
                description=f"{kind.value} step: {name_suffix}",
                rationale=(
                    f"template step for intent={intent_kind.value}"
                ),
                provenance=Provenance(
                    source="plan_builder", source_type=ProvenanceType.SYSTEM,
                    confidence=Confidence.MEDIUM,
                ),
            )
            tasks.append(task)
            name_to_id[name_suffix] = tid
            if phase not in milestones:
                milestones[phase] = Milestone(
                    id=f"m_{phase.value}",
                    name=f"{phase.value.title()} Phase",
                    phase=phase,
                    rationale=f"phase grouping for intent={intent_kind.value}",
                )
            milestones[phase].task_ids.append(tid)
            task.milestone_id = milestones[phase].id

        # 2. Attach NFR-driven tasks (security, performance)
        final_template_task = tasks[-1] if tasks else None
        nfr_tags = {t for it in spec.non_functional for t in it.tags}
        nfr_task_ids: list[str] = []
        if "security" in nfr_tags:
            self._add_nfr_task(
                tasks, milestones, counter, "security",
                TaskKind.SECURITY, "security-analysis",
                depends_on=name_to_id.get("core-implementation")
                or name_to_id.get("apply-change")
                or name_to_id.get("apply-fix")
                or name_to_id.get("apply-refactor")
                or name_to_id.get("apply-migration")
                or name_to_id.get("apply-optimization")
                or name_to_id.get("write-tests")
                or name_to_id.get("execute-deploy")
                or (tasks[0].id if tasks else None),
                phase=MilestonePhase.QUALITY,
                rationale="security NFR present in spec",
            )
            nfr_task_ids.append(tasks[-1].id)
        if "performance" in nfr_tags:
            self._add_nfr_task(
                tasks, milestones, counter + 1, "perf",
                TaskKind.PERFORMANCE, "performance-check",
                depends_on=name_to_id.get("core-implementation")
                or name_to_id.get("apply-change")
                or name_to_id.get("apply-optimization")
                or (tasks[0].id if tasks else None),
                phase=MilestonePhase.QUALITY,
                rationale="performance NFR present in spec",
            )
            nfr_task_ids.append(tasks[-1].id)
        # The plan's final template step (typically packaging/documentation)
        # should wait on the NFR-driven quality checks — otherwise these
        # newly-appended tasks are dangling leaves with no downstream
        # dependent, and the topological order has no reason to keep the
        # delivery step last. It's also the semantically correct order:
        # don't package/document before security/performance checks land.
        if final_template_task is not None and nfr_task_ids:
            final_template_task.depends_on = sorted(
                set(final_template_task.depends_on) | set(nfr_task_ids)
            )

        # 3. Order milestones
        ordered_phases = [
            MilestonePhase.REQUIREMENTS,
            MilestonePhase.DESIGN,
            MilestonePhase.IMPLEMENTATION,
            MilestonePhase.QUALITY,
            MilestonePhase.DELIVERY,
        ]
        ordered_milestones: list[Milestone] = []
        for i, ph in enumerate(ordered_phases):
            if ph in milestones:
                m = milestones[ph]
                if i > 0:
                    # depends on the previous milestone
                    prevs = [p for p in ordered_phases[:i] if p in milestones]
                    if prevs:
                        m.depends_on_milestones = [f"m_{prevs[-1].value}"]
                ordered_milestones.append(m)

        # 4. Rollback points after every milestone
        rollback_points: list[RollbackPoint] = []
        for m in ordered_milestones:
            rp = RollbackPoint(
                id=f"rp_{m.phase.value}",
                name=f"Rollback after {m.phase.value}",
                after_milestone_id=m.id,
                description=f"checkpoint at end of {m.phase.value} phase",
            )
            rollback_points.append(rp)
            m.rollback_point_id = rp.id

        # 5. Verification gates
        gates: list[VerificationGate] = []
        if MilestonePhase.IMPLEMENTATION in milestones:
            gates.append(VerificationGate(
                id="gate_impl_done",
                name="Implementation complete",
                condition=GateCondition.ALL_TASKS_DONE,
                attached_to_milestone_id=milestones[MilestonePhase.IMPLEMENTATION].id,
                blocking=True,
                rationale="must complete implementation before quality phase",
            ))
        if MilestonePhase.QUALITY in milestones:
            gates.append(VerificationGate(
                id="gate_tests_pass",
                name="All tests pass",
                condition=GateCondition.ALL_TESTS_PASS,
                attached_to_milestone_id=milestones[MilestonePhase.QUALITY].id,
                blocking=True,
                rationale="no delivery with failing tests",
            ))
            gates.append(VerificationGate(
                id="gate_no_critical_failures",
                name="No critical failures",
                condition=GateCondition.NO_CRITICAL_FAILURES,
                attached_to_milestone_id=milestones[MilestonePhase.QUALITY].id,
                blocking=True,
                rationale="critical failures block delivery",
            ))
        if MilestonePhase.DELIVERY in milestones:
            gates.append(VerificationGate(
                id="gate_verified",
                name="Verification signed",
                condition=GateCondition.VERIFICATION_SIGNED,
                attached_to_milestone_id=milestones[MilestonePhase.DELIVERY].id,
                blocking=True,
                rationale="deliverable requires signed verification",
            ))

        plan.tasks = tasks
        plan.milestones = ordered_milestones
        plan.gates = gates
        plan.rollback_points = rollback_points

        # 6. Compute graph properties (may raise on cycle — but our builder
        # cannot produce cycles; still validate defensively)
        cycle_check = _detect_cycles(tasks)
        if cycle_check:
            raise ValidationError(
                f"plan_builder produced a cyclic plan: {cycle_check}"
            )

        topo = _topological_order(tasks)
        if topo is None:
            raise ValidationError("plan is not a DAG")
        plan.execution_order = topo
        plan.parallel_groups = _parallel_groups(tasks, topo)
        plan.critical = _critical_path_method(tasks, topo)
        plan.total_duration = plan.critical.project_duration
        plan.rationale = (
            f"intent={intent_kind.value}; "
            f"tasks={len(tasks)}; "
            f"duration={plan.total_duration}; "
            f"parallel_groups={len(plan.parallel_groups)}; "
            f"critical_path_len={len(plan.critical.critical_path)}"
        )
        return plan

    def _add_nfr_task(
        self,
        tasks: list[Task],
        milestones: dict[MilestonePhase, Milestone],
        counter: int,
        tag: str,
        kind: TaskKind,
        name: str,
        depends_on: str | None,
        phase: MilestonePhase,
        rationale: str,
    ) -> None:
        tid = f"t{counter + 1:03d}"
        # Ensure unique ID (avoid collisions when both security & perf added)
        existing = {t.id for t in tasks}
        i = counter + 1
        while tid in existing:
            i += 1
            tid = f"t{i:03d}"
        t = Task(
            id=tid, name=name, kind=kind, duration=_DEFAULT_DURATIONS[kind],
            depends_on=[depends_on] if depends_on else [],
            description=f"NFR-driven {kind.value} task ({tag})",
            rationale=rationale,
            tags=[tag],
            provenance=Provenance(
                source="plan_builder", source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.MEDIUM,
            ),
        )
        tasks.append(t)
        if phase not in milestones:
            milestones[phase] = Milestone(
                id=f"m_{phase.value}", name=f"{phase.value.title()} Phase",
                phase=phase, rationale=f"phase for NFR task ({tag})",
            )
        milestones[phase].task_ids.append(tid)
        t.milestone_id = milestones[phase].id


# ════════════════════════════════════════════════════════════════════════════
# 5. PLANNER — algorithms facade
# ════════════════════════════════════════════════════════════════════════════
class Planner:
    """Entry point. Wraps PlanBuilder + graph algorithms."""

    def __init__(self) -> None:
        self.builder = PlanBuilder()

    def plan(
        self,
        spec: RequirementSpec,
        intent_ctx: IntentContext | None = None,
        *,
        project_id: str = "",
    ) -> Plan:
        return self.builder.build(spec, intent_ctx, project_id=project_id)

    # ---- pure algorithm API (testable in isolation) ----
    @staticmethod
    def topological_order(tasks: Sequence[Task]) -> list[str] | None:
        return _topological_order(tasks)

    @staticmethod
    def detect_cycles(tasks: Sequence[Task]) -> list[list[str]]:
        return _detect_cycles(tasks)

    @staticmethod
    def critical_path(tasks: Sequence[Task]) -> CriticalPathResult:
        topo = _topological_order(tasks)
        if topo is None:
            raise ValidationError("cannot compute CPM on a cyclic graph")
        return _critical_path_method(tasks, topo)

    @staticmethod
    def parallel_groups(tasks: Sequence[Task]) -> list[ParallelGroup]:
        topo = _topological_order(tasks)
        if topo is None:
            raise ValidationError("cannot compute parallel groups on a cyclic graph")
        return _parallel_groups(tasks, topo)

    # ---- validation ----
    @staticmethod
    def validate(plan: Plan) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        ids = [t.id for t in plan.tasks]
        if len(set(ids)) != len(ids):
            issues.append({"type": "duplicate_task_id"})
        id_set = set(ids)
        for t in plan.tasks:
            for d in t.depends_on:
                if d not in id_set:
                    issues.append({
                        "type": "missing_dependency",
                        "task": t.id, "dep": d,
                    })
        cycles = _detect_cycles(plan.tasks)
        if cycles:
            issues.append({"type": "cycle", "cycles": cycles})
        # milestone references
        m_ids = {m.id for m in plan.milestones}
        for t in plan.tasks:
            if t.milestone_id and t.milestone_id not in m_ids:
                issues.append({
                    "type": "bad_milestone_ref",
                    "task": t.id, "milestone": t.milestone_id,
                })
        for rp in plan.rollback_points:
            if rp.after_milestone_id not in m_ids:
                issues.append({
                    "type": "bad_rollback_ref", "rp": rp.id,
                    "milestone": rp.after_milestone_id,
                })
        for g in plan.gates:
            if g.attached_to_milestone_id not in m_ids:
                issues.append({
                    "type": "bad_gate_ref", "gate": g.id,
                    "milestone": g.attached_to_milestone_id,
                })
        return {"ok": not issues, "issues": issues, "issue_count": len(issues)}


# ════════════════════════════════════════════════════════════════════════════
# 6. PLAN REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class PlanRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, plan: Plan) -> str:
        if not plan.project_id:
            raise ValidationError("plan.project_id is required to persist")
        key = f"plan:{plan.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, plan.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=plan.project_id,
            tags=["plan", "c08"],
            provenance=plan.provenance,
        )
        if self.ontology is None:
            return plan.id
        ent = self.ontology.add(
            EntityKind.PLAN,
            _short(plan.rationale or f"Plan {plan.id[:8]}", 120),
            attributes={
                "plan_id": plan.id,
                "project_id": plan.project_id,
                "task_count": len(plan.tasks),
                "duration": plan.total_duration,
            },
            tags=["plan"],
            provenance=plan.provenance,
        )
        # Register tasks as TASK entities + CONTAINS relations
        task_ent_ids: dict[str, str] = {}
        for t in plan.tasks:
            te = self.ontology.add(
                EntityKind.TASK, _short(t.name, 120),
                attributes={
                    "task_id": t.id, "kind": t.kind.value,
                    "duration": t.duration, "milestone_id": t.milestone_id,
                },
                tags=["task", t.kind.value],
                provenance=t.provenance,
            )
            task_ent_ids[t.id] = te.id
            self.ontology.link(RelationKind.CONTAINS, ent.id, te.id)
        # Task dependencies
        for t in plan.tasks:
            if t.id not in task_ent_ids:
                continue
            for d in t.depends_on:
                if d in task_ent_ids:
                    try:
                        self.ontology.link(
                            RelationKind.DEPENDS_ON,
                            task_ent_ids[t.id], task_ent_ids[d],
                        )
                    except ValidationError:
                        pass
        return ent.id

    def load(self, plan_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"plan:{plan_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 7. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _mk_task(tid: str, kind: TaskKind, dur: int, deps: list[str] | None = None) -> Task:
    return Task(
        id=tid, name=tid, kind=kind, duration=dur,
        depends_on=list(deps or []),
    )


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

    print("Running C08 self-tests…")
    planner = Planner()

    # ---- helpers from C05/C06 for realistic tests ----
    from sebrain.c05 import RequirementParser
    from sebrain.c06 import IntentContextEngine
    parser = RequirementParser()
    ic_engine = IntentContextEngine()

    def _spec_intent(text: str) -> tuple[RequirementSpec, IntentContext]:
        spec = parser.parse(text)
        ic = ic_engine.analyze(text, project_id="test")
        return spec, ic

    # ---- algorithms in isolation ----
    def t_topo_simple() -> None:
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 1),
            _mk_task("b", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("c", TaskKind.IMPLEMENTATION, 1, ["b"]),
        ]
        order = planner.topological_order(tasks)
        assert order == ["a", "b", "c"]

    def t_topo_diamond() -> None:
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 1),
            _mk_task("b", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("c", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("d", TaskKind.IMPLEMENTATION, 1, ["b", "c"]),
        ]
        order = planner.topological_order(tasks)
        assert order is not None
        # a must precede b,c,d ; b and c must precede d
        assert order.index("a") < order.index("b") < order.index("d")
        assert order.index("a") < order.index("c") < order.index("d")

    def t_topo_cycle_returns_none() -> None:
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 1, ["c"]),
            _mk_task("b", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("c", TaskKind.IMPLEMENTATION, 1, ["b"]),
        ]
        assert planner.topological_order(tasks) is None

    def t_detect_cycle() -> None:
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 1, ["c"]),
            _mk_task("b", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("c", TaskKind.IMPLEMENTATION, 1, ["b"]),
        ]
        cycles = planner.detect_cycles(tasks)
        assert len(cycles) >= 1

    check("topo sort: linear chain", t_topo_simple)
    check("topo sort: diamond DAG", t_topo_diamond)
    check("topo sort: cycle → None", t_topo_cycle_returns_none)
    check("cycle detection", t_detect_cycle)

    # ---- CPM in isolation ----
    def t_cpm_linear() -> None:
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 3),
            _mk_task("b", TaskKind.DESIGN, 2, ["a"]),
            _mk_task("c", TaskKind.IMPLEMENTATION, 4, ["b"]),
        ]
        r = planner.critical_path(tasks)
        assert r.project_duration == 9
        assert r.ES["a"] == 0 and r.EF["a"] == 3
        assert r.ES["b"] == 3 and r.EF["b"] == 5
        assert r.ES["c"] == 5 and r.EF["c"] == 9
        # all slack 0 → all on critical path
        assert all(s == 0 for s in r.slack.values())
        assert set(r.critical_path) == {"a", "b", "c"}

    def t_cpm_parallel_with_slack() -> None:
        #  a(3) -> b(2) -> d(1)
        #     \-> c(1) -/
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 3),
            _mk_task("b", TaskKind.DESIGN, 2, ["a"]),
            _mk_task("c", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("d", TaskKind.IMPLEMENTATION, 1, ["b", "c"]),
        ]
        r = planner.critical_path(tasks)
        assert r.project_duration == 6
        # c has slack because b is longer
        assert r.slack["a"] == 0
        assert r.slack["b"] == 0
        assert r.slack["c"] == 1
        assert r.slack["d"] == 0
        assert set(r.critical_path) == {"a", "b", "d"}

    def t_cpm_multi_root() -> None:
        # two independent starts, join at end
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 5),
            _mk_task("b", TaskKind.ANALYSIS, 2),
            _mk_task("c", TaskKind.VERIFICATION, 1, ["a", "b"]),
        ]
        r = planner.critical_path(tasks)
        assert r.project_duration == 6
        assert r.slack["b"] == 3
        assert set(r.critical_path) == {"a", "c"}

    check("CPM: linear chain, duration = sum", t_cpm_linear)
    check("CPM: parallel branch has slack", t_cpm_parallel_with_slack)
    check("CPM: multi-root, later branch is critical", t_cpm_multi_root)

    # ---- parallel groups ----
    def t_parallel_groups() -> None:
        tasks = [
            _mk_task("a", TaskKind.ANALYSIS, 1),
            _mk_task("b", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("c", TaskKind.DESIGN, 1, ["a"]),
            _mk_task("d", TaskKind.IMPLEMENTATION, 1, ["b", "c"]),
        ]
        groups = planner.parallel_groups(tasks)
        assert [g.level for g in groups] == [0, 1, 2]
        assert groups[0].task_ids == ["a"]
        assert groups[1].task_ids == ["b", "c"]
        assert groups[2].task_ids == ["d"]

    check("parallel groups: diamond → 3 levels", t_parallel_groups)

    # ---- plan building from spec/intent ----
    def t_build_from_build_intent() -> None:
        text = (
            "Build a small REST API for managing tasks.\n"
            "Functional:\n- Users must be able to create tasks.\n"
            "Non-functional:\n- The API must respond within 200ms.\n"
        )
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        assert len(plan.tasks) >= 5
        # analysis → design → implementation → tests → verification present
        kinds = {t.kind for t in plan.tasks}
        assert TaskKind.ANALYSIS in kinds
        assert TaskKind.DESIGN in kinds
        assert TaskKind.IMPLEMENTATION in kinds
        assert TaskKind.TESTING in kinds
        assert TaskKind.VERIFICATION in kinds
        # performance NFR → performance task present
        assert TaskKind.PERFORMANCE in kinds
        # final task is delivery-ish (packaging or docs)
        last_tid = plan.execution_order[-1]
        last = plan.task(last_tid)
        assert last.kind in (TaskKind.PACKAGING, TaskKind.DOCUMENTATION)

    def t_build_from_fix_intent() -> None:
        text = "Fix the login bug causing users to be logged out."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        names = {t.name for t in plan.tasks}
        assert "reproduce-failure" in names
        assert "root-cause-analysis" in names
        assert "apply-fix" in names
        assert "regression-test" in names
        assert "verify-fix" in names

    def t_build_with_security_nfr() -> None:
        text = (
            "Build a service.\n"
            "Non-functional:\n- All traffic must use HTTPS.\n"
        )
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        kinds = {t.kind for t in plan.tasks}
        assert TaskKind.SECURITY in kinds

    def t_build_has_gates_and_rollbacks() -> None:
        text = "Build a small REST API for tasks."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        # at least: implementation gate, tests gate, verification gate
        gate_ids = {g.id for g in plan.gates}
        assert "gate_impl_done" in gate_ids
        assert "gate_tests_pass" in gate_ids
        assert "gate_verified" in gate_ids
        # rollback after each milestone
        assert len(plan.rollback_points) == len(plan.milestones)
        for m in plan.milestones:
            assert m.rollback_point_id is not None

    def t_build_validate() -> None:
        text = "Build a service."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        rep = planner.validate(plan)
        assert rep["ok"] is True, rep

    check("plan build: BUILD intent", t_build_from_build_intent)
    check("plan build: FIX intent", t_build_from_fix_intent)
    check("plan build: security NFR adds SECURITY task", t_build_with_security_nfr)
    check("plan build: gates + rollback points present", t_build_has_gates_and_rollbacks)
    check("plan validate: no issues", t_build_validate)

    # ---- critical path on built plan ----
    def t_built_plan_critical_path() -> None:
        text = "Build a small REST API for tasks."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        assert plan.total_duration > 0
        assert len(plan.critical.critical_path) >= 2
        # every task id on the critical path exists
        ids = {t.id for t in plan.tasks}
        for tid in plan.critical.critical_path:
            assert tid in ids
        # project duration = max EF
        assert plan.total_duration == max(plan.critical.EF.values())

    check("plan: critical path computed and consistent", t_built_plan_critical_path)

    # ---- execution order respects dependencies ----
    def t_execution_order_respects_deps() -> None:
        text = "Build a service."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        idx = {tid: i for i, tid in enumerate(plan.execution_order)}
        for t in plan.tasks:
            for d in t.depends_on:
                assert idx[d] < idx[t.id], (t.id, d)

    check("plan: execution_order respects all deps", t_execution_order_respects_deps)

    # ---- parallel groups on built plan ----
    def t_built_plan_parallelism() -> None:
        text = "Build a service."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        # should have >1 level (there's always analysis → design → ...)
        assert len(plan.parallel_groups) >= 2
        # flat union equals all task ids
        flat = {tid for g in plan.parallel_groups for tid in g.task_ids}
        assert flat == {t.id for t in plan.tasks}

    check("plan: parallel_groups cover all tasks", t_built_plan_parallelism)

    # ---- persistence ----
    def t_persist_and_reload() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                text = "Build a small REST API for tasks."
                spec = parser.parse(text)
                ic = ic_engine.analyze(text, project_id="proj-x")
                plan = planner.plan(spec, ic, project_id="proj-x")
                repo = PlanRepository(memory=mem, ontology=ont)
                ent_id = repo.save(plan)
                assert ent_id
                loaded = repo.load(plan.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == plan.id
                assert loaded["total_duration"] == plan.total_duration
                # ontology has PLAN entity and TASK entities
                assert ont.count(kind=EntityKind.PLAN) == 1
                assert ont.count(kind=EntityKind.TASK) >= len(plan.tasks)
            finally:
                s.shutdown()

    check("persist: plan → memory + ontology (with task entities + relations)",
          t_persist_and_reload)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        text = "Build a service."
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="p1")
        d = plan.to_dict()
        assert d["id"] == plan.id
        assert isinstance(d["tasks"], list) and len(d["tasks"]) > 0
        assert isinstance(d["milestones"], list)
        assert isinstance(d["execution_order"], list)
        assert isinstance(d["parallel_groups"], list)
        assert "critical" in d
        assert "project_duration" in d["critical"]
        s = plan.summary()
        assert "Project duration" in s
        assert "Tasks:" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- e2e canonical ----
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
            "Out of scope:\n- Mobile client\n"
        )
        spec, ic = _spec_intent(text)
        plan = planner.plan(spec, ic, project_id="demo")

        # structural guarantees
        assert len(plan.tasks) >= 7
        assert len(plan.milestones) >= 4
        assert len(plan.gates) >= 3
        assert len(plan.rollback_points) == len(plan.milestones)
        assert plan.total_duration > 0
        assert len(plan.critical.critical_path) >= 3
        assert len(plan.parallel_groups) >= 3

        # validate
        rep = planner.validate(plan)
        assert rep["ok"] is True, rep

        # execution order — every dep precedes its dependent
        idx = {tid: i for i, tid in enumerate(plan.execution_order)}
        for t in plan.tasks:
            for d in t.depends_on:
                assert idx[d] < idx[t.id]

        # NFR-driven tasks present
        kinds = {t.kind for t in plan.tasks}
        assert TaskKind.SECURITY in kinds      # HTTPS
        assert TaskKind.PERFORMANCE in kinds   # 200ms

    check("e2e: canonical REST API plan", t_e2e_canonical)

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
    print("SE Brain C08 — Planning Engine")
    print("=" * 78)

    from sebrain.c05 import RequirementParser
    from sebrain.c06 import IntentContextEngine

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
        "- Given a valid request, when POST /tasks is called, then a 201 response is returned.\n"
    )

    parser = RequirementParser()
    ic_engine = IntentContextEngine()
    planner = Planner()

    spec = parser.parse(text)
    ic = ic_engine.analyze(text, project_id="demo")
    plan = planner.plan(spec, ic, project_id="demo")

    print("\n[1] Summary:")
    print(plan.summary())

    print("\n[2] Tasks (execution order):")
    task_by_id = {t.id: t for t in plan.tasks}
    for tid in plan.execution_order:
        t = task_by_id[tid]
        deps = ",".join(t.depends_on) if t.depends_on else "-"
        crit = "*" if tid in set(plan.critical.critical_path) else " "
        print(f"    {crit} {t.id}  [{t.kind.value:15s}] dur={t.duration:2d} "
              f"deps=[{deps:10s}] {t.name}")

    print("\n[3] Milestones:")
    for m in plan.milestones:
        print(f"    {m.id}  {m.name}  tasks={len(m.task_ids)}  "
              f"rollback={m.rollback_point_id}")

    print("\n[4] Verification gates:")
    for g in plan.gates:
        print(f"    {g.id}  cond={g.condition.value}  blocking={g.blocking}")
        print(f"      → attached to {g.attached_to_milestone_id}")

    print("\n[5] Rollback points:")
    for rp in plan.rollback_points:
        print(f"    {rp.id}  after {rp.after_milestone_id}")

    print("\n[6] Critical path:")
    print(f"    duration = {plan.total_duration}")
    print(f"    path     = {plan.critical.critical_path}")
    print("    slack per task:")
    for tid in plan.execution_order:
        print(f"      {tid}: ES={plan.critical.ES[tid]}  "
              f"EF={plan.critical.EF[tid]}  "
              f"LS={plan.critical.LS[tid]}  "
              f"LF={plan.critical.LF[tid]}  "
              f"slack={plan.critical.slack[tid]}")

    print("\n[7] Parallel groups (tasks runnable at each level):")
    for g in plan.parallel_groups:
        print(f"    level {g.level}: {g.task_ids}")

    print("\n[8] Validation:", planner.validate(plan))

    # Persistence
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = PlanRepository(memory=mem, ontology=ont)
                ent_id = repo.save(plan)
                print(f"\n[9] Persisted plan → ontology entity: {ent_id[:12]}…")
                loaded = repo.load(plan.id, project_id="demo")
                print(f"    reloaded keys: {sorted(loaded.keys())}")
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
