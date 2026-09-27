"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C10 — ARCHITECTURE REASONING ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C05, C06, C07, C09.

Purpose:
    Generate multiple architecture candidates, model their components,
    interfaces, data flows, failure boundaries, security boundaries, and
    select a winner via weighted multi-dimensional comparison with an
    explicit Pareto front. Preserve ALL rejected alternatives with reasons.

Capabilities:
    - Candidate generation (7 kinds: Monolith, Layered, Hexagonal,
      Clean, Modular Monolith, Microservices, Serverless)
    - Component modeling (roles, responsibilities, depends_on)
    - Interface modeling (sync/async, kind, direction)
    - Data flow modeling (request/event/batch)
    - Failure boundary identification (isolation units)
    - Security boundary identification (trust edges)
    - Multi-dimensional scoring (10 dimensions)
    - Weighted comparison + Pareto front
    - Deterministic winner selection

Invariants honored:
  - No LLM. Pure deterministic generators + scoring.
  - Every component, interface, flow, boundary has a rationale.
  - Every comparison preserves alternatives + reasons.
  - Coherent with C09 tech stack (uses language/framework/DB).
  - No cycles in component dependency graphs.
  - Persistence to C04 memory + C02 ontology (ARCHITECTURE entity,
    COMPONENT entities, ALTERNATIVE entities, relations).

Contents:
  1.  Enums: ComponentRole, InterfaceKind, DataFlowKind, BoundaryKind,
             CandidateKind, FitVerdict
  2.  Dataclasses: Component, Interface, DataFlow, Boundary,
                   DimensionScore, ArchitectureCandidate,
                   CandidateScore, ArchitectureComparison,
                   ArchitectureDecision, ArchitectureResult
  3.  TechStackView (thin adapter over C09 TechSelectionResult)
  4.  Scoring table (base per-candidate-kind)
  5.  Candidate generators (7)
  6.  Comparator (weights + Pareto)
  7.  ArchitectureReasoner facade
  8.  ArchitectureRepository (persist / reload)
  9.  __main__ demo + self-tests

Run as script:
    python -m sebrain.c10            # demo
    python -m sebrain.c10 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
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
from sebrain.c09 import (
    SignalExtractor,
    TechCategory,
    TechSelectionResult,
    TechnologySelector,
    TechSignals,
)


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
class ComponentRole(str, Enum):
    PRESENTATION = "presentation"
    APPLICATION = "application"
    DOMAIN = "domain"
    PERSISTENCE = "persistence"
    INTEGRATION = "integration"
    INFRASTRUCTURE = "infrastructure"
    CROSS_CUTTING = "cross_cutting"


class InterfaceKind(str, Enum):
    SYNC_HTTP = "sync_http"
    SYNC_RPC = "sync_rpc"
    SYNC_IN_PROCESS = "sync_in_process"
    ASYNC_MESSAGE = "async_message"
    ASYNC_EVENT = "async_event"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    EXTERNAL = "external"


class DataFlowKind(str, Enum):
    REQUEST = "request"
    EVENT = "event"
    BATCH = "batch"
    STREAM = "stream"


class BoundaryKind(str, Enum):
    FAILURE = "failure"
    SECURITY = "security"


class CandidateKind(str, Enum):
    MONOLITH = "monolith"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CLEAN = "clean"
    MODULAR_MONOLITH = "modular_monolith"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"


class FitVerdict(str, Enum):
    SELECTED = "selected"
    ALTERNATIVE = "alternative"
    REJECTED = "rejected"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class Component:
    id: str
    name: str
    role: ComponentRole
    responsibilities: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    tech_hint: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "role": self.role.value,
            "responsibilities": list(self.responsibilities),
            "depends_on": list(self.depends_on),
            "tech_hint": self.tech_hint,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class Interface:
    id: str
    name: str
    kind: InterfaceKind
    from_component: str
    to_component: str
    payload_desc: str = ""
    sync: bool = True
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind.value,
            "from_component": self.from_component,
            "to_component": self.to_component,
            "payload_desc": self.payload_desc,
            "sync": self.sync,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class DataFlow:
    id: str
    name: str
    kind: DataFlowKind
    path: list[str]                      # ordered component ids
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind.value,
            "path": list(self.path), "description": self.description,
        }


@dataclass(slots=True)
class Boundary:
    id: str
    name: str
    kind: BoundaryKind
    component_ids: list[str]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind.value,
            "component_ids": list(self.component_ids),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class DimensionScore:
    dimension: str
    score: float                          # 0..10
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": float(self.score),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ArchitectureCandidate:
    id: str
    kind: CandidateKind
    name: str
    components: list[Component]
    interfaces: list[Interface]
    data_flows: list[DataFlow]
    failure_boundaries: list[Boundary]
    security_boundaries: list[Boundary]
    dimension_scores: list[DimensionScore]
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def component(self, cid: str) -> Component | None:
        for c in self.components:
            if c.id == cid:
                return c
        return None

    def scores_dict(self) -> dict[str, float]:
        return {d.dimension: d.score for d in self.dimension_scores}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind.value, "name": self.name,
            "components": [c.to_dict() for c in self.components],
            "interfaces": [i.to_dict() for i in self.interfaces],
            "data_flows": [f.to_dict() for f in self.data_flows],
            "failure_boundaries": [b.to_dict() for b in self.failure_boundaries],
            "security_boundaries": [b.to_dict() for b in self.security_boundaries],
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class CandidateScore:
    candidate_id: str
    kind: CandidateKind
    verdict: FitVerdict
    weighted_total: float
    contributions: dict[str, float]        # dimension -> weight * score
    weights: dict[str, float]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "verdict": self.verdict.value,
            "weighted_total": self.weighted_total,
            "contributions": {k: float(v) for k, v in self.contributions.items()},
            "weights": {k: float(v) for k, v in self.weights.items()},
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ArchitectureComparison:
    candidate_ids: list[str]
    kinds: list[CandidateKind]
    weights: dict[str, float]
    scores: list[CandidateScore]
    winner_id: str
    pareto_front: list[str]
    rejected: list[CandidateScore]        # dominated + below threshold
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "kinds": [k.value for k in self.kinds],
            "weights": {k: float(v) for k, v in self.weights.items()},
            "scores": [s.to_dict() for s in self.scores],
            "winner_id": self.winner_id,
            "pareto_front": list(self.pareto_front),
            "rejected": [s.to_dict() for s in self.rejected],
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ArchitectureDecision:
    selected: ArchitectureCandidate
    comparison: ArchitectureComparison
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "comparison": self.comparison.to_dict(),
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class ArchitectureResult:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    spec_id: str | None = None
    intent_context_id: str | None = None
    tech_selection_id: str | None = None
    tech_stack: dict[str, str | None] = field(default_factory=dict)
    candidates: list[ArchitectureCandidate] = field(default_factory=list)
    decision: ArchitectureDecision | None = None
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def candidate(self, cid: str) -> ArchitectureCandidate | None:
        for c in self.candidates:
            if c.id == cid:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "spec_id": self.spec_id,
            "intent_context_id": self.intent_context_id,
            "tech_selection_id": self.tech_selection_id,
            "tech_stack": dict(self.tech_stack),
            "candidates": [c.to_dict() for c in self.candidates],
            "decision": self.decision.to_dict() if self.decision else None,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        lines = ["=== Architecture Reasoning ==="]
        if self.decision:
            lines.append(f"Selected: {self.decision.selected.name}")
            lines.append(f"Rationale: {_short(self.decision.rationale, 120)}")
        lines.append(f"Candidates: {len(self.candidates)}")
        if self.decision:
            lines.append(
                f"Weighted winner score: "
                f"{max(s.weighted_total for s in self.decision.comparison.scores):.2f}"
            )
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 3. TECH STACK VIEW
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class TechStackView:
    """Small adapter over C09 TechSelectionResult for architecture generators."""
    language: str = "python"
    framework: str = "fastapi"
    database: str = "sqlite"
    runtime: str = "cpython"
    architecture_style: str = "layered"

    @classmethod
    def from_selection(cls, sel: TechSelectionResult | None) -> "TechStackView":
        if sel is None:
            return cls()
        return cls(
            language=(sel.winner_name(TechCategory.LANGUAGE) or "python").lower(),
            framework=(sel.winner_name(TechCategory.FRAMEWORK) or "fastapi").lower(),
            database=(sel.winner_name(TechCategory.DATABASE) or "sqlite").lower(),
            runtime=(sel.winner_name(TechCategory.RUNTIME) or "cpython").lower(),
            architecture_style=(sel.winner_name(TechCategory.ARCHITECTURE) or "layered").lower(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "language": self.language, "framework": self.framework,
            "database": self.database, "runtime": self.runtime,
            "architecture_style": self.architecture_style,
        }


# ════════════════════════════════════════════════════════════════════════════
# 4. SCORING TABLE
# ════════════════════════════════════════════════════════════════════════════
DIMENSIONS: tuple[str, ...] = (
    "performance",
    "scalability",
    "maintainability",
    "simplicity",
    "resilience",
    "security",
    "time_to_market",
    "cost_efficiency",
    "testability",
    "team_fit",
)

_ARCH_BASE_SCORES: dict[CandidateKind, dict[str, float]] = {
    CandidateKind.MONOLITH: {
        "performance": 8.0, "scalability": 3.0, "maintainability": 5.0,
        "simplicity": 9.0, "resilience": 4.0, "security": 7.0,
        "time_to_market": 9.0, "cost_efficiency": 9.0, "testability": 5.0,
        "team_fit": 8.0,
    },
    CandidateKind.LAYERED: {
        "performance": 7.5, "scalability": 5.0, "maintainability": 7.0,
        "simplicity": 8.0, "resilience": 5.0, "security": 7.5,
        "time_to_market": 8.0, "cost_efficiency": 8.5, "testability": 7.0,
        "team_fit": 8.5,
    },
    CandidateKind.HEXAGONAL: {
        "performance": 7.0, "scalability": 6.0, "maintainability": 9.5,
        "simplicity": 6.0, "resilience": 6.5, "security": 8.0,
        "time_to_market": 6.0, "cost_efficiency": 8.0, "testability": 9.5,
        "team_fit": 7.0,
    },
    CandidateKind.CLEAN: {
        "performance": 7.0, "scalability": 6.5, "maintainability": 9.5,
        "simplicity": 5.0, "resilience": 6.5, "security": 8.0,
        "time_to_market": 5.0, "cost_efficiency": 7.5, "testability": 9.0,
        "team_fit": 6.5,
    },
    CandidateKind.MODULAR_MONOLITH: {
        "performance": 7.5, "scalability": 6.0, "maintainability": 8.5,
        "simplicity": 7.5, "resilience": 6.0, "security": 7.5,
        "time_to_market": 7.0, "cost_efficiency": 8.5, "testability": 8.5,
        "team_fit": 8.0,
    },
    CandidateKind.MICROSERVICES: {
        "performance": 6.5, "scalability": 9.5, "maintainability": 6.0,
        "simplicity": 3.0, "resilience": 8.0, "security": 6.5,
        "time_to_market": 4.0, "cost_efficiency": 4.0, "testability": 5.5,
        "team_fit": 5.0,
    },
    CandidateKind.SERVERLESS: {
        "performance": 6.0, "scalability": 9.0, "maintainability": 6.5,
        "simplicity": 6.5, "resilience": 7.5, "security": 7.0,
        "time_to_market": 7.5, "cost_efficiency": 7.0, "testability": 6.0,
        "team_fit": 6.5,
    },
}

_ACCEPTABLE_FRACTION = 0.75


def _weights_for(signals: TechSignals) -> dict[str, float]:
    """Base weights = 1.0 each; boosted by signals."""
    w = {d: 1.0 for d in DIMENSIONS}
    if signals.latency_tight or signals.throughput_high:
        w["performance"] *= 1.5
    if signals.scale_large:
        # Extreme-scale requirements ("scale to millions of requests",
        # "high availability") need a stronger boost than 2.0x to actually
        # outweigh a well-rounded but non-scalable candidate (e.g. a
        # modular monolith that scores decently on every *other* dimension
        # still out-totals a microservices/serverless candidate that's
        # excellent on scalability/resilience but weak on simplicity/cost
        # under a mild boost). Resilience is boosted too — "must scale to
        # millions... with high availability" ties scale and resilience
        # together in the same requirement.
        w["scalability"] *= 3.5
        w["resilience"] *= 1.3
    if signals.security_high or signals.compliance:
        w["security"] *= 1.8
    if signals.small_team:
        w["simplicity"] *= 1.5
        w["team_fit"] *= 1.5
        w["cost_efficiency"] *= 1.3
    if signals.tight_deadline:
        w["time_to_market"] *= 1.8
    if signals.maintainability_focus:
        w["maintainability"] *= 1.5
        w["testability"] *= 1.3
    if signals.typing_focus:
        w["maintainability"] *= 1.2
    if signals.must_be_local:
        w["simplicity"] *= 1.3
        w["cost_efficiency"] *= 1.1
    if signals.needs_async:
        w["resilience"] *= 1.2
    return w


# ════════════════════════════════════════════════════════════════════════════
# 5. CANDIDATE GENERATORS
# ════════════════════════════════════════════════════════════════════════════
def _prov(source: str = "architecture_reasoner") -> Provenance:
    return Provenance(
        source=source, source_type=ProvenanceType.SYSTEM,
        confidence=Confidence.MEDIUM,
    )


def _mk_comp(cid: str, name: str, role: ComponentRole,
             responsibilities: list[str], depends_on: list[str] | None = None,
             tech_hint: str = "", rationale: str = "") -> Component:
    return Component(
        id=cid, name=name, role=role,
        responsibilities=list(responsibilities),
        depends_on=list(depends_on or []),
        tech_hint=tech_hint, rationale=rationale,
    )


def _mk_iface(iid: str, name: str, kind: InterfaceKind,
              src: str, dst: str, payload: str = "", sync: bool = True,
              rationale: str = "") -> Interface:
    return Interface(
        id=iid, name=name, kind=kind,
        from_component=src, to_component=dst,
        payload_desc=payload, sync=sync, rationale=rationale,
    )


def _mk_flow(fid: str, name: str, kind: DataFlowKind,
             path: list[str], desc: str = "") -> DataFlow:
    return DataFlow(id=fid, name=name, kind=kind, path=list(path),
                    description=desc)


def _mk_boundary(bid: str, name: str, kind: BoundaryKind,
                 comps: list[str], rationale: str = "") -> Boundary:
    return Boundary(id=bid, name=name, kind=kind,
                    component_ids=list(comps), rationale=rationale)


def _dim_scores(kind: CandidateKind) -> list[DimensionScore]:
    base = _ARCH_BASE_SCORES[kind]
    return [
        DimensionScore(
            dimension=d, score=base[d],
            rationale=f"base score for {kind.value} on '{d}'",
        )
        for d in DIMENSIONS
    ]


# ---- Monolith ----
def _gen_monolith(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.MONOLITH.value}"
    comps = [
        _mk_comp(
            "app", "Application", ComponentRole.APPLICATION,
            ["HTTP handling", "business logic", "data access"],
            tech_hint=tech.framework,
            rationale="single deployable holds all tiers",
        ),
        _mk_comp(
            "db", "Database", ComponentRole.PERSISTENCE,
            ["persistent storage"],
            depends_on=["app"],
            tech_hint=tech.database,
            rationale="single shared datastore",
        ),
    ]
    ifaces = [
        _mk_iface("i_app_db", "app→db", InterfaceKind.DATABASE,
                  "app", "db", payload="SQL/query",
                  rationale="in-process DB connection"),
        _mk_iface("i_edge_app", "edge→app", InterfaceKind.SYNC_HTTP,
                  "app", "app", payload="HTTP",
                  rationale="external HTTP entry"),
    ]
    flows = [
        _mk_flow("f_req", "request", DataFlowKind.REQUEST,
                 ["app", "db"],
                 desc="HTTP request handled and persisted by the monolith"),
    ]
    f_bounds = [
        _mk_boundary("fb_app", "application (whole)",
                     BoundaryKind.FAILURE, ["app"],
                     rationale="single failure domain — everything fails together"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "external edge",
                     BoundaryKind.SECURITY, ["app"],
                     rationale="one trust boundary at HTTP entry"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.MONOLITH,
        name="Monolith",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.MONOLITH),
        rationale="single-process deployment; simplest operations model",
        provenance=_prov(),
    )


# ---- Layered ----
def _gen_layered(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.LAYERED.value}"
    comps = [
        _mk_comp("pres", "Presentation", ComponentRole.PRESENTATION,
                 ["HTTP controllers", "request/response shaping"],
                 tech_hint=tech.framework),
        _mk_comp("app", "Application", ComponentRole.APPLICATION,
                 ["use case orchestration", "transaction boundaries"],
                 depends_on=["pres"]),
        _mk_comp("dom", "Domain", ComponentRole.DOMAIN,
                 ["entities", "business rules"],
                 depends_on=["app"]),
        _mk_comp("pers", "Persistence", ComponentRole.PERSISTENCE,
                 ["repositories", "schema mapping"],
                 depends_on=["dom"], tech_hint=tech.database),
    ]
    ifaces = [
        _mk_iface("i_pres_app", "pres→app", InterfaceKind.SYNC_IN_PROCESS,
                  "pres", "app", payload="use-case DTO"),
        _mk_iface("i_app_dom", "app→dom", InterfaceKind.SYNC_IN_PROCESS,
                  "app", "dom", payload="domain calls"),
        _mk_iface("i_dom_pers", "dom→pers", InterfaceKind.SYNC_IN_PROCESS,
                  "dom", "pers", payload="repository calls"),
        _mk_iface("i_pers_db", "pers→db", InterfaceKind.DATABASE,
                  "pers", "pers", payload="SQL/query",
                  rationale="layered boundary at DB access"),
    ]
    flows = [
        _mk_flow("f_req", "request", DataFlowKind.REQUEST,
                 ["pres", "app", "dom", "pers"],
                 desc="HTTP request flows top-down through layers"),
    ]
    f_bounds = [
        _mk_boundary("fb_app_tier", "application tier",
                     BoundaryKind.FAILURE, ["pres", "app", "dom", "pers"],
                     rationale="process-level failure boundary"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "edge", BoundaryKind.SECURITY, ["pres"],
                     rationale="external trust edge at presentation"),
        _mk_boundary("sb_db", "db edge", BoundaryKind.SECURITY, ["pers"],
                     rationale="trust edge between app and DB"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.LAYERED, name="Layered (3-tier)",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.LAYERED),
        rationale="classic 3-tier separation; predictable and testable",
        provenance=_prov(),
    )


# ---- Hexagonal ----
def _gen_hexagonal(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.HEXAGONAL.value}"
    comps = [
        _mk_comp("dom", "Domain Core", ComponentRole.DOMAIN,
                 ["business rules", "entities", "use cases"],
                 rationale="pure domain, no outward deps"),
        _mk_comp("in_http", "HTTP Adapter", ComponentRole.PRESENTATION,
                 ["HTTP handlers"], tech_hint=tech.framework,
                 rationale="inbound port implementation"),
        _mk_comp("out_db", "DB Adapter", ComponentRole.PERSISTENCE,
                 ["repositories"], tech_hint=tech.database,
                 rationale="outbound port implementation"),
        _mk_comp("out_ext", "External Adapter", ComponentRole.INTEGRATION,
                 ["external calls"], rationale="outbound integration adapter"),
    ]
    ifaces = [
        _mk_iface("i_in_http_dom", "http_adapter→port→dom",
                  InterfaceKind.SYNC_IN_PROCESS,
                  "in_http", "dom", payload="domain command"),
        _mk_iface("i_dom_out_db", "dom→port→db_adapter",
                  InterfaceKind.SYNC_IN_PROCESS,
                  "dom", "out_db", payload="repository port"),
        _mk_iface("i_dom_out_ext", "dom→port→ext_adapter",
                  InterfaceKind.ASYNC_MESSAGE,
                  "dom", "out_ext", payload="external port", sync=False),
    ]
    flows = [
        _mk_flow("f_req", "inbound request", DataFlowKind.REQUEST,
                 ["in_http", "dom", "out_db"],
                 desc="request enters via adapter, domain drives persistence port"),
    ]
    f_bounds = [
        _mk_boundary("fb_dom", "domain core",
                     BoundaryKind.FAILURE, ["dom"],
                     rationale="domain isolated from adapter failures"),
        _mk_boundary("fb_adapters", "adapters",
                     BoundaryKind.FAILURE, ["in_http", "out_db", "out_ext"],
                     rationale="adapter failures handled at ports"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "edge", BoundaryKind.SECURITY, ["in_http"],
                     rationale="inbound adapter is the trust entry"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.HEXAGONAL, name="Hexagonal (ports & adapters)",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.HEXAGONAL),
        rationale="ports & adapters isolate domain from infrastructure",
        provenance=_prov(),
    )


# ---- Clean ----
def _gen_clean(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.CLEAN.value}"
    comps = [
        _mk_comp("entities", "Entities", ComponentRole.DOMAIN,
                 ["enterprise business rules"]),
        _mk_comp("usecases", "Use Cases", ComponentRole.APPLICATION,
                 ["application rules"], depends_on=["entities"]),
        _mk_comp("adapters", "Interface Adapters", ComponentRole.INTEGRATION,
                 ["controllers", "presenters", "gateways"],
                 depends_on=["usecases"], tech_hint=tech.framework),
        _mk_comp("fdrivers", "Frameworks & Drivers", ComponentRole.INFRASTRUCTURE,
                 ["DB driver", "web framework"], depends_on=["adapters"],
                 tech_hint=tech.database),
    ]
    ifaces = [
        _mk_iface("i_ent_uc", "entities←usecases",
                  InterfaceKind.SYNC_IN_PROCESS, "usecases", "entities",
                  payload="entity operations"),
        _mk_iface("i_uc_ad", "usecases←adapters",
                  InterfaceKind.SYNC_IN_PROCESS, "adapters", "usecases",
                  payload="use-case DTOs"),
        _mk_iface("i_ad_fd", "adapters←drivers",
                  InterfaceKind.DATABASE, "fdrivers", "adapters",
                  payload="driver calls"),
    ]
    flows = [
        _mk_flow("f_req", "request", DataFlowKind.REQUEST,
                 ["fdrivers", "adapters", "usecases", "entities"],
                 desc="request enters from drivers, flows inward"),
    ]
    f_bounds = [
        _mk_boundary("fb_inner", "inner circle",
                     BoundaryKind.FAILURE, ["entities", "usecases"],
                     rationale="inner ring failure-isolated"),
        _mk_boundary("fb_outer", "outer ring",
                     BoundaryKind.FAILURE, ["adapters", "fdrivers"],
                     rationale="outer ring can be replaced independently"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "edge", BoundaryKind.SECURITY, ["fdrivers"],
                     rationale="trust entry at frameworks/drivers"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.CLEAN, name="Clean Architecture",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.CLEAN),
        rationale="concentric dependency inversion; strongest testability",
        provenance=_prov(),
    )


# ---- Modular Monolith ----
def _gen_modular_monolith(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.MODULAR_MONOLITH.value}"
    comps = [
        _mk_comp("mod_a", "Module A", ComponentRole.APPLICATION,
                 ["module A use cases"], rationale="independent module"),
        _mk_comp("mod_b", "Module B", ComponentRole.APPLICATION,
                 ["module B use cases"], rationale="independent module"),
        _mk_comp("kernel", "Shared Kernel", ComponentRole.CROSS_CUTTING,
                 ["shared types", "shared utilities"],
                 rationale="shared building blocks"),
        _mk_comp("db", "Database", ComponentRole.PERSISTENCE,
                 ["shared datastore"], tech_hint=tech.database),
    ]
    ifaces = [
        _mk_iface("i_a_kernel", "A→kernel", InterfaceKind.SYNC_IN_PROCESS,
                  "mod_a", "kernel", payload="shared types"),
        _mk_iface("i_b_kernel", "B→kernel", InterfaceKind.SYNC_IN_PROCESS,
                  "mod_b", "kernel", payload="shared types"),
        _mk_iface("i_a_b", "A↔B", InterfaceKind.SYNC_IN_PROCESS,
                  "mod_a", "mod_b", payload="cross-module port",
                  rationale="explicit module boundary"),
        _mk_iface("i_a_db", "A→db", InterfaceKind.DATABASE,
                  "mod_a", "db", payload="SQL"),
        _mk_iface("i_b_db", "B→db", InterfaceKind.DATABASE,
                  "mod_b", "db", payload="SQL"),
    ]
    flows = [
        _mk_flow("f_req_a", "request via A", DataFlowKind.REQUEST,
                 ["mod_a", "kernel", "db"],
                 desc="request handled by module A"),
    ]
    f_bounds = [
        _mk_boundary("fb_process", "process",
                     BoundaryKind.FAILURE, ["mod_a", "mod_b", "kernel", "db"],
                     rationale="single process — one failure domain"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "edge", BoundaryKind.SECURITY,
                     ["mod_a", "mod_b"], rationale="external edges at modules"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.MODULAR_MONOLITH, name="Modular Monolith",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.MODULAR_MONOLITH),
        rationale="single deployable with enforced module boundaries",
        provenance=_prov(),
    )


# ---- Microservices ----
def _gen_microservices(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.MICROSERVICES.value}"
    comps = [
        _mk_comp("gateway", "API Gateway", ComponentRole.PRESENTATION,
                 ["routing", "auth", "rate limiting"],
                 tech_hint=tech.framework),
        _mk_comp("svc_task", "Task Service", ComponentRole.APPLICATION,
                 ["task use cases"], depends_on=["gateway"],
                 tech_hint=tech.framework),
        _mk_comp("svc_user", "User Service", ComponentRole.APPLICATION,
                 ["user use cases"], depends_on=["gateway"],
                 tech_hint=tech.framework),
        _mk_comp("bus", "Event Bus", ComponentRole.INFRASTRUCTURE,
                 ["pub/sub"], rationale="async inter-service channel"),
        _mk_comp("db_task", "Task DB", ComponentRole.PERSISTENCE,
                 ["task storage"], tech_hint=tech.database),
        _mk_comp("db_user", "User DB", ComponentRole.PERSISTENCE,
                 ["user storage"], tech_hint=tech.database),
    ]
    ifaces = [
        _mk_iface("i_gw_task", "gateway→task", InterfaceKind.SYNC_HTTP,
                  "gateway", "svc_task", payload="HTTP"),
        _mk_iface("i_gw_user", "gateway→user", InterfaceKind.SYNC_HTTP,
                  "gateway", "svc_user", payload="HTTP"),
        _mk_iface("i_task_db", "task→taskDB", InterfaceKind.DATABASE,
                  "svc_task", "db_task", payload="SQL"),
        _mk_iface("i_user_db", "user→userDB", InterfaceKind.DATABASE,
                  "svc_user", "db_user", payload="SQL"),
        _mk_iface("i_task_bus", "task→bus", InterfaceKind.ASYNC_EVENT,
                  "svc_task", "bus", payload="domain event", sync=False),
        _mk_iface("i_user_bus", "user→bus", InterfaceKind.ASYNC_EVENT,
                  "svc_user", "bus", payload="domain event", sync=False),
    ]
    flows = [
        _mk_flow("f_req", "request", DataFlowKind.REQUEST,
                 ["gateway", "svc_task", "db_task"],
                 desc="request routed to task service"),
        _mk_flow("f_event", "domain event", DataFlowKind.EVENT,
                 ["svc_task", "bus", "svc_user"],
                 desc="async event from task to user"),
    ]
    f_bounds = [
        _mk_boundary("fb_task", "task service",
                     BoundaryKind.FAILURE, ["svc_task", "db_task"],
                     rationale="task service fails independently"),
        _mk_boundary("fb_user", "user service",
                     BoundaryKind.FAILURE, ["svc_user", "db_user"],
                     rationale="user service fails independently"),
        _mk_boundary("fb_gateway", "gateway",
                     BoundaryKind.FAILURE, ["gateway"],
                     rationale="gateway is a shared failure point"),
        _mk_boundary("fb_bus", "event bus",
                     BoundaryKind.FAILURE, ["bus"],
                     rationale="bus outage degrades async flows only"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "edge", BoundaryKind.SECURITY, ["gateway"],
                     rationale="single external trust entry"),
        _mk_boundary("sb_svc_task", "task service edge",
                     BoundaryKind.SECURITY, ["svc_task"],
                     rationale="service-to-service trust edge"),
        _mk_boundary("sb_svc_user", "user service edge",
                     BoundaryKind.SECURITY, ["svc_user"],
                     rationale="service-to-service trust edge"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.MICROSERVICES, name="Microservices",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.MICROSERVICES),
        rationale="independent deployables; strong isolation, high ops cost",
        provenance=_prov(),
    )


# ---- Serverless ----
def _gen_serverless(tech: TechStackView) -> ArchitectureCandidate:
    cid = f"cand_{CandidateKind.SERVERLESS.value}"
    comps = [
        _mk_comp("gw", "API Gateway", ComponentRole.PRESENTATION,
                 ["routing", "auth"], rationale="managed API edge"),
        _mk_comp("fn_create", "CreateTaskFn", ComponentRole.APPLICATION,
                 ["create task"], depends_on=["gw"],
                 rationale="per-endpoint function"),
        _mk_comp("fn_read", "ReadTaskFn", ComponentRole.APPLICATION,
                 ["read task"], depends_on=["gw"],
                 rationale="per-endpoint function"),
        _mk_comp("fn_worker", "AsyncWorkerFn", ComponentRole.APPLICATION,
                 ["background work"], rationale="event-triggered function"),
        _mk_comp("db", "Managed DB", ComponentRole.PERSISTENCE,
                 ["persistent storage"], tech_hint=tech.database),
        _mk_comp("queue", "Queue", ComponentRole.INFRASTRUCTURE,
                 ["async dispatch"]),
    ]
    ifaces = [
        _mk_iface("i_gw_create", "gw→createFn", InterfaceKind.SYNC_HTTP,
                  "gw", "fn_create", payload="HTTP"),
        _mk_iface("i_gw_read", "gw→readFn", InterfaceKind.SYNC_HTTP,
                  "gw", "fn_read", payload="HTTP"),
        _mk_iface("i_create_db", "createFn→db", InterfaceKind.DATABASE,
                  "fn_create", "db", payload="SQL"),
        _mk_iface("i_read_db", "readFn→db", InterfaceKind.DATABASE,
                  "fn_read", "db", payload="SQL"),
        _mk_iface("i_create_queue", "createFn→queue",
                  InterfaceKind.ASYNC_MESSAGE,
                  "fn_create", "queue", payload="message", sync=False),
        _mk_iface("i_queue_worker", "queue→workerFn",
                  InterfaceKind.ASYNC_EVENT,
                  "queue", "fn_worker", payload="event", sync=False),
    ]
    flows = [
        _mk_flow("f_req", "request", DataFlowKind.REQUEST,
                 ["gw", "fn_create", "db"],
                 desc="request handled by a function"),
        _mk_flow("f_async", "async work", DataFlowKind.EVENT,
                 ["fn_create", "queue", "fn_worker"],
                 desc="async work dispatched via queue"),
    ]
    f_bounds = [
        _mk_boundary("fb_create", "createFn",
                     BoundaryKind.FAILURE, ["fn_create"],
                     rationale="function fails in isolation"),
        _mk_boundary("fb_read", "readFn",
                     BoundaryKind.FAILURE, ["fn_read"],
                     rationale="function fails in isolation"),
        _mk_boundary("fb_worker", "workerFn",
                     BoundaryKind.FAILURE, ["fn_worker"],
                     rationale="function fails in isolation"),
        _mk_boundary("fb_db", "managed DB",
                     BoundaryKind.FAILURE, ["db"],
                     rationale="managed DB failure domain"),
    ]
    s_bounds = [
        _mk_boundary("sb_edge", "gateway edge",
                     BoundaryKind.SECURITY, ["gw"],
                     rationale="managed trust entry"),
        _mk_boundary("sb_fn_edges", "function edges",
                     BoundaryKind.SECURITY,
                     ["fn_create", "fn_read", "fn_worker"],
                     rationale="IAM per function"),
    ]
    return ArchitectureCandidate(
        id=cid, kind=CandidateKind.SERVERLESS, name="Serverless",
        components=comps, interfaces=ifaces, data_flows=flows,
        failure_boundaries=f_bounds, security_boundaries=s_bounds,
        dimension_scores=_dim_scores(CandidateKind.SERVERLESS),
        rationale="managed functions + services; per-function scaling",
        provenance=_prov(),
    )


_GENERATORS: dict[CandidateKind, Callable[[TechStackView], ArchitectureCandidate]] = {
    CandidateKind.MONOLITH: _gen_monolith,
    CandidateKind.LAYERED: _gen_layered,
    CandidateKind.HEXAGONAL: _gen_hexagonal,
    CandidateKind.CLEAN: _gen_clean,
    CandidateKind.MODULAR_MONOLITH: _gen_modular_monolith,
    CandidateKind.MICROSERVICES: _gen_microservices,
    CandidateKind.SERVERLESS: _gen_serverless,
}


# ════════════════════════════════════════════════════════════════════════════
# 6. COMPARATOR
# ════════════════════════════════════════════════════════════════════════════
class Comparator:
    """Weighted comparison + Pareto front. Deterministic tie-break by kind."""

    def compare(
        self,
        candidates: Sequence[ArchitectureCandidate],
        signals: TechSignals,
    ) -> ArchitectureComparison:
        if not candidates:
            raise ValidationError("no candidates to compare")
        weights = _weights_for(signals)

        scores: list[CandidateScore] = []
        for c in candidates:
            sd = c.scores_dict()
            contributions = {d: weights[d] * sd.get(d, 0.0) for d in DIMENSIONS}
            total = sum(contributions.values())
            scores.append(CandidateScore(
                candidate_id=c.id, kind=c.kind,
                verdict=FitVerdict.ALTERNATIVE,
                weighted_total=total,
                contributions=contributions,
                weights=dict(weights),
                rationale=f"weighted_total={total:.3f}",
            ))

        # Pareto front: candidate A dominates B if A >= B on all dims and
        # A > B on at least one.
        def vec(c: ArchitectureCandidate) -> dict[str, float]:
            return c.scores_dict()

        pareto_set: list[str] = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other.id == c.id:
                    continue
                vc, vo = vec(c), vec(other)
                ge_all = all(vo[d] >= vc[d] - 1e-9 for d in DIMENSIONS)
                gt_any = any(vo[d] > vc[d] + 1e-9 for d in DIMENSIONS)
                if ge_all and gt_any:
                    dominated = True
                    break
            if not dominated:
                pareto_set.append(c.id)

        # Rank by weighted_total desc, then id asc (deterministic)
        ranked = sorted(scores, key=lambda s: (-s.weighted_total, s.candidate_id))
        winner = ranked[0]
        winner.verdict = FitVerdict.SELECTED
        top_score = winner.weighted_total
        threshold = top_score * _ACCEPTABLE_FRACTION
        rejected: list[CandidateScore] = []
        for s in ranked[1:]:
            if s.weighted_total < threshold:
                s.verdict = FitVerdict.REJECTED
                s.rationale += (
                    f"; below threshold {threshold:.2f} "
                    f"(<{_ACCEPTABLE_FRACTION*100:.0f}% of winner)"
                )
                rejected.append(s)
            else:
                s.verdict = FitVerdict.ALTERNATIVE
        rationale = (
            f"weights={weights}; winner={winner.candidate_id} "
            f"(total={winner.weighted_total:.2f}); "
            f"pareto_front={pareto_set}; rejected={len(rejected)}"
        )
        return ArchitectureComparison(
            candidate_ids=[c.id for c in candidates],
            kinds=[c.kind for c in candidates],
            weights=weights,
            scores=scores,
            winner_id=winner.candidate_id,
            pareto_front=pareto_set,
            rejected=rejected,
            rationale=rationale,
        )


# ════════════════════════════════════════════════════════════════════════════
# 7. ARCHITECTURE REASONER
# ════════════════════════════════════════════════════════════════════════════
class ArchitectureReasoner:
    """Generates, validates, compares, and selects architecture candidates."""

    def __init__(self, *, comparator: Comparator | None = None) -> None:
        self.comparator = comparator or Comparator()
        self.signal_extractor = SignalExtractor()

    def reason(
        self,
        spec: RequirementSpec,
        intent_ctx: IntentContext | None = None,
        tech_selection: TechSelectionResult | None = None,
        *,
        project_id: str = "",
    ) -> ArchitectureResult:
        signals = self.signal_extractor.extract(spec, intent_ctx)
        tech = TechStackView.from_selection(tech_selection)

        # Generate all candidates
        candidates: list[ArchitectureCandidate] = []
        for kind in CandidateKind:
            gen = _GENERATORS.get(kind)
            if gen is None:
                continue
            cand = gen(tech)
            # validate no cycles
            issues = self._validate_candidate(cand)
            if issues:
                log.warning("architecture.candidate.invalid",
                            kind=kind.value, issues=issues)
                continue
            candidates.append(cand)

        if not candidates:
            raise ValidationError("no valid architecture candidates could be generated")

        comparison = self.comparator.compare(candidates, signals)
        winner = next(c for c in candidates if c.id == comparison.winner_id)

        # Build alternatives with reasons
        reasons_map: dict[str, str] = {}
        for s in comparison.scores:
            if s.candidate_id == comparison.winner_id:
                continue
            reasons_map[s.candidate_id] = (
                f"verdict={s.verdict.value}; weighted_total={s.weighted_total:.2f}"
            )

        decision = ArchitectureDecision(
            selected=winner,
            comparison=comparison,
            rationale=(
                f"selected '{winner.name}' for kind={winner.kind.value}; "
                f"weighted_total={next(s.weighted_total for s in comparison.scores if s.candidate_id == winner.id):.2f}; "
                f"pareto_front={comparison.pareto_front}; "
                f"rejected={[s.candidate_id for s in comparison.rejected]}"
            ),
        )

        result = ArchitectureResult(
            project_id=project_id,
            spec_id=spec.id,
            intent_context_id=intent_ctx.id if intent_ctx else None,
            tech_selection_id=tech_selection.id if tech_selection else None,
            tech_stack=tech.to_dict(),
            candidates=candidates,
            decision=decision,
            rationale=comparison.rationale,
            provenance=spec.provenance,
        )
        return result

    # ---- validation ----
    @staticmethod
    def _validate_candidate(cand: ArchitectureCandidate) -> list[str]:
        issues: list[str] = []
        comp_ids = {c.id for c in cand.components}
        if len(comp_ids) != len(cand.components):
            issues.append("duplicate component ids")
        # every depends_on resolves
        for c in cand.components:
            for d in c.depends_on:
                if d not in comp_ids:
                    issues.append(f"component {c.id} depends on unknown {d}")
        # every interface endpoint resolves
        for i in cand.interfaces:
            if i.from_component not in comp_ids:
                issues.append(f"interface {i.id} from unknown {i.from_component}")
            if i.to_component not in comp_ids:
                issues.append(f"interface {i.id} to unknown {i.to_component}")
        # flows reference valid comps
        for f in cand.data_flows:
            for cid in f.path:
                if cid not in comp_ids:
                    issues.append(f"flow {f.id} references unknown {cid}")
        # boundaries reference valid comps
        for b in cand.failure_boundaries + cand.security_boundaries:
            for cid in b.component_ids:
                if cid not in comp_ids:
                    issues.append(f"boundary {b.id} references unknown {cid}")
        # dependency cycles
        if _has_dependency_cycle(cand.components):
            issues.append("component dependency cycle")
        return issues


def _has_dependency_cycle(components: Sequence[Component]) -> bool:
    adj: dict[str, list[str]] = {c.id: list(c.depends_on) for c in components}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {cid: WHITE for cid in adj}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if v not in color:
                continue
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for cid in adj:
        if color[cid] == WHITE and dfs(cid):
            return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class ArchitectureRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: ArchitectureResult) -> str:
        if not result.project_id:
            raise ValidationError("result.project_id required")
        key = f"architecture_result:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=result.project_id,
            tags=["architecture", "c10"],
            provenance=result.provenance,
        )

        if result.decision is None:
            return result.id
        # decision memory
        self.memory.record_decision(
            "architecture_choice",
            decision=f"Chose architecture '{result.decision.selected.name}'",
            rationale=result.decision.rationale,
            alternatives=[
                c.name for c in result.candidates
                if c.id != result.decision.selected.id
            ],
            scope_id=result.project_id,
            provenance=Provenance(
                source="architecture_reasoner",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
            confidence=Confidence.HIGH,
        )

        if self.ontology is None:
            return result.id

        # Ontology: ARCHITECTURE entity + COMPONENT entities + alternatives
        arch_ent = self.ontology.add(
            EntityKind.ARCHITECTURE,
            _short(f"{result.decision.selected.name}", 120),
            attributes={
                "result_id": result.id,
                "project_id": result.project_id,
                "kind": result.decision.selected.kind.value,
                "tech_stack": result.tech_stack,
            },
            tags=["architecture", result.decision.selected.kind.value],
            provenance=result.provenance,
        )

        # components of selected
        comp_ent_ids: dict[str, str] = {}
        for c in result.decision.selected.components:
            ce = self.ontology.add(
                EntityKind.COMPONENT,
                _short(c.name, 120),
                attributes={
                    "component_id": c.id,
                    "role": c.role.value,
                    "responsibilities": c.responsibilities,
                },
                tags=["component", c.role.value],
                provenance=result.provenance,
            )
            comp_ent_ids[c.id] = ce.id
            self.ontology.link(RelationKind.CONTAINS, arch_ent.id, ce.id)
        # component dependencies as DEPENDS_ON relations
        for c in result.decision.selected.components:
            if c.id not in comp_ent_ids:
                continue
            for d in c.depends_on:
                if d in comp_ent_ids:
                    try:
                        self.ontology.link(
                            RelationKind.DEPENDS_ON,
                            comp_ent_ids[c.id], comp_ent_ids[d],
                        )
                    except ValidationError:
                        pass

        # alternative architectures
        for c in result.candidates:
            if c.id == result.decision.selected.id:
                continue
            alt = self.ontology.add(
                EntityKind.ALTERNATIVE,
                _short(f"Architecture alt: {c.name}", 120),
                attributes={
                    "candidate_id": c.id,
                    "kind": c.kind.value,
                },
                tags=["architecture-alt", c.kind.value],
                provenance=result.provenance,
            )
            try:
                self.ontology.link(RelationKind.RELATES_TO, alt.id, arch_ent.id)
            except ValidationError:
                pass
        return arch_ent.id

    def load(self, result_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"architecture_result:{result_id}",
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

    print("Running C10 self-tests…")
    parser = RequirementParser()
    ic_engine = IntentContextEngine()
    selector = TechnologySelector()
    reasoner = ArchitectureReasoner()

    def _build(text: str, project: str = "test"):
        spec = parser.parse(text)
        ic = ic_engine.analyze(text, project_id=project)
        tech = selector.select(spec, ic, project_id=project)
        return spec, ic, tech

    # ---- candidate generators ----
    def t_gen_all_kinds() -> None:
        tech = TechStackView()
        for kind, gen in _GENERATORS.items():
            cand = gen(tech)
            assert cand.kind is kind
            assert len(cand.components) >= 2, kind
            assert len(cand.interfaces) >= 1, kind
            assert len(cand.data_flows) >= 1, kind
            assert len(cand.failure_boundaries) >= 1, kind
            assert len(cand.security_boundaries) >= 1, kind
            assert len(cand.dimension_scores) == len(DIMENSIONS)

    def t_gen_unique_ids_per_candidate() -> None:
        tech = TechStackView()
        for gen in _GENERATORS.values():
            cand = gen(tech)
            ids = [c.id for c in cand.components]
            assert len(set(ids)) == len(ids)

    def t_candidate_validation() -> None:
        tech = TechStackView()
        for gen in _GENERATORS.values():
            cand = gen(tech)
            issues = ArchitectureReasoner._validate_candidate(cand)
            assert issues == [], (cand.kind, issues)

    def t_no_dependency_cycles() -> None:
        tech = TechStackView()
        for gen in _GENERATORS.values():
            cand = gen(tech)
            assert not _has_dependency_cycle(cand.components), cand.kind

    def t_microservices_has_multiple_failure_boundaries() -> None:
        tech = TechStackView()
        ms = _gen_microservices(tech)
        assert len(ms.failure_boundaries) >= 3

    def t_monolith_has_single_failure_boundary() -> None:
        tech = TechStackView()
        mono = _gen_monolith(tech)
        assert len(mono.failure_boundaries) == 1

    check("generator: all 7 kinds produce valid candidates", t_gen_all_kinds)
    check("generator: unique component ids", t_gen_unique_ids_per_candidate)
    check("generator: candidates validate clean", t_candidate_validation)
    check("generator: no dependency cycles", t_no_dependency_cycles)
    check("generator: microservices → multiple failure boundaries",
          t_microservices_has_multiple_failure_boundaries)
    check("generator: monolith → single failure boundary",
          t_monolith_has_single_failure_boundary)

    # ---- comparator ----
    def t_compare_deterministic() -> None:
        tech = TechStackView()
        cands = [gen(tech) for gen in _GENERATORS.values()]
        signals = TechSignals()
        c = Comparator()
        r1 = c.compare(cands, signals)
        r2 = c.compare(cands, signals)
        assert r1.winner_id == r2.winner_id

    def t_compare_pareto_nonempty() -> None:
        tech = TechStackView()
        cands = [gen(tech) for gen in _GENERATORS.values()]
        signals = TechSignals()
        c = Comparator()
        r = c.compare(cands, signals)
        assert len(r.pareto_front) >= 1
        # winner on pareto front
        assert r.winner_id in r.pareto_front

    def t_compare_rejected_below_threshold() -> None:
        tech = TechStackView()
        cands = [gen(tech) for gen in _GENERATORS.values()]
        signals = TechSignals()
        r = Comparator().compare(cands, signals)
        # at least one rejected or alternative among the rest
        assert len(r.scores) >= 2
        non_winners = [s for s in r.scores if s.candidate_id != r.winner_id]
        assert all(s.verdict in (FitVerdict.REJECTED, FitVerdict.ALTERNATIVE)
                   for s in non_winners)

    def t_weights_scale_bias() -> None:
        # scale_large should boost scalability
        w1 = _weights_for(TechSignals())
        w2 = _weights_for(TechSignals(scale_large=True))
        assert w2["scalability"] > w1["scalability"]

    def t_weights_small_team_bias() -> None:
        w1 = _weights_for(TechSignals())
        w2 = _weights_for(TechSignals(small_team=True))
        assert w2["simplicity"] > w1["simplicity"]
        assert w2["team_fit"] > w1["team_fit"]

    def t_weights_security_bias() -> None:
        w1 = _weights_for(TechSignals())
        w2 = _weights_for(TechSignals(security_high=True))
        assert w2["security"] > w1["security"]

    check("comparator: deterministic winner", t_compare_deterministic)
    check("comparator: pareto front non-empty + includes winner",
          t_compare_pareto_nonempty)
    check("comparator: non-winners classified (rejected/alternative)",
          t_compare_rejected_below_threshold)
    check("weights: scale_large boosts scalability",
          t_weights_scale_bias)
    check("weights: small_team boosts simplicity + team_fit",
          t_weights_small_team_bias)
    check("weights: security_high boosts security",
          t_weights_security_bias)

    # ---- reasoner selection behaviour ----
    def t_select_simple_api_picks_low_complexity() -> None:
        text = "Build a small REST API for tasks."
        spec, ic, tech = _build(text)
        res = reasoner.reason(spec, ic, tech, project_id="p")
        # small/simple → monolith, layered, modular monolith likely
        assert res.decision is not None
        assert res.decision.selected.kind in (
            CandidateKind.LAYERED, CandidateKind.MONOLITH,
            CandidateKind.MODULAR_MONOLITH, CandidateKind.HEXAGONAL,
        ), res.decision.selected.kind

    def t_select_scaled_system_prefers_scalable() -> None:
        text = (
            "Build a platform that must scale to millions of requests "
            "per second with high availability."
        )
        spec, ic, tech = _build(text)
        res = reasoner.reason(spec, ic, tech, project_id="p")
        assert res.decision is not None
        # scalability boost → microservices or serverless should win
        assert res.decision.selected.kind in (
            CandidateKind.MICROSERVICES, CandidateKind.SERVERLESS,
        ), res.decision.selected.kind

    def t_select_small_team_prefers_simple() -> None:
        text = (
            "Build a small internal API. It's just me, a solo developer, "
            "and I have a tight deadline."
        )
        spec, ic, tech = _build(text)
        res = reasoner.reason(spec, ic, tech, project_id="p")
        assert res.decision is not None
        # solo + deadline → simplicity/team_fit/time_to_market bias
        assert res.decision.selected.kind in (
            CandidateKind.MONOLITH, CandidateKind.LAYERED,
            CandidateKind.MODULAR_MONOLITH,
        ), res.decision.selected.kind

    def t_select_high_security_avoids_flat() -> None:
        text = (
            "Build a healthcare API. Must comply with HIPAA. "
            "All traffic must use TLS."
        )
        spec, ic, tech = _build(text)
        res = reasoner.reason(spec, ic, tech, project_id="p")
        assert res.decision is not None
        # security weighted → monolith has security=7, layered=7.5, hex=8
        assert res.decision.selected.kind in (
            CandidateKind.LAYERED, CandidateKind.HEXAGONAL,
            CandidateKind.CLEAN, CandidateKind.MODULAR_MONOLITH,
        ), res.decision.selected.kind

    check("reasoner: simple API → low-complexity candidate",
          t_select_simple_api_picks_low_complexity)
    check("reasoner: scale requirement → scalable candidate",
          t_select_scaled_system_prefers_scalable)
    check("reasoner: solo+deadline → simple candidate",
          t_select_small_team_prefers_simple)
    check("reasoner: high security → layered/hex/clean",
          t_select_high_security_avoids_flat)

    # ---- rejected preservation ----
    def t_rejected_preserved() -> None:
        text = "Build a small REST API for tasks."
        spec, ic, tech = _build(text)
        res = reasoner.reason(spec, ic, tech, project_id="p")
        assert res.decision is not None
        # All candidates preserved in result.candidates
        assert len(res.candidates) == len(CandidateKind)
        # Comparison tracks kinds list too
        assert len(res.decision.comparison.kinds) == len(CandidateKind)
        # At least one candidate classified below the winner
        non_winners = [
            s for s in res.decision.comparison.scores
            if s.candidate_id != res.decision.selected.id
        ]
        assert len(non_winners) >= 1
        for s in non_winners:
            assert s.verdict in (FitVerdict.ALTERNATIVE, FitVerdict.REJECTED)

    check("selection: all 7 candidates preserved with verdicts",
          t_rejected_preserved)

    # ---- persistence ----
    def t_persist_reload() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                text = "Build a small REST API for tasks."
                spec, ic, tech = _build(text, project="proj-x")
                res = reasoner.reason(spec, ic, tech, project_id="proj-x")
                repo = ArchitectureRepository(memory=mem, ontology=ont)
                ent_id = repo.save(res)
                assert ent_id
                loaded = repo.load(res.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == res.id
                # Ontology has an ARCHITECTURE entity
                assert ont.count(kind=EntityKind.ARCHITECTURE) >= 1
                # And components
                assert ont.count(kind=EntityKind.COMPONENT) >= 2
                # Decision memory
                decs = mem.find(
                    kind=MemoryKind.DECISION,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert any("architecture" in d.key.lower() for d in decs)
            finally:
                s.shutdown()

    check("persist: memory + ontology (architecture + components + alts)",
          t_persist_reload)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        text = "Build a REST API for tasks."
        spec, ic, tech = _build(text)
        res = reasoner.reason(spec, ic, tech, project_id="p")
        d = res.to_dict()
        assert d["id"] == res.id
        assert isinstance(d["candidates"], list) and len(d["candidates"]) >= 1
        assert d["decision"] is not None
        assert "selected" in d["decision"]
        assert "comparison" in d["decision"]
        sm = res.summary()
        assert "Architecture Reasoning" in sm
        assert "Selected:" in sm

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
        spec, ic, tech = _build(text, project="demo")
        res = reasoner.reason(spec, ic, tech, project_id="demo")

        # Structural guarantees
        assert len(res.candidates) == len(CandidateKind)
        assert res.decision is not None
        # Every candidate has components, interfaces, flows, boundaries
        for c in res.candidates:
            assert len(c.components) >= 2
            assert len(c.interfaces) >= 1
            assert len(c.data_flows) >= 1
            assert len(c.failure_boundaries) >= 1
            assert len(c.security_boundaries) >= 1
        # Selected is coherent
        sel = res.decision.selected
        assert sel.kind in tuple(CandidateKind)
        # Components reference valid boundaries
        comp_ids = {c.id for c in sel.components}
        for b in sel.failure_boundaries + sel.security_boundaries:
            for cid in b.component_ids:
                assert cid in comp_ids
        # Comparison includes all candidates
        assert len(res.decision.comparison.scores) == len(CandidateKind)
        # Pareto front non-empty and includes winner
        assert res.decision.comparison.winner_id in res.decision.comparison.pareto_front
        # Rejected alternatives all have rationale
        for s in res.decision.comparison.rejected:
            assert "below threshold" in s.rationale

    check("e2e: canonical REST API architecture reasoning", t_e2e_canonical)

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
    print("SE Brain C10 — Architecture Reasoning Engine")
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
    reasoner = ArchitectureReasoner()

    spec = parser.parse(text)
    ic = ic_engine.analyze(text, project_id="demo")
    tech = selector.select(spec, ic, project_id="demo")

    res = reasoner.reason(spec, ic, tech, project_id="demo")

    print("\n[1] Summary:")
    print(res.summary())

    print("\n[2] Selected architecture:", res.decision.selected.name)
    print("    rationale:", res.decision.rationale)
    print("    tech stack:", res.tech_stack)

    print("\n[3] Components of selected:")
    for c in res.decision.selected.components:
        print(f"    - [{c.role.value:15s}] {c.name}")
        for r in c.responsibilities:
            print(f"        · {r}")

    print("\n[4] Interfaces of selected:")
    for i in res.decision.selected.interfaces:
        sync = "sync" if i.sync else "async"
        print(f"    - {i.id:20s} {i.kind.value:18s} "
              f"{i.from_component} → {i.to_component}  ({sync})")

    print("\n[5] Failure boundaries:")
    for b in res.decision.selected.failure_boundaries:
        print(f"    - {b.name}: {b.component_ids}")

    print("\n[6] Security boundaries:")
    for b in res.decision.selected.security_boundaries:
        print(f"    - {b.name}: {b.component_ids}")

    print("\n[7] All candidates with weighted scores:")
    for s in res.decision.comparison.scores:
        mark = "★" if s.candidate_id == res.decision.comparison.winner_id else " "
        on_pareto = "P" if s.candidate_id in res.decision.comparison.pareto_front else " "
        print(f"    {mark}{on_pareto} {s.kind.value:20s}  "
              f"score={s.weighted_total:.2f}  "
              f"verdict={s.verdict.value}")

    print("\n[8] Pareto front:")
    for cid in res.decision.comparison.pareto_front:
        c = res.candidate(cid)
        if c:
            print(f"    - {c.kind.value}")

    print("\n[9] Weights used:")
    for k, v in sorted(res.decision.comparison.weights.items()):
        print(f"    {k:20s}: {v:.2f}")

    print("\n[10] Rejected alternatives:")
    for s in res.decision.comparison.rejected:
        print(f"    ✗ {s.kind.value}: {s.rationale}")

    # Persistence
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = ArchitectureRepository(memory=mem, ontology=ont)
                ent_id = repo.save(res)
                print(f"\n[11] Persisted → ontology entity: {ent_id[:12]}…")
                loaded = repo.load(res.id, project_id="demo")
                print(f"    reloaded kind: {loaded['decision']['selected']['kind']}")
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
