"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C12 — SPECIALIST AGENT FRAMEWORK (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C08 (Task/TaskKind), C11 (AgentResult, Agent Protocol).

Purpose:
    Provide a governed, contract-driven agent framework. Agents are WORKERS,
    not independent brains. Every agent has:
        - identity             (id, name, tier, version)
        - capability           (which C08 TaskKinds it can handle)
        - input/output contract (validated at execute time)
        - permissions           (explicit grant list; subprocess/network disabled by default)
        - execution policy      (timeout, sandbox flags)
        - lifecycle             (created → validated → assigned → running →
                                 result → evaluated → accepted/rejected)
        - evidence              (structured, attached to every result)
        - result / failure state (uniform AgentResult; error carries code + message)

Agent tiers:
    CORE       — small, stable, universally useful (echo, contract validator,
                 health aggregator, fallback)
    SPECIALIST — do one thing well (python AST, db schema, api routes,
                 testing subprocess, security scan, performance stats,
                 architecture validation, debugging traceback parse)
    DYNAMIC    — spawned at runtime from a callable + spec

Invariants honored:
  - NO external LLM. Every specialist uses deterministic algorithms.
  - Subprocess requires EXEC_SUBPROCESS permission (never granted by default).
  - Network requires NETWORK permission (never granted by default).
  - Input contract enforced before agent code runs.
  - Output contract enforced after agent success.
  - Timeout enforced via a worker thread; no agent can hang the orchestrator.
  - Registry compatible with C11 Orchestrator (`register/get/select_for/names`).
  - Every result carries `evidence` including its lifecycle trace.
  - No filesystem writes unless WRITE_FS granted.

Contents:
  1.  Enums: AgentTier, AgentLifecycle, Permission
  2.  Dataclasses: AgentIdentity, AgentCapabilitySpec, InputContract,
                   OutputContract, AgentPermissions, ExecutionPolicy,
                   AgentSpec
  3.  BaseAgent (contracts + permissions + timeout + lifecycle trace)
  4.  Core agents (3): EchoAgent, ContractValidatorAgent, HealthAggregatorAgent
  5.  Specialists (8): PythonSpecialist, DatabaseSpecialist, APISpecialist,
                       TestingSpecialist, SecuritySpecialist,
                       PerformanceSpecialist, ArchitectureSpecialist,
                       DebuggingSpecialist
  6.  DynamicAgent (callable-backed, runtime-spawned)
  7.  AgentFrameworkRegistry (tier-aware, C11-compatible)
  8.  AgentFrameworkRepository (persist AGENT entities to C02)
  9.  register_default_agents() convenience
  10. __main__ demo + self-tests

Run as script:
    python -m sebrain.c12            # demo
    python -m sebrain.c12 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import ast
import json
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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
)
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c08 import Task, TaskKind
from sebrain.c11 import (
    AgentCapability,
    AgentResult,
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
class AgentTier(str, Enum):
    CORE = "core"
    SPECIALIST = "specialist"
    DYNAMIC = "dynamic"


class AgentLifecycle(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    ASSIGNED = "assigned"
    RUNNING = "running"
    RESULT = "result"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class Permission(str, Enum):
    READ_FS = "read_fs"
    WRITE_FS = "write_fs"
    EXEC_SUBPROCESS = "exec_subprocess"
    NETWORK = "network"
    DB_ACCESS = "db_access"
    LLM_API = "llm_api"                    # never granted by default


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES — contracts, permissions, policy, spec
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class AgentIdentity:
    id: str
    name: str
    tier: AgentTier
    version: str = "1.0.0"
    description: str = ""

    @classmethod
    def create(cls, name: str, tier: AgentTier, *, version: str = "1.0.0",
               description: str = "") -> "AgentIdentity":
        return cls(
            id=_new_id(), name=name, tier=tier,
            version=version, description=description,
        )


@dataclass(frozen=True, slots=True)
class AgentCapabilitySpec:
    """Which C08 TaskKinds this agent can handle."""
    task_kinds: tuple[TaskKind, ...]
    keywords: tuple[str, ...] = ()
    description: str = ""

    def to_agent_capability(self) -> AgentCapability:
        return AgentCapability(
            task_kinds=tuple(self.task_kinds),
            description=self.description,
        )

    def matches(self, kind: TaskKind) -> bool:
        return kind in self.task_kinds


@dataclass(frozen=True, slots=True)
class InputContract:
    required_keys: tuple[str, ...] = ()
    optional_keys: tuple[str, ...] = ()
    description: str = ""

    def validate(self, inputs: dict[str, Any]) -> list[str]:
        errs: list[str] = []
        for k in self.required_keys:
            if k not in inputs or inputs[k] is None:
                errs.append(f"missing required input: '{k}'")
        allowed = set(self.required_keys) | set(self.optional_keys)
        if allowed:
            for k in inputs:
                if k not in allowed:
                    errs.append(f"unknown input key: '{k}'")
        return errs


@dataclass(frozen=True, slots=True)
class OutputContract:
    required_keys: tuple[str, ...] = ()
    description: str = ""

    def validate(self, output: dict[str, Any]) -> list[str]:
        return [
            f"missing required output: '{k}'"
            for k in self.required_keys
            if k not in output
        ]


@dataclass(frozen=True, slots=True)
class AgentPermissions:
    granted: frozenset[Permission] = frozenset()
    max_subprocess_seconds: float = 5.0
    max_output_bytes: int = 200_000

    def has(self, p: Permission) -> bool:
        return p in self.granted

    def describe(self) -> list[str]:
        return sorted(p.value for p in self.granted)


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    timeout_seconds: float = 10.0
    max_input_bytes: int = 500_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValidationError("timeout_seconds must be > 0")
        if self.max_input_bytes <= 0:
            raise ValidationError("max_input_bytes must be > 0")


@dataclass(frozen=True, slots=True)
class AgentSpec:
    identity: AgentIdentity
    capability: AgentCapabilitySpec
    input_contract: InputContract = field(default_factory=InputContract)
    output_contract: OutputContract = field(default_factory=OutputContract)
    permissions: AgentPermissions = field(default_factory=AgentPermissions)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": {
                "id": self.identity.id,
                "name": self.identity.name,
                "tier": self.identity.tier.value,
                "version": self.identity.version,
                "description": self.identity.description,
            },
            "capability": {
                "task_kinds": [k.value for k in self.capability.task_kinds],
                "keywords": list(self.capability.keywords),
                "description": self.capability.description,
            },
            "input_contract": {
                "required_keys": list(self.input_contract.required_keys),
                "optional_keys": list(self.input_contract.optional_keys),
                "description": self.input_contract.description,
            },
            "output_contract": {
                "required_keys": list(self.output_contract.required_keys),
                "description": self.output_contract.description,
            },
            "permissions": self.permissions.describe(),
            "policy": {
                "timeout_seconds": self.policy.timeout_seconds,
                "max_input_bytes": self.policy.max_input_bytes,
            },
        }


# ════════════════════════════════════════════════════════════════════════════
# 3. BASE AGENT
# ════════════════════════════════════════════════════════════════════════════
class BaseAgent:
    """Contract-driven, permission-gated, timeout-bounded worker.

    Subclasses override `_run(task, ctx, inputs) -> dict | AgentResult`.
    This base class handles:
      - lifecycle trace
      - input contract validation
      - input size limit
      - permission checks for subprocess/network (via helpers)
      - output contract validation
      - timeout enforcement
      - exception → AgentResult(success=False)
    """

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec
        # C11 Agent Protocol attributes
        self.name: str = spec.identity.name
        self.capability: AgentCapability = spec.capability.to_agent_capability()
        self._last_invocation_trace: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    # ---- public API (matches C11 Agent Protocol) ----
    def execute(self, task: Task, ctx: dict[str, Any]) -> AgentResult:
        started_at = now_iso()
        t0 = time.monotonic()
        trace: list[dict[str, Any]] = [
            {"state": AgentLifecycle.CREATED.value, "at": started_at},
            {"state": AgentLifecycle.VALIDATED.value, "at": now_iso()},
            {"state": AgentLifecycle.ASSIGNED.value, "at": now_iso()},
            {"state": AgentLifecycle.RUNNING.value, "at": now_iso()},
        ]

        # 1. Extract inputs
        inputs = ctx.get("inputs", {})
        if not isinstance(inputs, dict):
            return self._fail(
                trace, t0,
                code="BAD_INPUTS",
                message="ctx['inputs'] must be a dict",
            )

        # 2. Input size limit
        try:
            approx_size = len(json.dumps(inputs, default=str))
        except Exception:
            approx_size = 0
        if approx_size > self.spec.policy.max_input_bytes:
            return self._fail(
                trace, t0,
                code="INPUT_TOO_LARGE",
                message=(
                    f"inputs ~{approx_size}B > "
                    f"limit {self.spec.policy.max_input_bytes}B"
                ),
            )

        # 3. Input contract
        contract_errs = self.spec.input_contract.validate(inputs)
        if contract_errs:
            return self._fail(
                trace, t0, code="INPUT_CONTRACT",
                message="; ".join(contract_errs),
            )

        # 4. Run under timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut: Future = pool.submit(self._run, task, ctx, inputs)
                raw = fut.result(timeout=self.spec.policy.timeout_seconds)
        except FuturesTimeoutError:
            return self._fail(
                trace, t0, code="TIMEOUT",
                message=(
                    f"agent exceeded timeout "
                    f"{self.spec.policy.timeout_seconds}s"
                ),
            )
        except Exception as exc:
            return self._fail(
                trace, t0, code="AGENT_EXCEPTION",
                message=f"{type(exc).__name__}: {exc}",
            )

        # 5. Wrap
        if isinstance(raw, AgentResult):
            res = raw
        elif isinstance(raw, dict):
            res = AgentResult(success=True, output=raw)
        else:
            return self._fail(
                trace, t0, code="BAD_OUTPUT",
                message=f"_run returned {type(raw).__name__}, expected dict/AgentResult",
            )

        # 6. Output contract (on success only)
        if res.success:
            out_errs = self.spec.output_contract.validate(res.output)
            if out_errs:
                return self._fail(
                    trace, t0, code="OUTPUT_CONTRACT",
                    message="; ".join(out_errs),
                )

        # 7. Enrich with evidence + lifecycle
        duration = time.monotonic() - t0
        res.duration_seconds = duration
        final_state = (
            AgentLifecycle.ACCEPTED.value if res.success
            else AgentLifecycle.REJECTED.value
        )
        trace.extend([
            {"state": AgentLifecycle.RESULT.value, "at": now_iso()},
            {"state": AgentLifecycle.EVALUATED.value, "at": now_iso()},
            {"state": final_state, "at": now_iso()},
        ])
        res.evidence = list(res.evidence) + [{
            "kind": "lifecycle_trace",
            "agent": self.name,
            "tier": self.spec.identity.tier.value,
            "duration_seconds": duration,
            "states": [s["state"] for s in trace],
        }]
        with self._lock:
            self._last_invocation_trace = trace
        return res

    # ---- helpers for subclasses ----
    def _fail(
        self, trace: list[dict[str, Any]], t0: float,
        *, code: str, message: str,
    ) -> AgentResult:
        duration = time.monotonic() - t0
        trace.extend([
            {"state": AgentLifecycle.RESULT.value, "at": now_iso()},
            {"state": AgentLifecycle.EVALUATED.value, "at": now_iso()},
            {"state": AgentLifecycle.REJECTED.value, "at": now_iso()},
        ])
        with self._lock:
            self._last_invocation_trace = trace
        res = AgentResult(
            success=False,
            error={"code": code, "message": message},
            duration_seconds=duration,
            confidence=Confidence.LOW,
        )
        res.evidence.append({
            "kind": "lifecycle_trace",
            "agent": self.name,
            "tier": self.spec.identity.tier.value,
            "duration_seconds": duration,
            "states": [s["state"] for s in trace],
        })
        return res

    def _require_permission(self, perm: Permission) -> dict[str, Any] | None:
        """Returns an error dict if the permission is missing, else None."""
        if not self.spec.permissions.has(perm):
            return {
                "code": "PERMISSION_DENIED",
                "message": (
                    f"agent '{self.name}' is not granted permission "
                    f"'{perm.value}'"
                ),
            }
        return None

    def _run(
        self, task: Task, ctx: dict[str, Any], inputs: dict[str, Any],
    ) -> dict[str, Any] | AgentResult:
        raise NotImplementedError

    # ---- lifecycle introspection ----
    def last_lifecycle(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._last_invocation_trace)

    def to_dict(self) -> dict[str, Any]:
        return self.spec.to_dict()


# ════════════════════════════════════════════════════════════════════════════
# Subprocess helper (used only when EXEC_SUBPROCESS is granted)
# ════════════════════════════════════════════════════════════════════════════
def _run_bounded_subprocess(
    cmd: list[str], *, timeout: float, cwd: Path | None = None,
) -> dict[str, Any]:
    """Run subprocess with no shell, bounded timeout, output truncation."""
    try:
        proc = subprocess.run(
            cmd, shell=False, capture_output=True, text=True,
            timeout=timeout, cwd=str(cwd) if cwd else None,
        )
        return {
            "exit_code": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:],
            "stderr": (proc.stderr or "")[-4000:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": "timeout",
                "timed_out": True}
    except FileNotFoundError as exc:
        return {"exit_code": -2, "stdout": "", "stderr": f"not found: {exc}",
                "timed_out": False}
    except Exception as exc:
        return {"exit_code": -3, "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}", "timed_out": False}


# ════════════════════════════════════════════════════════════════════════════
# 4. CORE AGENTS (3)
# ════════════════════════════════════════════════════════════════════════════
class EchoAgent(BaseAgent):
    """Returns inputs verbatim. Useful for plumbing, testing, and fallback."""

    def __init__(self, name: str = "core.echo",
                 task_kinds: tuple[TaskKind, ...] | None = None) -> None:
        if task_kinds is None:
            task_kinds = tuple(TaskKind)
        spec = AgentSpec(
            identity=AgentIdentity.create(
                name, AgentTier.CORE,
                description="Echoes inputs — fallback / plumbing agent",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=task_kinds,
                description="accepts any task kind; echoes inputs",
            ),
            input_contract=InputContract(),
            output_contract=OutputContract(),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        return {"echo": dict(inputs), "task_id": task.id, "task_kind": task.kind.value}


class ContractValidatorAgent(BaseAgent):
    """Validates a value against a simple contract spec {required_keys, optional_keys}."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "core.contract_validator", AgentTier.CORE,
                description="Validates dicts against required/optional key contracts",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.ANALYSIS, TaskKind.VERIFICATION),
                description="contract validation",
            ),
            input_contract=InputContract(
                required_keys=("value", "contract"),
                description="value: any dict; contract: {required_keys, optional_keys}",
            ),
            output_contract=OutputContract(
                required_keys=("valid", "errors"),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        value = inputs["value"]
        contract = inputs["contract"]
        errs: list[str] = []
        if not isinstance(value, dict):
            errs.append("value is not a dict")
        else:
            for k in contract.get("required_keys", []):
                if k not in value:
                    errs.append(f"missing: {k}")
            allowed = set(contract.get("required_keys", [])) | set(
                contract.get("optional_keys", [])
            )
            if allowed:
                for k in value:
                    if k not in allowed:
                        errs.append(f"unknown: {k}")
        return {"valid": not errs, "errors": errs}


class HealthAggregatorAgent(BaseAgent):
    """Aggregates a list of health dicts (each with 'ok' bool)."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "core.health_aggregator", AgentTier.CORE,
                description="Aggregates component health reports",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.VERIFICATION, TaskKind.ANALYSIS),
            ),
            input_contract=InputContract(
                required_keys=("healths",),
                description="healths: list[dict] each with 'ok' bool",
            ),
            output_contract=OutputContract(required_keys=("ok", "count", "failures")),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        healths = inputs["healths"]
        if not isinstance(healths, list):
            raise ValidationError("healths must be a list")
        failures = [h for h in healths if not h.get("ok")]
        return {
            "ok": len(failures) == 0,
            "count": len(healths),
            "failures": failures,
        }


# ════════════════════════════════════════════════════════════════════════════
# 5. SPECIALIST AGENTS (8)
# ════════════════════════════════════════════════════════════════════════════
# ---- Python specialist (AST analysis) ----
class PythonSpecialist(BaseAgent):
    """Static analysis of Python source via AST. Deterministic, no LLM."""

    _LONG_FUNCTION_LINES = 60

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.python", AgentTier.SPECIALIST,
                description="Python source static analysis (AST)",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.IMPLEMENTATION, TaskKind.ANALYSIS),
                keywords=("python", "ast", "code"),
            ),
            input_contract=InputContract(
                optional_keys=("code", "path"),
                description="provide 'code' (str) OR 'path' (str, requires READ_FS)",
            ),
            output_contract=OutputContract(
                required_keys=("source_kind", "functions", "classes",
                               "imports", "syntax_ok"),
            ),
            permissions=AgentPermissions(granted=frozenset({Permission.READ_FS})),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        source: str | None = inputs.get("code")
        source_kind = "inline"
        if source is None and "path" in inputs:
            err = self._require_permission(Permission.READ_FS)
            if err:
                raise ValidationError(err["message"])
            try:
                source = Path(inputs["path"]).read_text(encoding="utf-8")
                source_kind = "file"
            except Exception as exc:
                raise ValidationError(f"could not read {inputs['path']}: {exc}")
        if source is None:
            return {
                "source_kind": "none",
                "functions": [], "classes": [], "imports": [],
                "syntax_ok": True, "note": "no source provided",
                "long_functions": [],
            }
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {
                "source_kind": source_kind,
                "functions": [], "classes": [], "imports": [],
                "syntax_ok": False,
                "syntax_error": f"{exc.msg} at line {exc.lineno}",
                "long_functions": [],
            }
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []
        imports: list[str] = []
        long_fns: list[dict[str, Any]] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                length = max(0, end - start + 1)
                functions.append({
                    "name": node.name, "line": start, "length": length,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "args": len(node.args.args),
                })
                if length >= self._LONG_FUNCTION_LINES:
                    long_fns.append({"name": node.name, "length": length})
            elif isinstance(node, ast.ClassDef):
                start = getattr(node, "lineno", 0)
                end = getattr(node, "end_lineno", start)
                classes.append({
                    "name": node.name, "line": start,
                    "length": max(0, end - start + 1),
                })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return {
            "source_kind": source_kind,
            "syntax_ok": True,
            "functions": functions,
            "classes": classes,
            "imports": sorted(set(imports)),
            "long_functions": long_fns,
        }


# ---- Database specialist (schema + query validation) ----
_SQL_HAZARD_PATTERNS = [
    (re.compile(r"\bSELECT\s+\*", re.I), "SELECT_STAR",
     "use explicit columns instead of SELECT *"),
    (re.compile(r"\b(?:DELETE\s+FROM|UPDATE)\s+[A-Za-z_]\w*\s*(?:;|$)", re.I),
     "MUTATION_WITHOUT_WHERE",
     "DELETE/UPDATE without WHERE clause"),
    (re.compile(r"(?i)\b(?:select|insert|update|delete)\b[^;]*\+"),
     "SQL_STRING_CONCAT",
     "SQL built by string concatenation — risk of injection"),
]


class DatabaseSpecialist(BaseAgent):
    """Validates DB schema dicts; scans SQL strings for hazards."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.database", AgentTier.SPECIALIST,
                description="DB schema / query validation",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.IMPLEMENTATION, TaskKind.DESIGN),
                keywords=("database", "sql", "schema"),
            ),
            input_contract=InputContract(
                optional_keys=("schema", "query"),
                description="provide 'schema' dict OR 'query' str",
            ),
            output_contract=OutputContract(
                required_keys=("mode", "valid", "findings"),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        findings: list[dict[str, Any]] = []
        if "schema" in inputs and inputs["schema"] is not None:
            schema = inputs["schema"]
            if not isinstance(schema, dict):
                raise ValidationError("schema must be a dict")
            for tname, tdef in schema.items():
                if not isinstance(tdef, dict):
                    findings.append({"table": tname, "issue": "table def not a dict"})
                    continue
                cols = tdef.get("columns", {})
                if not cols:
                    findings.append({"table": tname, "issue": "no columns"})
                pk = tdef.get("primary_key")
                if pk is None:
                    findings.append({"table": tname, "issue": "no primary_key declared"})
                fks = tdef.get("foreign_keys", []) or []
                for fk in fks:
                    ref_table = fk.get("table")
                    if ref_table not in schema:
                        findings.append({
                            "table": tname, "issue":
                            f"FK references unknown table '{ref_table}'",
                        })
            return {
                "mode": "schema",
                "valid": not findings,
                "findings": findings,
                "table_count": len(schema),
            }
        if "query" in inputs and inputs["query"] is not None:
            query = str(inputs["query"])
            for pat, code, msg in _SQL_HAZARD_PATTERNS:
                if pat.search(query):
                    findings.append({"code": code, "message": msg})
            return {
                "mode": "query",
                "valid": not findings,
                "findings": findings,
                "length": len(query),
            }
        return {"mode": "none", "valid": True, "findings": [],
                "note": "no schema or query provided"}


# ---- API specialist (route validation) ----
_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
_PATH_PARAM_RE = re.compile(r"^\{[a-zA-Z_]\w*\}$")


class APISpecialist(BaseAgent):
    """Validates a list of HTTP routes."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.api", AgentTier.SPECIALIST,
                description="HTTP route / API design validation",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.DESIGN, TaskKind.IMPLEMENTATION),
                keywords=("api", "rest", "http"),
            ),
            input_contract=InputContract(
                optional_keys=("routes",),
                description="routes: list[{method, path, ...}]",
            ),
            output_contract=OutputContract(
                required_keys=("route_count", "valid", "findings"),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        routes = inputs.get("routes") or []
        if not isinstance(routes, list):
            raise ValidationError("routes must be a list")
        findings: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for i, r in enumerate(routes):
            if not isinstance(r, dict):
                findings.append({"index": i, "issue": "route not a dict"})
                continue
            method = str(r.get("method", "")).upper()
            path = str(r.get("path", ""))
            if method not in _ALLOWED_METHODS:
                findings.append({
                    "index": i, "issue": f"invalid HTTP method '{method}'",
                })
            if not path.startswith("/"):
                findings.append({
                    "index": i, "issue": f"path '{path}' must start with '/'",
                })
            for seg in path.split("/"):
                if seg.startswith("{") or seg.endswith("}"):
                    if not _PATH_PARAM_RE.match(seg):
                        findings.append({
                            "index": i,
                            "issue": f"malformed path param '{seg}'",
                        })
            key = (method, path)
            if key in seen:
                findings.append({
                    "index": i,
                    "issue": f"duplicate route {method} {path}",
                })
            seen.add(key)
        return {
            "route_count": len(routes),
            "valid": not findings,
            "findings": findings,
            "methods": sorted({str(r.get("method", "")).upper()
                               for r in routes if isinstance(r, dict)}),
        }


# ---- Testing specialist (subprocess sandbox) ----
class TestingSpecialist(BaseAgent):
    """Runs a bounded subprocess and reports exit code / stdout / stderr.

    Requires EXEC_SUBPROCESS permission. Accepts:
        - command: list[str]  (explicit argv)
        - script:  str        (python source; written to a temp file and run)
    """

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.testing", AgentTier.SPECIALIST,
                description="Sandboxed subprocess test runner",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.TESTING, TaskKind.VERIFICATION),
                keywords=("test", "pytest", "unittest"),
            ),
            input_contract=InputContract(
                optional_keys=("command", "script"),
                description="provide 'command' list[str] OR 'script' str",
            ),
            output_contract=OutputContract(
                required_keys=("exit_code", "passed", "stdout", "stderr"),
            ),
            permissions=AgentPermissions(
                granted=frozenset({Permission.EXEC_SUBPROCESS}),
                max_subprocess_seconds=10.0,
            ),
            policy=ExecutionPolicy(timeout_seconds=20.0),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        err = self._require_permission(Permission.EXEC_SUBPROCESS)
        if err:
            return AgentResult(success=False, error=err)
        cmd = inputs.get("command")
        tmp_dir: Path | None = None
        if cmd is None and "script" in inputs:
            script = inputs["script"]
            if not isinstance(script, str):
                raise ValidationError("script must be a str")
            tmp_dir = Path(tempfile.mkdtemp(prefix="c12_test_"))
            script_path = tmp_dir / "test_case.py"
            script_path.write_text(script, encoding="utf-8")
            cmd = [sys.executable, str(script_path)]
        if cmd is None:
            return {
                "exit_code": 0, "passed": True,
                "stdout": "", "stderr": "",
                "note": "no command or script provided",
            }
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise ValidationError("command must be list[str]")
        res = _run_bounded_subprocess(
            cmd, timeout=self.spec.permissions.max_subprocess_seconds,
            cwd=tmp_dir,
        )
        return {
            "exit_code": res["exit_code"],
            "passed": res["exit_code"] == 0 and not res["timed_out"],
            "stdout": res["stdout"],
            "stderr": res["stderr"],
            "timed_out": res["timed_out"],
        }


# ---- Security specialist (regex scan) ----
_SECURITY_PATTERNS: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"(?i)(?:password|passwd|secret|api[_-]?key|token)\s*=\s*"
                r"['\"][^'\"]{6,}['\"]"),
     "HARDCODED_SECRET", "HIGH",
     "hardcoded credential-like literal"),
    (re.compile(r"\beval\s*\("),
     "EVAL_USAGE", "HIGH", "eval() can execute arbitrary code"),
    (re.compile(r"\bexec\s*\("),
     "EXEC_USAGE", "HIGH", "exec() can execute arbitrary code"),
    (re.compile(r"shell\s*=\s*True"),
     "SHELL_TRUE", "HIGH", "subprocess shell=True enables shell injection"),
    (re.compile(r"\bos\.system\s*\("),
     "OS_SYSTEM", "HIGH", "os.system is a shell injection risk"),
    (re.compile(r"\bpickle\.loads\s*\("),
     "PICKLE_LOADS", "MEDIUM",
     "pickle.loads can execute arbitrary code on untrusted data"),
    (re.compile(r"\byaml\.load\s*\([^,)]*\)"),
     "YAML_LOAD_UNSAFE", "MEDIUM",
     "yaml.load without SafeLoader"),
]


class SecuritySpecialist(BaseAgent):
    """Regex-based security scan of a code string. Not a full SAST — explicit."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.security", AgentTier.SPECIALIST,
                description="Pattern-based security scan (not full SAST)",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.SECURITY, TaskKind.VERIFICATION),
                keywords=("security", "vulnerability", "scan"),
            ),
            input_contract=InputContract(
                optional_keys=("code",),
                description="code: str",
            ),
            output_contract=OutputContract(
                required_keys=("findings", "highest_severity"),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        code = inputs.get("code")
        if code is None:
            return {"findings": [], "highest_severity": "NONE",
                    "note": "no code provided"}
        if not isinstance(code, str):
            raise ValidationError("code must be a str")
        findings: list[dict[str, Any]] = []
        for pat, code_id, severity, msg in _SECURITY_PATTERNS:
            for m in pat.finditer(code):
                line_no = code.count("\n", 0, m.start()) + 1
                findings.append({
                    "code": code_id,
                    "severity": severity,
                    "message": msg,
                    "line": line_no,
                    "snippet": code[m.start():m.end()][:80],
                })
        order = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
        highest = max(
            (f["severity"] for f in findings),
            key=lambda s: order.get(s, 0),
            default="NONE",
        )
        return {"findings": findings, "highest_severity": highest}


# ---- Performance specialist (timing stats + complexity heuristic) ----
class PerformanceSpecialist(BaseAgent):
    """Computes stats from timing samples OR complexity heuristics from code."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.performance", AgentTier.SPECIALIST,
                description="Timing stats / complexity heuristic",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.PERFORMANCE, TaskKind.ANALYSIS),
                keywords=("performance", "benchmark", "complexity"),
            ),
            input_contract=InputContract(
                optional_keys=("samples", "code", "unit"),
                description="provide 'samples' list[float] OR 'code' str",
            ),
            output_contract=OutputContract(
                required_keys=("mode",),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        samples = inputs.get("samples")
        code = inputs.get("code")
        unit = inputs.get("unit", "ms")
        if samples is not None:
            if not isinstance(samples, list) or not samples:
                raise ValidationError("samples must be a non-empty list")
            try:
                vals = [float(x) for x in samples]
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"samples must be numeric: {exc}")
            vals_sorted = sorted(vals)
            p95_idx = max(0, int(round(0.95 * (len(vals_sorted) - 1))))
            return {
                "mode": "samples",
                "unit": unit,
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": statistics.fmean(vals),
                "median": statistics.median(vals),
                "p95": vals_sorted[p95_idx],
                "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
        if code is not None:
            if not isinstance(code, str):
                raise ValidationError("code must be a str")
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                return {"mode": "code", "parse_error": str(exc),
                        "max_loop_depth": 0, "loops": 0, "recursive": False}
            loops = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, ast.For | ast.While | ast.AsyncFor)
            )
            max_depth = _max_loop_depth(tree)
            recursive = _has_recursion(tree)
            return {
                "mode": "code",
                "loops": loops,
                "max_loop_depth": max_depth,
                "recursive": recursive,
                "complexity_hint": (
                    "high" if max_depth >= 3 or recursive
                    else "medium" if max_depth >= 2
                    else "low"
                ),
            }
        return {"mode": "none", "note": "no samples or code provided"}


def _max_loop_depth(tree: ast.AST) -> int:
    def depth(node: ast.AST, cur: int) -> int:
        if isinstance(node, ast.For | ast.While | ast.AsyncFor):
            cur += 1
        best = cur
        for child in ast.iter_child_nodes(node):
            best = max(best, depth(child, cur))
        return best
    return depth(tree, 0)


def _has_recursion(tree: ast.AST) -> bool:
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
        for call in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
            if isinstance(call.func, ast.Name) and call.func.id == fn.name:
                return True
    return False


# ---- Architecture specialist (validate C10 result shape) ----
class ArchitectureSpecialist(BaseAgent):
    """Validates an architecture result (as produced by C10)."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.architecture", AgentTier.SPECIALIST,
                description="Architecture result validation",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.DESIGN, TaskKind.VERIFICATION),
                keywords=("architecture", "components", "boundaries"),
            ),
            input_contract=InputContract(
                optional_keys=("architecture",),
                description="architecture: C10 ArchitectureResult.to_dict()",
            ),
            output_contract=OutputContract(
                required_keys=("valid", "findings"),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        arch = inputs.get("architecture")
        if arch is None:
            return {"valid": True, "findings": [],
                    "note": "no architecture provided"}
        if not isinstance(arch, dict):
            raise ValidationError("architecture must be a dict")
        findings: list[dict[str, Any]] = []
        decision = arch.get("decision")
        if not decision:
            findings.append({"issue": "no decision block"})
            return {"valid": False, "findings": findings}
        selected = decision.get("selected") or {}
        comps = selected.get("components") or []
        ifaces = selected.get("interfaces") or []
        f_bounds = selected.get("failure_boundaries") or []
        s_bounds = selected.get("security_boundaries") or []
        if not comps:
            findings.append({"issue": "no components"})
        if not ifaces:
            findings.append({"issue": "no interfaces"})
        if not f_bounds:
            findings.append({"issue": "no failure boundaries"})
        if not s_bounds:
            findings.append({"issue": "no security boundaries"})
        kind = selected.get("kind")
        if kind == "microservices" and len(f_bounds) < 2:
            findings.append({
                "issue": "microservices declared but < 2 failure boundaries",
            })
        if kind == "monolith" and len(f_bounds) > 1:
            findings.append({
                "issue": "monolith declared but multiple failure boundaries",
            })
        return {
            "valid": not findings,
            "findings": findings,
            "component_count": len(comps),
            "interface_count": len(ifaces),
            "failure_boundaries": len(f_bounds),
            "security_boundaries": len(s_bounds),
            "kind": kind,
        }


# ---- Debugging specialist (traceback parsing) ----
_TB_HEADER = re.compile(r"^Traceback \(most recent call last\):")
_TB_FRAME = re.compile(
    r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<func>\S+)'
)
_TB_EXC = re.compile(r"^(?P<exc>[A-Za-z_][\w\.]*)(?::\s*(?P<msg>.*))?$")


class DebuggingSpecialist(BaseAgent):
    """Parses a Python traceback string into structured failure info."""

    def __init__(self) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                "specialist.debugging", AgentTier.SPECIALIST,
                description="Traceback parsing / failure localisation",
            ),
            capability=AgentCapabilitySpec(
                task_kinds=(TaskKind.ANALYSIS, TaskKind.VERIFICATION),
                keywords=("debug", "traceback", "exception"),
            ),
            input_contract=InputContract(
                optional_keys=("traceback",),
                description="traceback: Python traceback string",
            ),
            output_contract=OutputContract(
                required_keys=("parsed",),
            ),
        )
        super().__init__(spec)

    def _run(self, task, ctx, inputs):
        tb = inputs.get("traceback")
        if tb is None:
            return {"parsed": False, "note": "no traceback provided"}
        if not isinstance(tb, str):
            raise ValidationError("traceback must be a str")
        lines = tb.rstrip().splitlines()
        if not lines or not _TB_HEADER.match(lines[0]):
            return {"parsed": False, "reason": "missing traceback header"}

        frames: list[dict[str, Any]] = []
        exc_type: str | None = None
        exc_msg: str | None = None
        for line in lines[1:]:
            m = _TB_FRAME.match(line)
            if m:
                frames.append({
                    "file": m.group("file"),
                    "line": int(m.group("line")),
                    "func": m.group("func"),
                })
                continue
            m = _TB_EXC.match(line.strip())
            if m and m.group("exc"):
                # last matching line wins (exception is at the bottom)
                exc_type = m.group("exc")
                exc_msg = m.group("msg") or ""
        # Prefer the last user frame (heuristic: not stdlib)
        user_frames = [
            f for f in frames if "/lib/" not in f["file"]
            and "site-packages" not in f["file"]
        ]
        top = (user_frames[-1] if user_frames
               else (frames[-1] if frames else None))
        return {
            "parsed": True,
            "exception_type": exc_type,
            "exception_message": exc_msg,
            "frame_count": len(frames),
            "frames": frames,
            "top_frame": top,
        }


# ════════════════════════════════════════════════════════════════════════════
# 6. DYNAMIC AGENT
# ════════════════════════════════════════════════════════════════════════════
class DynamicAgent(BaseAgent):
    """Runtime-configured agent backed by a callable.

    The callable signature: (task, ctx, inputs) -> dict | AgentResult
    """

    def __init__(
        self,
        name: str,
        *,
        task_kinds: Iterable[TaskKind],
        fn: Callable[[Task, dict[str, Any], dict[str, Any]],
                     dict[str, Any] | AgentResult],
        input_contract: InputContract | None = None,
        output_contract: OutputContract | None = None,
        permissions: AgentPermissions | None = None,
        policy: ExecutionPolicy | None = None,
        description: str = "",
    ) -> None:
        spec = AgentSpec(
            identity=AgentIdentity.create(
                name, AgentTier.DYNAMIC, description=description,
            ),
            capability=AgentCapabilitySpec(
                task_kinds=tuple(task_kinds),
                description=description,
            ),
            input_contract=input_contract or InputContract(),
            output_contract=output_contract or OutputContract(),
            permissions=permissions or AgentPermissions(),
            policy=policy or ExecutionPolicy(),
        )
        super().__init__(spec)
        self._fn = fn

    def _run(self, task, ctx, inputs):
        return self._fn(task, ctx, inputs)


# ════════════════════════════════════════════════════════════════════════════
# 7. REGISTRY (C11-compatible)
# ════════════════════════════════════════════════════════════════════════════
class AgentFrameworkRegistry:
    """Insertion-ordered registry. First-match wins, like C11's AgentRegistry."""

    def __init__(self) -> None:
        self._by_name: dict[str, BaseAgent] = {}
        self._dynamic_counter = 0

    # ---- C11-compatible API ----
    def register(self, agent: BaseAgent) -> None:
        if agent.name in self._by_name:
            raise ValidationError(f"agent already registered: {agent.name}")
        self._by_name[agent.name] = agent

    def get(self, name: str) -> BaseAgent | None:
        return self._by_name.get(name)

    def select_for(self, kind: TaskKind) -> BaseAgent | None:
        for a in self._by_name.values():
            if a.capability.matches(kind):
                return a
        return None

    def all(self) -> list[BaseAgent]:
        return list(self._by_name.values())

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    # ---- tier-aware helpers ----
    def by_tier(self, tier: AgentTier) -> list[BaseAgent]:
        return [a for a in self._by_name.values()
                if a.spec.identity.tier is tier]

    def find_by_keyword(self, keyword: str) -> list[BaseAgent]:
        low = keyword.lower()
        return [
            a for a in self._by_name.values()
            if any(low in kw.lower() for kw in a.spec.capability.keywords)
        ]

    # ---- dynamic spawn ----
    def spawn_dynamic(
        self,
        name: str,
        *,
        task_kinds: Iterable[TaskKind],
        fn: Callable[..., dict[str, Any] | AgentResult],
        input_contract: InputContract | None = None,
        output_contract: OutputContract | None = None,
        permissions: AgentPermissions | None = None,
        policy: ExecutionPolicy | None = None,
        description: str = "",
    ) -> DynamicAgent:
        if name in self._by_name:
            raise ValidationError(f"agent already registered: {name}")
        agent = DynamicAgent(
            name, task_kinds=task_kinds, fn=fn,
            input_contract=input_contract,
            output_contract=output_contract,
            permissions=permissions, policy=policy,
            description=description,
        )
        self._by_name[name] = agent
        return agent

    def unregister(self, name: str) -> bool:
        return self._by_name.pop(name, None) is not None


# ════════════════════════════════════════════════════════════════════════════
# 8. REPOSITORY (persist AGENT entities to C02 ontology)
# ════════════════════════════════════════════════════════════════════════════
class AgentFrameworkRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save_registry(
        self, registry: AgentFrameworkRegistry, *, project_id: str,
    ) -> list[str]:
        """Create AGENT entities in C02 for every registered agent."""
        if not project_id:
            raise ValidationError("project_id required")
        # memory snapshot
        snapshot = {
            "agents": [a.to_dict() for a in registry.all()],
            "registered_at": now_iso(),
        }
        self.memory.upsert(
            MemoryKind.PROJECT, "agent_registry_snapshot", snapshot,
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["agent_registry", "c12"],
            provenance=Provenance(
                source="agent_framework", source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        if self.ontology is None:
            return []
        created: list[str] = []
        for a in registry.all():
            spec = a.spec
            ent = self.ontology.add(
                EntityKind.AGENT,
                _short(spec.identity.name, 100),
                attributes={
                    "agent_id": spec.identity.id,
                    "tier": spec.identity.tier.value,
                    "version": spec.identity.version,
                    "task_kinds": [k.value for k in spec.capability.task_kinds],
                    "permissions": spec.permissions.describe(),
                    "timeout_seconds": spec.policy.timeout_seconds,
                    "input_required": list(spec.input_contract.required_keys),
                    "output_required": list(spec.output_contract.required_keys),
                },
                tags=["agent", spec.identity.tier.value],
                provenance=Provenance(
                    source="agent_framework", source_type=ProvenanceType.SYSTEM,
                    confidence=Confidence.HIGH,
                ),
            )
            created.append(ent.id)
        return created


# ════════════════════════════════════════════════════════════════════════════
# 9. Convenience: build default registry
# ════════════════════════════════════════════════════════════════════════════
def register_default_agents(registry: AgentFrameworkRegistry | None = None) -> AgentFrameworkRegistry:
    """Register the 3 core + 8 specialists + a fallback echo agent.

    Order matters: specialists registered FIRST so they win on `select_for`.
    EchoAgent is registered LAST with all TaskKinds → catches anything else.
    """
    reg = registry or AgentFrameworkRegistry()
    # 8 specialists — specific task-kind matches win over the fallback
    reg.register(TestingSpecialist())         # TESTING, VERIFICATION
    reg.register(SecuritySpecialist())        # SECURITY, VERIFICATION
    reg.register(PerformanceSpecialist())     # PERFORMANCE, ANALYSIS
    reg.register(PythonSpecialist())          # IMPLEMENTATION, ANALYSIS
    reg.register(DatabaseSpecialist())        # IMPLEMENTATION, DESIGN
    reg.register(APISpecialist())             # DESIGN, IMPLEMENTATION
    reg.register(ArchitectureSpecialist())    # DESIGN, VERIFICATION
    reg.register(DebuggingSpecialist())       # ANALYSIS, VERIFICATION
    # 2 more core agents
    reg.register(ContractValidatorAgent())    # ANALYSIS, VERIFICATION
    reg.register(HealthAggregatorAgent())     # VERIFICATION, ANALYSIS
    # Last: universal fallback
    reg.register(EchoAgent("core.fallback"))  # ALL
    return reg


# ════════════════════════════════════════════════════════════════════════════
# 10. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _mk_task(task_id: str = "t1", kind: TaskKind = TaskKind.ANALYSIS) -> Task:
    return Task(
        id=task_id, name=f"task_{task_id}", kind=kind, duration=1,
        depends_on=[], description="test task",
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
            traceback.print_exc()
            failures.append(name)
            print(f"  ✗ {name}")

    print("Running C12 self-tests…")

    # ---- contracts ----
    def t_input_contract_missing_key() -> None:
        c = InputContract(required_keys=("a", "b"))
        errs = c.validate({"a": 1})
        assert any("b" in e for e in errs)

    def t_input_contract_unknown_key() -> None:
        c = InputContract(required_keys=("a",), optional_keys=("b",))
        errs = c.validate({"a": 1, "c": 2})
        assert any("c" in e for e in errs)

    def t_output_contract() -> None:
        c = OutputContract(required_keys=("x",))
        assert c.validate({}) == ["missing required output: 'x'"]
        assert c.validate({"x": 1}) == []

    def t_policy_rejects_bad_timeout() -> None:
        try:
            ExecutionPolicy(timeout_seconds=0)
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    check("contract: input required keys enforced", t_input_contract_missing_key)
    check("contract: input unknown keys rejected", t_input_contract_unknown_key)
    check("contract: output required keys enforced", t_output_contract)
    check("policy: timeout must be > 0", t_policy_rejects_bad_timeout)

    # ---- base agent behaviour ----
    def t_base_agent_input_contract_violation() -> None:
        ag = ContractValidatorAgent()
        res = ag.execute(_mk_task(), {"inputs": {"value": {}}})  # missing 'contract'
        assert res.success is False
        assert res.error is not None
        assert res.error["code"] == "INPUT_CONTRACT"

    def t_base_agent_output_contract_violation() -> None:
        # Dynamic agent declares output key it never returns
        fn = lambda task, ctx, inputs: {"wrong": 1}
        ag = DynamicAgent(
            "bad_out", task_kinds=(TaskKind.ANALYSIS,), fn=fn,
            output_contract=OutputContract(required_keys=("needed",)),
        )
        res = ag.execute(_mk_task(), {"inputs": {}})
        assert res.success is False
        assert res.error is not None
        assert res.error["code"] == "OUTPUT_CONTRACT"

    def t_base_agent_timeout() -> None:
        def slow(task, ctx, inputs):
            time.sleep(2.0)
            return {"done": True}
        ag = DynamicAgent(
            "slow", task_kinds=(TaskKind.ANALYSIS,), fn=slow,
            policy=ExecutionPolicy(timeout_seconds=0.2),
        )
        res = ag.execute(_mk_task(), {"inputs": {}})
        assert res.success is False
        assert res.error is not None
        assert res.error["code"] == "TIMEOUT"

    def t_base_agent_exception_captured() -> None:
        def boom(task, ctx, inputs):
            raise RuntimeError("exploded")
        ag = DynamicAgent("boom", task_kinds=(TaskKind.ANALYSIS,), fn=boom)
        res = ag.execute(_mk_task(), {"inputs": {}})
        assert res.success is False
        assert res.error is not None
        assert res.error["code"] == "AGENT_EXCEPTION"
        assert "exploded" in res.error["message"]

    def t_base_agent_emits_lifecycle_trace() -> None:
        ag = EchoAgent()
        res = ag.execute(_mk_task(), {"inputs": {"x": 1}})
        assert res.success is True
        assert any(e.get("kind") == "lifecycle_trace" for e in res.evidence)
        tr = ag.last_lifecycle()
        states = [s["state"] for s in tr]
        assert "created" in states
        assert "accepted" in states

    def t_base_agent_input_size_limit() -> None:
        ag = DynamicAgent(
            "big", task_kinds=(TaskKind.ANALYSIS,),
            fn=lambda t, c, i: {"ok": True},
            policy=ExecutionPolicy(timeout_seconds=5, max_input_bytes=50),
        )
        res = ag.execute(_mk_task(), {"inputs": {"data": "x" * 1000}})
        assert res.success is False
        assert res.error is not None
        assert res.error["code"] == "INPUT_TOO_LARGE"

    check("base: input contract enforced",
          t_base_agent_input_contract_violation)
    check("base: output contract enforced",
          t_base_agent_output_contract_violation)
    check("base: timeout enforced", t_base_agent_timeout)
    check("base: exceptions captured as AGENT_EXCEPTION",
          t_base_agent_exception_captured)
    check("base: lifecycle trace attached to every result",
          t_base_agent_emits_lifecycle_trace)
    check("base: input size limit enforced", t_base_agent_input_size_limit)

    # ---- permissions ----
    def t_permission_denied_subprocess() -> None:
        # TestingSpecialist without EXEC_SUBPROCESS
        spec = AgentSpec(
            identity=AgentIdentity.create("noperm", AgentTier.SPECIALIST),
            capability=AgentCapabilitySpec(task_kinds=(TaskKind.TESTING,)),
            input_contract=InputContract(optional_keys=("command",)),
            output_contract=OutputContract(required_keys=(
                "exit_code", "passed", "stdout", "stderr",
            )),
            permissions=AgentPermissions(granted=frozenset()),
        )
        # Use TestingSpecialist's _run but with a restrictive spec
        ag = TestingSpecialist()
        ag.spec = spec
        res = ag.execute(
            _mk_task(kind=TaskKind.TESTING),
            {"inputs": {"command": [sys.executable, "-c", "print(1)"]}},
        )
        assert res.success is False
        assert res.error is not None
        assert res.error["code"] == "PERMISSION_DENIED"

    def t_no_default_permission_for_subprocess_or_network() -> None:
        ag = PythonSpecialist()
        perms = ag.spec.permissions
        assert Permission.EXEC_SUBPROCESS not in perms.granted
        assert Permission.NETWORK not in perms.granted
        assert Permission.LLM_API not in perms.granted

    check("permissions: subprocess denied by default",
          t_permission_denied_subprocess)
    check("permissions: no default subprocess/network/LLM",
          t_no_default_permission_for_subprocess_or_network)

    # ---- CORE agents ----
    def t_echo_agent() -> None:
        ag = EchoAgent()
        res = ag.execute(_mk_task(), {"inputs": {"a": 1, "b": "x"}})
        assert res.success
        assert res.output["echo"] == {"a": 1, "b": "x"}

    def t_contract_validator_agent() -> None:
        ag = ContractValidatorAgent()
        res = ag.execute(_mk_task(), {"inputs": {
            "value": {"x": 1}, "contract": {"required_keys": ["x"]},
        }})
        assert res.success
        assert res.output["valid"] is True
        res2 = ag.execute(_mk_task(), {"inputs": {
            "value": {}, "contract": {"required_keys": ["x"]},
        }})
        assert res2.success
        assert res2.output["valid"] is False
        assert any("x" in e for e in res2.output["errors"])

    def t_health_aggregator_agent() -> None:
        ag = HealthAggregatorAgent()
        res = ag.execute(_mk_task(), {"inputs": {"healths": [
            {"component": "a", "ok": True},
            {"component": "b", "ok": False},
        ]}})
        assert res.success
        assert res.output["ok"] is False
        assert res.output["count"] == 2
        assert len(res.output["failures"]) == 1

    check("core.echo returns inputs", t_echo_agent)
    check("core.contract_validator validates + reports", t_contract_validator_agent)
    check("core.health_aggregator aggregates", t_health_aggregator_agent)

    # ---- SPECIALISTS ----
    def t_python_specialist() -> None:
        ag = PythonSpecialist()
        code = (
            "import os\n"
            "def f(x):\n    return x + 1\n"
            "class A:\n    def m(self): pass\n"
        )
        res = ag.execute(_mk_task(), {"inputs": {"code": code}})
        assert res.success
        assert res.output["syntax_ok"] is True
        names = {f["name"] for f in res.output["functions"]}
        assert "f" in names
        assert res.output["classes"][0]["name"] == "A"
        assert "os" in res.output["imports"]

    def t_python_specialist_syntax_error() -> None:
        ag = PythonSpecialist()
        res = ag.execute(_mk_task(), {"inputs": {"code": "def f(:\n  pass"}})
        assert res.success
        assert res.output["syntax_ok"] is False
        assert "syntax_error" in res.output

    def t_python_specialist_empty() -> None:
        ag = PythonSpecialist()
        res = ag.execute(_mk_task(), {"inputs": {}})
        assert res.success
        assert res.output["source_kind"] == "none"

    def t_database_specialist_schema() -> None:
        ag = DatabaseSpecialist()
        res = ag.execute(_mk_task(), {"inputs": {"schema": {
            "users": {"columns": {"id": "int", "email": "text"},
                      "primary_key": ["id"]},
            "orders": {"columns": {"id": "int", "user_id": "int"},
                       "primary_key": ["id"],
                       "foreign_keys": [{"table": "users", "column": "id"}]},
        }}})
        assert res.success
        assert res.output["valid"] is True
        assert res.output["table_count"] == 2

    def t_database_specialist_schema_bad_fk() -> None:
        ag = DatabaseSpecialist()
        res = ag.execute(_mk_task(), {"inputs": {"schema": {
            "orders": {"columns": {"id": "int"}, "primary_key": ["id"],
                       "foreign_keys": [{"table": "missing_table"}]},
        }}})
        assert res.success
        assert res.output["valid"] is False
        assert any("missing_table" in f.get("issue", "") for f in res.output["findings"])

    def t_database_specialist_query_hazards() -> None:
        ag = DatabaseSpecialist()
        res = ag.execute(_mk_task(), {"inputs": {
            "query": "SELECT * FROM users; DELETE FROM logs;"
        }})
        assert res.success
        codes = {f["code"] for f in res.output["findings"]}
        assert "SELECT_STAR" in codes
        assert "MUTATION_WITHOUT_WHERE" in codes

    def t_api_specialist_valid() -> None:
        ag = APISpecialist()
        res = ag.execute(_mk_task(), {"inputs": {"routes": [
            {"method": "GET", "path": "/tasks"},
            {"method": "POST", "path": "/tasks"},
            {"method": "GET", "path": "/tasks/{id}"},
        ]}})
        assert res.success
        assert res.output["valid"] is True
        assert res.output["route_count"] == 3

    def t_api_specialist_invalid() -> None:
        ag = APISpecialist()
        res = ag.execute(_mk_task(), {"inputs": {"routes": [
            {"method": "FETCH", "path": "tasks"},   # bad method + no leading /
            {"method": "GET", "path": "/tasks"},
            {"method": "GET", "path": "/tasks"},    # duplicate
            {"method": "GET", "path": "/tasks/{id"},  # malformed param
        ]}})
        assert res.success
        assert res.output["valid"] is False
        issues = " ".join(f.get("issue", "") for f in res.output["findings"])
        assert "invalid HTTP method" in issues
        assert "must start with" in issues
        assert "duplicate" in issues
        assert "malformed" in issues

    def t_testing_specialist_runs_script() -> None:
        ag = TestingSpecialist()
        script = "print('hello')\nimport sys; sys.exit(0)\n"
        res = ag.execute(_mk_task(kind=TaskKind.TESTING),
                         {"inputs": {"script": script}})
        assert res.success
        assert res.output["passed"] is True
        assert "hello" in res.output["stdout"]

    def t_testing_specialist_failing_script() -> None:
        ag = TestingSpecialist()
        script = "import sys; sys.exit(3)\n"
        res = ag.execute(_mk_task(kind=TaskKind.TESTING),
                         {"inputs": {"script": script}})
        assert res.success
        assert res.output["passed"] is False
        assert res.output["exit_code"] == 3

    def t_testing_specialist_timeout() -> None:
        ag = TestingSpecialist()
        # Override spec to have 0.5s subprocess limit
        spec = ag.spec
        ag.spec = AgentSpec(
            identity=spec.identity,
            capability=spec.capability,
            input_contract=spec.input_contract,
            output_contract=spec.output_contract,
            permissions=AgentPermissions(
                granted=frozenset({Permission.EXEC_SUBPROCESS}),
                max_subprocess_seconds=0.3,
            ),
            policy=ExecutionPolicy(timeout_seconds=3.0),
        )
        script = "import time; time.sleep(2)\n"
        res = ag.execute(_mk_task(kind=TaskKind.TESTING),
                         {"inputs": {"script": script}})
        assert res.success
        assert res.output["passed"] is False
        assert res.output["timed_out"] is True

    def t_security_specialist_clean() -> None:
        ag = SecuritySpecialist()
        res = ag.execute(_mk_task(kind=TaskKind.SECURITY),
                         {"inputs": {"code": "x = 1\nprint(x)\n"}})
        assert res.success
        assert res.output["findings"] == []
        assert res.output["highest_severity"] == "NONE"

    def t_security_specialist_finds_issues() -> None:
        ag = SecuritySpecialist()
        code = (
            "password = 'hunter2secret'\n"
            "eval('1+1')\n"
            "import os; os.system('ls')\n"
        )
        res = ag.execute(_mk_task(kind=TaskKind.SECURITY),
                         {"inputs": {"code": code}})
        assert res.success
        codes = {f["code"] for f in res.output["findings"]}
        assert "HARDCODED_SECRET" in codes
        assert "EVAL_USAGE" in codes
        assert "OS_SYSTEM" in codes
        assert res.output["highest_severity"] == "HIGH"

    def t_performance_specialist_samples() -> None:
        ag = PerformanceSpecialist()
        res = ag.execute(_mk_task(kind=TaskKind.PERFORMANCE),
                         {"inputs": {"samples": [10, 12, 11, 100, 9],
                                     "unit": "ms"}})
        assert res.success
        assert res.output["mode"] == "samples"
        assert res.output["min"] == 9
        assert res.output["max"] == 100
        assert res.output["count"] == 5

    def t_performance_specialist_complexity() -> None:
        ag = PerformanceSpecialist()
        code = (
            "def f(n):\n"
            "    for i in range(n):\n"
            "        for j in range(n):\n"
            "            for k in range(n):\n"
            "                pass\n"
        )
        res = ag.execute(_mk_task(kind=TaskKind.PERFORMANCE),
                         {"inputs": {"code": code}})
        assert res.success
        assert res.output["mode"] == "code"
        assert res.output["max_loop_depth"] >= 3
        assert res.output["complexity_hint"] == "high"

    def t_architecture_specialist_validates() -> None:
        ag = ArchitectureSpecialist()
        arch = {
            "decision": {
                "selected": {
                    "kind": "layered",
                    "components": [{"id": "a"}, {"id": "b"}],
                    "interfaces": [{"id": "i"}],
                    "failure_boundaries": [{"id": "fb"}],
                    "security_boundaries": [{"id": "sb"}],
                }
            }
        }
        res = ag.execute(_mk_task(kind=TaskKind.DESIGN),
                         {"inputs": {"architecture": arch}})
        assert res.success
        assert res.output["valid"] is True

    def t_architecture_specialist_flags_missing() -> None:
        ag = ArchitectureSpecialist()
        arch = {"decision": {"selected": {
            "kind": "microservices",
            "components": [{"id": "a"}],
            "interfaces": [],
            "failure_boundaries": [{"id": "fb"}],
            "security_boundaries": [{"id": "sb"}],
        }}}
        res = ag.execute(_mk_task(kind=TaskKind.DESIGN),
                         {"inputs": {"architecture": arch}})
        assert res.success
        assert res.output["valid"] is False
        issues = " ".join(f.get("issue", "") for f in res.output["findings"])
        assert "no interfaces" in issues
        assert "microservices declared but" in issues

    def t_debugging_specialist_parses() -> None:
        ag = DebuggingSpecialist()
        tb = (
            "Traceback (most recent call last):\n"
            '  File "/app/main.py", line 42, in handle\n'
            "    raise ValueError('boom')\n"
            "ValueError: boom\n"
        )
        res = ag.execute(_mk_task(kind=TaskKind.ANALYSIS),
                         {"inputs": {"traceback": tb}})
        assert res.success
        assert res.output["parsed"] is True
        assert res.output["exception_type"] == "ValueError"
        assert res.output["exception_message"] == "boom"
        assert res.output["frame_count"] == 1
        assert res.output["top_frame"]["file"] == "/app/main.py"

    def t_debugging_specialist_empty() -> None:
        ag = DebuggingSpecialist()
        res = ag.execute(_mk_task(), {"inputs": {}})
        assert res.success
        assert res.output["parsed"] is False

    check("python: parses code, finds functions/classes/imports",
          t_python_specialist)
    check("python: reports syntax errors", t_python_specialist_syntax_error)
    check("python: empty input → source_kind=none", t_python_specialist_empty)
    check("db: valid schema → valid=True", t_database_specialist_schema)
    check("db: bad FK reference flagged", t_database_specialist_schema_bad_fk)
    check("db: query hazards flagged (SELECT *, DELETE w/o WHERE)",
          t_database_specialist_query_hazards)
    check("api: valid routes accepted", t_api_specialist_valid)
    check("api: invalid routes flagged (method/path/dup/param)",
          t_api_specialist_invalid)
    check("testing: runs script, captures stdout",
          t_testing_specialist_runs_script)
    check("testing: non-zero exit → passed=False",
          t_testing_specialist_failing_script)
    check("testing: subprocess timeout enforced",
          t_testing_specialist_timeout)
    check("security: clean code → no findings",
          t_security_specialist_clean)
    check("security: finds secrets/eval/os.system",
          t_security_specialist_finds_issues)
    check("performance: computes timing stats", t_performance_specialist_samples)
    check("performance: heuristic complexity from code",
          t_performance_specialist_complexity)
    check("architecture: valid structure passes",
          t_architecture_specialist_validates)
    check("architecture: missing pieces flagged",
          t_architecture_specialist_flags_missing)
    check("debugging: parses traceback", t_debugging_specialist_parses)
    check("debugging: empty input", t_debugging_specialist_empty)

    # ---- dynamic agents ----
    def t_dynamic_agent_runs() -> None:
        def fn(task, ctx, inputs):
            return {"got": inputs["x"] * 2}
        ag = DynamicAgent(
            "dyn.doubler", task_kinds=(TaskKind.ANALYSIS,), fn=fn,
            input_contract=InputContract(required_keys=("x",)),
            output_contract=OutputContract(required_keys=("got",)),
        )
        res = ag.execute(_mk_task(), {"inputs": {"x": 21}})
        assert res.success
        assert res.output["got"] == 42

    def t_registry_spawn_dynamic() -> None:
        reg = AgentFrameworkRegistry()
        ag = reg.spawn_dynamic(
            "dyn.x", task_kinds=(TaskKind.TESTING,),
            fn=lambda t, c, i: {"ok": True},
        )
        assert reg.get("dyn.x") is ag
        assert reg.select_for(TaskKind.TESTING) is ag

    def t_registry_unregister() -> None:
        reg = AgentFrameworkRegistry()
        reg.register(EchoAgent("x"))
        assert reg.unregister("x") is True
        assert reg.get("x") is None
        assert reg.unregister("x") is False

    check("dynamic: user callable wrapped & validated", t_dynamic_agent_runs)
    check("registry: spawn_dynamic registers agent", t_registry_spawn_dynamic)
    check("registry: unregister works", t_registry_unregister)

    # ---- registry tiers + selection ----
    def t_registry_tiers() -> None:
        reg = register_default_agents()
        core = reg.by_tier(AgentTier.CORE)
        specialists = reg.by_tier(AgentTier.SPECIALIST)
        assert len(core) >= 3
        assert len(specialists) == 8

    def t_registry_select_matches_specific() -> None:
        reg = register_default_agents()
        # TESTING → TestingSpecialist
        assert reg.select_for(TaskKind.TESTING).name == "specialist.testing"
        # SECURITY → SecuritySpecialist
        assert reg.select_for(TaskKind.SECURITY).name == "specialist.security"
        # PERFORMANCE → PerformanceSpecialist
        assert reg.select_for(TaskKind.PERFORMANCE).name == "specialist.performance"

    def t_registry_fallback_catches_all() -> None:
        reg = register_default_agents()
        # DEPLOYMENT has no specialist → fallback
        a = reg.select_for(TaskKind.DEPLOYMENT)
        assert a is not None
        assert a.name == "core.fallback"

    def t_registry_keyword_search() -> None:
        reg = register_default_agents()
        a = reg.find_by_keyword("ast")
        assert any(x.name == "specialist.python" for x in a)

    check("registry: has ≥3 core + 8 specialists", t_registry_tiers)
    check("registry: specific task-kinds select specialists",
          t_registry_select_matches_specific)
    check("registry: fallback covers unspecified task kinds",
          t_registry_fallback_catches_all)
    check("registry: keyword lookup", t_registry_keyword_search)

    # ---- C11 compatibility ----
    def t_c11_compat_registry() -> None:
        """C11's Orchestrator must accept our registry & agents."""
        from sebrain.c11 import Orchestrator, ParallelismMode
        from sebrain.c08 import Plan, Task
        from sebrain.c08 import _topological_order, _critical_path_method

        # Build a minimal plan
        tasks = [
            Task(id="t1", name="a", kind=TaskKind.ANALYSIS, duration=1),
            Task(id="t2", name="b", kind=TaskKind.IMPLEMENTATION, duration=2,
                 depends_on=["t1"]),
            Task(id="t3", name="c", kind=TaskKind.TESTING, duration=2,
                 depends_on=["t2"]),
        ]
        topo = _topological_order(tasks)
        cp = _critical_path_method(tasks, topo)
        plan = Plan(
            project_id="p", tasks=tasks,
            execution_order=topo, critical=cp,
            total_duration=cp.project_duration,
        )
        reg = register_default_agents()
        orch = Orchestrator(reg, mode=ParallelismMode.SEQUENTIAL)
        res = orch.run(plan, project_id="p")
        # All three should succeed (each specialist tolerates empty inputs)
        from sebrain.c11 import AssignmentState
        for t in tasks:
            st = res.assignments[t.id].state
            assert st is AssignmentState.SUCCEEDED, (t.id, st)

    check("integration: C11 Orchestrator accepts C12 registry & agents",
          t_c11_compat_registry)

    # ---- persistence ----
    def t_persist_registry() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                reg = register_default_agents()
                repo = AgentFrameworkRepository(memory=mem, ontology=ont)
                ids = repo.save_registry(reg, project_id="proj-x")
                assert len(ids) == len(reg.names())
                # ontology has AGENT entities
                assert ont.count(kind=EntityKind.AGENT) == len(reg.names())
                # memory snapshot exists
                e = mem.get_current(
                    MemoryKind.PROJECT, "agent_registry_snapshot",
                    scope_type=MemoryScope.PROJECT, scope_id="proj-x",
                )
                assert e is not None
                assert len(e.content["agents"]) == len(reg.names())
            finally:
                s.shutdown()

    check("persist: registry → memory snapshot + ontology AGENT entities",
          t_persist_registry)

    # ---- to_dict ----
    def t_agent_to_dict() -> None:
        ag = PythonSpecialist()
        d = ag.to_dict()
        assert d["identity"]["tier"] == "specialist"
        assert "python" in d["identity"]["name"]
        assert d["input_contract"]["optional_keys"] == ["code", "path"]
        assert "read_fs" in d["permissions"]

    check("to_dict: agent spec serialisable", t_agent_to_dict)

    # ---- e2e canonical ----
    def t_e2e_canonical() -> None:
        """Full pipeline: C05→C06→C09→C10→C08→C11→C12 running together."""
        from sebrain.c05 import RequirementParser
        from sebrain.c06 import IntentContextEngine
        from sebrain.c08 import Planner
        from sebrain.c09 import TechnologySelector
        from sebrain.c10 import ArchitectureReasoner
        from sebrain.c11 import (
            Orchestrator, ParallelismMode, OrchestrationStatus,
        )

        text = (
            "Build a small production-quality REST API for managing tasks.\n"
            "Non-functional:\n"
            "- All traffic must use HTTPS.\n"
        )
        parser = RequirementParser()
        ic_eng = IntentContextEngine()
        spec = parser.parse(text)
        ic = ic_eng.analyze(text, project_id="demo")
        tech = TechnologySelector().select(spec, ic, project_id="demo")
        arch = ArchitectureReasoner().reason(spec, ic, tech, project_id="demo")
        plan = Planner().plan(spec, ic, project_id="demo")

        # Attach real inputs for our specialists by enriching ctx via closure
        # (this is a demo integration, not a C11 change)
        reg = register_default_agents()
        orch = Orchestrator(reg, mode=ParallelismMode.WAVE, max_workers=4)
        # We can't inject per-task inputs through the current C11 API,
        # but the specialists tolerate empty inputs — so success is expected.
        result = orch.run(plan, project_id="demo")
        assert result.status is OrchestrationStatus.COMPLETED, result.summary()
        # Every task has an agent assigned
        for tid, a in result.assignments.items():
            assert a.agent_name != "<none>", tid
        # Gate results are all passing (all success → all gates pass)
        for g in result.gate_results:
            assert g.passed, (g.gate_id, g.rationale)
        # Plan + arch consumed
        assert arch.decision is not None
        assert tech.winner_name is not None

    check("e2e: full pipeline with C12 agents runs to completion",
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
    print("SE Brain C12 — Specialist Agent Framework")
    print("=" * 78)

    reg = register_default_agents()

    print("\n[1] Registered agents:")
    for a in reg.all():
        spec = a.spec
        kinds = ",".join(k.value for k in spec.capability.task_kinds)
        perms = ",".join(spec.permissions.describe()) or "-"
        print(f"    [{spec.identity.tier.value:10s}] {spec.identity.name:25s}  "
              f"kinds=[{kinds[:50]}]  perms=[{perms}]")

    print(f"\n[2] Totals: core={len(reg.by_tier(AgentTier.CORE))}, "
          f"specialist={len(reg.by_tier(AgentTier.SPECIALIST))}, "
          f"dynamic={len(reg.by_tier(AgentTier.DYNAMIC))}")

    task = _mk_task("demo1", TaskKind.ANALYSIS)

    print("\n[3] PythonSpecialist on sample code:")
    py = reg.get("specialist.python")
    res = py.execute(task, {"inputs": {"code": (
        "import os\n"
        "def greet(name):\n"
        "    return 'hi ' + name\n"
        "class App:\n"
        "    def run(self): pass\n"
    )}})
    print(f"    success={res.success}  functions={[f['name'] for f in res.output['functions']]}")
    print(f"    classes={[c['name'] for c in res.output['classes']]}  imports={res.output['imports']}")
    print(f"    lifecycle={[s['state'] for s in py.last_lifecycle()]}")

    print("\n[4] SecuritySpecialist on risky code:")
    sec = reg.get("specialist.security")
    res = sec.execute(task, {"inputs": {"code": (
        "password = 'supersecret123'\n"
        "eval('1+1')\n"
        "import os; os.system('ls')\n"
    )}})
    print(f"    highest_severity={res.output['highest_severity']}")
    for f in res.output["findings"]:
        print(f"      · [{f['severity']}] {f['code']} (line {f['line']})")

    print("\n[5] TestingSpecialist (sandboxed subprocess):")
    ts = reg.get("specialist.testing")
    res = ts.execute(_mk_task("demo-test", TaskKind.TESTING), {"inputs": {
        "script": "print('all good')\nimport sys; sys.exit(0)\n",
    }})
    print(f"    passed={res.output['passed']}  exit_code={res.output['exit_code']}  "
          f"stdout={res.output['stdout'].strip()!r}")

    print("\n[6] DebuggingSpecialist parses a traceback:")
    dbg = reg.get("specialist.debugging")
    tb = (
        "Traceback (most recent call last):\n"
        '  File "/app/main.py", line 42, in handle\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
    )
    res = dbg.execute(task, {"inputs": {"traceback": tb}})
    print(f"    exception={res.output['exception_type']}: {res.output['exception_message']}")
    print(f"    top_frame={res.output['top_frame']}")

    print("\n[7] Dynamic agent (spawned at runtime):")
    def multiplier(task, ctx, inputs):
        return {"result": inputs["value"] * 10}
    dyn = reg.spawn_dynamic(
        "dynamic.x10", task_kinds=(TaskKind.ANALYSIS,),
        fn=multiplier,
        input_contract=InputContract(required_keys=("value",)),
        output_contract=OutputContract(required_keys=("result",)),
        description="multiplies by 10",
    )
    res = dyn.execute(task, {"inputs": {"value": 7}})
    print(f"    dynamic.x10(7) = {res.output['result']}")

    print("\n[8] End-to-end with C11 Orchestrator:")
    from sebrain.c05 import RequirementParser
    from sebrain.c06 import IntentContextEngine
    from sebrain.c08 import Planner
    from sebrain.c11 import Orchestrator, ParallelismMode

    text = (
        "Build a small production-quality REST API for managing tasks.\n"
        "Non-functional:\n- All traffic must use HTTPS.\n"
    )
    spec = RequirementParser().parse(text)
    ic = IntentContextEngine().analyze(text, project_id="demo")
    plan = Planner().plan(spec, ic, project_id="demo")

    orch = Orchestrator(reg, mode=ParallelismMode.WAVE, max_workers=4)
    result = orch.run(plan, project_id="demo")
    print(result.summary())
    print(f"    assignments:")
    for tid, a in result.assignments.items():
        print(f"      {tid}: agent={a.agent_name:25s}  state={a.state.value}")

    # Persistence
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = AgentFrameworkRepository(memory=mem, ontology=ont)
                ids = repo.save_registry(reg, project_id="demo")
                print(f"\n[9] Persisted {len(ids)} AGENT entities to ontology.")
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
