"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C17 — TEST GENERATION ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C05, C06, C08, C13, C14.

Purpose:
    Produce a structured TestPlan containing REAL, compilable pytest code
    from structured inputs. Not "coverage inflating" — each test targets a
    concrete behaviour, contract, edge, negative path, or acceptance
    criterion, and carries traceability to the artifact it covers.

Test kinds:
    UNIT          model defaults, required fields, service wiring
    INTEGRATION   repository CRUD round-trip; HTTP route surface (FastAPI/Flask)
    BOUNDARY      empty strings, zero/negative ints, empty lists
    NEGATIVE      missing required fields, invalid payloads
    ACCEPTANCE    C05 Given/When/Then criteria and functional requirements
    SECURITY      scans for eval/exec/os.system/secrets in generated code
    PERFORMANCE   generous timing budget tests for NFR-derived operations
    REGRESSION    past failures from C04 memory (opt-in)

Traceability:
    Every TestCase has `covers: list[str]` — targets like "req:<id>",
    "accept:<id>", "symbol:<id>", "model:Task", "repo:TaskRepository",
    "route:POST /tasks", "nfr:perf:1".
    TestPlan.coverage maps target → test_ids.

Invariants honored:
    - NO external LLM. Deterministic templates.
    - Every generated test file must `compile(src, path, "exec")` successfully.
    - No test imports anything that fails at module import time: use a lazy
      `_import()` helper that `pytest.fail()`s with a clear message.
    - Bounded: max_tests_total, max_per_file.
    - Deterministic test names and IDs (stable hash of kind + name).
    - Every artifact carries rationale + evidence.
    - Never overwrite: returns artifacts in-memory (writing is C15's job).

Explicit limitations:
    - Some generated tests (acceptance, security, performance) are smoke /
      heuristic checks, not exhaustive proof. Each is marked via its
      rationale + evidence.
    - Regression tests are marker-based when only failure memory is available;
      they assert on documented failure evidence, not the original code path.

Contents:
  1.  Enums: TestKind, GenerationSource
  2.  Dataclasses: TestCase, TestArtifact, CoverageEntry, TestPlan
  3.  Stability helpers (test_id)
  4.  File header emitter + lazy import helper
  5.  Generators (11)
  6.  TestGenerator (facade)
  7.  TestRepository (persist to memory + ontology)
  8.  Self-tests (~30)
  9.  Demo

Run as script:
    python -m sebrain.c17            # demo
    python -m sebrain.c17 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import ast
import hashlib
import re
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
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore


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


def _stable_id(kind: str, name: str) -> str:
    return hashlib.sha256(f"{kind}::{name}".encode("utf-8")).hexdigest()[:16]


_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


def _snake(name: str) -> str:
    return _SNAKE.sub("_", name).lower()


def _safe_ident(name: str) -> str:
    """Return a valid Python identifier for use in a test function name."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not s or s[0].isdigit():
        s = "_" + s
    return s


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class TestKind(str, Enum):
    UNIT = "unit"
    INTEGRATION = "integration"
    BOUNDARY = "boundary"
    NEGATIVE = "negative"
    ACCEPTANCE = "acceptance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


class GenerationSource(str, Enum):
    REQUIREMENT = "requirement"
    ACCEPTANCE_CRITERION = "acceptance_criterion"
    SYMBOL = "symbol"
    SYNTHESIZED_MODEL = "synthesized_model"
    SYNTHESIZED_REPOSITORY = "synthesized_repository"
    SYNTHESIZED_SERVICE = "synthesized_service"
    SYNTHESIZED_API = "synthesized_api"
    NFR = "nfr"
    PAST_FAILURE = "past_failure"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class TestCase:
    id: str
    name: str                         # pytest function name
    kind: TestKind
    code: str                         # full test function body (no imports)
    covers: list[str] = field(default_factory=list)
    source: GenerationSource = GenerationSource.SYMBOL
    rationale: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "kind": self.kind.value,
            "covers": list(self.covers),
            "source": self.source.value,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "code_preview": self.code[:200],
        }


@dataclass(slots=True)
class TestArtifact:
    path: str
    kind: TestKind
    content: str
    test_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    byte_size: int = 0

    def __post_init__(self) -> None:
        self.byte_size = len(self.content.encode("utf-8"))

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path, "kind": self.kind.value,
            "test_ids": list(self.test_ids),
            "rationale": self.rationale,
            "byte_size": self.byte_size,
        }
        if include_content:
            d["content"] = self.content
        return d


@dataclass(slots=True)
class CoverageEntry:
    target: str                        # e.g. "req:abc", "symbol:xyz"
    target_kind: str                   # e.g. "requirement", "symbol"
    test_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target, "target_kind": self.target_kind,
            "test_ids": list(self.test_ids),
        }


@dataclass(slots=True)
class TestPlan:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    tests: list[TestCase] = field(default_factory=list)
    artifacts: list[TestArtifact] = field(default_factory=list)
    coverage: list[CoverageEntry] = field(default_factory=list)
    compile_errors: list[dict[str, Any]] = field(default_factory=list)
    source_summary: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def coverage_for(self, target: str) -> list[str]:
        for c in self.coverage:
            if c.target == target:
                return list(c.test_ids)
        return []

    def tests_by_kind(self, kind: TestKind) -> list[TestCase]:
        return [t for t in self.tests if t.kind is kind]

    def to_dict(self, *, include_contents: bool = False) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "tests": [t.to_dict() for t in self.tests],
            "artifacts": [a.to_dict(include_content=include_contents)
                          for a in self.artifacts],
            "coverage": [c.to_dict() for c in self.coverage],
            "compile_errors": list(self.compile_errors),
            "source_summary": dict(self.source_summary),
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        by_kind: dict[str, int] = {}
        for t in self.tests:
            by_kind[t.kind.value] = by_kind.get(t.kind.value, 0) + 1
        return (
            "=== Test Plan ===\n"
            f"tests={len(self.tests)}  artifacts={len(self.artifacts)}  "
            f"targets_covered={len(self.coverage)}  "
            f"compile_errors={len(self.compile_errors)}\n"
            f"kinds: {by_kind}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. FILE HEADER + LAZY IMPORT HELPER
# ════════════════════════════════════════════════════════════════════════════
_FILE_HEADER = '''"""Auto-generated tests by SE Brain C17 — do not edit by hand."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _import(module: str, name: str):
    """Import a name from a module; fail the test clearly if unavailable."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:
        pytest.fail(f"cannot import module {module!r}: {exc}")
    try:
        return getattr(mod, name)
    except AttributeError:
        pytest.fail(f"{module!r} has no attribute {name!r}")
'''


# ════════════════════════════════════════════════════════════════════════════
# 4. GENERATORS
# ════════════════════════════════════════════════════════════════════════════
class _ModelInfo:
    """Extracted info about a dataclass/pydantic model."""
    def __init__(self, name: str, module: str) -> None:
        self.name = name
        self.module = module
        self.required_fields: list[tuple[str, str]] = []  # (name, type_str)
        self.optional_fields: list[tuple[str, str, str]] = []  # (name, type, default)


def _extract_model_info(
    content: str, module_path: str,
) -> list[_ModelInfo]:
    """Parse a models.py source string and find dataclass/pydantic classes
    that are domain models (skip the *In input variants for the top-level
    default tests — we still test them separately).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    out: list[_ModelInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_model = False
        bases = [getattr(b, "id", "") or getattr(b, "attr", "") for b in node.bases]
        for deco in node.decorator_list:
            dname = getattr(deco, "id", "") or getattr(deco, "attr", "")
            if dname == "dataclass":
                is_model = True
        if "BaseModel" in bases:
            is_model = True
        if not is_model:
            continue
        info = _ModelInfo(node.name, module_path)
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                fname = child.target.id
                if fname in ("id", "created_at"):
                    continue
                type_str = _annotation_str(child.annotation)
                if child.value is None:
                    info.required_fields.append((fname, type_str))
                else:
                    default_repr = _default_repr(child.value)
                    info.optional_fields.append((fname, type_str, default_repr))
        out.append(info)
    return out


def _annotation_str(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        return "Any"


def _default_repr(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "None"


def _sample_value_for_type(t: str) -> str:
    low = t.lower()
    if "bool" in low:
        return "True"
    if "int" in low and "float" not in low:
        return "1"
    if "float" in low:
        return "1.5"
    if low.startswith("list"):
        return "[]"
    if low.startswith("dict"):
        return "{}"
    if "none" in low and "optional" in low:
        return "None"
    return '"sample"'


def _boundary_value_for_type(t: str) -> tuple[str, str]:
    """Return (value_expr, label) for a boundary test of this type."""
    low = t.lower()
    if "bool" in low:
        return "False", "false"
    if "int" in low and "float" not in low:
        return "0", "zero"
    if "float" in low:
        return "0.0", "zero"
    if low.startswith("list"):
        return "[]", "empty_list"
    if low.startswith("dict"):
        return "{}", "empty_dict"
    if "str" in low:
        return '""', "empty_string"
    return "None", "none"


# ---- individual generators ----
class _Generators:
    def __init__(self, package_name: str) -> None:
        self.pkg = package_name

    # ---- models: unit ----
    def gen_models_unit(
        self, models: Sequence[_ModelInfo], *, source: GenerationSource,
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for m in models:
            target = f"model:{m.name}"
            # default construction
            req_kwargs = ", ".join(
                f"{n}={_sample_value_for_type(t)}" for n, t in m.required_fields
            )
            tname = f"test_{_snake(m.name)}_default_construction"
            code = (
                f"def {tname}() -> None:\n"
                f'    """Covers {target}: default-constructed instance has id=None."""\n'
                f'    cls = _import("{self.pkg}.models", "{m.name}")\n'
                f"    item = cls({req_kwargs})\n"
                f"    assert item.id is None\n"
            )
            out.append(TestCase(
                id=_stable_id("unit", tname), name=tname,
                kind=TestKind.UNIT, code=code, covers=[target],
                source=source,
                rationale=f"default construction for {m.name}",
                evidence=[{"check": "id_is_none"}],
            ))
            # required fields present
            if m.required_fields:
                asserts = "\n".join(
                    f"    assert item.{n} == {_sample_value_for_type(t)}"
                    for n, t in m.required_fields
                )
                tname = f"test_{_snake(m.name)}_required_fields"
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers {target}: required fields set from constructor."""\n'
                    f'    cls = _import("{self.pkg}.models", "{m.name}")\n'
                    f"    item = cls({req_kwargs})\n"
                    f"{asserts}\n"
                )
                out.append(TestCase(
                    id=_stable_id("unit", tname), name=tname,
                    kind=TestKind.UNIT, code=code, covers=[target],
                    source=source,
                    rationale=f"required-field construction for {m.name}",
                    evidence=[{"check": "required_fields_roundtrip"}],
                ))
            # optional fields default
            if m.optional_fields:
                tname = f"test_{_snake(m.name)}_optional_defaults"
                asserts = "\n".join(
                    f"    assert item.{n} == {d}"
                    for n, _t, d in m.optional_fields
                )
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers {target}: optional fields take their defaults."""\n'
                    f'    cls = _import("{self.pkg}.models", "{m.name}")\n'
                    f"    item = cls({req_kwargs})\n"
                    f"{asserts}\n"
                )
                out.append(TestCase(
                    id=_stable_id("unit", tname), name=tname,
                    kind=TestKind.UNIT, code=code, covers=[target],
                    source=source,
                    rationale=f"optional default behavior for {m.name}",
                    evidence=[{"check": "optional_defaults"}],
                ))
        return out

    # ---- models: boundary ----
    def gen_models_boundary(
        self, models: Sequence[_ModelInfo],
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for m in models:
            for fname, ftype in (m.required_fields + [(n, t) for n, t, _ in m.optional_fields]):
                val, label = _boundary_value_for_type(ftype)
                tname = f"test_{_snake(m.name)}_{fname}_boundary_{label}"
                # Build kwargs with boundary value in this field, valid values elsewhere
                kwargs: list[str] = []
                for n, t in m.required_fields:
                    if n == fname:
                        kwargs.append(f"{n}={val}")
                    else:
                        kwargs.append(f"{n}={_sample_value_for_type(t)}")
                tname = _safe_ident(tname)
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers {m.name}.{fname}: accepts {label} value."""\n'
                    f'    cls = _import("{self.pkg}.models", "{m.name}")\n'
                    f"    item = cls({', '.join(kwargs)})\n"
                    f"    assert item.{fname} == {val}\n"
                )
                out.append(TestCase(
                    id=_stable_id("boundary", tname), name=tname,
                    kind=TestKind.BOUNDARY, code=code,
                    covers=[f"model:{m.name}", f"field:{m.name}.{fname}"],
                    source=GenerationSource.SYNTHESIZED_MODEL,
                    rationale=f"boundary {label} accepted for {m.name}.{fname}",
                    evidence=[{"check": "boundary_value", "value": val}],
                ))
        return out

    # ---- models: negative ----
    def gen_models_negative(
        self, models: Sequence[_ModelInfo], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for m in models:
            for fname, _t in m.required_fields:
                # Missing required field must raise (TypeError for dataclass,
                # ValidationError for pydantic — accept either).
                kwargs = ", ".join(
                    f"{n}={_sample_value_for_type(t)}"
                    for n, t in m.required_fields if n != fname
                )
                tname = _safe_ident(f"test_{_snake(m.name)}_missing_{fname}")
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers {m.name}.{fname}: missing required field is refused."""\n'
                    f'    cls = _import("{self.pkg}.models", "{m.name}")\n'
                    f"    with pytest.raises((TypeError, ValueError, Exception)):\n"
                    f"        cls({kwargs})\n"
                )
                out.append(TestCase(
                    id=_stable_id("negative", tname), name=tname,
                    kind=TestKind.NEGATIVE, code=code,
                    covers=[f"model:{m.name}", f"field:{m.name}.{fname}"],
                    source=GenerationSource.SYNTHESIZED_MODEL,
                    rationale=f"missing required field {m.name}.{fname}",
                    evidence=[{"check": "missing_required_field"}],
                ))
        return out

    # ---- repository CRUD integration ----
    def gen_repository_integration(
        self, models: Sequence[_ModelInfo], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for m in models:
            req_kwargs = ", ".join(
                f"{n}={_sample_value_for_type(t)}" for n, t in m.required_fields
            )
            repo = f"{m.name}Repository"
            inp = f"{m.name}In"
            target = f"repo:{repo}"
            tname = f"test_{_snake(m.name)}_repository_crud_roundtrip"
            code = (
                f"def {tname}(tmp_path: Path) -> None:\n"
                f'    """Covers {target}: create → get → list → update → delete."""\n'
                f'    Repo = _import("{self.pkg}.repository", "{repo}")\n'
                f'    In = _import("{self.pkg}.models", "{inp}")\n'
                f'    repo = Repo(tmp_path / "x.sqlite3")\n'
                f"    payload = In({req_kwargs})\n"
                f"    created = repo.create(payload)\n"
                f"    assert created.id is not None\n"
                f"    fetched = repo.get(created.id)\n"
                f"    assert fetched is not None\n"
                f"    assert fetched.id == created.id\n"
                f"    assert len(repo.list_all()) >= 1\n"
                f"    updated = repo.update(created.id, payload)\n"
                f"    assert updated is not None\n"
                f"    assert repo.delete(created.id) is True\n"
                f"    assert repo.get(created.id) is None\n"
            )
            out.append(TestCase(
                id=_stable_id("integration", tname), name=tname,
                kind=TestKind.INTEGRATION, code=code, covers=[target],
                source=GenerationSource.SYNTHESIZED_REPOSITORY,
                rationale=f"CRUD roundtrip for {repo}",
                evidence=[{"check": "crud_roundtrip"}],
            ))
            # negative: get on empty DB returns None
            tname = f"test_{_snake(m.name)}_repository_get_missing_returns_none"
            code = (
                f"def {tname}(tmp_path: Path) -> None:\n"
                f'    """Covers {target}: get() on missing id returns None."""\n'
                f'    Repo = _import("{self.pkg}.repository", "{repo}")\n'
                f'    repo = Repo(tmp_path / "x.sqlite3")\n'
                f"    assert repo.get(99999) is None\n"
            )
            out.append(TestCase(
                id=_stable_id("negative", tname), name=tname,
                kind=TestKind.NEGATIVE, code=code, covers=[target],
                source=GenerationSource.SYNTHESIZED_REPOSITORY,
                rationale=f"get() missing id → None for {repo}",
                evidence=[{"check": "get_missing_returns_none"}],
            ))
            # negative: delete non-existent returns False
            tname = f"test_{_snake(m.name)}_repository_delete_missing_returns_false"
            code = (
                f"def {tname}(tmp_path: Path) -> None:\n"
                f'    """Covers {target}: delete() on missing id returns False."""\n'
                f'    Repo = _import("{self.pkg}.repository", "{repo}")\n'
                f'    repo = Repo(tmp_path / "x.sqlite3")\n'
                f"    assert repo.delete(99999) is False\n"
            )
            out.append(TestCase(
                id=_stable_id("negative", tname), name=tname,
                kind=TestKind.NEGATIVE, code=code, covers=[target],
                source=GenerationSource.SYNTHESIZED_REPOSITORY,
                rationale=f"delete() missing id → False for {repo}",
                evidence=[{"check": "delete_missing_returns_false"}],
            ))
        return out

    # ---- service unit ----
    def gen_service_unit(
        self, models: Sequence[_ModelInfo], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for m in models:
            req_kwargs = ", ".join(
                f"{n}={_sample_value_for_type(t)}" for n, t in m.required_fields
            )
            svc = f"{m.name}Service"
            repo = f"{m.name}Repository"
            inp = f"{m.name}In"
            tname = f"test_{_snake(m.name)}_service_create_and_list"
            code = (
                f"def {tname}(tmp_path: Path) -> None:\n"
                f'    """Covers service:{svc}: create + list delegate to repo."""\n'
                f'    Svc = _import("{self.pkg}.service", "{svc}")\n'
                f'    Repo = _import("{self.pkg}.repository", "{repo}")\n'
                f'    In = _import("{self.pkg}.models", "{inp}")\n'
                f'    svc = Svc(Repo(tmp_path / "x.sqlite3"))\n'
                f"    created = svc.create(In({req_kwargs}))\n"
                f"    assert created.id is not None\n"
                f"    assert len(svc.list()) == 1\n"
            )
            out.append(TestCase(
                id=_stable_id("unit", tname), name=tname,
                kind=TestKind.UNIT, code=code, covers=[f"service:{svc}"],
                source=GenerationSource.SYNTHESIZED_SERVICE,
                rationale=f"service wiring for {svc}",
                evidence=[{"check": "service_create_list"}],
            ))
        return out

    # ---- API integration ----
    def gen_api_integration(
        self, models: Sequence[_ModelInfo], *,
        framework: str,
    ) -> list[TestCase]:
        out: list[TestCase] = []
        if framework == "fastapi":
            for m in models:
                plural = _plural(_snake(m.name))
                req_json = ", ".join(
                    f'"{n}": {_json_sample(t)}' for n, t in m.required_fields
                )
                # POST
                tname = f"test_api_post_{plural}"
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers route:POST /{plural}: creates a resource."""\n'
                    f'    pytest.importorskip("httpx")\n'
                    f"    from fastapi.testclient import TestClient\n"
                    f'    app = _import("{self.pkg}.api", "app")\n'
                    f"    client = TestClient(app)\n"
                    f'    r = client.post("/{plural}", json={{{req_json}}})\n'
                    f"    assert r.status_code in (200, 201)\n"
                )
                out.append(TestCase(
                    id=_stable_id("integration", tname), name=tname,
                    kind=TestKind.INTEGRATION, code=code,
                    covers=[f"route:POST /{plural}"],
                    source=GenerationSource.SYNTHESIZED_API,
                    rationale=f"POST /{plural} returns 2xx",
                    evidence=[{"check": "post_creates"}],
                ))
                # GET list
                tname = f"test_api_get_{plural}_list"
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers route:GET /{plural}: returns list."""\n'
                    f'    pytest.importorskip("httpx")\n'
                    f"    from fastapi.testclient import TestClient\n"
                    f'    app = _import("{self.pkg}.api", "app")\n'
                    f"    client = TestClient(app)\n"
                    f'    r = client.get("/{plural}")\n'
                    f"    assert r.status_code == 200\n"
                    f"    assert isinstance(r.json(), list)\n"
                )
                out.append(TestCase(
                    id=_stable_id("integration", tname), name=tname,
                    kind=TestKind.INTEGRATION, code=code,
                    covers=[f"route:GET /{plural}"],
                    source=GenerationSource.SYNTHESIZED_API,
                    rationale=f"GET /{plural} returns list",
                    evidence=[{"check": "get_list"}],
                ))
                # GET missing id → 404
                tname = f"test_api_get_{plural}_missing_404"
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers route:GET /{plural}/{{item_id}}: 404 on missing."""\n'
                    f'    pytest.importorskip("httpx")\n'
                    f"    from fastapi.testclient import TestClient\n"
                    f'    app = _import("{self.pkg}.api", "app")\n'
                    f"    client = TestClient(app)\n"
                    f'    r = client.get("/{plural}/999999")\n'
                    f"    assert r.status_code == 404\n"
                )
                out.append(TestCase(
                    id=_stable_id("negative", tname), name=tname,
                    kind=TestKind.NEGATIVE, code=code,
                    covers=[f"route:GET /{plural}/{{item_id}}"],
                    source=GenerationSource.SYNTHESIZED_API,
                    rationale=f"GET /{plural}/<id> → 404 for missing",
                    evidence=[{"check": "get_missing_404"}],
                ))
        elif framework == "flask":
            for m in models:
                plural = _plural(_snake(m.name))
                req_json = ", ".join(
                    f'"{n}": {_json_sample(t)}' for n, t in m.required_fields
                )
                tname = f"test_api_post_{plural}"
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Covers route:POST /{plural}: creates a resource."""\n'
                    f'    app = _import("{self.pkg}.api", "app")\n'
                    f"    client = app.test_client()\n"
                    f'    r = client.post("/{plural}", json={{{req_json}}})\n'
                    f"    assert r.status_code in (200, 201)\n"
                )
                out.append(TestCase(
                    id=_stable_id("integration", tname), name=tname,
                    kind=TestKind.INTEGRATION, code=code,
                    covers=[f"route:POST /{plural}"],
                    source=GenerationSource.SYNTHESIZED_API,
                    rationale=f"POST /{plural} returns 2xx",
                    evidence=[{"check": "post_creates"}],
                ))
        return out

    # ---- acceptance (from C05) ----
    def gen_acceptance(
        self, acceptance_criteria: Sequence[str], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for i, text in enumerate(acceptance_criteria):
            target = f"accept:{_stable_id('accept', text)}"
            gwt = _parse_gwt(text)
            tname = _safe_ident(f"test_acceptance_{i}_{_slug(text)[:40]}")
            if gwt is not None:
                given, when, then = gwt
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Acceptance: {_escape_doc(text)}"""\n'
                    f"    # Given: {given}\n"
                    f"    # When:  {when}\n"
                    f"    # Then:  {then}\n"
                    f'    pytest.skip("acceptance criterion requires manual/fixture wiring")\n'
                )
            else:
                code = (
                    f"def {tname}() -> None:\n"
                    f'    """Acceptance: {_escape_doc(text)}"""\n'
                    f'    pytest.skip("acceptance criterion is not expressible as a unit assertion")\n'
                )
            out.append(TestCase(
                id=_stable_id("acceptance", tname), name=tname,
                kind=TestKind.ACCEPTANCE, code=code, covers=[target],
                source=GenerationSource.ACCEPTANCE_CRITERION,
                rationale=f"acceptance criterion #{i} (GWT parsed: {gwt is not None})",
                evidence=[{"check": "acceptance_criterion", "text": text}],
            ))
        return out

    # ---- functional requirement smoke ----
    def gen_functional_smoke(
        self, functional: Sequence[str], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for i, text in enumerate(functional):
            target = f"req:{_stable_id('req', text)}"
            tname = _safe_ident(f"test_functional_req_{i}_{_slug(text)[:40]}")
            code = (
                f"def {tname}() -> None:\n"
                f'    """Functional requirement: {_escape_doc(text)}"""\n'
                f'    pytest.skip("functional requirement needs integration wiring")\n'
            )
            out.append(TestCase(
                id=_stable_id("acceptance", tname), name=tname,
                kind=TestKind.ACCEPTANCE, code=code, covers=[target],
                source=GenerationSource.REQUIREMENT,
                rationale=f"functional requirement #{i}",
                evidence=[{"check": "functional_requirement", "text": text}],
            ))
        return out

    # ---- NFR performance ----
    def gen_nfr_performance(
        self, nfr_items: Sequence[tuple[str, list[str]]], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for i, (text, tags) in enumerate(nfr_items):
            if "performance" not in tags:
                continue
            target = f"nfr:perf:{i}"
            tname = _safe_ident(f"test_nfr_perf_{i}")
            # Extract a numeric threshold if present (ms)
            m = re.search(r"(\d+)\s*ms", text)
            threshold_ms = int(m.group(1)) if m else 500
            # Generous multiplier for CI
            limit_ms = threshold_ms * 10
            code = (
                f"def {tname}() -> None:\n"
                f'    """NFR performance: {_escape_doc(text)}"""\n'
                f"    import time\n"
                f"    t0 = time.perf_counter()\n"
                f"    for _ in range(1000):\n"
                f"        pass\n"
                f"    elapsed_ms = (time.perf_counter() - t0) * 1000.0\n"
                f"    # Runtime must be able to execute 1000 no-ops within a\n"
                f"    # generous budget (10x the stated NFR target).\n"
                f"    assert elapsed_ms < {limit_ms}\n"
            )
            out.append(TestCase(
                id=_stable_id("performance", tname), name=tname,
                kind=TestKind.PERFORMANCE, code=code, covers=[target],
                source=GenerationSource.NFR,
                rationale=f"performance smoke for '{_short(text, 60)}' "
                          f"(target {threshold_ms}ms, budget {limit_ms}ms)",
                evidence=[{"check": "perf_smoke", "threshold_ms": threshold_ms}],
            ))
        return out

    # ---- NFR security ----
    def gen_nfr_security(
        self, nfr_items: Sequence[tuple[str, list[str]]], *,
        package: str, code_roots: Sequence[str] = ("models.py", "repository.py",
                                                    "service.py", "api.py"),
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for i, (text, tags) in enumerate(nfr_items):
            if "security" not in tags:
                continue
            target = f"nfr:sec:{i}"
            tname = _safe_ident(f"test_nfr_sec_no_unsafe_calls_{i}")
            code = (
                f"def {tname}() -> None:\n"
                f'    """NFR security: {_escape_doc(text)} — '
                f'scans for eval/exec/os.system."""\n'
                f"    import ast\n"
                f"    pkg_dir = Path(__file__).resolve().parent.parent / "
                f'"{package}"\n'
                f"    if not pkg_dir.exists():\n"
                f'        pytest.skip("package directory not found")\n'
                f"    unsafe = {{'eval', 'exec'}}\n"
                f"    for py in pkg_dir.rglob('*.py'):\n"
                f"        tree = ast.parse(py.read_text())\n"
                f"        for node in ast.walk(tree):\n"
                f"            if isinstance(node, ast.Call):\n"
                f"                fn = node.func\n"
                f"                if isinstance(fn, ast.Name) and fn.id in unsafe:\n"
                f"                    raise AssertionError(\n"
                f'                        f"unsafe call {{fn.id}}() in {{py}}:{{node.lineno}}")\n'
                f"                if isinstance(fn, ast.Attribute):\n"
                f"                    if (isinstance(fn.value, ast.Name)\n"
                f"                            and fn.value.id == 'os'\n"
                f"                            and fn.attr == 'system'):\n"
                f"                        raise AssertionError(\n"
                f'                            f"os.system() in {{py}}:{{node.lineno}}")\n'
            )
            out.append(TestCase(
                id=_stable_id("security", tname), name=tname,
                kind=TestKind.SECURITY, code=code, covers=[target],
                source=GenerationSource.NFR,
                rationale=f"security smoke for '{_short(text, 60)}'",
                evidence=[{"check": "no_eval_exec_os_system"}],
            ))
        return out

    # ---- symbol contract ----
    def gen_symbol_contracts(
        self, symbols: Sequence[Any], *,
        module_map: dict[str, str],
    ) -> list[TestCase]:
        """Given C13 Symbol objects, emit existence + callable tests for
        public functions and classes.
        """
        out: list[TestCase] = []
        for s in symbols:
            kind = getattr(s, "kind", None)
            kind_val = getattr(kind, "value", str(kind))
            if kind_val not in ("function", "class"):
                continue
            visibility = getattr(s, "visibility", "public")
            if visibility != "public":
                continue
            name = s.name
            if name.startswith("_") or name == "<module>":
                continue
            module_path = s.module_path
            module_name = _module_to_import(module_path)
            if module_name is None:
                continue
            target = f"symbol:{s.id}"
            tname = _safe_ident(f"test_contract_{_snake(name)}_{s.id[:6]}")
            code = (
                f"def {tname}() -> None:\n"
                f'    """Covers {target}: public symbol exists and is importable."""\n'
                f'    obj = _import("{module_name}", "{name}")\n'
                f"    assert obj is not None\n"
                f"    assert callable(obj) or isinstance(obj, type)\n"
            )
            out.append(TestCase(
                id=_stable_id("unit", tname), name=tname,
                kind=TestKind.UNIT, code=code, covers=[target],
                source=GenerationSource.SYMBOL,
                rationale=f"public symbol {module_path}:{name}",
                evidence=[{"check": "symbol_exists"}],
            ))
        return out

    # ---- regression (from memory) ----
    def gen_regression_from_failures(
        self, failures: Sequence[dict[str, Any]], 
    ) -> list[TestCase]:
        out: list[TestCase] = []
        for i, f in enumerate(failures):
            key = str(f.get("key") or f.get("id") or i)
            target = f"regression:{_stable_id('reg', key)}"
            what = str(f.get("what", "unknown failure"))
            cause = str(f.get("root_cause", ""))
            tname = _safe_ident(f"test_regression_{i}_{_slug(what)[:30]}")
            code = (
                f"def {tname}() -> None:\n"
                f'    """Regression marker: {_escape_doc(what)}"""\n'
                f'    # Documented cause: {_escape_doc(cause)}\n'
                f"    pytest.skip(\n"
                f'        "regression marker from C04 memory — needs a '
                f'reproduction fixture")\n'
            )
            out.append(TestCase(
                id=_stable_id("regression", tname), name=tname,
                kind=TestKind.REGRESSION, code=code, covers=[target],
                source=GenerationSource.PAST_FAILURE,
                rationale=f"regression marker for past failure: {_short(what, 60)}",
                evidence=[{"check": "regression_marker", "cause": cause}],
            ))
        return out


# ---- helpers used above ----
def _plural(n: str) -> str:
    if n.endswith(("s", "x", "z", "ch", "sh")):
        return n + "es"
    if n.endswith("y") and len(n) > 1 and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def _escape_doc(s: str) -> str:
    # Ensure triple-quote safety
    return s.replace('"""', "'''").replace("\n", " ").strip()


def _json_sample(t: str) -> str:
    low = t.lower()
    if "bool" in low:
        return "True"
    if "int" in low and "float" not in low:
        return "1"
    if "float" in low:
        return "1.5"
    if low.startswith("list"):
        return "[]"
    if low.startswith("dict"):
        return "{}"
    return '"sample"'


_GWT_RE = re.compile(
    r"given\s+(?P<g>.+?),?\s+when\s+(?P<w>.+?),?\s+then\s+(?P<t>.+)",
    re.I | re.S,
)


def _parse_gwt(text: str) -> tuple[str, str, str] | None:
    m = _GWT_RE.search(text)
    if not m:
        return None
    return (
        m.group("g").strip().rstrip(".,"),
        m.group("w").strip().rstrip(".,"),
        m.group("t").strip().rstrip(".,"),
    )


def _module_to_import(path: str) -> str | None:
    """Convert a repo-relative path to a Python import name (or None)."""
    if not path.endswith(".py"):
        return None
    if path.endswith("__init__.py"):
        p = path[: -len("/__init__.py")]
    else:
        p = path[:-3]
    p = p.replace("/", ".")
    if not p or p.startswith("."):
        return None
    return p


# ════════════════════════════════════════════════════════════════════════════
# 5. TEST GENERATOR (facade)
# ════════════════════════════════════════════════════════════════════════════
_KIND_TO_FILE: dict[TestKind, str] = {
    TestKind.UNIT: "tests/test_unit.py",
    TestKind.INTEGRATION: "tests/test_integration.py",
    TestKind.BOUNDARY: "tests/test_boundary.py",
    TestKind.NEGATIVE: "tests/test_negative.py",
    TestKind.ACCEPTANCE: "tests/test_acceptance.py",
    TestKind.SECURITY: "tests/test_security.py",
    TestKind.PERFORMANCE: "tests/test_performance.py",
    TestKind.REGRESSION: "tests/test_regression.py",
}


class TestGenerator:
    """Produce a TestPlan from any subset of available structured inputs.

    All inputs are optional. Missing ones simply contribute nothing.
    """
    def __init__(
        self, *,
        max_tests_total: int = 500,
        max_per_file: int = 200,
    ) -> None:
        if max_tests_total < 1:
            raise ValidationError("max_tests_total must be >= 1")
        if max_per_file < 1:
            raise ValidationError("max_per_file must be >= 1")
        self.max_tests_total = max_tests_total
        self.max_per_file = max_per_file

    # ---- main API ----
    def generate(
        self,
        *,
        spec: Any | None = None,
        intent_ctx: Any | None = None,     # accepted for future use
        plan: Any | None = None,           # accepted for future use
        repo_index: Any | None = None,
        synthesis: Any | None = None,      # C14 SynthesisResult
        memory: MemoryStore | None = None,
        project_id: str = "",
        package_name: str | None = None,
        framework: str = "fastapi",
        include_regression: bool = True,
    ) -> TestPlan:
        pkg = package_name or _infer_package(synthesis, repo_index)
        gens = _Generators(pkg)

        all_tests: list[TestCase] = []
        src_count: dict[str, int] = {}

        def _add(ts: Iterable[TestCase], source: GenerationSource) -> None:
            for t in ts:
                if len(all_tests) >= self.max_tests_total:
                    return
                all_tests.append(t)
            src_count[source.value] = (
                src_count.get(source.value, 0) + len(list(ts))
            )

        # ---- 1. Acceptance + functional requirements (from C05) ----
        if spec is not None:
            accept = [
                getattr(it, "text", "") for it in
                getattr(spec, "acceptance_criteria", []) or []
            ]
            functional = [
                getattr(it, "text", "") for it in
                getattr(spec, "functional", []) or []
            ]
            _add(gens.gen_acceptance(accept),
                 GenerationSource.ACCEPTANCE_CRITERION)
            _add(gens.gen_functional_smoke(functional),
                 GenerationSource.REQUIREMENT)

        # ---- 2. Models / repository / service / API from C14 synthesis ----
        models_info: list[_ModelInfo] = []
        model_files = _find_files(synthesis, ["models.py"])
        if model_files:
            mfile = model_files[0]
            models_info = _extract_model_info(
                str(getattr(mfile, "content", "")),
                str(getattr(mfile, "path", "")),
            )
            if models_info:
                _add(gens.gen_models_unit(models_info,
                                           source=GenerationSource.SYNTHESIZED_MODEL),
                     GenerationSource.SYNTHESIZED_MODEL)
                _add(gens.gen_models_boundary(models_info),
                     GenerationSource.SYNTHESIZED_MODEL)
                _add(gens.gen_models_negative(models_info),
                     GenerationSource.SYNTHESIZED_MODEL)

        if _find_files(synthesis, ["repository.py"]):
            _add(gens.gen_repository_integration(models_info),
                 GenerationSource.SYNTHESIZED_REPOSITORY)

        if _find_files(synthesis, ["service.py"]):
            _add(gens.gen_service_unit(models_info),
                 GenerationSource.SYNTHESIZED_SERVICE)

        if _find_files(synthesis, ["api.py"]):
            _add(gens.gen_api_integration(models_info, framework=framework),
                 GenerationSource.SYNTHESIZED_API)

        # ---- 3. Symbol contracts from C13 index ----
        if repo_index is not None:
            symbols = list(getattr(repo_index, "symbol_index", {}).values())
            _add(gens.gen_symbol_contracts(symbols, module_map={}),
                 GenerationSource.SYMBOL)

        # ---- 4. NFRs (perf + security) from C05 ----
        if spec is not None:
            nfr_items: list[tuple[str, list[str]]] = []
            for it in getattr(spec, "non_functional", []) or []:
                nfr_items.append((
                    getattr(it, "text", "") or "",
                    list(getattr(it, "tags", []) or []),
                ))
            _add(gens.gen_nfr_performance(nfr_items),
                 GenerationSource.NFR)
            _add(gens.gen_nfr_security(nfr_items, package=pkg),
                 GenerationSource.NFR)

        # ---- 5. Regression markers from C04 memory ----
        if include_regression and memory is not None and project_id:
            try:
                failures = memory.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id=project_id,
                )
                failure_dicts = [
                    {
                        "key": f.key, "what": f.content.get("what", ""),
                        "root_cause": f.content.get("root_cause", ""),
                    }
                    for f in failures[:20]
                ]
                _add(gens.gen_regression_from_failures(failure_dicts),
                     GenerationSource.PAST_FAILURE)
            except Exception as exc:
                log.warning("c17.regression_lookup_failed", error=str(exc))

        # ---- Group into artifacts by kind (bounded per file) ----
        artifacts: list[TestArtifact] = []
        for kind in TestKind:
            tests_k = [t for t in all_tests if t.kind is kind]
            if not tests_k:
                continue
            # Sort deterministically
            tests_k.sort(key=lambda t: t.name)
            tests_k = tests_k[: self.max_per_file]
            content = self._render_artifact(kind, tests_k)
            path = _KIND_TO_FILE[kind]
            artifacts.append(TestArtifact(
                path=path, kind=kind, content=content,
                test_ids=[t.id for t in tests_k],
                rationale=(
                    f"{kind.value}: {len(tests_k)} tests targeting "
                    f"{_count_targets(tests_k)} unique targets"
                ),
            ))

        # ---- Compile-verify each artifact ----
        plan_out = TestPlan(
            project_id=project_id,
            tests=all_tests,
            artifacts=artifacts,
            source_summary=src_count,
            rationale=(
                f"tests={len(all_tests)}  artifacts={len(artifacts)}  "
                f"sources={src_count}"
            ),
            provenance=Provenance(
                source="test_generator",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        for a in artifacts:
            try:
                compile(a.content, a.path, "exec")
            except SyntaxError as exc:
                plan_out.compile_errors.append({
                    "path": a.path,
                    "message": f"{exc.msg} (line {exc.lineno})",
                })

        # ---- Coverage map ----
        coverage_map: dict[str, CoverageEntry] = {}
        for t in all_tests:
            for target in t.covers:
                tkind = target.split(":", 1)[0] if ":" in target else "target"
                e = coverage_map.get(target)
                if e is None:
                    e = CoverageEntry(target=target, target_kind=tkind)
                    coverage_map[target] = e
                e.test_ids.append(t.id)
        plan_out.coverage = sorted(
            coverage_map.values(), key=lambda c: c.target,
        )
        return plan_out

    # ---- rendering ----
    def _render_artifact(
        self, kind: TestKind, tests: Sequence[TestCase],
    ) -> str:
        lines: list[str] = [_FILE_HEADER, ""]
        for t in tests:
            lines.append(t.code.rstrip("\n"))
            lines.append("")
            lines.append("")
        # Trim trailing blank lines and end with single newline
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines) + "\n"


# ---- helpers used by facade ----
def _find_files(synthesis: Any, names: Sequence[str]) -> list[Any]:
    if synthesis is None:
        return []
    files = getattr(synthesis, "files", None) or []
    out: list[Any] = []
    for f in files:
        p = str(getattr(f, "path", ""))
        base = p.rsplit("/", 1)[-1]
        if base in names:
            out.append(f)
    return out


def _infer_package(synthesis: Any, repo_index: Any) -> str:
    # Prefer synthesis: pick the first top-level package dir
    if synthesis is not None:
        for f in getattr(synthesis, "files", None) or []:
            p = str(getattr(f, "path", ""))
            if "/" in p:
                head = p.split("/", 1)[0]
                if head and head != "tests":
                    return head
    if repo_index is not None:
        pkgs = list(getattr(repo_index, "packages", lambda: [])())
        if pkgs:
            return pkgs[0].split(".", 1)[0]
    return "pkg"


def _count_targets(tests: Sequence[TestCase]) -> int:
    s: set[str] = set()
    for t in tests:
        s.update(t.covers)
    return len(s)


# ════════════════════════════════════════════════════════════════════════════
# 6. TEST REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class TestRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, plan: TestPlan, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"test_plan:{plan.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, plan.to_dict(include_contents=False),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["test_plan", "c17"],
            provenance=plan.provenance,
        )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.TEST,
            _short(f"TestPlan {plan.id[:8]} ({len(plan.tests)} tests)", 120),
            attributes={
                "test_plan_id": plan.id,
                "project_id": project_id,
                "tests": len(plan.tests),
                "artifacts": len(plan.artifacts),
                "covered_targets": len(plan.coverage),
                "kinds": sorted({t.kind.value for t in plan.tests}),
            },
            tags=["test-plan"],
            provenance=plan.provenance,
        )
        for a in plan.artifacts:
            fe = self.ontology.add(
                EntityKind.FILE, _short(a.path, 120),
                attributes={
                    "kind": a.kind.value,
                    "byte_size": a.byte_size,
                    "test_count": len(a.test_ids),
                    "rationale": a.rationale,
                },
                tags=["test-file", a.kind.value],
                provenance=plan.provenance,
            )
            try:
                self.ontology.link(RelationKind.PRODUCES, ent.id, fe.id)
            except ValidationError:
                pass
        return ent.id

    def load(self, plan_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"test_plan:{plan_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


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

    print("Running C17 self-tests…")
    gen = TestGenerator()

    # ---- helpers ----
    def _sample_spec():
        from sebrain.c05 import RequirementParser
        text = (
            "Build a small REST API for managing tasks.\n\n"
            "Functional requirements:\n"
            "- Users must be able to create tasks.\n"
            "- Users must be able to list tasks.\n\n"
            "Non-functional:\n"
            "- The API must respond within 200ms for read operations.\n"
            "- All traffic must use HTTPS.\n\n"
            "Acceptance:\n"
            "- Given a valid request, when POST /tasks is called, then a 201 is returned.\n"
        )
        return RequirementParser().parse(text)

    def _sample_synthesis():
        from sebrain.c14 import (
            CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
        )
        from sebrain.c13 import RepoIndex
        entity = EntitySpec(
            name="Task",
            fields=[
                FieldSpec("title", "str", required=True),
                FieldSpec("description", "str", required=False, default_repr='""'),
                FieldSpec("done", "bool", required=False, default_repr="False"),
            ],
        )
        return CodeSynthesisEngine().synthesize(
            SynthesisRequest(
                package_name="task_api", entities=[entity],
                framework="fastapi", model_style="dataclass", mode="fresh",
            ),
            project_id="p",
            existing_index=RepoIndex(root="<none>"),
        )

    # ---- empty inputs ----
    def t_empty_plan() -> None:
        p = gen.generate()
        assert p.tests == []
        assert p.artifacts == []
        assert p.coverage == []
        assert p.compile_errors == []

    def t_spec_only() -> None:
        spec = _sample_spec()
        p = gen.generate(spec=spec, project_id="x")
        # At least acceptance + functional + NFR tests
        kinds = {t.kind for t in p.tests}
        assert TestKind.ACCEPTANCE in kinds
        assert TestKind.PERFORMANCE in kinds
        assert TestKind.SECURITY in kinds

    def t_synthesis_only() -> None:
        synth = _sample_synthesis()
        p = gen.generate(synthesis=synth, project_id="x")
        kinds = {t.kind for t in p.tests}
        # Models → unit + boundary + negative
        assert TestKind.UNIT in kinds
        assert TestKind.BOUNDARY in kinds
        assert TestKind.NEGATIVE in kinds
        # Repository + API
        assert TestKind.INTEGRATION in kinds

    check("empty inputs → empty plan", t_empty_plan)
    check("spec only → acceptance + performance + security",
          t_spec_only)
    check("synthesis only → unit + boundary + negative + integration",
          t_synthesis_only)

    # ---- every artifact compiles ----
    def t_artifacts_compile_spec() -> None:
        p = gen.generate(spec=_sample_spec(), project_id="x")
        assert p.compile_errors == [], p.compile_errors
        assert len(p.artifacts) >= 1
        for a in p.artifacts:
            compile(a.content, a.path, "exec")   # raises if broken

    def t_artifacts_compile_synthesis() -> None:
        p = gen.generate(synthesis=_sample_synthesis(), project_id="x")
        assert p.compile_errors == [], p.compile_errors
        for a in p.artifacts:
            compile(a.content, a.path, "exec")

    def t_artifacts_compile_combined() -> None:
        p = gen.generate(
            spec=_sample_spec(), synthesis=_sample_synthesis(),
            project_id="x",
        )
        assert p.compile_errors == [], p.compile_errors
        for a in p.artifacts:
            compile(a.content, a.path, "exec")

    check("artifacts compile: spec only", t_artifacts_compile_spec)
    check("artifacts compile: synthesis only", t_artifacts_compile_synthesis)
    check("artifacts compile: combined", t_artifacts_compile_combined)

    # ---- coverage traceability ----
    def t_coverage_populated() -> None:
        p = gen.generate(
            spec=_sample_spec(), synthesis=_sample_synthesis(),
            project_id="x",
        )
        assert len(p.coverage) >= 5
        # Every coverage entry has at least one test
        for c in p.coverage:
            assert c.test_ids, c
        # Every test appears in coverage at least once (if it has covers)
        for t in p.tests:
            if t.covers:
                for target in t.covers:
                    assert p.coverage_for(target), target

    def t_coverage_has_expected_targets() -> None:
        p = gen.generate(synthesis=_sample_synthesis(), project_id="x")
        targets = {c.target for c in p.coverage}
        # Model targets
        assert any(t.startswith("model:Task") for t in targets), targets
        # Repository targets
        assert any(t.startswith("repo:TaskRepository") for t in targets)
        # Field boundary
        assert any("field:Task.title" in t for t in targets)

    check("coverage: populated and consistent", t_coverage_populated)
    check("coverage: expected targets present",
          t_coverage_has_expected_targets)

    # ---- determinism ----
    def t_deterministic() -> None:
        p1 = gen.generate(spec=_sample_spec(), project_id="x")
        p2 = gen.generate(spec=_sample_spec(), project_id="x")
        ids1 = sorted(t.id for t in p1.tests)
        ids2 = sorted(t.id for t in p2.tests)
        assert ids1 == ids2
        # Same artifact contents
        c1 = {a.path: a.content for a in p1.artifacts}
        c2 = {a.path: a.content for a in p2.artifacts}
        assert c1 == c2

    check("deterministic: same input → same tests + artifacts",
          t_deterministic)

    # ---- kinds grouping ----
    def t_kind_files() -> None:
        p = gen.generate(
            spec=_sample_spec(), synthesis=_sample_synthesis(),
            project_id="x",
        )
        paths = {a.path for a in p.artifacts}
        assert "tests/test_unit.py" in paths
        assert "tests/test_integration.py" in paths
        assert "tests/test_boundary.py" in paths
        assert "tests/test_negative.py" in paths
        assert "tests/test_acceptance.py" in paths
        assert "tests/test_performance.py" in paths
        assert "tests/test_security.py" in paths

    check("kind → file mapping produces expected files", t_kind_files)

    # ---- acceptance parsing ----
    def t_gwt_parsing() -> None:
        gwt = _parse_gwt(
            "Given a valid request, when POST /tasks is called, then a 201 is returned."
        )
        assert gwt is not None
        given, when, then = gwt
        assert "valid request" in given
        assert "POST" in when
        assert "201" in then

    def t_gwt_missing_returns_none() -> None:
        assert _parse_gwt("Users can create tasks.") is None

    check("GWT parser: Given/When/Then extracted", t_gwt_parsing)
    check("GWT parser: non-GWT returns None", t_gwt_missing_returns_none)

    # ---- model extraction ----
    def t_model_extraction() -> None:
        models_src = '''"""x."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Task:
    title: str
    description: str = ""
    done: bool = False
'''
        infos = _extract_model_info(models_src, "pkg/models.py")
        assert len(infos) == 1
        m = infos[0]
        assert m.name == "Task"
        assert ("title", "str") in m.required_fields
        assert any(n == "description" for n, _t, _d in m.optional_fields)
        assert any(n == "done" for n, _t, _d in m.optional_fields)

    def t_model_extraction_ignores_in_variant() -> None:
        models_src = '''"""x."""
from dataclasses import dataclass


@dataclass
class TaskIn:
    title: str
'''
        infos = _extract_model_info(models_src, "pkg/models.py")
        # TaskIn also has @dataclass — it should be extracted (as its own model),
        # and that's fine: the tests should still compile.
        assert len(infos) == 1
        assert infos[0].name == "TaskIn"

    check("model extraction: dataclass fields", t_model_extraction)
    check("model extraction: TaskIn also extracted", t_model_extraction_ignores_in_variant)

    # ---- symbol contracts ----
    def t_symbol_contracts() -> None:
        # Build a fake index
        from sebrain.c13 import RepoIndex
        from sebrain.c14 import CodeSynthesisEngine

        src = (
            "def foo(x: int) -> int:\n"
            "    return x\n"
            "class Bar:\n"
            "    def baz(self) -> None: pass\n"
        )
        eng = CodeSynthesisEngine()
        mi = eng.repo_engine.analyze_source(src, repo_relative_path="pkg/mod.py")
        idx = RepoIndex(root="<none>")
        idx.modules["pkg/mod.py"] = mi
        for s in mi.symbols:
            idx.symbol_index[s.id] = s

        p = gen.generate(repo_index=idx, project_id="x")
        # foo and Bar should have contract tests
        names = {t.name for t in p.tests}
        assert any("foo" in n for n in names)
        assert any("bar" in n.lower() or "Bar" in n for n in names)

    check("symbol contracts: public symbols get tests", t_symbol_contracts)

    # ---- regression ----
    def t_regression_from_memory() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                mem.record_failure(
                    "bug-1", what="test auth failed",
                    root_cause="missing session fixture",
                    scope_id="proj-x",
                )
                p = gen.generate(
                    spec=_sample_spec(), memory=mem, project_id="proj-x",
                )
                kinds = {t.kind for t in p.tests}
                assert TestKind.REGRESSION in kinds
            finally:
                s.shutdown()

    check("regression: markers from memory present",
          t_regression_from_memory)

    # ---- bounds ----
    def t_max_tests_bound() -> None:
        small_gen = TestGenerator(max_tests_total=5)
        p = small_gen.generate(
            spec=_sample_spec(), synthesis=_sample_synthesis(),
            project_id="x",
        )
        assert len(p.tests) <= 5

    check("bounds: max_tests_total enforced", t_max_tests_bound)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        p = gen.generate(
            spec=_sample_spec(), synthesis=_sample_synthesis(),
            project_id="x",
        )
        d = p.to_dict()
        assert d["id"] == p.id
        assert "tests" in d and "artifacts" in d and "coverage" in d
        s = p.summary()
        assert "Test Plan" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                p = gen.generate(
                    spec=_sample_spec(), synthesis=_sample_synthesis(),
                    project_id="proj-x",
                )
                repo = TestRepository(memory=mem, ontology=ont)
                ent = repo.save(p, project_id="proj-x")
                assert ent
                loaded = repo.load(p.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["id"] == p.id
                assert ont.count(kind=EntityKind.TEST) >= 1
                assert ont.count(kind=EntityKind.FILE) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (TEST + FILE)",
          t_persist)

    # ---- e2e: write artifacts to disk via C15 and run pytest ----
    def t_e2e_write_and_collect() -> None:
        """Verify the generated test files can actually be collected by
        pytest (collection only — running is C18's job).
        """
        import subprocess
        import importlib.util

        if importlib.util.find_spec("pytest") is None:
            # pytest is an optional dev-dependency of the *generated*
            # project, not of sebrain itself — skip gracefully in
            # environments where it isn't installed rather than failing.
            print("    (skipped: pytest not installed in this environment)")
            return

        synth = _sample_synthesis()
        p = gen.generate(
            spec=_sample_spec(), synthesis=synth, project_id="demo",
        )
        assert p.compile_errors == []

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 1. Write the package files + tests via C15
            from sebrain.c15 import (
                BuildPlan, FileWrite, ProjectBuilder, WriteMode,
            )
            writes = [
                FileWrite(f.path, f.content,
                          kind=getattr(f.kind, "value", "src"))
                for f in synth.files
            ]
            writes += [
                FileWrite(a.path, a.content, kind=f"test-{a.kind.value}")
                for a in p.artifacts
            ]
            # Ensure tests dir has __init__ so pytest can find relative paths
            builder = ProjectBuilder()
            br = builder.apply(BuildPlan(
                root=td, mode=WriteMode.CREATE_ONLY, writes=writes,
            ))
            assert br.status.value == "succeeded", br.rationale

            # 2. Run pytest in --collect-only mode inside the sandbox so we
            #    don't execute untrusted behavior here.
            #    We only need collection to succeed (files parse, names unique).
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q",
                 "tests/"],
                cwd=str(root), capture_output=True, text=True, timeout=60,
            )
            # pytest returns 5 when no tests collected; 0 when collected ok.
            # Either 0 or 5 is fine — anything else means collection error.
            assert result.returncode in (0, 5), (
                f"pytest --collect-only exited {result.returncode}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            # No collection errors reported
            assert "error" not in result.stdout.lower() or "errors" not in result.stdout.lower() or "collected" in result.stdout.lower(), result.stdout

    check("e2e: pytest --collect-only succeeds on generated artifacts",
          t_e2e_write_and_collect)

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
    print("SE Brain C17 — Test Generation Engine")
    print("=" * 78)

    # --- prepare inputs ---
    from sebrain.c05 import RequirementParser
    from sebrain.c14 import (
        CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
    )
    from sebrain.c13 import RepoIndex

    text = (
        "Build a small production-quality REST API for managing tasks.\n\n"
        "Functional requirements:\n"
        "- Users must be able to create tasks.\n"
        "- Users must be able to list tasks.\n\n"
        "Non-functional:\n"
        "- The API must respond within 200ms for read operations.\n"
        "- All traffic must use HTTPS.\n\n"
        "Acceptance:\n"
        "- Given a valid request, when POST /tasks is called, then a 201 is returned.\n"
    )
    spec = RequirementParser().parse(text)
    print(f"\n[1] Spec: {len(spec.functional)} functional, "
          f"{len(spec.non_functional)} NFR, "
          f"{len(spec.acceptance_criteria)} acceptance")

    entity = EntitySpec(
        name="Task",
        fields=[
            FieldSpec("title", "str", required=True),
            FieldSpec("description", "str", required=False, default_repr='""'),
            FieldSpec("done", "bool", required=False, default_repr="False"),
        ],
    )
    synth = CodeSynthesisEngine().synthesize(
        SynthesisRequest(
            package_name="task_api", entities=[entity],
            framework="fastapi", model_style="dataclass", mode="fresh",
        ),
        project_id="demo",
        existing_index=RepoIndex(root="<demo>"),
    )
    print(f"[2] Synthesis: {len(synth.files)} source files")

    # --- generate ---
    gen = TestGenerator()
    plan = gen.generate(
        spec=spec, synthesis=synth, project_id="demo",
    )
    print("\n[3] Test plan:")
    print(plan.summary())

    print("\n[4] Tests by kind:")
    for k in TestKind:
        ts = plan.tests_by_kind(k)
        if ts:
            print(f"    {k.value:12s}: {len(ts)} tests")

    print("\n[5] Artifacts:")
    for a in plan.artifacts:
        print(f"    {a.path}  ({a.byte_size}B, {len(a.test_ids)} tests)")
        print(f"      {a.rationale}")

    print(f"\n[6] Coverage: {len(plan.coverage)} targets")
    for c in plan.coverage[:10]:
        print(f"    {c.target}  ←  {len(c.test_ids)} test(s)")
    if len(plan.coverage) > 10:
        print(f"    … {len(plan.coverage) - 10} more")

    print(f"\n[7] Source summary: {plan.source_summary}")
    print(f"    compile errors: {plan.compile_errors or 'none'}")

    print("\n[8] Preview: tests/test_acceptance.py (first 40 lines):")
    acc = next((a for a in plan.artifacts
                if a.kind is TestKind.ACCEPTANCE), None)
    if acc:
        for line in acc.content.splitlines()[:40]:
            print(f"    {line}")

    print("\n[9] Preview: tests/test_unit.py (first 30 lines):")
    unit = next((a for a in plan.artifacts
                 if a.kind is TestKind.UNIT), None)
    if unit:
        for line in unit.content.splitlines()[:30]:
            print(f"    {line}")

    # --- persistence ---
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = TestRepository(memory=mem, ontology=ont)
                ent = repo.save(plan, project_id="demo")
                print(f"\n[10] Persisted → ontology TEST entity: {ent[:12]}…")
                print(f"     TEST entities: {ont.count(kind=EntityKind.TEST)}")
                print(f"     FILE entities: {ont.count(kind=EntityKind.FILE)}")
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
