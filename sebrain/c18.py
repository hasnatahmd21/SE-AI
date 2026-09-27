"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C18 — TEST EXECUTION ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C13, C16, C17.

Purpose:
    Run pytest suites under the C16 sandbox and collect structured, per-test
    results: identity, outcome, duration, failure + traceback, environment,
    affected component, and regression status. Support selective testing
    (based on changed components from C13) while retaining full-suite
    validation gates for high-risk changes.

Capabilities:
    - Full-suite execution (delegates to C16 sandbox)
    - Selective execution (AST-scans test files' imports to map changed
      source paths → test files)
    - Full-suite validation gate (forced override)
    - Per-test structured results: nodeid, outcome, duration, failure text,
      traceback, component (derived from test file path)
    - Environment capture (python, platform, pytest version, cwd)
    - Regression detection: compare against a previous TestRunResult —
      any test that went PASSED → FAILED/ERROR is a regression
    - Bounded execution (timeout, output limits, sandbox policy)
    - Persistence to C04 memory + C02 ontology (TEST_RESULT entities)

Invariants honored:
    - NO external LLM. Pure deterministic parsing + sandboxed subprocess.
    - Sandbox is REQUIRED — never run pytest in the host process.
    - Every TestResult carries nodeid + outcome + duration + component.
    - Never fabricates results: if pytest fails to produce a report,
      the run fails loudly with the sandbox's captured stdout/stderr.
    - No writes outside the target root (plugin + report cleaned up in
      a finally block).
    - Selective testing falls back to FULL when:
        * any root-level config / conftest.py / test-dir file changed
        * no test file maps to any change
        * a caller sets force_full=True

Explicit limitations:
    - Selective mapping uses import scanning, not true call-graph analysis.
      If a test imports a helper that (transitively) imports the changed
      module, C18 will not see the dependency — callers should pass
      force_full=True when unsure.
    - Regression classification relies on prior results being supplied by
      the caller (typically loaded from C04 memory).
    - Only pytest is supported. Other frameworks are out of scope here.

Contents:
  1.  Enums: TestOutcome, RunStatus, SelectionMode
  2.  Dataclasses: TestResult, EnvironmentInfo, SelectionResult,
                   TestRunResult, TestRunPolicy
  3.  pytest JSON reporter plugin (embedded, written at run time)
  4.  Selection engine (import-based mapping)
  5.  TestExecutionEngine (facade)
  6.  TestRunRepository (persist / reload)
  7.  Self-tests (~26)
  8.  Demo

Run as script:
    python -m sebrain.c18            # demo
    python -m sebrain.c18 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import ast
import json
import platform
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
from sebrain.c16 import Sandbox, SandboxPolicy, ExecutionStatus


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
class TestOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"          # setup/teardown failure
    SKIPPED = "skipped"
    XFAIL = "xfail"
    UNKNOWN = "unknown"


class RunStatus(str, Enum):
    SUCCEEDED = "succeeded"        # pytest exit 0 (all passed, some may skip)
    TESTS_FAILED = "tests_failed"  # exit 1
    INTERRUPTED = "interrupted"    # exit 2
    INTERNAL_ERROR = "internal_error"  # exit 3
    USAGE_ERROR = "usage_error"    # exit 4
    NO_TESTS = "no_tests"          # exit 5
    TIMED_OUT = "timed_out"        # sandbox timeout
    SANDBOX_ERROR = "sandbox_error"  # sandbox could not run pytest
    PARSE_ERROR = "parse_error"    # report json missing / malformed


class SelectionMode(str, Enum):
    FULL = "full"
    SELECTIVE = "selective"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class TestResult:
    nodeid: str
    outcome: TestOutcome
    duration_seconds: float = 0.0
    failure_message: str = ""
    failure_traceback: str = ""
    test_file: str = ""
    component: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodeid": self.nodeid,
            "outcome": self.outcome.value,
            "duration_seconds": self.duration_seconds,
            "failure_message": self.failure_message,
            "failure_traceback": self.failure_traceback,
            "test_file": self.test_file,
            "component": self.component,
        }


@dataclass(slots=True)
class EnvironmentInfo:
    python_version: str = ""
    python_full_version: str = ""
    platform: str = ""
    system: str = ""
    pytest_version: str = ""
    cwd: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "python_full_version": self.python_full_version,
            "platform": self.platform,
            "system": self.system,
            "pytest_version": self.pytest_version,
            "cwd": self.cwd,
        }


@dataclass(slots=True)
class SelectionResult:
    mode: SelectionMode
    selected_files: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    reason: str = ""
    full_suite_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "selected_files": list(self.selected_files),
            "changed_paths": list(self.changed_paths),
            "reason": self.reason,
            "full_suite_required": self.full_suite_required,
        }


@dataclass(slots=True)
class TestRunPolicy:
    timeout_seconds: float = 300.0
    max_output_bytes: int = 1_000_000
    extra_pytest_args: tuple[str, ...] = ()
    test_dir: str = "tests"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValidationError("timeout_seconds must be > 0")
        if self.max_output_bytes < 1:
            raise ValidationError("max_output_bytes must be >= 1")


@dataclass(slots=True)
class TestRunResult:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    root: str = ""
    status: RunStatus = RunStatus.SUCCEEDED
    selection: SelectionResult = field(
        default_factory=lambda: SelectionResult(mode=SelectionMode.FULL)
    )
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)
    results: list[TestResult] = field(default_factory=list)
    regressions: list[TestResult] = field(default_factory=list)
    exit_code: int = 0
    timed_out: bool = False
    duration_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    # ---- aggregates ----
    def count(self, outcome: TestOutcome) -> int:
        return sum(1 for r in self.results if r.outcome is outcome)

    @property
    def passed(self) -> int:
        return self.count(TestOutcome.PASSED)

    @property
    def failed(self) -> int:
        return self.count(TestOutcome.FAILED)

    @property
    def errors(self) -> int:
        return self.count(TestOutcome.ERROR)

    @property
    def skipped(self) -> int:
        return self.count(TestOutcome.SKIPPED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "root": self.root, "status": self.status.value,
            "selection": self.selection.to_dict(),
            "environment": self.environment.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "regressions": [r.to_dict() for r in self.regressions],
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Test Run Result ===\n"
            f"status={self.status.value}  mode={self.selection.mode.value}  "
            f"exit={self.exit_code}\n"
            f"total={len(self.results)}  passed={self.passed}  "
            f"failed={self.failed}  errors={self.errors}  "
            f"skipped={self.skipped}  regressions={len(self.regressions)}\n"
            f"duration={self.duration_seconds:.3f}s  "
            f"timed_out={self.timed_out}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDED PYTEST REPORTER PLUGIN
# ════════════════════════════════════════════════════════════════════════════
_PLUGIN_SOURCE = '''\
"""C18 pytest reporter plugin — writes a JSON report on session finish."""
from __future__ import annotations

import json
import os

import pytest  # noqa: F401

_BY_TEST: dict = {}


def pytest_runtest_logreport(report):
    nid = report.nodeid
    rec = _BY_TEST.setdefault(nid, {
        "nodeid": nid,
        "outcome": "passed",
        "duration": 0.0,
        "failure": None,
    })
    try:
        rec["duration"] += float(report.duration or 0.0)
    except (TypeError, ValueError):
        pass

    outcome = report.outcome
    if report.when == "setup" and report.outcome == "failed":
        outcome = "error"

    priority = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
    if priority.get(outcome, 0) > priority.get(rec["outcome"], 0):
        rec["outcome"] = outcome

    if report.failed and rec["failure"] is None:
        try:
            rec["failure"] = str(report.longrepr)
        except Exception:
            rec["failure"] = "<unreprable>"


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("C18_REPORT_PATH", ".c18_report.json")
    payload = {
        "results": list(_BY_TEST.values()),
        "exitstatus": int(exitstatus),
        "pytest_version": getattr(pytest, "__version__", "unknown"),
    }
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except OSError:
        pass
'''


# ════════════════════════════════════════════════════════════════════════════
# 4. SELECTION ENGINE
# ════════════════════════════════════════════════════════════════════════════
_FULL_SUITE_TRIGGERS = frozenset({
    "pyproject.toml", "setup.py", "setup.cfg", "pytest.ini",
    "tox.ini", "conftest.py", ".flake8", "mypy.ini", "ruff.toml",
})


def _path_to_module(rel: str) -> str | None:
    p = rel.replace("\\", "/")
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    elif p.endswith("__init__.py"):
        return None
    elif p.endswith(".py"):
        p = p[:-3]
    else:
        return None
    return p.replace("/", ".") if p else None


def _extract_imports_from_test(test_path: Path) -> set[str]:
    """Return the set of top-level module names imported by a test file.

    Only collects `import X` and `from X import ...` (absolute imports).
    Relative imports are ignored — test files rarely use them, and if they
    do, the caller should treat the change as high-risk and force full.
    """
    try:
        src = test_path.read_text(encoding="utf-8")
    except OSError:
        return set()
    try:
        tree = ast.parse(src, filename=str(test_path))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.add(node.module)
    return out


def _imports_match(imports: set[str], target_module: str) -> bool:
    """True iff target_module, a strict ancestor of it, OR a strict
    descendant of it is imported.

    Ancestor check: changing "pkg.math" matches a test that does
    "import pkg" (imports the whole package).
    Descendant check: changing "pkg" (i.e. pkg/__init__.py) matches a
    test that does "from pkg.math import ..." — editing a package's
    __init__ can affect any of its submodules' consumers too, so the
    match has to work in both directions, not just ancestor→target.
    """
    if not target_module:
        return False
    parts = target_module.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in imports:
            return True
    prefix_with_dot = target_module + "."
    return any(imp == target_module or imp.startswith(prefix_with_dot)
               for imp in imports)


class SelectionEngine:
    """Deterministic changed-paths → test-files mapper (import-based)."""

    def select(
        self, *, root: Path, changed_paths: Sequence[str],
        test_dir: str = "tests",
    ) -> SelectionResult:
        if not changed_paths:
            return SelectionResult(
                mode=SelectionMode.FULL,
                selected_files=[],
                changed_paths=[],
                reason="no changed paths supplied; running full suite",
                full_suite_required=True,
            )
        # Normalize
        norm = [p.replace("\\", "/").lstrip("./") for p in changed_paths]

        # Rule 1: high-risk changes → full suite
        triggers: list[str] = []
        for p in norm:
            base = p.rsplit("/", 1)[-1]
            if base in _FULL_SUITE_TRIGGERS:
                triggers.append(p)
            elif p.startswith(f"{test_dir}/") and base.startswith("conftest"):
                triggers.append(p)
        if triggers:
            return SelectionResult(
                mode=SelectionMode.FULL,
                selected_files=[],
                changed_paths=norm,
                reason=(
                    "full suite required; high-risk change(s): "
                    + ", ".join(triggers[:5])
                    + (" …" if len(triggers) > 5 else "")
                ),
                full_suite_required=True,
            )

        # Rule 2: directly changed tests run, plus tests importing changed modules
        selected: set[str] = set()
        changed_sources: list[str] = []
        for p in norm:
            if p.startswith(f"{test_dir}/") and p.endswith(".py"):
                selected.add(p)
            else:
                changed_sources.append(p)

        test_files: list[Path] = []
        test_root = root / test_dir
        if test_root.is_dir():
            test_files = sorted(test_root.rglob("test_*.py"))

        if changed_sources and test_files:
            for tf in test_files:
                rel_tf = str(tf.relative_to(root)).replace("\\", "/")
                imports = _extract_imports_from_test(tf)
                if not imports:
                    continue
                for src in changed_sources:
                    mod = _path_to_module(src)
                    if mod and _imports_match(imports, mod):
                        selected.add(rel_tf)
                        break

        if not selected:
            return SelectionResult(
                mode=SelectionMode.FULL,
                selected_files=[],
                changed_paths=norm,
                reason=(
                    "no test files mapped to changes via import scanning; "
                    "running full suite"
                ),
                full_suite_required=True,
            )

        return SelectionResult(
            mode=SelectionMode.SELECTIVE,
            selected_files=sorted(selected),
            changed_paths=norm,
            reason=(
                f"selective: {len(selected)} test file(s) mapped to "
                f"{len(changed_sources)} source change(s)"
            ),
            full_suite_required=False,
        )


# ════════════════════════════════════════════════════════════════════════════
# 5. TEST EXECUTION ENGINE
# ════════════════════════════════════════════════════════════════════════════
_STATUS_FROM_EXIT: dict[int, RunStatus] = {
    0: RunStatus.SUCCEEDED,
    1: RunStatus.TESTS_FAILED,
    2: RunStatus.INTERRUPTED,
    3: RunStatus.INTERNAL_ERROR,
    4: RunStatus.USAGE_ERROR,
    5: RunStatus.NO_TESTS,
}


class TestExecutionEngine:
    """Facade: run / run_full / run_selective / select."""

    PLUGIN_NAME = "_c18_reporter.py"
    REPORT_NAME = ".c18_report.json"

    def __init__(self, *, sandbox: Sandbox | None = None) -> None:
        self.sandbox = sandbox or Sandbox()
        self.selector = SelectionEngine()

    # ---- public API ----
    def select(
        self, *, root: str | Path, changed_paths: Sequence[str],
        test_dir: str = "tests",
    ) -> SelectionResult:
        r = Path(root).resolve()
        return self.selector.select(
            root=r, changed_paths=changed_paths, test_dir=test_dir,
        )

    def run_full(
        self, *, root: str | Path, project_id: str = "",
        policy: TestRunPolicy | None = None,
        previous: TestRunResult | None = None,
    ) -> TestRunResult:
        return self._execute(
            root=Path(root).resolve(),
            project_id=project_id,
            policy=policy or TestRunPolicy(),
            selection=SelectionResult(
                mode=SelectionMode.FULL,
                changed_paths=[],
                reason="full suite requested",
                full_suite_required=True,
            ),
            previous=previous,
        )

    def run_selective(
        self, *, root: str | Path, changed_paths: Sequence[str],
        project_id: str = "",
        policy: TestRunPolicy | None = None,
        previous: TestRunResult | None = None,
        force_full: bool = False,
    ) -> TestRunResult:
        p = policy or TestRunPolicy()
        r = Path(root).resolve()
        if force_full:
            sel = SelectionResult(
                mode=SelectionMode.FULL,
                selected_files=[],
                changed_paths=[c.replace("\\", "/") for c in changed_paths],
                reason="force_full=True",
                full_suite_required=True,
            )
        else:
            sel = self.selector.select(
                root=r, changed_paths=changed_paths, test_dir=p.test_dir,
            )
        return self._execute(
            root=r, project_id=project_id, policy=p,
            selection=sel, previous=previous,
        )

    def run_gate(
        self, *, root: str | Path, project_id: str = "",
        policy: TestRunPolicy | None = None,
        previous: TestRunResult | None = None,
    ) -> TestRunResult:
        """Full-suite validation gate. Always runs everything."""
        return self.run_full(
            root=root, project_id=project_id,
            policy=policy, previous=previous,
        )

    # ---- core executor ----
    def _execute(
        self, *, root: Path, project_id: str, policy: TestRunPolicy,
        selection: SelectionResult, previous: TestRunResult | None,
    ) -> TestRunResult:
        if not root.exists() or not root.is_dir():
            raise ValidationError(f"root not found or not a directory: {root}")

        env = EnvironmentInfo(
            python_version=sys.version.split()[0],
            python_full_version=sys.version.replace("\n", " "),
            platform=platform.platform(),
            system=platform.system(),
            pytest_version="unknown",
            cwd=str(root),
        )

        result = TestRunResult(
            project_id=project_id, root=str(root),
            selection=selection, environment=env,
            provenance=Provenance(
                source="test_execution_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )

        plugin_path = root / self.PLUGIN_NAME
        report_path = root / self.REPORT_NAME
        wrote_plugin = False
        try:
            plugin_path.write_text(_PLUGIN_SOURCE, encoding="utf-8")
            wrote_plugin = True

            # Remove any stale report
            try:
                report_path.unlink()
            except OSError:
                pass

            cmd = [
                sys.executable, "-m", "pytest",
                "-p", "no:cacheprovider",
                "-p", "_c18_reporter",
                "--tb=long",
                "-v",
                "-rf",
                "-ra",
                *policy.extra_pytest_args,
            ]
            if selection.mode is SelectionMode.SELECTIVE:
                cmd.extend(selection.selected_files)
            else:
                test_dir = policy.test_dir
                if (root / test_dir).is_dir():
                    cmd.append(f"{test_dir}/")
                # else: no explicit paths → pytest discovers from cwd

            sandbox_policy = SandboxPolicy(
                timeout_seconds=policy.timeout_seconds,
                max_output_bytes=policy.max_output_bytes,
                env_additions=(
                    ("PYTHONPATH", str(root)),
                    ("PYTHONDONTWRITEBYTECODE", "1"),
                    ("C18_REPORT_PATH", str(report_path)),
                ),
                allow_shell=False,
            )

            sb_result = self.sandbox.run(
                cmd, policy=sandbox_policy, workdir=str(root),
            )
            result.exit_code = sb_result.exit_code
            result.timed_out = sb_result.timed_out
            result.duration_seconds = sb_result.duration_seconds
            result.stdout_tail = sb_result.stdout[-4000:]
            result.stderr_tail = sb_result.stderr[-4000:]

            if sb_result.status is ExecutionStatus.TIMED_OUT:
                result.status = RunStatus.TIMED_OUT
                result.rationale = (
                    f"pytest exceeded sandbox timeout "
                    f"({policy.timeout_seconds}s)"
                )
                return result

            if sb_result.status is ExecutionStatus.REJECTED:
                result.status = RunStatus.SANDBOX_ERROR
                result.rationale = f"sandbox rejected: {sb_result.rationale}"
                return result

            if not report_path.exists():
                result.status = RunStatus.PARSE_ERROR
                result.rationale = (
                    "pytest did not write a report — likely import error "
                    "or missing pytest. See stdout/stderr tails."
                )
                return result

            try:
                payload = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result.status = RunStatus.PARSE_ERROR
                result.rationale = f"could not parse report: {exc}"
                return result

            env.pytest_version = str(payload.get("pytest_version", "unknown"))
            result.results = [
                _row_to_test_result(r) for r in payload.get("results", [])
            ]
            result.results.sort(key=lambda r: r.nodeid)

            # Classify by pytest exit code
            exit_code = int(payload.get("exitstatus", sb_result.exit_code))
            result.exit_code = exit_code
            result.status = _STATUS_FROM_EXIT.get(
                exit_code, RunStatus.INTERNAL_ERROR,
            )

            # Regression diff
            if previous is not None:
                result.regressions = _diff_regressions(
                    previous=previous, current=result,
                )

            result.rationale = (
                f"pytest exit={exit_code}  mode={selection.mode.value}  "
                f"selected={len(selection.selected_files)}  "
                f"total={len(result.results)}  "
                f"failed={result.failed}  errors={result.errors}  "
                f"regressions={len(result.regressions)}"
            )
            return result

        finally:
            # Cleanup: always remove the plugin + report we created
            try:
                if wrote_plugin:
                    plugin_path.unlink()
            except OSError:
                pass
            try:
                report_path.unlink()
            except OSError:
                pass
            # Also drop any __pycache__ we may have created
            try:
                pyc = root / "__pycache__"
                if pyc.is_dir():
                    for f in pyc.glob("_c18_reporter*"):
                        f.unlink()
            except OSError:
                pass


# ---- row → TestResult ----
_TEST_FILE_RE = re.compile(r"^(?P<file>.+?)::")


def _row_to_test_result(row: dict[str, Any]) -> TestResult:
    nid = str(row.get("nodeid", ""))
    outcome_raw = str(row.get("outcome", "passed"))
    try:
        outcome = TestOutcome(outcome_raw)
    except ValueError:
        outcome = TestOutcome.UNKNOWN

    duration = float(row.get("duration", 0.0) or 0.0)
    failure = row.get("failure") or ""
    failure_msg, failure_tb = _split_failure(str(failure))

    m = _TEST_FILE_RE.match(nid)
    test_file = m.group("file") if m else ""
    component = _component_from_test_file(test_file)

    return TestResult(
        nodeid=nid, outcome=outcome, duration_seconds=duration,
        failure_message=failure_msg, failure_traceback=failure_tb,
        test_file=test_file, component=component,
    )


def _split_failure(text: str) -> tuple[str, str]:
    """Split a pytest longrepr into (message, traceback).

    Heuristic: the last non-empty line before the pytest footer is the
    message (e.g. `AssertionError: assert 1 == 2`). Everything before is
    the traceback.
    """
    if not text:
        return ("", "")
    lines = text.rstrip().splitlines()
    # The message is usually the last line starting with a capital letter
    # and containing 'Error' or 'assert' or ':'.
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith("=") or line.startswith("-"):
            continue
        if line.startswith("E "):
            line = line[2:].strip()
        msg = line
        tb = "\n".join(lines[:i])
        return (msg, tb)
    return (lines[-1].strip(), "")


def _component_from_test_file(test_file: str) -> str:
    if not test_file:
        return ""
    # tests/test_unit.py → unit
    # tests/subdir/test_foo.py → subdir/foo
    p = test_file.replace("\\", "/")
    if p.startswith("tests/"):
        p = p[len("tests/"):]
    base = p.rsplit("/", 1)[-1]
    if base.startswith("test_"):
        base = base[len("test_"):]
    if base.endswith(".py"):
        base = base[:-3]
    prefix = p.rsplit("/", 1)[0] if "/" in p else ""
    return f"{prefix}/{base}" if prefix else base


def _diff_regressions(
    *, previous: TestRunResult, current: TestRunResult,
) -> list[TestResult]:
    prev_map = {r.nodeid: r for r in previous.results}
    out: list[TestResult] = []
    for cur in current.results:
        prev = prev_map.get(cur.nodeid)
        if prev is None:
            continue
        prev_ok = prev.outcome in (TestOutcome.PASSED, TestOutcome.SKIPPED,
                                   TestOutcome.XFAIL)
        cur_bad = cur.outcome in (TestOutcome.FAILED, TestOutcome.ERROR)
        if prev_ok and cur_bad:
            out.append(cur)
    out.sort(key=lambda r: r.nodeid)
    return out


# ════════════════════════════════════════════════════════════════════════════
# 6. TEST RUN REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class TestRunRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, run: TestRunResult, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"test_run:{run.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, run.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["test_run", "c18", run.status.value],
            provenance=run.provenance,
        )
        # Record failures for C33
        for r in run.results:
            if r.outcome in (TestOutcome.FAILED, TestOutcome.ERROR):
                self.memory.record_failure(
                    f"test_fail:{run.id}:{r.nodeid}",
                    what=f"test {r.nodeid} {r.outcome.value}",
                    root_cause=r.failure_message or "unknown",
                    fix=None,
                    scope_id=project_id,
                    provenance=Provenance(
                        source="test_execution_engine",
                        source_type=ProvenanceType.SYSTEM,
                        confidence=Confidence.HIGH,
                    ),
                    confidence=Confidence.HIGH,
                )
        if self.ontology is None:
            return key

        ent = self.ontology.add(
            EntityKind.EXECUTION,
            _short(
                f"TestRun {run.id[:8]} ({run.status.value}, "
                f"{len(run.results)} tests)", 120,
            ),
            attributes={
                "test_run_id": run.id,
                "project_id": project_id,
                "status": run.status.value,
                "mode": run.selection.mode.value,
                "exit_code": run.exit_code,
                "total": len(run.results),
                "passed": run.passed,
                "failed": run.failed,
                "errors": run.errors,
                "skipped": run.skipped,
                "regressions": len(run.regressions),
                "timed_out": run.timed_out,
                "duration_seconds": run.duration_seconds,
            },
            tags=["test-run", run.status.value],
            provenance=run.provenance,
        )
        # Record regression TEST_RESULT entities
        for r in run.regressions:
            tre = self.ontology.add(
                EntityKind.TEST_RESULT,
                _short(r.nodeid, 120),
                attributes={
                    "outcome": r.outcome.value,
                    "regression": True,
                    "component": r.component,
                    "duration_seconds": r.duration_seconds,
                },
                tags=["test-result", "regression"],
                provenance=run.provenance,
            )
            try:
                self.ontology.link(RelationKind.PRODUCES, ent.id, tre.id)
            except ValidationError:
                pass
        return ent.id

    def load(self, run_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"test_run:{run_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None

    def load_latest(self, *, project_id: str) -> TestRunResult | None:
        """Load the most recent test_run for a project (best-effort)."""
        entries = self.memory.find(
            kind=MemoryKind.PROJECT,
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            key_like="test_run:",
        )
        if not entries:
            return None
        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return _run_from_dict(entries[0].content)


def _run_from_dict(d: dict[str, Any]) -> TestRunResult:
    """Reconstruct a TestRunResult from memory JSON (best-effort)."""
    run = TestRunResult(
        id=d.get("id", _new_id()),
        project_id=d.get("project_id", ""),
        root=d.get("root", ""),
        status=RunStatus(d.get("status", RunStatus.INTERNAL_ERROR.value)),
        environment=EnvironmentInfo(**(d.get("environment") or {})),
        exit_code=int(d.get("exit_code", 0)),
        timed_out=bool(d.get("timed_out", False)),
        duration_seconds=float(d.get("duration_seconds", 0.0)),
        rationale=d.get("rationale", ""),
        created_at=d.get("created_at", now_iso()),
    )
    sel = d.get("selection") or {}
    try:
        run.selection = SelectionResult(
            mode=SelectionMode(sel.get("mode", "full")),
            selected_files=list(sel.get("selected_files", [])),
            changed_paths=list(sel.get("changed_paths", [])),
            reason=sel.get("reason", ""),
            full_suite_required=bool(sel.get("full_suite_required", True)),
        )
    except ValueError:
        pass
    for r in d.get("results", []):
        try:
            outcome = TestOutcome(r.get("outcome", "unknown"))
        except ValueError:
            outcome = TestOutcome.UNKNOWN
        run.results.append(TestResult(
            nodeid=r.get("nodeid", ""), outcome=outcome,
            duration_seconds=float(r.get("duration_seconds", 0.0)),
            failure_message=r.get("failure_message", ""),
            failure_traceback=r.get("failure_traceback", ""),
            test_file=r.get("test_file", ""), component=r.get("component", ""),
        ))
    for r in d.get("regressions", []):
        try:
            outcome = TestOutcome(r.get("outcome", "unknown"))
        except ValueError:
            outcome = TestOutcome.UNKNOWN
        run.regressions.append(TestResult(
            nodeid=r.get("nodeid", ""), outcome=outcome,
            duration_seconds=float(r.get("duration_seconds", 0.0)),
            failure_message=r.get("failure_message", ""),
            test_file=r.get("test_file", ""), component=r.get("component", ""),
        ))
    return run


# ════════════════════════════════════════════════════════════════════════════
# 7. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _write(d: Path, rel: str, content: str) -> Path:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _mk_small_repo(root: Path, *, make_fail: bool = False,
                   make_error: bool = False) -> None:
    """Create a tiny project with pytest tests, so C18 has something to run."""
    _write(root, "pkg/__init__.py", '"""pkg."""\n')
    _write(root, "pkg/math.py", (
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "\n"
        "def sub(a: int, b: int) -> int:\n"
        "    return a - b\n"
    ))
    _write(root, "tests/__init__.py", "")
    # Passing tests
    _write(root, "tests/test_math.py", (
        "from pkg.math import add, sub\n"
        "\n"
        "def test_add() -> None:\n"
        "    assert add(1, 2) == 3\n"
        "\n"
        "def test_sub() -> None:\n"
        "    assert sub(5, 3) == 2\n"
    ))
    if make_fail:
        _write(root, "tests/test_fail.py", (
            "from pkg.math import add\n"
            "\n"
            "def test_fail_on_purpose() -> None:\n"
            "    assert add(1, 2) == 999\n"
        ))
    if make_error:
        _write(root, "tests/test_error.py", (
            "import pytest\n"
            "\n"
            "@pytest.fixture\n"
            "def boom():\n"
            "    raise RuntimeError('setup boom')\n"
            "\n"
            "def test_setup_error(boom) -> None:\n"
            "    assert True\n"
        ))


def _run_self_tests() -> int:
    import importlib.util
    has_pytest = importlib.util.find_spec("pytest") is not None

    failures: list[str] = []
    passed = 0
    skipped = 0

    def check(name: str, fn: Callable[[], None], *, requires_pytest: bool = False) -> None:
        nonlocal passed, skipped
        if requires_pytest and not has_pytest:
            # This module's core job is invoking real pytest as a
            # subprocess (it writes its own reporter plugin and relies on
            # pytest's actual hook API) — there's no meaningful way to
            # exercise that without pytest installed. Skip rather than
            # report a false failure; these tests run for real wherever
            # pytest is available (it's listed in requirements.txt).
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

    print("Running C18 self-tests…")
    engine = TestExecutionEngine()

    # ---- helpers ----
    def _tmp_repo(**kw):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        _mk_small_repo(root, **kw)
        return td, root

    # ---- selection ----
    def t_select_no_changes_falls_back_full() -> None:
        td, root = _tmp_repo()
        try:
            sel = engine.select(root=root, changed_paths=[])
            assert sel.mode is SelectionMode.FULL
            assert sel.full_suite_required is True
        finally:
            td.cleanup()

    def t_select_config_change_forces_full() -> None:
        td, root = _tmp_repo()
        try:
            sel = engine.select(root=root, changed_paths=["pyproject.toml"])
            assert sel.mode is SelectionMode.FULL
            assert sel.full_suite_required is True
            assert "high-risk" in sel.reason or "full suite" in sel.reason
        finally:
            td.cleanup()

    def t_select_conftest_forces_full() -> None:
        td, root = _tmp_repo()
        try:
            sel = engine.select(root=root, changed_paths=["conftest.py"])
            assert sel.full_suite_required is True
        finally:
            td.cleanup()

    def t_select_source_maps_to_test() -> None:
        td, root = _tmp_repo()
        try:
            sel = engine.select(
                root=root, changed_paths=["pkg/math.py"],
            )
            assert sel.mode is SelectionMode.SELECTIVE
            assert "tests/test_math.py" in sel.selected_files
        finally:
            td.cleanup()

    def t_select_changed_test_runs_it() -> None:
        td, root = _tmp_repo()
        try:
            sel = engine.select(
                root=root, changed_paths=["tests/test_math.py"],
            )
            assert sel.mode is SelectionMode.SELECTIVE
            assert sel.selected_files == ["tests/test_math.py"]
        finally:
            td.cleanup()

    def t_select_unmapped_falls_back_full() -> None:
        td, root = _tmp_repo()
        try:
            # A source module no test imports
            _write(root, "pkg/unused.py", "X = 1\n")
            sel = engine.select(root=root, changed_paths=["pkg/unused.py"])
            assert sel.mode is SelectionMode.FULL
            assert sel.full_suite_required is True
        finally:
            td.cleanup()

    def t_select_init_change_maps_to_package_tests() -> None:
        td, root = _tmp_repo()
        try:
            # test_math.py imports from pkg.math, so `pkg/__init__.py`
            # change (module `pkg`) should match via prefix
            sel = engine.select(root=root, changed_paths=["pkg/__init__.py"])
            # pkg/__init__.py → module "pkg" → test imports "pkg.math" prefix
            # contains "pkg" so it should match
            assert sel.mode is SelectionMode.SELECTIVE
            assert "tests/test_math.py" in sel.selected_files
        finally:
            td.cleanup()

    def t_select_backslash_normalized() -> None:
        td, root = _tmp_repo()
        try:
            sel = engine.select(
                root=root, changed_paths=["pkg\\math.py"],
            )
            assert sel.mode is SelectionMode.SELECTIVE
            assert "tests/test_math.py" in sel.selected_files
        finally:
            td.cleanup()

    check("select: no changes → full", t_select_no_changes_falls_back_full)
    check("select: pyproject.toml → full", t_select_config_change_forces_full)
    check("select: conftest.py → full", t_select_conftest_forces_full)
    check("select: source maps to importing test", t_select_source_maps_to_test)
    check("select: changed test file runs that file",
          t_select_changed_test_runs_it)
    check("select: unmapped source → full",
          t_select_unmapped_falls_back_full)
    check("select: pkg/__init__.py → package tests",
          t_select_init_change_maps_to_package_tests)
    check("select: backslashes normalized",
          t_select_backslash_normalized)

    # ---- full-suite run (happy path) ----
    def t_run_full_all_pass() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_full(root=root, project_id="p")
            assert res.status is RunStatus.SUCCEEDED, res.rationale
            assert res.exit_code == 0
            assert res.passed == 2
            assert res.failed == 0
            assert res.errors == 0
            nodeids = {r.nodeid for r in res.results}
            assert any("test_math.py::test_add" in n for n in nodeids)
            assert any("test_math.py::test_sub" in n for n in nodeids)
        finally:
            td.cleanup()

    def t_run_full_detects_failures() -> None:
        td, root = _tmp_repo(make_fail=True)
        try:
            res = engine.run_full(root=root, project_id="p")
            assert res.status is RunStatus.TESTS_FAILED
            assert res.exit_code == 1
            assert res.failed >= 1
            # Failure message captured
            failed = next(
                r for r in res.results if r.outcome is TestOutcome.FAILED
            )
            assert "test_fail_on_purpose" in failed.nodeid
            assert failed.failure_message
            assert "999" in failed.failure_message or "assert" in failed.failure_message.lower()
        finally:
            td.cleanup()

    def t_run_full_detects_setup_errors() -> None:
        td, root = _tmp_repo(make_error=True)
        try:
            res = engine.run_full(root=root, project_id="p")
            assert res.errors >= 1
            er = next(
                r for r in res.results if r.outcome is TestOutcome.ERROR
            )
            assert "test_setup_error" in er.nodeid
        finally:
            td.cleanup()

    check("run_full: all pass → SUCCEEDED", t_run_full_all_pass, requires_pytest=True)
    check("run_full: failure detected + traceback", t_run_full_detects_failures, requires_pytest=True)
    check("run_full: setup error classified as ERROR",
          t_run_full_detects_setup_errors, requires_pytest=True)

    # ---- environment ----
    def t_environment_captured() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_full(root=root)
            assert res.environment.python_version
            assert res.environment.platform
            assert res.environment.system
            assert res.environment.pytest_version != "unknown"
            assert res.environment.cwd == str(root.resolve())
        finally:
            td.cleanup()

    check("environment: python/platform/pytest captured",
          t_environment_captured, requires_pytest=True)

    # ---- component derivation ----
    def t_component_derived() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_full(root=root)
            components = {r.component for r in res.results}
            # tests/test_math.py → "math"
            assert "math" in components, components
        finally:
            td.cleanup()

    check("component: derived from test file", t_component_derived, requires_pytest=True)

    # ---- selective execution ----
    def t_run_selective_only_selected() -> None:
        td, root = _tmp_repo(make_fail=True)
        try:
            # change pkg/math.py → only tests/test_math.py selected
            res = engine.run_selective(
                root=root, changed_paths=["pkg/math.py"], project_id="p",
            )
            assert res.selection.mode is SelectionMode.SELECTIVE
            assert res.selection.selected_files == [
                "tests/test_fail.py", "tests/test_math.py"
            ]
            # Both tests import the changed module, so both are correctly selected.
            nodeids = {r.nodeid for r in res.results}
            assert any("test_math.py::test_add" in n for n in nodeids), nodeids
            assert any("test_fail.py::test_fail_on_purpose" in n for n in nodeids), nodeids
            assert res.passed == 2
            assert res.failed == 1
        finally:
            td.cleanup()

    def t_run_selective_force_full() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_selective(
                root=root, changed_paths=["pkg/math.py"],
                project_id="p", force_full=True,
            )
            assert res.selection.mode is SelectionMode.FULL
            assert res.selection.full_suite_required is True
        finally:
            td.cleanup()

    def t_run_selective_config_falls_back_full() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_selective(
                root=root, changed_paths=["pyproject.toml"], project_id="p",
            )
            assert res.selection.mode is SelectionMode.FULL
            assert res.selection.full_suite_required is True
        finally:
            td.cleanup()

    check("run_selective: only selected tests run",
          t_run_selective_only_selected, requires_pytest=True)
    check("run_selective: force_full overrides selection",
          t_run_selective_force_full)
    check("run_selective: config change falls back to full",
          t_run_selective_config_falls_back_full)

    # ---- regression detection ----
    def t_regression_detected() -> None:
        td, root = _tmp_repo()
        try:
            # Run 1 — all pass
            prev = engine.run_full(root=root, project_id="p")
            assert prev.status is RunStatus.SUCCEEDED
            # Break one test
            _write(root, "tests/test_math.py", (
                "from pkg.math import add, sub\n"
                "\n"
                "def test_add() -> None:\n"
                "    assert add(1, 2) == 3\n"
                "\n"
                "def test_sub() -> None:\n"
                "    assert sub(5, 3) == 999   # intentionally broken\n"
            ))
            cur = engine.run_full(root=root, project_id="p",
                                   previous=prev)
            assert cur.status is RunStatus.TESTS_FAILED
            assert len(cur.regressions) == 1
            assert "test_sub" in cur.regressions[0].nodeid
        finally:
            td.cleanup()

    def t_no_false_regression() -> None:
        td, root = _tmp_repo()
        try:
            prev = engine.run_full(root=root, project_id="p")
            cur = engine.run_full(root=root, project_id="p", previous=prev)
            assert cur.status is RunStatus.SUCCEEDED
            assert cur.regressions == []
        finally:
            td.cleanup()

    def t_regression_does_not_fire_for_new_test() -> None:
        td, root = _tmp_repo()
        try:
            prev = engine.run_full(root=root, project_id="p")
            _write(root, "tests/test_new.py", (
                "def test_brand_new() -> None:\n"
                "    assert False\n"
            ))
            cur = engine.run_full(root=root, project_id="p", previous=prev)
            # New failing test ≠ regression (no prior PASSED state)
            assert all("test_brand_new" not in r.nodeid
                       for r in cur.regressions)
        finally:
            td.cleanup()

    check("regression: PASSED → FAILED flagged", t_regression_detected, requires_pytest=True)
    check("regression: no false positives on re-run",
          t_no_false_regression, requires_pytest=True)
    check("regression: new failing test ≠ regression",
          t_regression_does_not_fire_for_new_test)

    # ---- run_gate (full-suite validation) ----
    def t_run_gate() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_gate(root=root, project_id="p")
            assert res.selection.mode is SelectionMode.FULL
            assert res.status is RunStatus.SUCCEEDED
        finally:
            td.cleanup()

    check("run_gate: always full-suite", t_run_gate, requires_pytest=True)

    # ---- cleanup ----
    def t_cleanup_plugin_and_report() -> None:
        td, root = _tmp_repo()
        try:
            engine.run_full(root=root)
            # Plugin + report must be removed after run
            assert not (root / engine.PLUGIN_NAME).exists()
            assert not (root / engine.REPORT_NAME).exists()
        finally:
            td.cleanup()

    check("cleanup: plugin + report removed",
          t_cleanup_plugin_and_report)

    # ---- to_dict / summary ----
    def t_to_dict_summary() -> None:
        td, root = _tmp_repo()
        try:
            res = engine.run_full(root=root)
            d = res.to_dict()
            assert d["status"] == "succeeded"
            assert isinstance(d["results"], list)
            assert d["environment"]["python_version"]
            s = res.summary()
            assert "Test Run Result" in s
        finally:
            td.cleanup()

    check("to_dict + summary", t_to_dict_summary, requires_pytest=True)

    # ---- persistence ----
    def t_persist_and_reload() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                # Create a tiny project in another temp dir
                with tempfile.TemporaryDirectory() as ptd:
                    root = Path(ptd)
                    _mk_small_repo(root)
                    res = engine.run_full(root=root, project_id="proj-x")
                    repo = TestRunRepository(memory=mem, ontology=ont)
                    ent = repo.save(res, project_id="proj-x")
                    assert ent
                    loaded = repo.load(res.id, project_id="proj-x")
                    assert loaded is not None
                    assert loaded["status"] == "succeeded"
                    # Ontology has an EXECUTION entity
                    assert ont.count(kind=EntityKind.EXECUTION) >= 1
            finally:
                s.shutdown()

    def t_persist_load_latest() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                repo = TestRunRepository(memory=mem, ontology=ont)
                with tempfile.TemporaryDirectory() as ptd:
                    root = Path(ptd)
                    _mk_small_repo(root, make_fail=True)
                    run1 = engine.run_full(root=root, project_id="proj-y")
                    repo.save(run1, project_id="proj-y")
                    latest = repo.load_latest(project_id="proj-y")
                    assert latest is not None
                    assert latest.id == run1.id
                    assert latest.status is RunStatus.TESTS_FAILED
            finally:
                s.shutdown()

    def t_persist_regression_creates_test_result_entity() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                repo = TestRunRepository(memory=mem, ontology=ont)
                with tempfile.TemporaryDirectory() as ptd:
                    root = Path(ptd)
                    _mk_small_repo(root)
                    prev = engine.run_full(root=root, project_id="z")
                    repo.save(prev, project_id="z")
                    # Break a test
                    _write(root, "tests/test_math.py", (
                        "from pkg.math import add\n"
                        "\n"
                        "def test_add() -> None:\n"
                        "    assert add(1, 2) == 3\n"
                        "\n"
                        "def test_sub() -> None:\n"
                        "    assert False\n"
                    ))
                    cur = engine.run_full(
                        root=root, project_id="z", previous=prev,
                    )
                    repo.save(cur, project_id="z")
                    # Regression entity present
                    assert ont.count(kind=EntityKind.TEST_RESULT) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology EXECUTION",
          t_persist_and_reload, requires_pytest=True)
    check("persist: load_latest returns most recent",
          t_persist_load_latest, requires_pytest=True)
    check("persist: regressions create TEST_RESULT entities",
          t_persist_regression_creates_test_result_entity, requires_pytest=True)

    # ---- timeout ----
    def t_timeout() -> None:
        td, root = _tmp_repo()
        try:
            _write(root, "tests/test_slow.py", (
                "import time\n"
                "\n"
                "def test_very_slow() -> None:\n"
                "    time.sleep(60)\n"
            ))
            res = engine.run_full(
                root=root, policy=TestRunPolicy(timeout_seconds=1.5),
            )
            assert res.status is RunStatus.TIMED_OUT
            assert res.timed_out is True
        finally:
            td.cleanup()

    check("timeout: sandbox timeout → TIMED_OUT", t_timeout, requires_pytest=True)

    # ---- no tests ----
    def t_no_tests() -> None:
        td = tempfile.TemporaryDirectory()
        try:
            root = Path(td.name)
            _write(root, "empty/__init__.py", "")
            res = engine.run_full(root=root)
            # pytest with no tests → exit code 5
            assert res.status is RunStatus.NO_TESTS, res.rationale
        finally:
            td.cleanup()

    check("no_tests: exit 5 → NO_TESTS", t_no_tests, requires_pytest=True)

    # ---- e2e with C14 + C15 + C17 ----
    def t_e2e_full_pipeline() -> None:
        from sebrain.c05 import RequirementParser
        from sebrain.c13 import RepoIndex
        from sebrain.c14 import (
            CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
        )
        from sebrain.c15 import (
            BuildPlan, FileWrite, ProjectBuilder, WriteMode,
        )
        from sebrain.c17 import TestGenerator

        text = (
            "Build a small REST API for tasks.\n"
            "Users must be able to create, read, and update tasks.\n"
            "Non-functional:\n- All traffic must use HTTPS.\n"
            "Acceptance:\n- Given a valid request, when POST /tasks is called, then a 201 is returned.\n"
        )
        spec = RequirementParser().parse(text)

        entity = EntitySpec(
            name="Task",
            fields=[
                FieldSpec("title", "str", required=True),
                FieldSpec("done", "bool", required=False, default_repr="False"),
            ],
        )
        synth = CodeSynthesisEngine().synthesize(
            SynthesisRequest(
                package_name="task_api", entities=[entity],
                framework="fastapi", model_style="dataclass", mode="fresh",
            ),
            project_id="demo",
            existing_index=RepoIndex(root="<none>"),
        )
        testgen = TestGenerator()
        tplan = testgen.generate(
            spec=spec, synthesis=synth, project_id="demo",
        )
        assert tplan.compile_errors == []

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            writes = [
                FileWrite(f.path, f.content,
                          kind=getattr(f.kind, "value", "src"))
                for f in synth.files
            ]
            writes += [
                FileWrite(a.path, a.content, kind=f"test-{a.kind.value}")
                for a in tplan.artifacts
            ]
            builder = ProjectBuilder()
            br = builder.apply(BuildPlan(
                root=td, mode=WriteMode.CREATE_ONLY, writes=writes,
            ))
            assert br.status.value == "succeeded", br.rationale

            # Run full suite
            res = engine.run_full(root=root, project_id="demo")
            # We don't guarantee every generated test passes (some are
            # acceptance markers using pytest.skip). But collection must
            # succeed and pytest must produce a report.
            assert res.status in (
                RunStatus.SUCCEEDED, RunStatus.TESTS_FAILED, RunStatus.NO_TESTS,
            ), res.rationale
            # Should have collected at least the model + repo tests
            assert len(res.results) >= 5, len(res.results)
            # Skips for acceptance markers
            assert res.skipped >= 1

    check("e2e: C14→C15→C17→C18 runs generated suite",
          t_e2e_full_pipeline, requires_pytest=True)

    print()
    skip_note = f", {skipped} skipped (pytest not installed)" if skipped else ""
    print(f"Self-tests: {passed} passed, {len(failures)} failed{skip_note}")
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
    print("SE Brain C18 — Test Execution Engine")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_repo"
        root.mkdir()
        _mk_small_repo(root, make_fail=True)

        engine = TestExecutionEngine()

        print(f"\n[1] Demo repo at {root}")
        for p in sorted(root.rglob("*.py")):
            print(f"    {p.relative_to(root)}")

        print("\n[2] Full suite run:")
        res = engine.run_full(root=root, project_id="demo")
        print(res.summary())

        print("\n[3] Environment:")
        for k, v in res.environment.to_dict().items():
            if k in ("python_full_version", "platform"):
                print(f"    {k}: {_short(str(v), 70)}")
            else:
                print(f"    {k}: {v}")

        print("\n[4] Per-test results:")
        for r in res.results:
            mark = {"passed": "✓", "failed": "✗", "error": "E",
                    "skipped": "s"}.get(r.outcome.value, "?")
            print(f"    {mark} [{r.component:12s}] {r.nodeid}  "
                  f"({r.duration_seconds*1000:.1f}ms)")
            if r.failure_message:
                print(f"        {r.failure_message}")

        print("\n[5] Selection:")
        sel = engine.select(root=root, changed_paths=["pkg/math.py"])
        print(f"    mode={sel.mode.value}  files={sel.selected_files}")
        print(f"    reason={sel.reason}")

        sel2 = engine.select(root=root, changed_paths=["pyproject.toml"])
        print(f"    pyproject change → mode={sel2.mode.value}")
        print(f"    reason={sel2.reason}")

        print("\n[6] Selective run:")
        sel_res = engine.run_selective(
            root=root, changed_paths=["pkg/math.py"], project_id="demo",
        )
        print(sel_res.summary())
        for r in sel_res.results:
            print(f"    {r.outcome.value:7s}  {r.nodeid}")

        print("\n[7] Regression detection:")
        # Fix the failure
        _write(root, "tests/test_fail.py", (
            "from pkg.math import add\n"
            "\n"
            "def test_fail_on_purpose() -> None:\n"
            "    assert add(1, 2) == 3\n"
        ))
        prev = res
        cur = engine.run_full(
            root=root, project_id="demo", previous=prev,
        )
        print(f"    previous failed: {prev.failed}")
        print(f"    current failed : {cur.failed}")
        print(f"    regressions    : {len(cur.regressions)}")

        print("\n[8] Persistence:")
        with tempfile.TemporaryDirectory() as std:
            cfg = Config(data_dir=Path(std) / "sebrain", log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                with execution_scope(project_id="demo"):
                    mem = MemoryStore(app.storage)
                    ont = Ontology(app.storage)
                    repo = TestRunRepository(memory=mem, ontology=ont)
                    ent = repo.save(cur, project_id="demo")
                    print(f"    ontology entity: {ent[:12]}…")
                    latest = repo.load_latest(project_id="demo")
                    print(f"    reloaded status: {latest.status.value}  "
                          f"tests={len(latest.results)}")
                    print(f"    EXECUTION entities: "
                          f"{ont.count(kind=EntityKind.EXECUTION)}")
                    print(f"    TEST_RESULT entities: "
                          f"{ont.count(kind=EntityKind.TEST_RESULT)}")
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
