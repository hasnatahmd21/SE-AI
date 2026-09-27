"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C20 — CODE REPAIR ENGINE (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C13, C14, C15, C16, C18, C19.

Purpose:
    Generate multiple repair candidates for a diagnosed failure, evaluate
    each ONE in an isolated copy of the repository, and ACCEPT a candidate
    only if it (a) fixes the target failure AND (b) causes no regressions.

Hard rules (from the specification):
    - Generate multiple candidates where appropriate
    - Apply → Test → Analyze → Compare → Accept/Reject
    - NEVER accept a repair merely because the failing test disappeared
    - Check regression impact
    - Preserve the previous version for rollback

Built-in repair strategies (deterministic, honest about scope):
    1) DependencyMissingModuleStrategy
         For DEPENDENCY failures with an extractable module name.
         Emits: append the module to requirements.txt (create if missing).
         NOTE: this does NOT fix tests that import a missing module at
         collection time. It records the missing dependency as a first
         step. Its acceptance is evaluated the same way as any other
         candidate — it will usually be rejected because the target
         test still fails. That is correct behaviour (Rule: never
         accept just because the failing test disappeared).
    2) ConfigMissingKeyStrategy
         For CONFIG failures with a KeyError-style message.
         Emits: replace `obj['KEY']` with `obj.get('KEY')` on the failing
         line (exact-anchor patch).
    3) UserSuppliedStrategy
         The caller passes pre-built RepairCandidates (e.g. from a human
         or a future LLM adapter). The engine evaluates them the same way.

Non-repairable categories (SYNTAX, LOGIC, TYPE, ENVIRONMENT, INTEGRATION,
RESOURCE, CONCURRENCY, SECURITY, RUNTIME, UNKNOWN) yield ZERO auto-candidates
and an explicit reason. This is a documented limitation, not a fake.

Evaluation flow:
    baseline     ← C18 full-suite on an UNMODIFIED copy
    for each candidate:
        fresh copy → apply patches → C18 full-suite → compare vs baseline
        target-fixed = target nodeid improved to PASSED
        regressions  = prior PASSED/SKIPPED that now FAILED/ERROR
        accepted     = target-fixed AND regressions == 0
    rank accepted  = (fewest bytes changed, fewest files, candidate id)
    pick best      = first accepted (or None)
    baseline is NEVER modified; caller's root is NEVER modified

Invariants honored:
    - NO external LLM
    - Original root is never mutated
    - Every candidate carries rationale + evidence + byte-diff stats
    - Rejections carry a structured reason (never silent)
    - Byte-exact anchor patches: `old_text` must appear EXACTLY ONCE
    - Bounded: max_patches, max_candidates, max_total_patch_bytes
    - Deterministic: same inputs → same candidate order and verdicts

Explicit limitations:
    - Only 2 categories are safely auto-repairable without semantics.
    - Target-fixed requires the caller to provide the target test nodeid.
    - Patch generation is not LLM-driven; it is pattern-based and can miss
      valid fixes. That is by design (no external AI).

Contents:
  1.  Enums: RepairKind, RepairStatus, EvaluationVerdict
  2.  Dataclasses: FilePatch, RepairCandidate, PatchApplyResult,
                   CandidateEvaluation, BaselineSnapshot, RepairResult
  3.  Copy + patch application
  4.  Built-in strategies (2 + user-supplied passthrough)
  5.  RepairEvaluator (uses C18)
  6.  RepairEngine facade
  7.  RepairRepository (persist to C04 memory + C02 ontology)
  8.  Self-tests (~30)
  9.  Demo

Run as script:
    python -m sebrain.c20            # demo
    python -m sebrain.c20 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import os
import re
import shutil
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


_SKIP_COPY_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "venv", "env", "node_modules", ".tox",
    ".idea", ".vscode", "dist", "build", ".eggs", ".sebrain_backups",
})


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a repo tree to dst, skipping heavy/irrelevant dirs.

    Repair evaluation must never silently omit source files or follow symlinks
    outside the repository: either the complete supported tree is copied or
    the operation fails loudly so the candidate cannot receive a false verdict.
    """
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        raise OSError(f"source repository is not a directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=False)

    def _copy_dir_contents(src_dir: Path, dst_dir: Path) -> None:
        try:
            with os.scandir(src_dir) as entries:
                for entry in entries:
                    name = entry.name
                    if name in _SKIP_COPY_DIRS:
                        continue
                    source = Path(entry.path)
                    target = dst_dir / name
                    if entry.is_symlink():
                        raise OSError(
                            f"symlink is not allowed in repair evaluation: {source}"
                        )
                    if entry.is_dir(follow_symlinks=False):
                        target.mkdir(parents=True, exist_ok=False)
                        _copy_dir_contents(source, target)
                    elif entry.is_file(follow_symlinks=False):
                        try:
                            shutil.copy2(source, target)
                        except OSError as exc:
                            raise OSError(
                                f"failed to copy file {source}: {exc}"
                            ) from exc
                    else:
                        raise OSError(
                            f"unsupported filesystem entry in repair tree: {source}"
                        )
        except OSError:
            shutil.rmtree(dst_dir, ignore_errors=True)
            raise

    _copy_dir_contents(src, dst)


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class RepairKind(str, Enum):
    """Repair strategy identifier — where the candidate came from."""
    DEPENDENCY_ADD_REQUIREMENT = "dependency_add_requirement"
    CONFIG_USE_GET = "config_use_get"
    USER_SUPPLIED = "user_supplied"


class RepairStatus(str, Enum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EvaluationVerdict(str, Enum):
    ACCEPTED = "accepted"
    REJECTED_TARGET_NOT_FIXED = "rejected_target_not_fixed"
    REJECTED_REGRESSION = "rejected_regression"
    REJECTED_PATCH_FAILED = "rejected_patch_failed"
    REJECTED_TEST_RUN_ERROR = "rejected_test_run_error"
    REJECTED_NO_TARGET = "rejected_no_target"
    NOT_EVALUATED = "not_evaluated"


class _PatchApplyStatus(str, Enum):
    OK = "ok"
    FILE_MISSING = "file_missing"
    ANCHOR_NOT_FOUND = "anchor_not_found"
    ANCHOR_AMBIGUOUS = "anchor_ambiguous"
    WRITE_FAILED = "write_failed"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class FilePatch:
    """Replace `old_text` with `new_text` in `path` (exact single-anchor)."""
    path: str
    old_text: str
    new_text: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValidationError("FilePatch.path is required")
        if self.old_text == self.new_text:
            raise ValidationError("FilePatch old_text == new_text (no-op)")

    def byte_size(self) -> int:
        return len(self.old_text.encode("utf-8")) + \
               len(self.new_text.encode("utf-8"))

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "old_len": len(self.old_text),
            "new_len": len(self.new_text),
            "rationale": self.rationale,
        }
        if include_content:
            d["old_text"] = self.old_text
            d["new_text"] = self.new_text
        return d


@dataclass(slots=True)
class RepairCandidate:
    id: str = field(default_factory=_new_id)
    kind: RepairKind = RepairKind.USER_SUPPLIED
    patches: list[FilePatch] = field(default_factory=list)
    rationale: str = ""
    confidence: Confidence = Confidence.LOW
    evidence: list[dict[str, Any]] = field(default_factory=list)
    status: RepairStatus = RepairStatus.PROPOSED

    def total_bytes(self) -> int:
        return sum(p.byte_size() for p in self.patches)

    def files_touched(self) -> list[str]:
        return sorted({p.path for p in self.patches})

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind.value,
            "patches": [p.to_dict(include_content=include_content)
                        for p in self.patches],
            "rationale": self.rationale,
            "confidence": self.confidence.value,
            "evidence": list(self.evidence),
            "status": self.status.value,
            "total_bytes": self.total_bytes(),
            "files_touched": self.files_touched(),
        }


@dataclass(slots=True)
class PatchApplyResult:
    status: _PatchApplyStatus
    path: str = ""
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "path": self.path,
                "message": self.message}


@dataclass(slots=True)
class BaselineSnapshot:
    """Compact per-nodeid outcome map of the reference test run."""
    id: str = field(default_factory=_new_id)
    root: str = ""
    outcomes: dict[str, str] = field(default_factory=dict)   # nodeid → outcome
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    status: str = ""            # C18 RunStatus value

    @classmethod
    def from_run(cls, run: Any, *, root: str) -> "BaselineSnapshot":
        snap = cls(root=root)
        snap.status = getattr(getattr(run, "status", None), "value", "")
        for r in getattr(run, "results", []) or []:
            nodeid = getattr(r, "nodeid", "")
            oc = getattr(r, "outcome", None)
            snap.outcomes[nodeid] = getattr(oc, "value", str(oc))
        snap.total = len(snap.outcomes)
        for oc in snap.outcomes.values():
            if oc == "passed": snap.passed += 1
            elif oc == "failed": snap.failed += 1
            elif oc == "error": snap.errors += 1
            elif oc == "skipped": snap.skipped += 1
        return snap

    def outcome_of(self, nodeid: str) -> str:
        return self.outcomes.get(nodeid, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "root": self.root,
            "outcomes": dict(self.outcomes),
            "passed": self.passed, "failed": self.failed,
            "errors": self.errors, "skipped": self.skipped,
            "total": self.total, "status": self.status,
        }


@dataclass(slots=True)
class CandidateEvaluation:
    candidate_id: str
    verdict: EvaluationVerdict
    target_nodeid: str = ""
    target_status_before: str = ""
    target_status_after: str = ""
    regression_count: int = 0
    regressions: list[str] = field(default_factory=list)
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    apply_results: list[PatchApplyResult] = field(default_factory=list)
    reason: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "verdict": self.verdict.value,
            "target_nodeid": self.target_nodeid,
            "target_status_before": self.target_status_before,
            "target_status_after": self.target_status_after,
            "regression_count": self.regression_count,
            "regressions": list(self.regressions),
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "total_errors": self.total_errors,
            "apply_results": [a.to_dict() for a in self.apply_results],
            "reason": self.reason,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class RepairResult:
    id: str = field(default_factory=_new_id)
    project_id: str = ""
    root: str = ""
    debug_report_id: str = ""
    category: str = ""
    auto_repairable: bool = False
    non_repairable_reason: str = ""
    target_nodeid: str = ""
    baseline: BaselineSnapshot | None = None
    candidates: list[RepairCandidate] = field(default_factory=list)
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    accepted_id: str | None = None
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def accepted(self) -> RepairCandidate | None:
        if not self.accepted_id:
            return None
        for c in self.candidates:
            if c.id == self.accepted_id:
                return c
        return None

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "id": self.id, "project_id": self.project_id,
            "root": self.root,
            "debug_report_id": self.debug_report_id,
            "category": self.category,
            "auto_repairable": self.auto_repairable,
            "non_repairable_reason": self.non_repairable_reason,
            "target_nodeid": self.target_nodeid,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "candidates": [c.to_dict(include_content=include_content)
                           for c in self.candidates],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "accepted_id": self.accepted_id,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Repair Result ===\n"
            f"category={self.category}  auto_repairable={self.auto_repairable}\n"
            f"target={self.target_nodeid or '<none>'}\n"
            f"candidates={len(self.candidates)}  "
            f"evaluations={len(self.evaluations)}  "
            f"accepted={self.accepted_id[:8] if self.accepted_id else '<none>'}\n"
            + (
                f"baseline: passed={self.baseline.passed} "
                f"failed={self.baseline.failed} "
                f"errors={self.baseline.errors}"
                if self.baseline else "baseline: <none>"
            )
            + (f"\nnote: {self.non_repairable_reason}"
               if self.non_repairable_reason else "")
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. COPY + PATCH APPLICATION
# ════════════════════════════════════════════════════════════════════════════
def apply_patches(
    root: Path, patches: Sequence[FilePatch],
) -> list[PatchApplyResult]:
    """Apply patches to a copy in-place (root already points at a temp copy).

    Anchors must match exactly once. A missing target may only be created when
    the patch explicitly uses an empty old_text anchor (the repository's
    create-file convention). Paths and symlinks are confined to ``root``.
    """
    results: list[PatchApplyResult] = []
    root = root.resolve()

    for p in patches:
        raw = Path(p.path)
        if raw.is_absolute():
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.FILE_MISSING, path=p.path,
                message="absolute patch paths are not allowed",
            ))
            continue
        # Reject path traversal and symlink escapes before any filesystem I/O.
        target = root.joinpath(*raw.parts)
        try:
            resolved_target = target.resolve(strict=False)
            resolved_target.relative_to(root)
        except (OSError, ValueError):
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.FILE_MISSING, path=p.path,
                message="patch path escapes repository root",
            ))
            continue

        # Existing symlink targets are never modified. Parent symlinks are
        # also rejected by the resolved containment check above.
        if target.is_symlink():
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.FILE_MISSING, path=p.path,
                message="target is a symlink; refusing to patch",
            ))
            continue

        if not target.is_file():
            # Explicit empty-anchor + missing file = create-file operation.
            if p.old_text == "":
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Re-check after creating parents for a symlink race.
                    cursor = target.parent
                    while cursor != root:
                        if cursor.is_symlink():
                            raise OSError("symlink parent")
                        cursor = cursor.parent
                    target.write_text(p.new_text, encoding="utf-8")
                except OSError as exc:
                    results.append(PatchApplyResult(
                        status=_PatchApplyStatus.WRITE_FAILED,
                        path=p.path, message=f"create failed: {exc}",
                    ))
                else:
                    results.append(PatchApplyResult(
                        status=_PatchApplyStatus.OK, path=p.path,
                        message="created",
                    ))
                continue
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.FILE_MISSING,
                path=p.path, message=f"file does not exist: {target}",
            ))
            continue

        try:
            content = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.WRITE_FAILED,
                path=p.path, message=f"read failed: {exc}",
            ))
            continue

        count = content.count(p.old_text)
        if count == 0:
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.ANCHOR_NOT_FOUND,
                path=p.path, message="anchor not found",
            ))
            continue
        if count > 1:
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.ANCHOR_AMBIGUOUS,
                path=p.path,
                message=f"anchor found {count} times; refusing to patch",
            ))
            continue
        new_content = content.replace(p.old_text, p.new_text, 1)
        try:
            target.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            results.append(PatchApplyResult(
                status=_PatchApplyStatus.WRITE_FAILED,
                path=p.path, message=f"write failed: {exc}",
            ))
            continue
        results.append(PatchApplyResult(
            status=_PatchApplyStatus.OK, path=p.path, message="applied",
        ))
    return results


# ════════════════════════════════════════════════════════════════════════════
# 4. REPAIR STRATEGIES
# ════════════════════════════════════════════════════════════════════════════
_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
_KEYERROR_RE = re.compile(r"^['\"](?P<key>[^'\"]+)['\"]$")


def _extract_missing_module(msg: str) -> str:
    m = _MISSING_MODULE_RE.search(msg or "")
    return m.group(1) if m else ""


def _extract_keyerror_key(msg: str) -> str:
    m = _KEYERROR_RE.match((msg or "").strip())
    return m.group("key") if m else ""


class DependencyMissingModuleStrategy:
    """For DEPENDENCY failures with an extractable module name.

    Produces exactly one candidate: append the module to requirements.txt.
    If requirements.txt doesn't exist, create it. If the module is already
    declared, produce no candidate (already accounted for).
    """

    KIND = RepairKind.DEPENDENCY_ADD_REQUIREMENT

    def propose(
        self, *, report: Any, root: Path,
    ) -> list[RepairCandidate]:
        cat = getattr(report, "category", None)
        # Accept either enum or its .value
        cat_val = getattr(cat, "value", str(cat))
        if cat_val != "dependency":
            return []
        sig = getattr(report, "signature", None)
        if sig is None:
            return []
        module = _extract_missing_module(getattr(sig, "exception_message", ""))
        if not module:
            return []

        req_path = root / "requirements.txt"
        if req_path.is_file():
            try:
                content = req_path.read_text(encoding="utf-8")
            except OSError:
                content = ""
            lines = [ln.strip() for ln in content.splitlines()]
            if module in lines:
                # Already declared → no safe automatic candidate
                return []
            if content and not content.endswith("\n"):
                content += "\n"
            new_content = content + f"{module}\n"
            patch = FilePatch(
                path="requirements.txt",
                old_text=content,
                new_text=new_content,
                rationale=f"append '{module}' to requirements.txt",
            )
        else:
            patch = FilePatch(
                path="requirements.txt",
                old_text="",  # anchor is the whole file
                new_text=f"{module}\n",
                rationale=f"create requirements.txt with '{module}'",
            )

        return [RepairCandidate(
            kind=self.KIND,
            patches=[patch],
            rationale=(
                f"Declare missing dependency '{module}' in requirements.txt "
                f"so future installs resolve it."
            ),
            confidence=Confidence.MEDIUM,
            evidence=[
                {"kind": "strategy", "value": self.KIND.value},
                {"kind": "missing_module", "value": module},
                {"kind": "source_failure_id",
                 "value": str(getattr(report, "id", ""))},
            ],
        )]


# Regex for `base['KEY']` (single-quoted or double-quoted, single line)
_SUBSCRIPT_RE_TEMPLATE = (
    r"(?P<base>[A-Za-z_][A-Za-z0-9_.]*)\[\s*['\"]{key}['\"]\s*\]"
)


class ConfigMissingKeyStrategy:
    """For CONFIG failures from KeyError with a localized source line.

    Emits one candidate per distinct `x['KEY']` occurrence found on the
    localized line.
    """

    KIND = RepairKind.CONFIG_USE_GET

    def propose(
        self, *, report: Any, root: Path,
    ) -> list[RepairCandidate]:
        cat = getattr(report, "category", None)
        cat_val = getattr(cat, "value", str(cat))
        if cat_val != "config":
            return []
        sig = getattr(report, "signature", None)
        if sig is None:
            return []
        if getattr(sig, "exception_type", "") != "KeyError":
            return []
        key = _extract_keyerror_key(getattr(sig, "exception_message", ""))
        if not key:
            return []
        loc = getattr(report, "localized", None)
        if loc is None or not getattr(loc, "is_user_code", False):
            return []

        # Determine repo-relative path
        try:
            abs_file = Path(loc.file).resolve()
            rel = abs_file.relative_to(root.resolve())
        except (OSError, ValueError):
            return []
        rel_str = str(rel).replace("\\", "/")

        try:
            text = abs_file.read_text(encoding="utf-8")
        except OSError:
            return []
        lines = text.splitlines()
        lineno = int(getattr(loc, "lineno", 0) or 0)
        if lineno <= 0 or lineno > len(lines):
            return []
        src_line = lines[lineno - 1]

        pattern = re.compile(
            _SUBSCRIPT_RE_TEMPLATE.format(key=re.escape(key))
        )
        matches = list(pattern.finditer(src_line))
        if not matches:
            return []

        seen_anchors: set[str] = set()
        patches: list[FilePatch] = []
        for m in matches:
            base = m.group("base")
            anchor = m.group(0)
            if anchor in seen_anchors:
                continue
            seen_anchors.add(anchor)
            quote = anchor[anchor.index('[') + 1]
            replacement = f"{base}.get({quote}{key}{quote})"
            patches.append(FilePatch(
                path=rel_str,
                old_text=anchor,
                new_text=replacement,
                rationale=(
                    f"replace {anchor} with .get(...) so a missing "
                    f"'{key}' does not raise KeyError"
                ),
            ))
        if not patches:
            return []

        return [RepairCandidate(
            kind=self.KIND,
            patches=patches,
            rationale=(
                f"Use dict.get() for optional key '{key}' on the failing "
                f"line so absence does not raise KeyError."
            ),
            confidence=Confidence.LOW,
            evidence=[
                {"kind": "strategy", "value": self.KIND.value},
                {"kind": "missing_key", "value": key},
                {"kind": "anchor_line", "value": src_line[:200]},
                {"kind": "source_failure_id",
                 "value": str(getattr(report, "id", ""))},
            ],
        )]


class UserSuppliedStrategy:
    """Passthrough for external candidates (human or future LLM adapter)."""

    def __init__(self, candidates: Sequence[RepairCandidate]) -> None:
        self._candidates = list(candidates)

    def propose(self, *, report: Any, root: Path) -> list[RepairCandidate]:
        out: list[RepairCandidate] = []
        for c in self._candidates:
            if c.kind is RepairKind.USER_SUPPLIED:
                out.append(c)
            else:
                out.append(RepairCandidate(
                    id=c.id, kind=RepairKind.USER_SUPPLIED,
                    patches=list(c.patches), rationale=c.rationale,
                    confidence=c.confidence, evidence=list(c.evidence),
                ))
        return out


# ════════════════════════════════════════════════════════════════════════════
# 5. REPAIR EVALUATOR (uses C18)
# ════════════════════════════════════════════════════════════════════════════
class RepairEvaluator:
    """Runs baseline + candidate copies and compares via C18."""

    def __init__(
        self,
        *,
        test_engine: Any | None = None,
        test_policy: Any | None = None,
    ) -> None:
        if test_engine is None:
            from sebrain.c18 import TestExecutionEngine, TestRunPolicy
            test_engine = TestExecutionEngine()
            if test_policy is None:
                test_policy = TestRunPolicy(timeout_seconds=120.0)
        self.test_engine = test_engine
        self.test_policy = test_policy

    # ---- baseline ----
    def run_baseline(self, *, root: Path) -> BaselineSnapshot:
        with tempfile.TemporaryDirectory() as td:
            copy_root = Path(td) / "baseline"
            _copy_tree(root, copy_root)
            run = self.test_engine.run_full(
                root=copy_root, project_id="c20-baseline",
                policy=self.test_policy,
            )
            return BaselineSnapshot.from_run(run, root=str(copy_root))

    # ---- per-candidate ----
    def evaluate(
        self, *, root: Path, candidate: RepairCandidate,
        target_nodeid: str = "",
        baseline: BaselineSnapshot | None = None,
    ) -> CandidateEvaluation:
        import time
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as td:
            copy_root = Path(td) / "candidate"
            try:
                _copy_tree(root, copy_root)
            except OSError as exc:
                return CandidateEvaluation(
                    candidate_id=candidate.id,
                    verdict=EvaluationVerdict.REJECTED_TEST_RUN_ERROR,
                    target_nodeid=target_nodeid,
                    reason=(
                        "could not create isolated evaluation copy: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    duration_seconds=time.monotonic() - t0,
                )

            apply_results = apply_patches(copy_root, candidate.patches)
            failed_apply = [a for a in apply_results
                            if a.status is not _PatchApplyStatus.OK]
            if failed_apply:
                reason = "; ".join(
                    f"{a.path}: {a.status.value}: {a.message}"
                    for a in failed_apply
                )
                return CandidateEvaluation(
                    candidate_id=candidate.id,
                    verdict=EvaluationVerdict.REJECTED_PATCH_FAILED,
                    target_nodeid=target_nodeid,
                    apply_results=apply_results,
                    reason=reason,
                    duration_seconds=time.monotonic() - t0,
                )

            try:
                run = self.test_engine.run_full(
                    root=copy_root, project_id="c20-candidate",
                    policy=self.test_policy,
                )
            except Exception as exc:
                return CandidateEvaluation(
                    candidate_id=candidate.id,
                    verdict=EvaluationVerdict.REJECTED_TEST_RUN_ERROR,
                    target_nodeid=target_nodeid,
                    apply_results=apply_results,
                    reason=f"test run failed: {type(exc).__name__}: {exc}",
                    duration_seconds=time.monotonic() - t0,
                )

            new_snap = BaselineSnapshot.from_run(run, root=str(copy_root))

            # Compute regressions ourselves (more explicit than C18's diff,
            # which only fires when previous is passed in)
            regressions: list[str] = []
            if baseline is not None:
                for nodeid, prev_oc in baseline.outcomes.items():
                    new_oc = new_snap.outcomes.get(nodeid, "")
                    if prev_oc in ("passed", "skipped") and new_oc in (
                        "failed", "error",
                    ):
                        regressions.append(nodeid)

            # Target check
            target_before = baseline.outcome_of(target_nodeid) if (
                baseline and target_nodeid) else ""
            target_after = new_snap.outcome_of(target_nodeid) if (
                target_nodeid) else ""
            target_fixed = (
                bool(target_nodeid)
                and target_after == "passed"
                and target_before in ("failed", "error")
            )

            # Verdict
            if target_nodeid and not target_fixed:
                verdict = EvaluationVerdict.REJECTED_TARGET_NOT_FIXED
                reason = (
                    f"target '{target_nodeid}' not fixed "
                    f"(before={target_before or '?'} "
                    f"after={target_after or '?'})"
                )
            elif not target_nodeid:
                verdict = EvaluationVerdict.REJECTED_NO_TARGET
                reason = (
                    "no target nodeid provided — cannot verify target is "
                    "fixed; refusing to accept on vague criteria"
                )
            elif regressions:
                verdict = EvaluationVerdict.REJECTED_REGRESSION
                names = ", ".join(regressions[:3])
                reason = (
                    f"{len(regressions)} regression(s): {names}"
                    + (" …" if len(regressions) > 3 else "")
                )
            else:
                verdict = EvaluationVerdict.ACCEPTED
                reason = "target fixed, no regressions"

            return CandidateEvaluation(
                candidate_id=candidate.id,
                verdict=verdict,
                target_nodeid=target_nodeid,
                target_status_before=target_before,
                target_status_after=target_after,
                regression_count=len(regressions),
                regressions=regressions,
                total_passed=new_snap.passed,
                total_failed=new_snap.failed,
                total_errors=new_snap.errors,
                apply_results=apply_results,
                reason=reason,
                duration_seconds=time.monotonic() - t0,
            )


# ════════════════════════════════════════════════════════════════════════════
# 6. REPAIR ENGINE (facade)
# ════════════════════════════════════════════════════════════════════════════
_NON_REPAIRABLE_CATEGORIES: dict[str, str] = {
    "syntax":
        "syntax errors require parser-aware rewriting; not auto-generated",
    "type":
        "type mismatches require semantic knowledge of surrounding code",
    "logic":
        "logic defects require understanding intent; not auto-generated",
    "runtime":
        "generic runtime errors need context; no safe automatic patch",
    "environment":
        "environment issues are external; nothing to patch in the repo",
    "integration":
        "integration failures are external; retry/timeout tuning is a "
        "policy decision, not a code fix",
    "concurrency":
        "concurrency fixes require reasoning about async flow",
    "resource":
        "resource fixes require workload/semantics awareness",
    "security":
        "security fixes must be reviewed by a human",
    "unknown":
        "no categorization → no automatic repair",
}


class RepairEngine:
    """Full pipeline: propose → baseline → evaluate → accept/reject → pick best."""

    def __init__(
        self, *,
        evaluator: RepairEvaluator | None = None,
        max_candidates: int = 8,
        max_total_patch_bytes: int = 100_000,
    ) -> None:
        if max_candidates < 1:
            raise ValidationError("max_candidates must be >= 1")
        if max_total_patch_bytes < 1:
            raise ValidationError("max_total_patch_bytes must be >= 1")
        self.evaluator = evaluator or RepairEvaluator()
        self.max_candidates = max_candidates
        self.max_total_patch_bytes = max_total_patch_bytes
        self.strategies = [
            DependencyMissingModuleStrategy(),
            ConfigMissingKeyStrategy(),
        ]

    # ---- propose ----
    def propose(
        self, *, report: Any, root: str | Path,
        user_candidates: Sequence[RepairCandidate] | None = None,
    ) -> tuple[list[RepairCandidate], bool, str]:
        """Return (candidates, auto_repairable, non_repairable_reason)."""
        root_p = Path(root).resolve()
        candidates: list[RepairCandidate] = []
        for s in self.strategies:
            try:
                candidates.extend(s.propose(report=report, root=root_p))
            except Exception as exc:
                log.warning("c20.strategy_failed",
                            strategy=type(s).__name__, error=str(exc))
        # User-supplied last (their order preserved)
        if user_candidates:
            try:
                candidates.extend(
                    UserSuppliedStrategy(user_candidates)
                    .propose(report=report, root=root_p)
                )
            except Exception as exc:
                log.warning("c20.user_strategy_failed", error=str(exc))

        # Bound total bytes and count
        bounded: list[RepairCandidate] = []
        total_bytes = 0
        for c in candidates[: self.max_candidates]:
            if total_bytes + c.total_bytes() > self.max_total_patch_bytes:
                break
            total_bytes += c.total_bytes()
            bounded.append(c)

        auto_repairable = any(
            c.kind is not RepairKind.USER_SUPPLIED for c in bounded
        )
        reason = ""
        if not bounded:
            cat = getattr(report, "category", None)
            cat_val = getattr(cat, "value", str(cat))
            reason = _NON_REPAIRABLE_CATEGORIES.get(
                cat_val,
                "no strategy produced a candidate",
            )
        return bounded, auto_repairable, reason

    # ---- full run ----
    def run(
        self, *, report: Any, root: str | Path,
        project_id: str = "",
        target_nodeid: str = "",
        user_candidates: Sequence[RepairCandidate] | None = None,
        baseline: BaselineSnapshot | None = None,
    ) -> RepairResult:
        root_p = Path(root).resolve()
        if not root_p.is_dir():
            raise ValidationError(f"root not a directory: {root_p}")

        cat = getattr(report, "category", None)
        cat_val = getattr(cat, "value", str(cat))

        result = RepairResult(
            project_id=project_id, root=str(root_p),
            debug_report_id=str(getattr(report, "id", "")),
            category=cat_val,
            target_nodeid=target_nodeid,
            provenance=Provenance(
                source="code_repair_engine",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )

        # 1. Propose
        candidates, auto, reason = self.propose(
            report=report, root=root_p, user_candidates=user_candidates,
        )
        result.candidates = candidates
        result.auto_repairable = auto
        result.non_repairable_reason = reason
        if not candidates:
            result.rationale = (
                f"no candidates for category '{cat_val}'; "
                f"reason: {reason}"
            )
            return result

        # 2. Baseline
        if baseline is None:
            try:
                baseline = self.evaluator.run_baseline(root=root_p)
            except Exception as exc:
                result.rationale = (
                    f"baseline run failed: {type(exc).__name__}: {exc}"
                )
                return result
        result.baseline = baseline

        # 3. Evaluate
        evaluations: list[CandidateEvaluation] = []
        for c in candidates:
            ev = self.evaluator.evaluate(
                root=root_p, candidate=c,
                target_nodeid=target_nodeid, baseline=baseline,
            )
            evaluations.append(ev)
            c.status = (
                RepairStatus.ACCEPTED
                if ev.verdict is EvaluationVerdict.ACCEPTED
                else RepairStatus.REJECTED
            )
        result.evaluations = evaluations

        # 4. Rank accepted (bytes asc, files asc, id asc) → best
        accepted_pairs = [
            (c, ev) for c, ev in zip(candidates, evaluations)
            if ev.verdict is EvaluationVerdict.ACCEPTED
        ]
        if accepted_pairs:
            accepted_pairs.sort(key=lambda pair: (
                pair[0].total_bytes(),
                len(pair[0].files_touched()),
                pair[0].id,
            ))
            result.accepted_id = accepted_pairs[0][0].id
        else:
            result.accepted_id = None

        accepted = len(accepted_pairs)
        result.rationale = (
            f"category={cat_val}  candidates={len(candidates)}  "
            f"accepted={accepted}  "
            f"baseline(p={baseline.passed}, "
            f"f={baseline.failed}, e={baseline.errors})  "
            f"target={target_nodeid or '<none>'}"
        )
        return result


# ════════════════════════════════════════════════════════════════════════════
# 7. REPAIR REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class RepairRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: RepairResult, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"repair_result:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(include_content=False),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["repair", "c20",
                  "accepted" if result.accepted_id else "no_fix"],
            provenance=result.provenance,
        )
        # Record decision memory when accepted
        if result.accepted_id:
            acc = result.accepted()
            if acc:
                self.memory.record_decision(
                    f"repair:{result.id}",
                    decision=(
                        f"accept repair '{acc.kind.value}' for "
                        f"{result.category}"
                    ),
                    rationale=acc.rationale,
                    alternatives=[
                        c.rationale for c in result.candidates
                        if c.id != result.accepted_id
                    ][:5],
                    scope_id=project_id,
                    provenance=result.provenance,
                    confidence=Confidence.MEDIUM,
                )
        if self.ontology is None:
            return key

        ent = self.ontology.add(
            EntityKind.FIX,
            _short(
                f"Repair {result.id[:8]} "
                f"({'accepted' if result.accepted_id else 'no_fix'}, "
                f"{result.category})", 120,
            ),
            attributes={
                "repair_id": result.id,
                "project_id": project_id,
                "category": result.category,
                "auto_repairable": result.auto_repairable,
                "candidate_count": len(result.candidates),
                "accepted_id": result.accepted_id,
                "target_nodeid": result.target_nodeid,
                "baseline": (result.baseline.to_dict()
                             if result.baseline else None),
            },
            tags=["repair", result.category],
            provenance=result.provenance,
        )
        # One ALTERNATIVE entity per candidate
        for c in result.candidates:
            ce = self.ontology.add(
                EntityKind.ALTERNATIVE,
                _short(f"Repair candidate {c.kind.value}", 120),
                attributes={
                    "kind": c.kind.value,
                    "status": c.status.value,
                    "patches": len(c.patches),
                    "bytes": c.total_bytes(),
                },
                tags=["repair-candidate", c.kind.value],
                provenance=result.provenance,
            )
            try:
                self.ontology.link(RelationKind.RELATES_TO, ce.id, ent.id)
            except ValidationError:
                pass
        return ent.id

    def load(self, repair_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"repair_result:{repair_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 8. SELF-TESTS
# ════════════════════════════════════════════════════════════════════════════
def _write(d: Path, rel: str, content: str) -> Path:
    p = d / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _mk_logic_repo(root: Path) -> None:
    _write(root, "pyproject.toml", "[project]\nname='x'\n")
    _write(root, "pkg/__init__.py", '"""pkg."""\n')
    _write(root, "pkg/calc.py", (
        "def add(a: int, b: int) -> int:\n"
        "    return a - b   # wrong on purpose\n"
        "\n"
        "def sub(a: int, b: int) -> int:\n"
        "    return a - b\n"
    ))
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_calc.py", (
        "from pkg.calc import add, sub\n"
        "\n"
        "def test_add() -> None:\n"
        "    assert add(1, 2) == 3\n"
        "\n"
        "def test_sub() -> None:\n"
        "    assert sub(5, 3) == 2\n"
    ))


def _mk_dep_repo(root: Path) -> None:
    _write(root, "pyproject.toml", "[project]\nname='x'\n")
    _write(root, "pkg/__init__.py", '"""pkg."""\n')
    _write(root, "pkg/broken.py", (
        "import definitely_missing_module_xyz_123  # noqa: F401\n"
        "\n"
        "def f() -> int:\n"
        "    return 1\n"
    ))
    _write(root, "tests/__init__.py", "")
    _write(root, "tests/test_broken.py", (
        "from pkg.broken import f\n"
        "\n"
        "def test_f() -> None:\n"
        "    assert f() == 1\n"
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

    print("Running C20 self-tests…")
    engine = RepairEngine()

    # ---- patch application ----
    def t_patch_apply_ok() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "X = 1\nY = 2\n")
            results = apply_patches(root, [FilePatch(
                path="a.py", old_text="X = 1", new_text="X = 99",
            )])
            assert results[0].status is _PatchApplyStatus.OK
            assert "X = 99" in (root / "a.py").read_text()

    def t_patch_anchor_missing() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "X = 1\n")
            results = apply_patches(root, [FilePatch(
                path="a.py", old_text="NOT_THERE", new_text="Y = 2",
            )])
            assert results[0].status is _PatchApplyStatus.ANCHOR_NOT_FOUND
            assert (root / "a.py").read_text() == "X = 1\n"

    def t_patch_anchor_ambiguous() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "a.py", "X = 1\nX = 1\n")
            results = apply_patches(root, [FilePatch(
                path="a.py", old_text="X = 1", new_text="X = 2",
            )])
            assert results[0].status is _PatchApplyStatus.ANCHOR_AMBIGUOUS
            assert (root / "a.py").read_text().count("X = 1") == 2

    def t_patch_file_missing() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = apply_patches(root, [FilePatch(
                path="nope.py", old_text="A", new_text="B",
            )])
            assert results[0].status is _PatchApplyStatus.FILE_MISSING

    def t_patch_noop_rejected() -> None:
        try:
            FilePatch(path="a.py", old_text="x", new_text="x")
        except ValidationError:
            return
        raise AssertionError("expected ValidationError for no-op patch")

    check("patch: applies when anchor unique", t_patch_apply_ok)
    check("patch: refuses missing anchor", t_patch_anchor_missing)
    check("patch: refuses ambiguous anchor", t_patch_anchor_ambiguous)
    check("patch: refuses missing file", t_patch_file_missing)
    def t_patch_create_missing_file() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = apply_patches(root, [FilePatch(
                path="nested/requirements.txt", old_text="", new_text="pkg-x\n",
            )])
            assert results[0].status is _PatchApplyStatus.OK
            assert (root / "nested/requirements.txt").read_text() == "pkg-x\n"

    def t_patch_rejects_path_escape() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            outside = Path(td).parent / (Path(td).name + "_escape.txt")
            try:
                results = apply_patches(root, [FilePatch(
                    path="../escape.txt", old_text="", new_text="owned\n",
                )])
                assert results[0].status is _PatchApplyStatus.FILE_MISSING
                assert not outside.exists()
            finally:
                outside.unlink(missing_ok=True)

    check("patch: create missing file", t_patch_create_missing_file)
    check("patch: reject path escape", t_patch_rejects_path_escape)
    check("patch: no-op rejected at construction", t_patch_noop_rejected)

    def t_copy_tree_rejects_symlink() -> None:
        if not hasattr(os, "symlink"):
            return
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src = base / "src"
            dst = base / "dst"
            outside = base / "outside.txt"
            src.mkdir()
            outside.write_text("secret", encoding="utf-8")
            try:
                (src / "escape.txt").symlink_to(outside)
            except (OSError, NotImplementedError):
                return
            try:
                _copy_tree(src, dst)
            except OSError:
                assert not (dst / "escape.txt").exists()
            else:
                raise AssertionError("symlink source was copied")

    check("copy: rejects symlink sources", t_copy_tree_rejects_symlink)

    # ---- dependency strategy ----
    def t_dep_strategy_creates_requirements() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_dep_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/pkg/broken.py", line 1, in <module>\n'
                "    import definitely_missing_module_xyz_123\n"
                "ModuleNotFoundError: No module named "
                "'definitely_missing_module_xyz_123'\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            cands, auto, _ = engine.propose(report=report, root=root)
            assert auto is True
            assert len(cands) == 1
            c = cands[0]
            assert c.kind is RepairKind.DEPENDENCY_ADD_REQUIREMENT
            p = c.patches[0]
            assert p.path == "requirements.txt"
            assert "definitely_missing_module_xyz_123" in p.new_text

    def t_dep_strategy_appends_when_exists() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_dep_repo(root)
            _write(root, "requirements.txt", "existing_pkg\n")
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/pkg/broken.py", line 1, in <module>\n'
                "    import definitely_missing_module_xyz_123\n"
                "ModuleNotFoundError: No module named "
                "'definitely_missing_module_xyz_123'\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            cands, _, _ = engine.propose(report=report, root=root)
            assert len(cands) == 1
            p = cands[0].patches[0]
            assert "existing_pkg" in p.old_text
            assert p.new_text.endswith(
                "definitely_missing_module_xyz_123\n"
            )

    def t_dep_strategy_skips_if_already_declared() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_dep_repo(root)
            _write(root, "requirements.txt",
                   "definitely_missing_module_xyz_123\n")
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/pkg/broken.py", line 1, in <module>\n'
                "    import definitely_missing_module_xyz_123\n"
                "ModuleNotFoundError: No module named "
                "'definitely_missing_module_xyz_123'\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            cands, auto, _ = engine.propose(report=report, root=root)
            assert cands == []
            assert auto is False

    def t_dep_strategy_ignores_non_dep_category() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            assert report.category.value == "logic"
            cands, auto, reason = engine.propose(report=report, root=root)
            assert cands == []
            assert auto is False
            assert "logic" in reason.lower()

    check("dep strategy: creates requirements.txt",
          t_dep_strategy_creates_requirements)
    check("dep strategy: appends when file exists",
          t_dep_strategy_appends_when_exists)
    check("dep strategy: skips if already declared",
          t_dep_strategy_skips_if_already_declared)
    check("dep strategy: ignores non-DEPENDENCY categories",
          t_dep_strategy_ignores_non_dep_category)

    # ---- user-supplied ----
    def t_user_supplied_passthrough() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)

            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                    rationale="fix the subtraction to addition",
                )],
                rationale="user-provided fix",
                confidence=Confidence.HIGH,
            )
            cands, auto, _ = engine.propose(
                report=report, root=root, user_candidates=[user_c],
            )
            assert len(cands) == 1
            assert cands[0].kind is RepairKind.USER_SUPPLIED
            assert auto is False  # user-supplied is not "auto"

    check("user-supplied: passthrough + re-tag",
          t_user_supplied_passthrough)

    # ---- full run: accept real fix ----
    def t_run_user_logic_fix_accepted() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                    rationale="fix subtraction → addition",
                )],
                rationale="obvious logic fix",
                confidence=Confidence.HIGH,
            )
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_calc.py::test_add",
                user_candidates=[user_c],
            )
            assert result.accepted_id is not None
            ev = result.evaluations[0]
            assert ev.verdict is EvaluationVerdict.ACCEPTED
            assert ev.target_status_before == "failed"
            assert ev.target_status_after == "passed"
            assert ev.regression_count == 0

    def t_run_regression_rejected() -> None:
        """A 'fix' that makes the target pass but breaks another test
        MUST be rejected."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            # Sloppy fix: change both functions to `a + b`, breaking sub()
            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[
                    FilePatch(
                        path="pkg/calc.py",
                        old_text=("    return a - b   # wrong on purpose\n"
                                  "\n"
                                  "def sub"),
                        new_text=("    return a + b\n"
                                  "\n"
                                  "def sub"),
                    ),
                    FilePatch(
                        path="pkg/calc.py",
                        old_text=("def sub(a: int, b: int) -> int:\n"
                                  "    return a - b"),
                        new_text=("def sub(a: int, b: int) -> int:\n"
                                  "    return a + b"),
                    ),
                ],
                rationale="change both to +",
                confidence=Confidence.LOW,
            )
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_calc.py::test_add",
                user_candidates=[user_c],
            )
            assert result.accepted_id is None
            ev = result.evaluations[0]
            assert ev.verdict is EvaluationVerdict.REJECTED_REGRESSION
            assert ev.regression_count >= 1

    def t_run_patch_failure_rejected() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="ANCHOR_NOT_PRESENT",
                    new_text="X",
                )],
            )
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_calc.py::test_add",
                user_candidates=[user_c],
            )
            ev = result.evaluations[0]
            assert ev.verdict is EvaluationVerdict.REJECTED_PATCH_FAILED

    def t_run_no_target_rejected() -> None:
        """Even a working fix must be rejected if no target is provided:
        we cannot verify the specific failure was addressed."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                )],
            )
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="",     # no target
                user_candidates=[user_c],
            )
            ev = result.evaluations[0]
            assert ev.verdict is EvaluationVerdict.REJECTED_NO_TARGET
            assert result.accepted_id is None

    def t_run_original_untouched() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            before = (root / "pkg" / "calc.py").read_text()
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                )],
            )
            engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_calc.py::test_add",
                user_candidates=[user_c],
            )
            after = (root / "pkg" / "calc.py").read_text()
            assert before == after, "original root was mutated!"

    check("run: user logic fix accepted (target passes, no regressions)",
          t_run_user_logic_fix_accepted, requires_pytest=True)
    check("run: repair causing regression → REJECTED_REGRESSION",
          t_run_regression_rejected, requires_pytest=True)
    check("run: patch failure → REJECTED_PATCH_FAILED",
          t_run_patch_failure_rejected)
    check("run: no target → REJECTED_NO_TARGET (never vague accept)",
          t_run_no_target_rejected)
    check("run: original root is never mutated",
          t_run_original_untouched)

    # ---- ranking ----
    def t_ranking_prefers_smallest() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            c_small = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                )],
                rationale="small",
            )
            c_big = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text=(
                        "def add(a: int, b: int) -> int:\n"
                        "    return a - b   # wrong on purpose\n"
                    ),
                    new_text=(
                        "def add(a: int, b: int) -> int:\n"
                        '    """Return a + b."""\n'
                        "    # fix applied\n"
                        "    return a + b\n"
                    ),
                )],
                rationale="rewrite",
            )
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_calc.py::test_add",
                user_candidates=[c_big, c_small],
            )
            assert result.accepted_id == c_small.id

    check("ranking: smallest accepted candidate wins",
          t_ranking_prefers_smallest, requires_pytest=True)

    # ---- non-repairable path ----
    def t_non_repairable_logic() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            result = engine.run(
                report=report, root=root, project_id="p",
            )
            assert result.auto_repairable is False
            assert result.non_repairable_reason
            assert result.candidates == []

    check("non-repairable: LOGIC → no auto candidates",
          t_non_repairable_logic)

    # ---- config strategy (skips when not KeyError, proposes otherwise) ----
    def t_config_strategy_proposes_get() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root, "pyproject.toml", "[project]\nname='x'\n")
            _write(root, "pkg/__init__.py", '"""pkg."""\n')
            _write(root, "pkg/settings.py", (
                "def get_db_url(cfg: dict) -> str:\n"
                "    return cfg['DATABASE_URL']\n"
            ))
            _write(root, "tests/__init__.py", "")
            _write(root, "tests/test_settings.py", (
                "from pkg.settings import get_db_url\n"
                "\n"
                "def test_get_db_url_no_key() -> None:\n"
                "    # test expects missing key → 'unknown'\n"
                "    # so the fix should use .get()\n"
                "    assert get_db_url({}) == 'unknown'\n"
            ))
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/pkg/settings.py", line 2, '
                f'in get_db_url\n'
                "    return cfg['DATABASE_URL']\n"
                "KeyError: 'DATABASE_URL'\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            assert report.category.value == "config", report.category
            cands, auto, _ = engine.propose(report=report, root=root)
            assert auto is True
            assert len(cands) == 1
            c = cands[0]
            assert c.kind is RepairKind.CONFIG_USE_GET
            p = c.patches[0]
            assert p.path == "pkg/settings.py"
            assert p.old_text == "cfg['DATABASE_URL']"
            assert p.new_text == "cfg.get('DATABASE_URL')"

    check("config strategy: proposes .get() patch",
          t_config_strategy_proposes_get)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/tests/test_calc.py", line 4, in test_add\n'
                "    assert add(1, 2) == 3\n"
                "AssertionError: assert -1 == 3\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                )],
            )
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_calc.py::test_add",
                user_candidates=[user_c],
            )
            d = result.to_dict()
            assert d["id"] == result.id
            assert isinstance(d["candidates"], list)
            assert isinstance(d["evaluations"], list)
            s = result.summary()
            assert "Repair Result" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                with tempfile.TemporaryDirectory() as ptd:
                    root = Path(ptd)
                    _mk_logic_repo(root)
                    from sebrain.c19 import Debugger as _Dbg
                    tb = (
                        "Traceback (most recent call last):\n"
                        f'  File "{root}/tests/test_calc.py", line 4, '
                        f'in test_add\n'
                        "    assert add(1, 2) == 3\n"
                        "AssertionError: assert -1 == 3\n"
                    )
                    report = _Dbg(root=root).analyze_text(tb)
                    user_c = RepairCandidate(
                        kind=RepairKind.USER_SUPPLIED,
                        patches=[FilePatch(
                            path="pkg/calc.py",
                            old_text=("    return a - b   "
                                      "# wrong on purpose"),
                            new_text="    return a + b",
                        )],
                    )
                    result = engine.run(
                        report=report, root=root, project_id="proj-x",
                        target_nodeid="tests/test_calc.py::test_add",
                        user_candidates=[user_c],
                    )
                    repo = RepairRepository(memory=mem, ontology=ont)
                    ent = repo.save(result, project_id="proj-x")
                    assert ent
                    loaded = repo.load(result.id, project_id="proj-x")
                    assert loaded is not None
                    assert loaded["accepted_id"] == result.accepted_id
                    assert ont.count(kind=EntityKind.FIX) >= 1
                    assert ont.count(kind=EntityKind.ALTERNATIVE) >= 1
                    decs = mem.find(
                        kind=MemoryKind.DECISION,
                        scope_type=MemoryScope.PROJECT,
                        scope_id="proj-x",
                    )
                    assert any(d.key.startswith("repair:") for d in decs)
            finally:
                s.shutdown()

    check("persist: memory + ontology FIX + ALTERNATIVE",
          t_persist, requires_pytest=True)

    # ---- E2E with C19 ----
    def t_e2e_from_c19_report() -> None:
        from sebrain.c18 import TestExecutionEngine, RunStatus
        from sebrain.c19 import Debugger as _Dbg

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_logic_repo(root)
            run = TestExecutionEngine().run_full(root=root, project_id="demo")
            assert run.status is RunStatus.TESTS_FAILED
            failed = next(r for r in run.results
                          if r.outcome.value == "failed")

            report = _Dbg(root=root).analyze_test_result(failed)
            assert report.category.value == "logic"

            user_c = RepairCandidate(
                kind=RepairKind.USER_SUPPLIED,
                patches=[FilePatch(
                    path="pkg/calc.py",
                    old_text="    return a - b   # wrong on purpose",
                    new_text="    return a + b",
                )],
            )
            result = engine.run(
                report=report, root=root, project_id="demo",
                target_nodeid=failed.nodeid,
                user_candidates=[user_c],
            )
            assert result.accepted_id == user_c.id
            ev = result.evaluations[0]
            assert ev.verdict is EvaluationVerdict.ACCEPTED
            assert ev.target_status_before == "failed"
            assert ev.target_status_after == "passed"

    check("e2e: C18 failure → C19 diagnosis → C20 accepted repair",
          t_e2e_from_c19_report, requires_pytest=True)

    # ---- honest rejection of dep fix (doesn't actually fix tests) ----
    def t_dep_fix_is_honestly_rejected() -> None:
        """Appending to requirements.txt doesn't make the failing test
        pass — C20 must NOT accept it just because it's a 'fix'."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_dep_repo(root)
            from sebrain.c19 import Debugger as _Dbg
            tb = (
                "Traceback (most recent call last):\n"
                f'  File "{root}/pkg/broken.py", line 1, in <module>\n'
                "    import definitely_missing_module_xyz_123\n"
                "ModuleNotFoundError: No module named "
                "'definitely_missing_module_xyz_123'\n"
            )
            report = _Dbg(root=root).analyze_text(tb)
            result = engine.run(
                report=report, root=root, project_id="p",
                target_nodeid="tests/test_broken.py::test_f",
            )
            assert len(result.candidates) == 1
            ev = result.evaluations[0]
            # The target test still fails (module still not importable)
            assert ev.verdict is EvaluationVerdict.REJECTED_TARGET_NOT_FIXED
            assert result.accepted_id is None

    check("honest: dep fix rejected because target still fails",
          t_dep_fix_is_honestly_rejected, requires_pytest=True)

    print()
    skip_note = f", {skipped} skipped (pytest not installed)" if skipped else ""
    print(f"Self-tests: {passed} passed, {len(failures)} failed{skip_note}")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 9. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    print("=" * 78)
    print("SE Brain C20 — Code Repair Engine")
    print("=" * 78)

    from sebrain.c18 import TestExecutionEngine
    from sebrain.c19 import Debugger as _Dbg

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_repo"
        root.mkdir()
        _mk_logic_repo(root)

        test_engine = TestExecutionEngine()
        dbg = _Dbg(root=root)
        engine = RepairEngine()

        print("\n[1] Run the (broken) test suite:")
        run = test_engine.run_full(root=root, project_id="demo")
        print(f"    status={run.status.value}  passed={run.passed}  "
              f"failed={run.failed}")
        failed = next(r for r in run.results if r.outcome.value == "failed")
        print(f"    failing node: {failed.nodeid}")

        print("\n[2] Diagnose with C19:")
        report = dbg.analyze_test_result(failed)
        print(report.summary())
        for c in report.candidates[:2]:
            print(f"    candidate[{c.category.value}]: "
                  f"{_short(c.description, 80)}")
            print(f"        probe: {_short(c.suggested_probe, 90)}")

        print("\n[3] Auto-propose (built-in strategies):")
        cands, auto, reason = engine.propose(report=report, root=root)
        print(f"    candidates={len(cands)}  auto={auto}")
        print(f"    reason: {reason}")

        print("\n[4] Supply a manual fix and evaluate:")
        user_c = RepairCandidate(
            kind=RepairKind.USER_SUPPLIED,
            patches=[FilePatch(
                path="pkg/calc.py",
                old_text="    return a - b   # wrong on purpose",
                new_text="    return a + b",
                rationale="fix the operator",
            )],
            rationale="obvious logic fix",
            confidence=Confidence.HIGH,
        )
        result = engine.run(
            report=report, root=root, project_id="demo",
            target_nodeid=failed.nodeid,
            user_candidates=[user_c],
        )
        print(result.summary())
        for ev in result.evaluations:
            print(f"    candidate {ev.candidate_id[:8]}: {ev.verdict.value}")
            print(f"        before={ev.target_status_before} "
                  f"after={ev.target_status_after}  "
                  f"regressions={ev.regression_count}")
            print(f"        reason={ev.reason}")

        acc = result.accepted()
        if acc:
            print(f"\n[5] Accepted candidate: {acc.kind.value}  "
                  f"patches={len(acc.patches)}")
            for p in acc.patches:
                print(f"    {p.path}: "
                      f"{_short(p.old_text, 40)} → {_short(p.new_text, 40)}")

        print("\n[6] Verify caller's root is untouched:")
        still_broken = "a - b" in (root / "pkg" / "calc.py").read_text()
        print(f"    calc.py still contains 'a - b': {still_broken}")

        print("\n[7] Regressing 'fix' demonstration:")
        _mk_logic_repo(root)
        run2 = test_engine.run_full(root=root, project_id="demo")
        failed2 = next(r for r in run2.results
                       if r.outcome.value == "failed")
        report2 = dbg.analyze_test_result(failed2)
        bad_c = RepairCandidate(
            kind=RepairKind.USER_SUPPLIED,
            patches=[
                FilePatch(
                    path="pkg/calc.py",
                    old_text=("    return a - b   # wrong on purpose\n\n"
                              "def sub"),
                    new_text=("    return a + b\n\ndef sub"),
                ),
                FilePatch(
                    path="pkg/calc.py",
                    old_text=("def sub(a: int, b: int) -> int:\n"
                              "    return a - b"),
                    new_text=("def sub(a: int, b: int) -> int:\n"
                              "    return a + b"),
                ),
            ],
        )
        result2 = engine.run(
            report=report2, root=root, project_id="demo",
            target_nodeid=failed2.nodeid,
            user_candidates=[bad_c],
        )
        if result2.evaluations:
            ev2 = result2.evaluations[0]
            print(f"    verdict={ev2.verdict.value}")
            print(f"    reason={ev2.reason}")
            print(f"    accepted={result2.accepted_id is not None}")

        # Persistence
        print("\n[8] Persistence:")
        with tempfile.TemporaryDirectory() as std:
            cfg = Config(data_dir=Path(std) / "sebrain", log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                with execution_scope(project_id="demo"):
                    mem = MemoryStore(app.storage)
                    ont = Ontology(app.storage)
                    repo = RepairRepository(memory=mem, ontology=ont)
                    ent = repo.save(result, project_id="demo")
                    print(f"    FIX entity: {ent[:12]}…")
                    print(f"    FIX count: {ont.count(kind=EntityKind.FIX)}")
                    print(f"    ALTERNATIVE count: "
                          f"{ont.count(kind=EntityKind.ALTERNATIVE)}")
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
