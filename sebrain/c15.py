"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C15 — REPOSITORY / PROJECT BUILDER (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04, C14.

Purpose:
    Write in-memory SynthesizedFile objects (from C14) into a real project
    on disk — with hard safety guarantees:

      - Path traversal protection (no '..', no absolute paths, no drive letters)
      - Symlink guards (no writing through symlinks, no symlink parents)
      - Control-character / null-byte rejection in any path segment
      - Bounded operations (max files, max total bytes)
      - Atomic per-file writes (temp + os.replace)
      - Transactional rollback: on any mid-build failure, restore or delete
      - Optional persistent backups (mode=BACKUP_AND_OVERWRITE)
      - Dry-run mode (compute changes without touching disk)

Capabilities:
    - Directories (create parents safely)
    - Files (text, UTF-8)
    - Metadata generators (pyproject.toml, requirements.txt, README.md, .gitignore)
    - plan_from_synthesis(): build a BuildPlan from a C14 SynthesisResult
    - Apply with modes: CREATE_ONLY (default), SKIP_EXISTING, OVERWRITE,
      BACKUP_AND_OVERWRITE
    - Dry-run preview
    - Rollback on failure (restores overwritten, deletes created)
    - SHA256 hashing for change auditing
    - Persistence to C04 memory + C02 ontology (EXECUTION + FILE entities)

Invariants honored:
  - NO destructive operations beyond explicit overwrite/backup
  - NO shell=True anywhere
  - Every filesystem mutation is bounded and reversible within a build
  - Pre-flight validation: bad plans fail BEFORE any write
  - Deterministic: same plan + same state → same result
  - No external LLM
  - Rollback is best-effort (documented) — failures during rollback are
    logged but do not raise

Explicit limitations:
  - Cross-file atomicity is approximated via rollback, not filesystem MVCC.
  - Mid-build failure rollback cannot undo externally-observed side effects.
  - Persistent backups are stored under <root>/.sebrain_backups/<build_id>/.

Contents:
  1.  Enums: WriteMode, ChangeKind, BuildStatus
  2.  Dataclasses: FileWrite, BuildPlan, FileChange, BuildResult, RollbackReport
  3.  Path safety helpers
  4.  Atomic write helper
  5.  Metadata generators (pyproject, requirements, README, gitignore)
  6.  plan_from_synthesis
  7.  ProjectBuilder (facade: apply / dry-run / rollback)
  8.  BuildRepository (persist)
  9.  Self-tests (~28 tests)
  10. Demo

Run as script:
    python -m sebrain.c15            # demo
    python -m sebrain.c15 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
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


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _short(s: str, n: int = 100) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class WriteMode(str, Enum):
    CREATE_ONLY = "create_only"                    # fail if exists
    SKIP_EXISTING = "skip_existing"                # leave existing alone
    OVERWRITE = "overwrite"                        # replace in place
    BACKUP_AND_OVERWRITE = "backup_and_overwrite"  # backup then replace


class ChangeKind(str, Enum):
    CREATE = "create"
    OVERWRITE = "overwrite"
    SKIP = "skip"


class BuildStatus(str, Enum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"           # some files skipped, no failures
    FAILED = "failed"             # pre-flight failure — nothing written
    ROLLED_BACK = "rolled_back"   # mid-build failure → rolled back


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(slots=True)
class FileWrite:
    """One file to create. Path is repo-relative POSIX ('/'-separated)."""
    path: str
    content: str
    kind: str = "text"
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValidationError("FileWrite.path is required")
        if self.content is None:
            raise ValidationError(f"FileWrite.content cannot be None: {self.path}")

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path, "kind": self.kind,
            "rationale": self.rationale,
            "byte_size": len(self.content.encode("utf-8")),
        }
        if include_content:
            d["content"] = self.content
        return d


@dataclass(slots=True)
class BuildPlan:
    root: str
    writes: list[FileWrite] = field(default_factory=list)
    mode: WriteMode = WriteMode.CREATE_ONLY
    dry_run: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.root:
            raise ValidationError("BuildPlan.root is required")

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "root": self.root,
            "mode": self.mode.value,
            "dry_run": self.dry_run,
            "rationale": self.rationale,
            "writes": [w.to_dict(include_content=include_content)
                       for w in self.writes],
        }


@dataclass(slots=True)
class FileChange:
    path: str
    kind: ChangeKind
    old_hash: str = ""
    new_hash: str = ""
    byte_size: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "kind": self.kind.value,
            "old_hash": self.old_hash, "new_hash": self.new_hash,
            "byte_size": self.byte_size, "reason": self.reason,
        }


@dataclass(slots=True)
class RollbackReport:
    restored_files: int = 0
    deleted_files: int = 0
    removed_dirs: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "restored_files": self.restored_files,
            "deleted_files": self.deleted_files,
            "removed_dirs": self.removed_dirs,
            "errors": list(self.errors),
        }


@dataclass(slots=True)
class BuildResult:
    id: str = field(default_factory=_new_id)
    root: str = ""
    mode: WriteMode = WriteMode.CREATE_ONLY
    dry_run: bool = False
    status: BuildStatus = BuildStatus.SUCCEEDED
    changes: list[FileChange] = field(default_factory=list)
    applied: int = 0
    skipped: int = 0
    failed: list[dict[str, Any]] = field(default_factory=list)
    rollback: RollbackReport | None = None
    backup_dir: str = ""
    duration_seconds: float = 0.0
    rationale: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "root": self.root,
            "mode": self.mode.value, "dry_run": self.dry_run,
            "status": self.status.value,
            "changes": [c.to_dict() for c in self.changes],
            "applied": self.applied, "skipped": self.skipped,
            "failed": list(self.failed),
            "rollback": self.rollback.to_dict() if self.rollback else None,
            "backup_dir": self.backup_dir,
            "duration_seconds": self.duration_seconds,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
            "created_at": self.created_at,
        }

    def summary(self) -> str:
        return (
            "=== Build Result ===\n"
            f"status={self.status.value}  mode={self.mode.value}  "
            f"dry_run={self.dry_run}\n"
            f"root={self.root}\n"
            f"applied={self.applied}  skipped={self.skipped}  "
            f"failed={len(self.failed)}\n"
            f"changes={len(self.changes)}  "
            f"duration={self.duration_seconds*1000:.1f}ms"
            + (f"\nbackup_dir={self.backup_dir}" if self.backup_dir else "")
            + (f"\nrollback={self.rollback.to_dict()}" if self.rollback else "")
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. PATH SAFETY HELPERS
# ════════════════════════════════════════════════════════════════════════════
_MAX_PATH_SEGMENT = 255
_RESERVED_WINDOWS_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5",
    "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5",
    "lpt6", "lpt7", "lpt8", "lpt9",
})


def validate_rel_path(rel: str) -> tuple[str, ...]:
    """Return a validated tuple of path segments or raise ValidationError.

    Rules:
      - Non-empty, no leading/trailing slashes
      - Forward-slash separated (backslashes normalized)
      - No absolute paths, no drive letters
      - No '.' or '..' segments
      - No null bytes or ASCII control chars
      - Segment length <= 255
      - No reserved Windows device names (case-insensitive)
    """
    if not isinstance(rel, str) or not rel:
        raise ValidationError("path must be a non-empty string")
    rel = rel.replace("\\", "/")
    if rel.startswith("/"):
        raise ValidationError(f"absolute path not allowed: {rel!r}")
    if len(rel) >= 2 and rel[1] == ":":
        raise ValidationError(f"drive-letter path not allowed: {rel!r}")
    pp = PurePosixPath(rel)
    parts = pp.parts
    if not parts:
        raise ValidationError(f"no path segments in {rel!r}")
    for p in parts:
        if p in (".", ".."):
            raise ValidationError(f"path traversal not allowed: {rel!r}")
        if "\0" in p:
            raise ValidationError(f"null byte in path segment: {rel!r}")
        if any(ord(c) < 32 for c in p):
            raise ValidationError(f"control char in path segment: {rel!r}")
        if len(p) > _MAX_PATH_SEGMENT:
            raise ValidationError(
                f"segment too long ({len(p)}>{_MAX_PATH_SEGMENT}): {p!r}"
            )
        # Windows reserved names (only if it's the basename; harmless to check all)
        stem = p.split(".", 1)[0].lower()
        if stem in _RESERVED_WINDOWS_NAMES:
            raise ValidationError(f"reserved name not allowed: {p!r}")
    return tuple(parts)


def safe_target(root_real: Path, rel: str) -> Path:
    """Compose root/rel after validating, and reject any symlink in the path."""
    parts = validate_rel_path(rel)
    target = root_real.joinpath(*parts)
    # Walk from target up to root; reject symlinks
    cursor = target
    while cursor != root_real and cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValidationError(f"symlink in path not allowed: {cursor}")
        cursor = cursor.parent
    return target


# ════════════════════════════════════════════════════════════════════════════
# 4. ATOMIC WRITE HELPER
# ════════════════════════════════════════════════════════════════════════════
def atomic_write_bytes(target: Path, content: bytes) -> None:
    """Write content atomically: temp file in same dir → fsync → os.replace."""
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".sebrain_c15_", suffix=".tmp", dir=str(parent),
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync may fail on some filesystems; safe to continue
                pass
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ════════════════════════════════════════════════════════════════════════════
# 5. METADATA GENERATORS
# ════════════════════════════════════════════════════════════════════════════
def pyproject_toml(
    name: str, *,
    version: str = "0.1.0",
    description: str = "",
    requires_python: str = ">=3.11",
    dependencies: Sequence[str] | None = None,
) -> str:
    deps = list(dependencies or [])
    lines = [
        "[build-system]",
        'requires = ["setuptools>=68", "wheel"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f'name = "{name}"',
        f'version = "{version}"',
    ]
    if description:
        lines.append(f'description = "{description}"')
    lines.append(f'requires-python = "{requires_python}"')
    lines.append("dependencies = [")
    for d in deps:
        lines.append(f'    "{d}",')
    lines.append("]")
    lines.append("")
    return "\n".join(lines)


def requirements_txt(dependencies: Sequence[str]) -> str:
    return "\n".join(dependencies) + ("\n" if dependencies else "")


def readme_md(name: str, description: str = "") -> str:
    body = f"# {name}\n\n"
    if description:
        body += f"{description}\n\n"
    body += "## Installation\n\n```bash\npip install -e .\n```\n\n"
    body += "## Usage\n\nSee the source under the package directory.\n"
    return body


def gitignore_python() -> str:
    return (
        "__pycache__/\n"
        "*.py[cod]\n"
        "*.egg-info/\n"
        ".eggs/\n"
        "build/\n"
        "dist/\n"
        ".venv/\n"
        "venv/\n"
        ".pytest_cache/\n"
        ".mypy_cache/\n"
        ".ruff_cache/\n"
        ".sebrain_backups/\n"
        "*.sqlite3\n"
        "*.db\n"
        ".env\n"
    )


# ════════════════════════════════════════════════════════════════════════════
# 6. PLAN FROM SYNTHESIS (C14 → C15 bridge)
# ════════════════════════════════════════════════════════════════════════════
def plan_from_synthesis(
    synthesis: Any,
    *,
    root: str | Path,
    mode: WriteMode = WriteMode.CREATE_ONLY,
    dry_run: bool = False,
    include_metadata: bool = True,
    project_name: str | None = None,
    description: str = "",
    extra_files: dict[str, str] | None = None,
    dependencies: Sequence[str] | None = None,
) -> BuildPlan:
    """Build a BuildPlan from a C14 SynthesisResult (duck-typed).

    Accepts any object with `.files` where each file has `.path`, `.content`,
    `.kind`, `.rationale` (i.e., C14 SynthesizedFile).
    """
    writes: list[FileWrite] = []
    files = getattr(synthesis, "files", None)
    if not files:
        raise ValidationError("synthesis has no files to build")

    for f in files:
        writes.append(FileWrite(
            path=str(f.path),
            content=str(f.content),
            kind=str(getattr(f, "kind", "text")),
            rationale=str(getattr(f, "rationale", "")),
        ))

    if include_metadata:
        name = project_name or _guess_project_name(files)
        writes.append(FileWrite(
            path="pyproject.toml",
            content=pyproject_toml(
                name, description=description, dependencies=dependencies,
            ),
            kind="metadata",
            rationale="project metadata (PEP 621)",
        ))
        if dependencies:
            writes.append(FileWrite(
                path="requirements.txt",
                content=requirements_txt(dependencies),
                kind="metadata",
                rationale="runtime dependencies pin (informational)",
            ))
        writes.append(FileWrite(
            path="README.md",
            content=readme_md(name, description=description),
            kind="documentation",
            rationale="project readme",
        ))
        writes.append(FileWrite(
            path=".gitignore",
            content=gitignore_python(),
            kind="metadata",
            rationale="standard Python ignores",
        ))

    if extra_files:
        for p, content in extra_files.items():
            writes.append(FileWrite(
                path=p, content=content, kind="extra",
                rationale="extra file supplied by caller",
            ))

    return BuildPlan(
        root=str(root), writes=writes, mode=mode, dry_run=dry_run,
        rationale=f"plan from synthesis ({len(writes)} writes)",
    )


def _guess_project_name(files: Sequence[Any]) -> str:
    """Pick a sensible project name from top-level package dir or 'project'."""
    for f in files:
        p = str(getattr(f, "path", ""))
        if "/" in p:
            head = p.split("/", 1)[0]
            if head and head != "tests":
                return head
    return "project"


# ════════════════════════════════════════════════════════════════════════════
# 7. PROJECT BUILDER (facade)
# ════════════════════════════════════════════════════════════════════════════
class ProjectBuilder:
    """Apply BuildPlans to disk with hard safety + rollback.

    Two phases:
      Phase A — PRE-FLIGHT: validate root, bounds, all paths, all targets,
                compute per-file ChangeKind. No filesystem mutations.
      Phase B — APPLY: create dirs, atomic file writes, track rollback state.
                On any failure, restore/deletes and report ROLLED_BACK.
    """

    def __init__(
        self,
        *,
        max_files: int = 2000,
        max_total_bytes: int = 20_000_000,
    ) -> None:
        if max_files < 1:
            raise ValidationError("max_files must be >= 1")
        if max_total_bytes < 1:
            raise ValidationError("max_total_bytes must be >= 1")
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes

    # ---- public API ----
    def apply(self, plan: BuildPlan) -> BuildResult:
        t0 = time.monotonic()
        result = BuildResult(
            root=plan.root, mode=plan.mode, dry_run=plan.dry_run,
            provenance=Provenance(
                source="project_builder", source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        # --- resolve root ---
        try:
            root = Path(plan.root).resolve()
        except (OSError, RuntimeError) as exc:
            result.status = BuildStatus.FAILED
            result.failed.append({"stage": "root", "error": f"resolve failed: {exc}"})
            result.duration_seconds = time.monotonic() - t0
            result.rationale = "root could not be resolved"
            return result
        if not root.exists():
            result.status = BuildStatus.FAILED
            result.failed.append({"stage": "root", "error": f"root does not exist: {root}"})
            result.duration_seconds = time.monotonic() - t0
            result.rationale = "root missing"
            return result
        if not root.is_dir():
            result.status = BuildStatus.FAILED
            result.failed.append({"stage": "root", "error": f"root is not a directory: {root}"})
            result.duration_seconds = time.monotonic() - t0
            result.rationale = "root not a directory"
            return result

        # --- bounds ---
        if len(plan.writes) > self.max_files:
            result.status = BuildStatus.FAILED
            result.failed.append({
                "stage": "bounds",
                "error": f"{len(plan.writes)} writes > max_files={self.max_files}",
            })
            result.duration_seconds = time.monotonic() - t0
            result.rationale = "too many files"
            return result
        total_bytes = sum(len(w.content.encode("utf-8")) for w in plan.writes)
        if total_bytes > self.max_total_bytes:
            result.status = BuildStatus.FAILED
            result.failed.append({
                "stage": "bounds",
                "error": f"{total_bytes}B > max_total_bytes={self.max_total_bytes}",
            })
            result.duration_seconds = time.monotonic() - t0
            result.rationale = "total bytes exceed limit"
            return result

        # --- pre-flight ---
        validated: list[tuple[FileWrite, Path, ChangeKind, str]] = []
        for w in plan.writes:
            try:
                target = safe_target(root, w.path)
            except ValidationError as exc:
                result.status = BuildStatus.FAILED
                result.failed.append({
                    "stage": "preflight", "path": w.path,
                    "error": f"unsafe path: {exc}",
                })
                result.duration_seconds = time.monotonic() - t0
                result.rationale = "path validation failed"
                return result
            # Determine change
            try:
                exists = target.exists() or target.is_symlink()
            except OSError as exc:
                result.status = BuildStatus.FAILED
                result.failed.append({
                    "stage": "preflight", "path": w.path,
                    "error": f"stat failed: {exc}",
                })
                result.duration_seconds = time.monotonic() - t0
                result.rationale = "stat failure"
                return result
            if exists:
                if target.is_symlink():
                    result.status = BuildStatus.FAILED
                    result.failed.append({
                        "stage": "preflight", "path": w.path,
                        "error": "target is a symlink",
                    })
                    result.duration_seconds = time.monotonic() - t0
                    result.rationale = "symlink target rejected"
                    return result
                if target.is_dir():
                    result.status = BuildStatus.FAILED
                    result.failed.append({
                        "stage": "preflight", "path": w.path,
                        "error": "target is a directory",
                    })
                    result.duration_seconds = time.monotonic() - t0
                    result.rationale = "target is a directory"
                    return result
                if plan.mode == WriteMode.CREATE_ONLY:
                    result.status = BuildStatus.FAILED
                    result.failed.append({
                        "stage": "preflight", "path": w.path,
                        "error": "target exists (mode=create_only)",
                    })
                    result.duration_seconds = time.monotonic() - t0
                    result.rationale = "create_only refuses existing target"
                    return result
                if plan.mode == WriteMode.SKIP_EXISTING:
                    change = ChangeKind.SKIP
                else:  # OVERWRITE or BACKUP_AND_OVERWRITE
                    change = ChangeKind.OVERWRITE
            else:
                change = ChangeKind.CREATE
            validated.append((w, target, change, w.path))

        # --- dry run: report without touching disk ---
        if plan.dry_run:
            for w, target, change, rel in validated:
                content_bytes = w.content.encode("utf-8")
                if change is ChangeKind.SKIP:
                    result.changes.append(FileChange(
                        path=rel, kind=change, reason="exists (dry-run skip)",
                    ))
                    result.skipped += 1
                    continue
                old_hash = ""
                if change is ChangeKind.OVERWRITE:
                    try:
                        old_hash = _sha256(target.read_bytes())
                    except OSError:
                        old_hash = ""
                result.changes.append(FileChange(
                    path=rel, kind=change,
                    old_hash=old_hash,
                    new_hash=_sha256(content_bytes),
                    byte_size=len(content_bytes),
                    reason="dry-run: would write",
                ))
                result.applied += 1
            result.status = (
                BuildStatus.PARTIAL if result.skipped else BuildStatus.SUCCEEDED
            )
            result.rationale = "dry-run complete (no disk writes)"
            result.duration_seconds = time.monotonic() - t0
            return result

        # --- phase B: apply, tracking rollback state ---
        snapshot: dict[Path, bytes] = {}      # original bytes for overwrite
        created_files: list[Path] = []
        created_dirs: list[Path] = []
        backup_dir: Path | None = None
        if plan.mode is WriteMode.BACKUP_AND_OVERWRITE:
            backup_dir = root / ".sebrain_backups" / result.id
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                result.status = BuildStatus.FAILED
                result.failed.append({"stage": "backup", "error": str(exc)})
                result.duration_seconds = time.monotonic() - t0
                result.rationale = "could not create backup dir"
                return result
            result.backup_dir = str(backup_dir)

        for w, target, change, rel in validated:
            if change is ChangeKind.SKIP:
                result.changes.append(FileChange(
                    path=rel, kind=change, reason="exists (skip_existing)",
                ))
                result.skipped += 1
                continue
            # Prepare state
            parents_to_create: list[Path] = []
            cursor = target.parent
            while cursor != root and not cursor.exists():
                parents_to_create.append(cursor)
                cursor = cursor.parent
            try:
                old_hash = ""
                if change is ChangeKind.OVERWRITE:
                    old_bytes = target.read_bytes()
                    snapshot[target] = old_bytes
                    old_hash = _sha256(old_bytes)
                    if backup_dir is not None:
                        bp = backup_dir / Path(*validate_rel_path(rel))
                        bp.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write_bytes(bp, old_bytes)
                # Create dirs (reverse order = top-down)
                for d in reversed(parents_to_create):
                    d.mkdir(exist_ok=False)
                    created_dirs.append(d)
                # Write
                content_bytes = w.content.encode("utf-8")
                atomic_write_bytes(target, content_bytes)
                if change is ChangeKind.CREATE:
                    created_files.append(target)
                new_hash = _sha256(content_bytes)
                result.changes.append(FileChange(
                    path=rel, kind=change,
                    old_hash=old_hash, new_hash=new_hash,
                    byte_size=len(content_bytes),
                    reason=w.rationale or "written",
                ))
                result.applied += 1
            except Exception as exc:
                # ROLLBACK
                result.failed.append({
                    "stage": "apply", "path": rel,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                result.rollback = self._rollback(
                    root, snapshot, created_files, created_dirs,
                )
                result.status = BuildStatus.ROLLED_BACK
                result.rationale = (
                    f"apply failed at '{rel}'; rolled back "
                    f"{result.rollback.to_dict()}"
                )
                result.duration_seconds = time.monotonic() - t0
                return result

        # --- final status ---
        if result.skipped:
            result.status = BuildStatus.PARTIAL
        else:
            result.status = BuildStatus.SUCCEEDED
        result.rationale = (
            f"applied={result.applied}  skipped={result.skipped}  "
            f"mode={plan.mode.value}  files={len(plan.writes)}"
        )
        result.duration_seconds = time.monotonic() - t0
        return result

    # ---- rollback ----
    def _rollback(
        self, root: Path,
        snapshot: dict[Path, bytes],
        created_files: list[Path],
        created_dirs: list[Path],
    ) -> RollbackReport:
        report = RollbackReport()
        # 1. Restore overwritten files (reverse order)
        for path, content in reversed(list(snapshot.items())):
            try:
                atomic_write_bytes(path, content)
                report.restored_files += 1
            except Exception as exc:
                report.errors.append(
                    f"restore failed for {path}: {type(exc).__name__}: {exc}"
                )
        # 2. Delete files we created
        for path in reversed(created_files):
            try:
                if path.exists():
                    path.unlink()
                    report.deleted_files += 1
            except Exception as exc:
                report.errors.append(
                    f"delete failed for {path}: {type(exc).__name__}: {exc}"
                )
        # 3. Remove created dirs (deepest first; only if empty)
        for d in sorted(created_dirs, key=lambda p: len(p.parts), reverse=True):
            try:
                if d.exists() and d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
                    report.removed_dirs += 1
            except Exception as exc:
                report.errors.append(
                    f"rmdir failed for {d}: {type(exc).__name__}: {exc}"
                )
        return report


# ════════════════════════════════════════════════════════════════════════════
# 8. BUILD REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class BuildRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: BuildResult, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"build:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["build", "c15", result.status.value],
            provenance=result.provenance,
        )
        # Record failures for C33
        if result.status in (BuildStatus.FAILED, BuildStatus.ROLLED_BACK):
            for fail in result.failed:
                self.memory.record_failure(
                    f"build_fail:{result.id}:{fail.get('path', fail.get('stage', '?'))}",
                    what=f"build {result.id[:8]} failed",
                    root_cause=str(fail.get("error", "unknown")),
                    fix=None,
                    scope_id=project_id,
                    provenance=Provenance(
                        source="project_builder", source_type=ProvenanceType.SYSTEM,
                        confidence=Confidence.HIGH,
                    ),
                    confidence=Confidence.HIGH,
                )
        if self.ontology is None:
            return key
        ent = self.ontology.add(
            EntityKind.EXECUTION,
            _short(
                f"Build {result.id[:8]} ({result.status.value}, "
                f"{result.applied} applied)", 120,
            ),
            attributes={
                "build_id": result.id,
                "project_id": project_id,
                "root": result.root,
                "status": result.status.value,
                "mode": result.mode.value,
                "dry_run": result.dry_run,
                "applied": result.applied,
                "skipped": result.skipped,
                "failed": len(result.failed),
                "changes": [c.to_dict() for c in result.changes],
            },
            tags=["build", result.status.value],
            provenance=result.provenance,
        )
        for c in result.changes:
            if c.kind in (ChangeKind.CREATE, ChangeKind.OVERWRITE):
                fe = self.ontology.add(
                    EntityKind.FILE, _short(c.path, 120),
                    attributes={
                        "kind": c.kind.value,
                        "hash": c.new_hash,
                        "byte_size": c.byte_size,
                    },
                    tags=["built-file"],
                    provenance=result.provenance,
                )
                try:
                    self.ontology.link(RelationKind.PRODUCES, ent.id, fe.id)
                except ValidationError:
                    pass
        return ent.id

    def load(self, build_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"build:{build_id}",
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

    print("Running C15 self-tests…")
    builder = ProjectBuilder()

    def _root():
        td = tempfile.TemporaryDirectory()
        return td, Path(td.name)

    # ---- path validation ----
    def t_reject_absolute() -> None:
        try:
            validate_rel_path("/etc/passwd")
        except ValidationError:
            return
        raise AssertionError("expected rejection")

    def t_reject_dotdot() -> None:
        for p in ("../etc/passwd", "a/../../b", "a/../b"):
            try:
                validate_rel_path(p)
            except ValidationError:
                continue
            raise AssertionError(f"expected rejection for {p}")

    def t_reject_drive_letter() -> None:
        try:
            validate_rel_path("C:/Windows/System32")
        except ValidationError:
            return
        raise AssertionError("expected rejection")

    def t_reject_null_and_control() -> None:
        for p in ("a\0b.py", "a\x01b.py", "file\x1f.py"):
            try:
                validate_rel_path(p)
            except ValidationError:
                continue
            raise AssertionError(f"expected rejection for {p!r}")

    def t_reject_reserved_windows() -> None:
        for p in ("CON.py", "aux.txt", "NUL", "com1.py"):
            try:
                validate_rel_path(p)
            except ValidationError:
                continue
            raise AssertionError(f"expected rejection for {p}")

    def t_reject_long_segment() -> None:
        try:
            validate_rel_path("a" * 300 + ".py")
        except ValidationError:
            return
        raise AssertionError("expected rejection")

    def t_accept_valid_paths() -> None:
        assert validate_rel_path("pkg/mod.py") == ("pkg", "mod.py")
        assert validate_rel_path("a/b/c/d.py") == ("a", "b", "c", "d.py")
        assert validate_rel_path("just_file.py") == ("just_file.py",)

    check("path: absolute rejected", t_reject_absolute)
    check("path: '..' traversal rejected", t_reject_dotdot)
    check("path: Windows drive-letter rejected", t_reject_drive_letter)
    check("path: null/control chars rejected", t_reject_null_and_control)
    check("path: reserved Windows names rejected", t_reject_reserved_windows)
    check("path: overlong segments rejected", t_reject_long_segment)
    check("path: valid paths accepted", t_accept_valid_paths)

    # ---- happy path ----
    def t_build_succeeds_create_only() -> None:
        td, root = _root()
        try:
            plan = BuildPlan(
                root=str(root),
                mode=WriteMode.CREATE_ONLY,
                writes=[
                    FileWrite("pkg/__init__.py", '"""pkg."""\n'),
                    FileWrite("pkg/mod.py", "X = 1\n"),
                    FileWrite("tests/test_mod.py", "def test_x(): assert True\n"),
                ],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.SUCCEEDED, res.rationale
            assert res.applied == 3
            assert (root / "pkg" / "__init__.py").exists()
            assert (root / "pkg" / "mod.py").read_text() == "X = 1\n"
            assert (root / "tests" / "test_mod.py").exists()
        finally:
            td.cleanup()

    def t_create_only_refuses_existing() -> None:
        td, root = _root()
        try:
            (root / "pkg").mkdir()
            (root / "pkg" / "mod.py").write_text("OLD\n")
            plan = BuildPlan(
                root=str(root), mode=WriteMode.CREATE_ONLY,
                writes=[FileWrite("pkg/mod.py", "NEW\n")],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.FAILED
            assert any("create_only" in str(f.get("error", ""))
                       for f in res.failed)
            # unchanged
            assert (root / "pkg" / "mod.py").read_text() == "OLD\n"
        finally:
            td.cleanup()

    def t_overwrite_replaces() -> None:
        td, root = _root()
        try:
            (root / "mod.py").write_text("OLD\n")
            plan = BuildPlan(
                root=str(root), mode=WriteMode.OVERWRITE,
                writes=[FileWrite("mod.py", "NEW\n")],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.SUCCEEDED, res.rationale
            assert res.applied == 1
            assert (root / "mod.py").read_text() == "NEW\n"
            # old_hash reported
            ch = next(c for c in res.changes if c.path == "mod.py")
            assert ch.old_hash != ""
            assert ch.new_hash != ch.old_hash
        finally:
            td.cleanup()

    def t_skip_existing_leaves_target() -> None:
        td, root = _root()
        try:
            (root / "mod.py").write_text("OLD\n")
            plan = BuildPlan(
                root=str(root), mode=WriteMode.SKIP_EXISTING,
                writes=[FileWrite("mod.py", "NEW\n")],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.PARTIAL
            assert res.skipped == 1
            assert res.applied == 0
            assert (root / "mod.py").read_text() == "OLD\n"
        finally:
            td.cleanup()

    def t_backup_and_overwrite() -> None:
        td, root = _root()
        try:
            (root / "mod.py").write_text("ORIGINAL\n")
            plan = BuildPlan(
                root=str(root), mode=WriteMode.BACKUP_AND_OVERWRITE,
                writes=[FileWrite("mod.py", "NEW\n")],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.SUCCEEDED, res.rationale
            assert res.backup_dir != ""
            assert (root / "mod.py").read_text() == "NEW\n"
            # backup contains original
            backups = list((Path(res.backup_dir)).rglob("mod.py"))
            assert backups
            assert backups[0].read_text() == "ORIGINAL\n"
        finally:
            td.cleanup()

    check("build: happy path (create_only, 3 files)", t_build_succeeds_create_only)
    check("build: create_only refuses existing target",
          t_create_only_refuses_existing)
    check("build: overwrite replaces + reports hashes", t_overwrite_replaces)
    check("build: skip_existing leaves target untouched",
          t_skip_existing_leaves_target)
    check("build: backup_and_overwrite preserves original",
          t_backup_and_overwrite)

    # ---- dry-run ----
    def t_dry_run_no_writes() -> None:
        td, root = _root()
        try:
            plan = BuildPlan(
                root=str(root), mode=WriteMode.CREATE_ONLY, dry_run=True,
                writes=[FileWrite("pkg/mod.py", "X = 1\n")],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.SUCCEEDED
            assert res.applied == 1
            # Nothing actually written
            assert not (root / "pkg").exists()
            assert not (root / "pkg" / "mod.py").exists()
        finally:
            td.cleanup()

    check("dry-run: reports changes but writes nothing", t_dry_run_no_writes)

    # ---- bounds ----
    def t_max_files_bound() -> None:
        td, root = _root()
        try:
            b = ProjectBuilder(max_files=2)
            plan = BuildPlan(root=str(root), mode=WriteMode.CREATE_ONLY, writes=[
                FileWrite(f"f{i}.py", "x\n") for i in range(3)
            ])
            res = b.apply(plan)
            assert res.status is BuildStatus.FAILED
            assert any("max_files" in str(f.get("error", "")) for f in res.failed)
        finally:
            td.cleanup()

    def t_max_bytes_bound() -> None:
        td, root = _root()
        try:
            b = ProjectBuilder(max_total_bytes=10)
            plan = BuildPlan(root=str(root), mode=WriteMode.CREATE_ONLY, writes=[
                FileWrite("big.py", "x" * 100),
            ])
            res = b.apply(plan)
            assert res.status is BuildStatus.FAILED
            assert any("max_total_bytes" in str(f.get("error", ""))
                       for f in res.failed)
        finally:
            td.cleanup()

    check("bounds: max_files enforced", t_max_files_bound)
    check("bounds: max_total_bytes enforced", t_max_bytes_bound)

    # ---- symlink guards ----
    def t_reject_symlink_parent() -> None:
        td, root = _root()
        try:
            outside = Path(td.name).parent / f"c15_out_{uuid.uuid4().hex[:6]}"
            outside.mkdir()
            try:
                (root / "linkdir").symlink_to(outside)
                plan = BuildPlan(root=str(root), mode=WriteMode.CREATE_ONLY, writes=[
                    FileWrite("linkdir/evil.py", "boom\n"),
                ])
                res = builder.apply(plan)
                assert res.status is BuildStatus.FAILED
                assert any("symlink" in str(f.get("error", ""))
                           for f in res.failed)
                # outside dir untouched
                assert not (outside / "evil.py").exists()
            finally:
                # clean up
                try:
                    (root / "linkdir").unlink()
                except OSError:
                    pass
                try:
                    outside.rmdir()
                except OSError:
                    pass
        finally:
            td.cleanup()

    def t_reject_symlink_target() -> None:
        td, root = _root()
        try:
            outside = Path(td.name).parent / f"c15_target_{uuid.uuid4().hex[:6]}"
            outside.write_text("SECRET\n")
            try:
                (root / "mod.py").symlink_to(outside)
                plan = BuildPlan(
                    root=str(root), mode=WriteMode.OVERWRITE,
                    writes=[FileWrite("mod.py", "NEW\n")],
                )
                res = builder.apply(plan)
                assert res.status is BuildStatus.FAILED
                assert any("symlink" in str(f.get("error", ""))
                           for f in res.failed)
                # external file untouched
                assert outside.read_text() == "SECRET\n"
            finally:
                try:
                    (root / "mod.py").unlink()
                except OSError:
                    pass
                try:
                    outside.unlink()
                except OSError:
                    pass
        finally:
            td.cleanup()

    check("symlink: parent symlink rejected", t_reject_symlink_parent)
    check("symlink: target symlink rejected", t_reject_symlink_target)

    # ---- rollback ----
    def t_rollback_removes_created_on_failure() -> None:
        td, root = _root()
        try:
            # Create a regular file that will block our second write's mkdir
            (root / "blocker").write_text("i am a file\n")
            plan = BuildPlan(
                root=str(root), mode=WriteMode.CREATE_ONLY,
                writes=[
                    FileWrite("pkg/ok.py", "OK = 1\n"),
                    # This path attempts to use `blocker` as a directory
                    FileWrite("blocker/child.py", "X = 1\n"),
                ],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.ROLLED_BACK, res.rationale
            assert res.rollback is not None
            # The first file must be gone
            assert not (root / "pkg" / "ok.py").exists()
            # The pkg dir must be gone too
            assert not (root / "pkg").exists()
            # The blocker file must still be there
            assert (root / "blocker").read_text() == "i am a file\n"
        finally:
            td.cleanup()

    def t_rollback_restores_overwritten_on_failure() -> None:
        td, root = _root()
        try:
            (root / "a.py").write_text("OLD_A\n")
            (root / "blocker").write_text("i am a file\n")
            plan = BuildPlan(
                root=str(root), mode=WriteMode.OVERWRITE,
                writes=[
                    FileWrite("a.py", "NEW_A\n"),
                    FileWrite("blocker/child.py", "X = 1\n"),
                ],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.ROLLED_BACK, res.rationale
            # a.py must be restored to old content
            assert (root / "a.py").read_text() == "OLD_A\n"
        finally:
            td.cleanup()

    check("rollback: created files removed on mid-build failure",
          t_rollback_removes_created_on_failure)
    check("rollback: overwritten files restored on mid-build failure",
          t_rollback_restores_overwritten_on_failure)

    # ---- metadata ----
    def t_pyproject_content() -> None:
        s = pyproject_toml(
            "task_api", description="A small API",
            dependencies=["fastapi", "uvicorn"],
        )
        assert 'name = "task_api"' in s
        assert "fastapi" in s
        assert "[build-system]" in s

    def t_requirements_content() -> None:
        s = requirements_txt(["a", "b>=1"])
        assert s.splitlines() == ["a", "b>=1"]

    def t_readme_content() -> None:
        s = readme_md("x", description="Y")
        assert "# x" in s and "Y" in s

    def t_gitignore_content() -> None:
        s = gitignore_python()
        assert "__pycache__/" in s
        assert ".sebrain_backups/" in s

    check("metadata: pyproject.toml valid shape", t_pyproject_content)
    check("metadata: requirements.txt one-per-line", t_requirements_content)
    check("metadata: README.md has title + description", t_readme_content)
    check("metadata: .gitignore includes backups dir", t_gitignore_content)

    # ---- plan_from_synthesis ----
    def t_plan_from_synthesis() -> None:
        # Build a fake synthesis (duck-typed)
        from dataclasses import dataclass

        @dataclass
        class FakeFile:
            path: str
            content: str
            kind: str = "model"
            rationale: str = ""

        @dataclass
        class FakeSynth:
            files: list

        synth = FakeSynth(files=[
            FakeFile("task_api/models.py", "class Task: pass\n"),
            FakeFile("task_api/__init__.py", '"""pkg."""\n'),
        ])
        with tempfile.TemporaryDirectory() as td:
            plan = plan_from_synthesis(
                synth, root=td, mode=WriteMode.CREATE_ONLY,
                project_name="task_api",
                description="demo",
                dependencies=["fastapi"],
            )
            paths = {w.path for w in plan.writes}
            assert "task_api/models.py" in paths
            assert "task_api/__init__.py" in paths
            assert "pyproject.toml" in paths
            assert "requirements.txt" in paths
            assert "README.md" in paths
            assert ".gitignore" in paths
            # Apply it
            res = builder.apply(plan)
            assert res.status is BuildStatus.SUCCEEDED, res.rationale
            assert (Path(td) / "task_api" / "models.py").read_text().startswith(
                "class Task"
            )
            assert (Path(td) / "pyproject.toml").exists()

    check("plan_from_synthesis: full build with metadata",
          t_plan_from_synthesis)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                plan = BuildPlan(
                    root=str(Path(td) / "target"),
                    mode=WriteMode.CREATE_ONLY,
                    writes=[FileWrite("pkg/mod.py", "X = 1\n")],
                )
                Path(plan.root).mkdir()
                res = builder.apply(plan)
                repo = BuildRepository(memory=mem, ontology=ont)
                ent = repo.save(res, project_id="proj-x")
                assert ent
                loaded = repo.load(res.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["status"] == "succeeded"
                assert ont.count(kind=EntityKind.EXECUTION) >= 1
                assert ont.count(kind=EntityKind.FILE) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology (EXECUTION + FILE)", t_persist)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        with tempfile.TemporaryDirectory() as td:
            plan = BuildPlan(
                root=td, mode=WriteMode.CREATE_ONLY,
                writes=[FileWrite("a.py", "X = 1\n")],
            )
            res = builder.apply(plan)
            d = res.to_dict()
            assert d["status"] == "succeeded"
            assert isinstance(d["changes"], list)
            s = res.summary()
            assert "Build Result" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- e2e with real C14 synthesis ----
    def t_e2e_synthesis_to_build() -> None:
        from sebrain.c14 import (
            CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
            FileKind,
        )
        from sebrain.c13 import RepoIndex

        # 1. Synthesise
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
        assert synth.status.value == "succeeded"

        # 2. Build to disk
        with tempfile.TemporaryDirectory() as td:
            plan = plan_from_synthesis(
                synth, root=td, mode=WriteMode.CREATE_ONLY,
                project_name="task_api",
                description="Task API demo",
                dependencies=["fastapi", "uvicorn"],
            )
            res = builder.apply(plan)
            assert res.status is BuildStatus.SUCCEEDED, res.rationale

            # 3. Verify layout
            root = Path(td)
            assert (root / "task_api" / "models.py").exists()
            assert (root / "task_api" / "repository.py").exists()
            assert (root / "task_api" / "api.py").exists()
            assert (root / "task_api" / "__init__.py").exists()
            assert (root / "task_api" / "__main__.py").exists()
            assert (root / "tests" / "test_task_api.py").exists()
            assert (root / "pyproject.toml").exists()
            assert (root / "README.md").exists()
            assert (root / ".gitignore").exists()

            # 4. Verify content is real Python
            for p in ("task_api/models.py", "task_api/repository.py",
                      "task_api/api.py"):
                code = (root / p).read_text()
                compile(code, p, "exec")  # must compile

            # 5. Byte sizes reported match
            for c in res.changes:
                if c.kind.value in ("create", "overwrite"):
                    real = (root / c.path).stat().st_size
                    assert real == c.byte_size, (c.path, real, c.byte_size)

    check("e2e: C14 synthesis → C15 build lands on disk", t_e2e_synthesis_to_build)

    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# 10. DEMO
# ════════════════════════════════════════════════════════════════════════════
def _tree(root: Path, *, max_entries: int = 50) -> list[str]:
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        depth = len(rel.parts) - 1
        prefix = "  " * depth + ("└─ " if depth else "")
        suffix = "/" if p.is_dir() else f"  ({p.stat().st_size}B)"
        out.append(f"{prefix}{p.name}{suffix}")
        if len(out) >= max_entries:
            out.append(f"  … (truncated at {max_entries})")
            break
    return out


def _demo() -> None:
    print("=" * 78)
    print("SE Brain C15 — Repository / Project Builder")
    print("=" * 78)

    # 1. Synthesize via C14
    from sebrain.c14 import (
        CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
    )
    from sebrain.c13 import RepoIndex

    task = EntitySpec(
        name="Task",
        fields=[
            FieldSpec("title", "str", required=True),
            FieldSpec("description", "str", required=False, default_repr='""'),
            FieldSpec("done", "bool", required=False, default_repr="False"),
        ],
    )
    synth = CodeSynthesisEngine().synthesize(
        SynthesisRequest(
            package_name="task_api", entities=[task],
            framework="fastapi", model_style="dataclass", mode="fresh",
        ),
        project_id="demo",
        existing_index=RepoIndex(root="<demo>"),
    )
    print(f"\n[1] Synthesis: {len(synth.files)} files, "
          f"status={synth.status.value}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "demo_project"
        root.mkdir()

        # 2. Build plan
        plan = plan_from_synthesis(
            synth, root=root, mode=WriteMode.CREATE_ONLY,
            project_name="task_api",
            description="Small REST API for managing tasks",
            dependencies=["fastapi", "uvicorn", "pydantic"],
        )
        print(f"\n[2] Plan: {len(plan.writes)} writes, mode={plan.mode.value}")

        # 3. Dry-run preview
        dry_plan = BuildPlan(
            root=str(root), writes=plan.writes,
            mode=plan.mode, dry_run=True,
        )
        dry = ProjectBuilder().apply(dry_plan)
        print(f"\n[3] Dry-run: {dry.applied} would apply, "
              f"{dry.skipped} skip, status={dry.status.value}")

        # 4. Real apply
        builder = ProjectBuilder()
        res = builder.apply(plan)
        print(f"\n[4] Applied:")
        print(res.summary())

        print(f"\n[5] Changes:")
        for c in res.changes:
            print(f"    [{c.kind.value:9s}] {c.path}  "
                  f"{c.byte_size}B  hash={c.new_hash[:12]}…")

        print(f"\n[6] File tree:")
        for line in _tree(root):
            print(f"    {line}")

        # 5. Persistence
        print(f"\n[7] Persist to memory + ontology…")
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = BuildRepository(memory=mem, ontology=ont)
                ent = repo.save(res, project_id="demo")
                print(f"    ontology entity: {ent[:12]}…")
                print(f"    FILE entities: {ont.count(kind=EntityKind.FILE)}")
                print(f"    EXECUTION entities: "
                      f"{ont.count(kind=EntityKind.EXECUTION)}")
        finally:
            app.stop()

        # 6. Idempotency check: same plan on same target now fails (create_only)
        print(f"\n[8] Re-applying same plan (create_only):")
        res2 = builder.apply(plan)
        print(f"    status={res2.status.value}  "
              f"failures={[f.get('error') for f in res2.failed]}")

        # 7. Overwrite mode instead
        print(f"\n[9] Re-applying with overwrite:")
        ow_plan = BuildPlan(
            root=str(root), writes=plan.writes, mode=WriteMode.OVERWRITE,
        )
        res3 = builder.apply(ow_plan)
        print(f"    status={res3.status.value}  applied={res3.applied}")

    print("\nDone.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if "--test" in sys.argv:
        sys.exit(0 if _run_self_tests() == 0 else 1)
    else:
        _demo()
