"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C16 — EXECUTION SANDBOX (Single-File Complete Implementation)
================================================================================

Depends on C01, C02, C04.

Purpose:
    Execute subprocesses under hard, OS-enforced controls. Never run untrusted
    code in the host process. Every execution is bounded in time, memory,
    CPU, file writes, and file descriptors (POSIX). Output is bounded.
    Environment is minimal. Filesystem is isolated to a throw-away workdir.

Capabilities:
    - Process isolation (fork+exec, no shell, new session)
    - Wall-clock timeout (kill process group on expiry)
    - CPU time limit (RLIMIT_CPU)
    - Memory limit (RLIMIT_AS)
    - File-write limit (RLIMIT_FSIZE)
    - Open-files limit (RLIMIT_NOFILE)
    - Core-dump disabled (RLIMIT_CORE = 0)
    - Bounded stdout / stderr capture (no unbounded buffering)
    - Minimal environment (no secret inheritance)
    - Temp workdir with cleanup on completion
    - Optional read-only workdir (chmod 0555 during run)
    - Real per-run CPU usage (getrusage deltas)
    - Exit code + signal capture
    - Termination reason classification
    - Convenience: run_python(source), run_module(name)

Explicit limitations (honest, per prompt Rule #59):
    - RLIMIT_* only on POSIX. On Windows the policy flags are accepted but
      NOT enforced (documented in result.evidence as "rlimits_not_supported").
    - Network isolation is NOT enforced. `allow_network` is informational.
      Real network blocking requires namespaces/seccomp/containers.
    - Absolute-path filesystem writes (e.g. to /tmp) are NOT blocked by
      this sandbox. Only the workdir is chmod-restricted when requested.
    - `preexec_fn` (used for rlimits) is documented as unsafe in
      multi-threaded programs. Callers spawning sandboxes from many threads
      simultaneously may set apply_rlimits=False.
    - peak_rss is the process lifetime high-water mark of the parent's
      reaped children (getrusage(RUSAGE_CHILDREN).ru_maxrss). It is NOT a
      per-run peak if other children ran concurrently in the same process.

Invariants honored:
  - NO external LLM
  - NO shell=True anywhere
  - Every rejection has a structured reason (never silent)
  - Every result has evidence + rationale
  - Bounded: everything has a limit
  - Deterministic: same command + same policy → same class of result

Contents:
  1.  Enums: ExecutionStatus, TerminationReason
  2.  Dataclasses: SandboxPolicy, SandboxResult
  3.  Helpers (env, rlimits, bounded reader, executable resolution)
  4.  Sandbox (facade: run / run_python / run_module / validate_command)
  5.  SandboxRepository (persist EXECUTION entity)
  6.  Self-tests (~30)
  7.  Demo

Run as script:
    python -m sebrain.c16            # demo
    python -m sebrain.c16 --test     # self-tests
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    import resource as _resource
    _HAVE_RESOURCE = True
except ImportError:  # Windows
    _resource = None  # type: ignore[assignment]
    _HAVE_RESOURCE = False

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


# Shell names we refuse by default.
_SHELL_NAMES = frozenset({
    "sh", "bash", "dash", "zsh", "ksh", "csh", "tcsh", "fish",
    "cmd", "cmd.exe", "powershell", "powershell.exe",
    "pwsh", "pwsh.exe", "wsl", "wsl.exe",
})


# ════════════════════════════════════════════════════════════════════════════
# 1. ENUMS
# ════════════════════════════════════════════════════════════════════════════
class ExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"                 # exit_code != 0, no timeout, no signal
    TIMED_OUT = "timed_out"           # wall-clock timeout triggered
    CRASHED = "crashed"               # killed by signal (segfault, SIGXCPU, ...)
    REJECTED = "rejected"             # pre-flight rejection (bad command etc.)
    INTERNAL_ERROR = "internal_error" # sandbox raised before/during exec


class TerminationReason(str, Enum):
    NORMAL_EXIT = "normal_exit"
    WALL_CLOCK_TIMEOUT = "wall_clock_timeout"
    KILLED_BY_SIGNAL = "killed_by_signal"
    RLIMIT_CPU = "rlimit_cpu"
    RLIMIT_AS = "rlimit_as"
    RLIMIT_FSIZE = "rlimit_fsize"
    RLIMIT_NOFILE = "rlimit_nofile"
    RLIMIT_NPROC = "rlimit_nproc"
    EXECUTION_ERROR = "execution_error"
    PREFLIGHT_REJECTION = "preflight_rejection"
    UNKNOWN = "unknown"


# ════════════════════════════════════════════════════════════════════════════
# 2. DATACLASSES
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    """Immutable sandbox policy. Defaults are conservative."""
    timeout_seconds: float = 10.0
    max_cpu_seconds: int | None = None
    max_memory_bytes: int | None = None
    max_file_bytes: int | None = 50_000_000        # RLIMIT_FSIZE (50 MB)
    max_open_files: int | None = 256
    max_processes: int | None = None               # UID-wide on Linux; off by default
    max_output_bytes: int = 100_000                # 100 KB per stream
    read_only_fs: bool = False
    keep_workdir: bool = False
    allow_network: bool = False                    # informational only
    allow_shell: bool = False                      # refuse shells by default
    env_additions: tuple[tuple[str, str], ...] = () # immutable kv
    apply_rlimits: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValidationError("timeout_seconds must be > 0")
        if self.max_output_bytes < 1:
            raise ValidationError("max_output_bytes must be >= 1")
        if self.max_cpu_seconds is not None and self.max_cpu_seconds <= 0:
            raise ValidationError("max_cpu_seconds must be > 0")
        if self.max_memory_bytes is not None and self.max_memory_bytes <= 0:
            raise ValidationError("max_memory_bytes must be > 0")
        if self.max_file_bytes is not None and self.max_file_bytes <= 0:
            raise ValidationError("max_file_bytes must be > 0")

    def env_dict(self) -> dict[str, str]:
        return dict(self.env_additions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_file_bytes": self.max_file_bytes,
            "max_open_files": self.max_open_files,
            "max_processes": self.max_processes,
            "max_output_bytes": self.max_output_bytes,
            "read_only_fs": self.read_only_fs,
            "keep_workdir": self.keep_workdir,
            "allow_network": self.allow_network,
            "allow_shell": self.allow_shell,
            "env_additions": list(self.env_additions),
            "apply_rlimits": self.apply_rlimits,
        }


@dataclass(slots=True)
class SandboxResult:
    id: str = field(default_factory=_new_id)
    command: list[str] = field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED
    exit_code: int = 0
    signal_number: int = -1
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    duration_seconds: float = 0.0
    user_cpu_seconds: float = 0.0
    system_cpu_seconds: float = 0.0
    peak_rss_bytes: int = 0
    workdir: str = ""
    policy: SandboxPolicy | None = None
    termination_reason: TerminationReason = TerminationReason.UNKNOWN
    rationale: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = field(default_factory=now_iso)
    ended_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "command": list(self.command),
            "status": self.status.value,
            "exit_code": self.exit_code,
            "signal_number": self.signal_number,
            "stdout": self.stdout, "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "user_cpu_seconds": self.user_cpu_seconds,
            "system_cpu_seconds": self.system_cpu_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "workdir": self.workdir,
            "policy": self.policy.to_dict() if self.policy else None,
            "termination_reason": self.termination_reason.value,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "started_at": self.started_at, "ended_at": self.ended_at,
        }

    def summary(self) -> str:
        return (
            "=== Sandbox Result ===\n"
            f"status={self.status.value}  reason={self.termination_reason.value}\n"
            f"cmd={' '.join(self.command) if self.command else '<rejected>'}\n"
            f"exit_code={self.exit_code}  signal={self.signal_number}\n"
            f"duration={self.duration_seconds*1000:.1f}ms  "
            f"cpu_user={self.user_cpu_seconds:.3f}s  "
            f"cpu_sys={self.system_cpu_seconds:.3f}s\n"
            f"stdout={len(self.stdout)}B"
            f"{' (truncated)' if self.stdout_truncated else ''}  "
            f"stderr={len(self.stderr)}B"
            f"{' (truncated)' if self.stderr_truncated else ''}"
        )


# ════════════════════════════════════════════════════════════════════════════
# 3. HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _build_env(policy: SandboxPolicy, workdir: Path) -> dict[str, str]:
    """Minimal, deterministic env. No inheritance from os.environ."""
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",             # determinism
        "HOME": str(workdir),              # confine $HOME
        "TMPDIR": str(workdir),            # confine temp files
        "TEMP": str(workdir),
        "TMP": str(workdir),
        # Defensive: disable common Python behaviors that touch network
        "PIP_NO_INDEX": "1" if not policy.allow_network else "0",
        "REQUESTS_CA_BUNDLE": "",          # best-effort: clear proxies
        "no_proxy": "*",
    }
    for k, v in policy.env_additions:
        env[k] = v
    return env


def _preexec_factory(policy: SandboxPolicy) -> Callable[[], None] | None:
    """Return a preexec_fn that applies RLIMIT_* constraints, or None if
    unsupported / disabled.
    """
    if not _HAVE_RESOURCE or not policy.apply_rlimits or _resource is None:
        return None

    # Capture policy fields (closure)
    cpu = policy.max_cpu_seconds
    mem = policy.max_memory_bytes
    fsz = policy.max_file_bytes
    nfiles = policy.max_open_files
    nproc = policy.max_processes

    def _preexec() -> None:
        # --- CPU time (soft == hard → SIGKILL at limit) ---
        if cpu is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_CPU, (cpu, cpu),
                )
            except (ValueError, OSError):
                pass
        # --- Address space ---
        if mem is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_AS, (mem, mem),
                )
            except (ValueError, OSError):
                pass
        # --- File size ---
        if fsz is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_FSIZE, (fsz, fsz),
                )
            except (ValueError, OSError):
                pass
        # --- Open files ---
        if nfiles is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_NOFILE, (nfiles, nfiles),
                )
            except (ValueError, OSError):
                pass
        # --- Processes (dangerous: UID-wide on Linux; opt-in only) ---
        if nproc is not None:
            try:
                _resource.setrlimit(
                    _resource.RLIMIT_NPROC, (nproc, nproc),
                )
            except (ValueError, OSError):
                pass
        # --- Disable core dumps ---
        try:
            _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
        except (ValueError, OSError):
            pass

    return _preexec


def _read_bounded(
    pipe: Any, max_bytes: int, sink: dict[str, Any],
) -> None:
    """Read from a binary stream until EOF, capping stored bytes."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    try:
        # `BufferedReader.read(n)` may wait for more data than is currently
        # available.  That can race with a short-lived timeout: the child may
        # already have emitted useful output, but the reader is still waiting
        # for a larger chunk.  `read1()` asks the buffered layer for whatever
        # is currently available and therefore preserves partial output
        # reliably while still draining the pipe.
        reader = getattr(pipe, "read1", pipe.read)
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            if total >= max_bytes:
                truncated = True
                continue  # drain without storing
            take = max_bytes - total
            if len(chunk) <= take:
                chunks.append(chunk)
                total += len(chunk)
            else:
                chunks.append(chunk[:take])
                total += take
                truncated = True
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass
    sink["data"] = b"".join(chunks)
    sink["truncated"] = truncated


def _classify_signal(sig: int) -> TerminationReason:
    """Map a signal number to the most likely cause."""
    if _HAVE_RESOURCE and _resource is not None:
        try:
            if sig == getattr(signal, "SIGXCPU", -1):
                return TerminationReason.RLIMIT_CPU
            if sig == getattr(signal, "SIGXFSZ", -1):
                return TerminationReason.RLIMIT_FSIZE
        except Exception:
            pass
    return TerminationReason.KILLED_BY_SIGNAL


# ════════════════════════════════════════════════════════════════════════════
# 4. SANDBOX
# ════════════════════════════════════════════════════════════════════════════
class Sandbox:
    """Hard-controlled subprocess executor.

    Usage:
        result = Sandbox().run_python("print('hi')")
        assert result.status is ExecutionStatus.SUCCEEDED
    """

    def __init__(self, *, base_workdir: Path | None = None) -> None:
        self.base_workdir = Path(base_workdir) if base_workdir else None

    # ---- public API ----
    def run(
        self,
        command: list[str] | tuple[str, ...],
        *,
        policy: SandboxPolicy | None = None,
        workdir: str | Path | None = None,
        stdin: bytes | str | None = None,
    ) -> SandboxResult:
        policy = policy or SandboxPolicy()

        # -------- 1. Pre-flight: validate command --------
        try:
            cmd = self.validate_command(command, policy=policy)
        except ValidationError as exc:
            return self._reject(
                list(command) if isinstance(command, (list, tuple)) else [str(command)],
                policy, reason=f"command rejected: {exc}",
            )

        # -------- 2. Pre-flight: workdir --------
        workdir_provided = workdir is not None
        if workdir_provided:
            wd = Path(workdir).resolve()  # type: ignore[arg-type]
            if not wd.exists():
                try:
                    wd.mkdir(parents=True, exist_ok=False)
                except OSError as exc:
                    return self._reject(
                        cmd, policy,
                        reason=f"workdir could not be created: {exc}",
                    )
            if not wd.is_dir():
                return self._reject(
                    cmd, policy,
                    reason=f"workdir is not a directory: {wd}",
                )
        else:
            try:
                if self.base_workdir:
                    self.base_workdir.mkdir(parents=True, exist_ok=True)
                    wd = Path(tempfile.mkdtemp(
                        prefix="sebrain_c16_", dir=str(self.base_workdir),
                    ))
                else:
                    wd = Path(tempfile.mkdtemp(prefix="sebrain_c16_"))
            except OSError as exc:
                return self._reject(
                    cmd, policy, reason=f"temp workdir failed: {exc}",
                )

        # -------- 3. Optional read-only FS --------
        original_mode: int | None = None
        if policy.read_only_fs:
            try:
                original_mode = stat.S_IMODE(os.stat(wd).st_mode)
                os.chmod(wd, 0o555)
            except OSError as exc:
                log.warning("c16.chmod_failed", error=str(exc))

        # -------- 4. Prepare stdin, env, rlimits --------
        stdin_pipe = subprocess.PIPE if stdin is not None else subprocess.DEVNULL
        env = _build_env(policy, wd)

        preexec = _preexec_factory(policy)
        rlimits_applied = preexec is not None

        # -------- 5. Launch --------
        started_at = now_iso()
        t0 = time.monotonic()
        ru_before = _snapshot_rusage()

        proc: subprocess.Popen[bytes] | None = None
        try:
            try:
                proc = subprocess.Popen(   # noqa: S603 (no shell, validated argv)
                    cmd,
                    stdin=stdin_pipe,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=str(wd),
                    shell=False,
                    start_new_session=True,   # new session → killpg works
                    preexec_fn=preexec,
                    close_fds=True,
                )
            except FileNotFoundError as exc:
                return self._reject(cmd, policy, reason=f"executable missing: {exc}")
            except PermissionError as exc:
                return self._reject(cmd, policy, reason=f"permission denied: {exc}")
            except OSError as exc:
                return self._reject(cmd, policy, reason=f"exec error: {exc}")

            # Write stdin (if provided), then close
            if stdin is not None:
                try:
                    stdin_bytes = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
                    assert proc.stdin is not None
                    proc.stdin.write(stdin_bytes)
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    try:
                        if proc.stdin is not None:
                            proc.stdin.close()
                    except Exception:
                        pass

            # Launch bounded readers (threads so we never deadlock on pipe buffer)
            out_sink: dict[str, Any] = {"data": b"", "truncated": False}
            err_sink: dict[str, Any] = {"data": b"", "truncated": False}
            assert proc.stdout is not None and proc.stderr is not None
            t_out = threading.Thread(
                target=_read_bounded,
                args=(proc.stdout, policy.max_output_bytes, out_sink),
                daemon=True,
            )
            t_err = threading.Thread(
                target=_read_bounded,
                args=(proc.stderr, policy.max_output_bytes, err_sink),
                daemon=True,
            )
            t_out.start()
            t_err.start()

            # -------- 6. Wait (with timeout) --------
            timed_out = False
            termination_reason = TerminationReason.NORMAL_EXIT
            try:
                proc.wait(timeout=policy.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                termination_reason = TerminationReason.WALL_CLOCK_TIMEOUT
                self._terminate_tree(proc)
                # Give the OS a moment to reap
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass

            # -------- 7. Join readers --------
            t_out.join(timeout=5.0)
            t_err.join(timeout=5.0)

            # -------- 8. Compute metadata --------
            duration = time.monotonic() - t0
            ended_at = now_iso()
            ru_after = _snapshot_rusage()

            user_cpu = max(0.0, ru_after["user"] - ru_before["user"])
            sys_cpu = max(0.0, ru_after["system"] - ru_before["system"])
            peak_rss = ru_after["max_rss_bytes"]

            exit_code = proc.returncode if proc.returncode is not None else -999
            signal_number = -1
            if exit_code < 0:
                signal_number = -exit_code
                if not timed_out:
                    termination_reason = _classify_signal(signal_number)

            # -------- 9. Classify overall status --------
            if timed_out:
                status = ExecutionStatus.TIMED_OUT
            elif signal_number > 0:
                status = ExecutionStatus.CRASHED
            elif exit_code == 0:
                status = ExecutionStatus.SUCCEEDED
            else:
                status = ExecutionStatus.FAILED

            stdout_text = out_sink["data"].decode("utf-8", errors="replace")
            stderr_text = err_sink["data"].decode("utf-8", errors="replace")

            result = SandboxResult(
                command=list(cmd),
                status=status,
                exit_code=exit_code,
                signal_number=signal_number,
                stdout=stdout_text,
                stderr=stderr_text,
                stdout_truncated=bool(out_sink["truncated"]),
                stderr_truncated=bool(err_sink["truncated"]),
                timed_out=timed_out,
                duration_seconds=duration,
                user_cpu_seconds=user_cpu,
                system_cpu_seconds=sys_cpu,
                peak_rss_bytes=peak_rss,
                workdir=str(wd),
                policy=policy,
                termination_reason=termination_reason,
                rationale=self._explain(
                    status, exit_code, signal_number, timed_out, policy,
                ),
                started_at=started_at, ended_at=ended_at,
            )
            result.evidence.extend(self._evidence(
                cmd, policy, wd, rlimits_applied, ru_before, ru_after,
            ))
            return result

        except Exception as exc:
            # Internal sandbox error
            log.exception("c16.internal_error")
            ended_at = now_iso()
            result = SandboxResult(
                command=list(cmd),
                status=ExecutionStatus.INTERNAL_ERROR,
                exit_code=-999,
                workdir=str(wd),
                policy=policy,
                termination_reason=TerminationReason.EXECUTION_ERROR,
                duration_seconds=time.monotonic() - t0,
                rationale=f"internal sandbox error: {type(exc).__name__}: {exc}",
                started_at=started_at, ended_at=ended_at,
            )
            result.evidence.append({
                "kind": "internal_error",
                "exception": type(exc).__name__,
                "message": str(exc),
            })
            return result

        finally:
            # Restore mode if we changed it
            if original_mode is not None:
                try:
                    os.chmod(wd, original_mode)
                except OSError:
                    pass
            # Cleanup temp workdir (only if we made it)
            if not workdir_provided and not policy.keep_workdir:
                shutil.rmtree(wd, ignore_errors=True)

    # ---- convenience wrappers ----
    def run_python(
        self,
        source: str,
        *,
        policy: SandboxPolicy | None = None,
        argv: list[str] | None = None,
        stdin: bytes | str | None = None,
        workdir: str | Path | None = None,
    ) -> SandboxResult:
        """Write `source` to a temp file in the workdir and execute it."""
        policy = policy or SandboxPolicy()
        # Use a caller-provided workdir if given, else create one so the
        # generated file lives inside it (and gets cleaned).
        wd_provided = workdir is not None
        if wd_provided:
            wd = Path(workdir).resolve()  # type: ignore[arg-type]
            wd.mkdir(parents=True, exist_ok=True)
        else:
            base = self.base_workdir
            if base:
                base.mkdir(parents=True, exist_ok=True)
                wd = Path(tempfile.mkdtemp(prefix="sebrain_c16_py_", dir=str(base)))
            else:
                wd = Path(tempfile.mkdtemp(prefix="sebrain_c16_py_"))
        try:
            script_path = wd / "runner.py"
            script_path.write_text(source, encoding="utf-8")
            cmd = [sys.executable, "-I", str(script_path), *(argv or [])]
            # -I = isolated: no site-packages user, no env PYTHONPATH
            return self.run(
                cmd, policy=policy,
                workdir=wd if wd_provided else str(wd),
                stdin=stdin,
            )
        finally:
            if not wd_provided and not policy.keep_workdir:
                shutil.rmtree(wd, ignore_errors=True)

    def run_module(
        self,
        module_name: str,
        *,
        policy: SandboxPolicy | None = None,
        args: list[str] | None = None,
        stdin: bytes | str | None = None,
        workdir: str | Path | None = None,
    ) -> SandboxResult:
        """Run `python -m <module_name> [args...]`."""
        if not module_name or not isinstance(module_name, str):
            return self._reject([], policy or SandboxPolicy(),
                                reason="module_name must be a non-empty str")
        cmd = [sys.executable, "-I", "-m", module_name, *(args or [])]
        return self.run(
            cmd, policy=policy or SandboxPolicy(),
            workdir=workdir, stdin=stdin,
        )

    # ---- validation ----
    def validate_command(
        self, command: list[str] | tuple[str, ...],
        *, policy: SandboxPolicy,
    ) -> list[str]:
        if not isinstance(command, (list, tuple)):
            raise ValidationError("command must be a list or tuple of str")
        cmd = list(command)
        if not cmd:
            raise ValidationError("command must not be empty")
        if not all(isinstance(c, str) for c in cmd):
            raise ValidationError("every command element must be a str")
        if any("\0" in c for c in cmd):
            raise ValidationError("command contains a null byte")
        exe = cmd[0]
        exe_base = Path(exe).name.lower()
        if exe_base in _SHELL_NAMES and not policy.allow_shell:
            raise ValidationError(
                f"shell execution refused by default: {exe!r} "
                f"(set allow_shell=True to override)"
            )
        # Existence check (PATH or explicit path)
        if not Path(exe).is_file():
            if shutil.which(exe) is None:
                raise ValidationError(f"executable not found: {exe!r}")
        return cmd

    # ---- termination ----
    @staticmethod
    def _terminate_tree(proc: subprocess.Popen[bytes]) -> None:
        """Kill the child's process group (SIGKILL). Fallback to .kill()."""
        if proc.poll() is not None:
            return
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass

    # ---- pre-flight rejection ----
    def _reject(
        self, command: list[str], policy: SandboxPolicy, *, reason: str,
    ) -> SandboxResult:
        return SandboxResult(
            command=list(command),
            status=ExecutionStatus.REJECTED,
            exit_code=-1,
            policy=policy,
            termination_reason=TerminationReason.PREFLIGHT_REJECTION,
            rationale=reason,
            evidence=[{"kind": "preflight_rejection", "reason": reason}],
        )

    # ---- evidence + rationale helpers ----
    @staticmethod
    def _evidence(
        cmd: list[str], policy: SandboxPolicy, wd: Path,
        rlimits_applied: bool,
        ru_before: dict[str, float], ru_after: dict[str, float],
    ) -> list[dict[str, Any]]:
        ev: list[dict[str, Any]] = [
            {"kind": "workdir", "path": str(wd)},
            {"kind": "policy", "value": policy.to_dict()},
            {"kind": "rlimits_applied", "value": rlimits_applied},
            {"kind": "rlimits_supported", "value": _HAVE_RESOURCE},
            {"kind": "platform", "value": platform.system()},
        ]
        if not rlimits_applied and policy.apply_rlimits:
            ev.append({
                "kind": "limitation",
                "value": "rlimits were not applied (unsupported platform "
                         "or disabled); only wall-clock timeout enforced",
            })
        if policy.allow_network:
            ev.append({
                "kind": "limitation",
                "value": "allow_network=True is informational; network is "
                         "not enforced by this pure-Python sandbox",
            })
        return ev

    @staticmethod
    def _explain(
        status: ExecutionStatus, exit_code: int, signal_number: int,
        timed_out: bool, policy: SandboxPolicy,
    ) -> str:
        if timed_out:
            return f"wall-clock timeout after {policy.timeout_seconds}s; killed process group"
        if signal_number > 0:
            return f"process killed by signal {signal_number} (crashed)"
        if status is ExecutionStatus.SUCCEEDED:
            return "process exited with code 0"
        return f"process exited with non-zero code {exit_code}"


# ---- rusage snapshot ----
def _snapshot_rusage() -> dict[str, float]:
    if not _HAVE_RESOURCE or _resource is None:
        return {"user": 0.0, "system": 0.0, "max_rss_bytes": 0.0}
    try:
        ru = _resource.getrusage(_resource.RUSAGE_CHILDREN)
        # ru_maxrss units differ: KB on Linux, bytes on macOS
        rss = float(ru.ru_maxrss)
        if sys.platform == "darwin":
            rss_bytes = rss  # already bytes on macOS
        else:
            rss_bytes = rss * 1024.0
        return {
            "user": float(ru.ru_utime),
            "system": float(ru.ru_stime),
            "max_rss_bytes": rss_bytes,
        }
    except Exception:
        return {"user": 0.0, "system": 0.0, "max_rss_bytes": 0.0}


# ════════════════════════════════════════════════════════════════════════════
# 5. SANDBOX REPOSITORY
# ════════════════════════════════════════════════════════════════════════════
class SandboxRepository:
    def __init__(self, memory: MemoryStore, ontology: Ontology | None = None) -> None:
        self.memory = memory
        self.ontology = ontology

    def save(self, result: SandboxResult, *, project_id: str) -> str:
        if not project_id:
            raise ValidationError("project_id required")
        key = f"sandbox_run:{result.id}"
        self.memory.create(
            MemoryKind.PROJECT, key, result.to_dict(),
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
            tags=["sandbox", "c16", result.status.value],
            provenance=Provenance(
                source="execution_sandbox",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        # Failures for C33
        if result.status in (
            ExecutionStatus.FAILED, ExecutionStatus.CRASHED,
            ExecutionStatus.TIMED_OUT, ExecutionStatus.INTERNAL_ERROR,
        ):
            self.memory.record_failure(
                f"sandbox_fail:{result.id}",
                what=f"sandbox run {result.id[:8]} {result.status.value}",
                root_cause=result.termination_reason.value,
                fix=None,
                scope_id=project_id,
                provenance=Provenance(
                    source="execution_sandbox",
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
                f"Sandbox {result.id[:8]} ({result.status.value})", 120,
            ),
            attributes={
                "sandbox_id": result.id,
                "project_id": project_id,
                "status": result.status.value,
                "exit_code": result.exit_code,
                "signal": result.signal_number,
                "timed_out": result.timed_out,
                "duration_seconds": result.duration_seconds,
                "user_cpu_seconds": result.user_cpu_seconds,
                "system_cpu_seconds": result.system_cpu_seconds,
                "command": result.command,
                "workdir": result.workdir,
                "termination_reason": result.termination_reason.value,
            },
            tags=["sandbox", result.status.value],
            provenance=Provenance(
                source="execution_sandbox",
                source_type=ProvenanceType.SYSTEM,
                confidence=Confidence.HIGH,
            ),
        )
        return ent.id

    def load(self, sandbox_id: str, *, project_id: str) -> dict[str, Any] | None:
        e = self.memory.get_current(
            MemoryKind.PROJECT, f"sandbox_run:{sandbox_id}",
            scope_type=MemoryScope.PROJECT, scope_id=project_id,
        )
        return dict(e.content) if e else None


# ════════════════════════════════════════════════════════════════════════════
# 6. SELF-TESTS
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

    print("Running C16 self-tests…")
    sb = Sandbox()

    # ---- policy validation ----
    def t_policy_rejects_bad_timeout() -> None:
        try:
            SandboxPolicy(timeout_seconds=0)
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    def t_policy_immutable() -> None:
        p = SandboxPolicy(timeout_seconds=5.0)
        try:
            p.timeout_seconds = 99  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("expected frozen dataclass to refuse mutation")

    check("policy: rejects bad timeout", t_policy_rejects_bad_timeout)
    check("policy: frozen (immutable)", t_policy_immutable)

    # ---- command validation ----
    def t_reject_empty_command() -> None:
        r = sb.run([], policy=SandboxPolicy())
        assert r.status is ExecutionStatus.REJECTED
        assert "empty" in r.rationale

    def t_reject_non_list_command() -> None:
        r = sb.run("python -c print", policy=SandboxPolicy())  # type: ignore[arg-type]
        assert r.status is ExecutionStatus.REJECTED

    def t_reject_missing_executable() -> None:
        r = sb.run(["/definitely/not/a/binary/xyz"], policy=SandboxPolicy())
        assert r.status is ExecutionStatus.REJECTED
        assert "not found" in r.rationale or "missing" in r.rationale

    def t_reject_shell_by_default() -> None:
        r = sb.run(["sh", "-c", "echo hi"], policy=SandboxPolicy())
        assert r.status is ExecutionStatus.REJECTED
        assert "shell" in r.rationale.lower()

    def t_allow_shell_when_opt_in() -> None:
        # Only run if /bin/sh exists
        if not shutil.which("sh"):
            return
        r = sb.run(
            ["sh", "-c", "echo shell-ok"],
            policy=SandboxPolicy(allow_shell=True, timeout_seconds=5.0),
        )
        # Should succeed with exit 0 and stdout containing shell-ok
        assert r.status is ExecutionStatus.SUCCEEDED
        assert "shell-ok" in r.stdout

    check("validate: empty command rejected", t_reject_empty_command)
    check("validate: non-list rejected", t_reject_non_list_command)
    check("validate: missing executable rejected",
          t_reject_missing_executable)
    check("validate: shell refused by default", t_reject_shell_by_default)
    check("validate: shell allowed when opt-in", t_allow_shell_when_opt_in)

    # ---- happy path ----
    def t_run_python_print() -> None:
        r = sb.run_python("print('hello from sandbox')\n")
        assert r.status is ExecutionStatus.SUCCEEDED, r.rationale
        assert r.exit_code == 0
        assert "hello from sandbox" in r.stdout
        assert r.timed_out is False
        assert r.termination_reason is TerminationReason.NORMAL_EXIT

    def t_run_python_exit_code() -> None:
        r = sb.run_python("import sys; sys.exit(7)\n")
        assert r.status is ExecutionStatus.FAILED
        assert r.exit_code == 7

    def t_captures_stderr() -> None:
        r = sb.run_python(
            "import sys; sys.stderr.write('oops\\n'); sys.exit(1)\n"
        )
        assert r.exit_code == 1
        assert "oops" in r.stderr
        assert "oops" not in r.stdout

    def t_stdin_passed() -> None:
        r = sb.run_python(
            "import sys; data = sys.stdin.read(); print(data.upper())\n",
            stdin="hello\n",
        )
        assert r.status is ExecutionStatus.SUCCEEDED
        assert "HELLO" in r.stdout

    def t_argv_passed() -> None:
        r = sb.run_python(
            "import sys; print('|'.join(sys.argv[1:]))\n",
            argv=["alpha", "beta"],
        )
        assert "alpha|beta" in r.stdout

    check("run: print → SUCCEEDED", t_run_python_print)
    check("run: exit code captured", t_run_python_exit_code)
    check("run: stderr captured", t_captures_stderr)
    check("run: stdin delivered", t_stdin_passed)
    check("run: argv delivered", t_argv_passed)

    # ---- timeout ----
    def t_timeout_kills_process() -> None:
        r = sb.run_python(
            "import time\nprint('started', flush=True)\ntime.sleep(10)\n",
            policy=SandboxPolicy(timeout_seconds=2.0),
        )
        assert r.status is ExecutionStatus.TIMED_OUT, r.rationale
        assert r.timed_out is True
        assert r.termination_reason is TerminationReason.WALL_CLOCK_TIMEOUT
        assert r.duration_seconds < 5.0
        assert "started" in r.stdout

    def t_timeout_kills_children() -> None:
        # Spawn a child that sleeps; the parent exits quickly.
        # Our killpg should terminate the child too.
        script = (
            "import subprocess, sys, time\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "print('child_pid', p.pid, flush=True)\n"
            "time.sleep(30)\n"
        )
        r = sb.run_python(script, policy=SandboxPolicy(timeout_seconds=0.5))
        assert r.status is ExecutionStatus.TIMED_OUT
        # Child should have been killed along with parent (same pgid)
        # Extract pid and check
        for line in r.stdout.splitlines():
            if line.startswith("child_pid"):
                try:
                    pid = int(line.split()[1])
                except (ValueError, IndexError):
                    continue
                # Poll for a few seconds rather than a single fixed sleep —
                # SIGKILL delivery + reaping can take longer than 0.3s
                # under container/scheduler overhead even when the kill
                # itself is correct, which made this test flaky rather
                # than actually wrong.
                if hasattr(os, "kill"):
                    deadline = time.monotonic() + 5.0
                    while True:
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            break   # child is gone → success
                        except PermissionError:
                            break   # can't check → skip
                        if time.monotonic() >= deadline:
                            raise AssertionError(
                                f"grandchild pid {pid} survived killpg"
                            )
                        time.sleep(0.2)

    check("timeout: process killed, partial stdout preserved",
          t_timeout_kills_process)
    check("timeout: process group (children) also killed",
          t_timeout_kills_children)

    # ---- output bounds ----
    def t_output_truncated() -> None:
        r = sb.run_python(
            "print('x' * 1000000)\n",
            policy=SandboxPolicy(max_output_bytes=1000),
        )
        assert r.status is ExecutionStatus.SUCCEEDED
        assert r.stdout_truncated is True
        assert len(r.stdout) <= 1000

    def t_output_no_truncation_flag_when_small() -> None:
        r = sb.run_python("print('small')\n",
                          policy=SandboxPolicy(max_output_bytes=10000))
        assert r.stdout_truncated is False

    check("output: truncated at max_output_bytes",
          t_output_truncated)
    check("output: no false-positive truncation flag",
          t_output_no_truncation_flag_when_small)

    # ---- env isolation ----
    def t_env_minimal() -> None:
        r = sb.run_python(
            "import os\n"
            "keys = sorted(k for k in os.environ if k.startswith('SEBRAIN_SECRET'))\n"
            "print('LEAKED' if keys else 'clean')\n"
            "print('HOME=' + os.environ.get('HOME', '<unset>'))\n"
        )
        # We never inject such vars, so this passes as long as we don't inherit
        # (our env is built fresh, so even if parent had them they don't leak)
        assert "clean" in r.stdout
        # HOME points to the workdir (deleted by now, but should be a path)
        assert "HOME=" in r.stdout

    def t_env_additions_applied() -> None:
        r = sb.run_python(
            "import os\nprint(os.environ.get('MY_TEST_VAR', '<missing>'))\n",
            policy=SandboxPolicy(env_additions=(("MY_TEST_VAR", "xyz"),)),
        )
        assert "xyz" in r.stdout

    check("env: no secret leakage from parent", t_env_minimal)
    check("env: additions are applied", t_env_additions_applied)

    # ---- workdir ----
    def t_temp_workdir_cleaned() -> None:
        r = sb.run_python(
            "import os; print(os.getcwd())\n",
            policy=SandboxPolicy(),
        )
        assert r.status is ExecutionStatus.SUCCEEDED
        wd = r.workdir
        assert wd
        # Should be gone (temp workdir)
        assert not Path(wd).exists()

    def t_keep_workdir() -> None:
        r = sb.run_python(
            "open('marker.txt', 'w').write('hi')\n",
            policy=SandboxPolicy(keep_workdir=True),
        )
        try:
            assert r.status is ExecutionStatus.SUCCEEDED
            wd = Path(r.workdir)
            assert wd.exists()
            assert (wd / "marker.txt").read_text() == "hi"
        finally:
            shutil.rmtree(r.workdir, ignore_errors=True)

    def t_read_only_fs_blocks_writes() -> None:
        if platform.system() == "Windows":
            return
        # Running as root would bypass chmod — skip if so
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return
        r = sb.run_python(
            "open('forbidden.txt', 'w').write('nope')\n",
            policy=SandboxPolicy(read_only_fs=True),
        )
        # Write should fail → non-zero exit and error in stderr
        assert r.status is ExecutionStatus.FAILED, r.rationale
        assert "Permission" in r.stderr or "read-only" in r.stderr.lower()

    check("workdir: temp cleaned up", t_temp_workdir_cleaned)
    check("workdir: keep_workdir preserves it", t_keep_workdir)
    check("workdir: read_only_fs blocks writes",
          t_read_only_fs_blocks_writes)

    # ---- rlimits ----
    def t_cpu_limit() -> None:
        if not _HAVE_RESOURCE:
            print("      (skipped: no resource module)")
            return
        r = sb.run_python(
            "n = 0\nwhile True:\n    n += 1\n",
            policy=SandboxPolicy(timeout_seconds=15, max_cpu_seconds=1),
        )
        # Should be killed by SIGKILL/SIGXCPU well before the wall timeout
        assert r.duration_seconds < 10.0
        assert r.status in (ExecutionStatus.CRASHED, ExecutionStatus.FAILED,
                            ExecutionStatus.TIMED_OUT), r.rationale
        assert r.timed_out is False, "wall timeout shouldn't have fired"

    def t_file_size_limit() -> None:
        if not _HAVE_RESOURCE:
            return
        script = (
            "with open('big.txt', 'w') as f:\n"
            "    for _ in range(10000):\n"
            "        f.write('x' * 1024)\n"
        )
        r = sb.run_python(
            script,
            policy=SandboxPolicy(max_file_bytes=100_000, timeout_seconds=10),
        )
        # RLIMIT_FSIZE → write fails with OSError (SIGXFSZ or EFBIG)
        assert r.status in (ExecutionStatus.FAILED, ExecutionStatus.CRASHED), \
            r.rationale

    check("rlimit: cpu limit terminates busy loop", t_cpu_limit)
    check("rlimit: file-size limit blocks huge writes", t_file_size_limit)

    # ---- rusage ----
    def t_rusage_populated() -> None:
        if not _HAVE_RESOURCE:
            return
        r = sb.run_python(
            "s = 0\nfor i in range(100000): s += i\nprint(s)\n"
        )
        assert r.status is ExecutionStatus.SUCCEEDED
        # CPU delta must be non-negative
        assert r.user_cpu_seconds >= 0.0
        assert r.system_cpu_seconds >= 0.0
        # Duration should be > 0
        assert r.duration_seconds > 0.0

    check("rusage: cpu usage captured (POSIX)", t_rusage_populated)

    # ---- crash detection ----
    def t_signal_crash() -> None:
        if platform.system() == "Windows":
            return
        # Send ourselves SIGSEGV via ctypes. libc's function is named
        # "raise" (no trailing underscore) — but "raise" is a Python
        # keyword, so it can't be accessed as a plain attribute
        # (`lib.raise_` looks up a symbol that doesn't exist and raises
        # AttributeError instead of actually sending the signal, which
        # just crashes the subprocess with a normal Python traceback and
        # exit code 1 rather than triggering a real SIGSEGV).
        r = sb.run_python(
            "import ctypes, sys\n"
            "getattr(ctypes.CDLL(None), 'raise')(11)\n"   # SIGSEGV
        )
        assert r.status is ExecutionStatus.CRASHED, r.rationale
        assert r.signal_number > 0
        assert r.termination_reason is TerminationReason.KILLED_BY_SIGNAL

    check("crash: signal captured (POSIX)", t_signal_crash)

    # ---- run_module ----
    def t_run_module_validates() -> None:
        r = sb.run_module("")
        assert r.status is ExecutionStatus.REJECTED

    def t_run_module_unknown() -> None:
        r = sb.run_module(
            "definitely_not_a_module_xyz_123",
            policy=SandboxPolicy(timeout_seconds=5),
        )
        # Python exits non-zero for missing module
        assert r.status is ExecutionStatus.FAILED
        assert r.exit_code != 0

    check("run_module: rejects empty name", t_run_module_validates)
    check("run_module: unknown module → non-zero exit", t_run_module_unknown)

    # ---- to_dict/summary ----
    def t_to_dict_summary() -> None:
        r = sb.run_python("print('x')\n")
        d = r.to_dict()
        assert d["status"] == "succeeded"
        assert d["exit_code"] == 0
        assert "policy" in d
        s = r.summary()
        assert "Sandbox Result" in s

    check("to_dict + summary", t_to_dict_summary)

    # ---- persistence ----
    def t_persist() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                r = sb.run_python("print('persist me')\n")
                repo = SandboxRepository(memory=mem, ontology=ont)
                ent = repo.save(r, project_id="proj-x")
                assert ent
                loaded = repo.load(r.id, project_id="proj-x")
                assert loaded is not None
                assert loaded["status"] == "succeeded"
                assert ont.count(kind=EntityKind.EXECUTION) >= 1
            finally:
                s.shutdown()

    def t_persist_records_failures() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                mem = MemoryStore(s)
                ont = Ontology(s)
                r = sb.run_python(
                    "import sys; sys.exit(9)\n",
                    policy=SandboxPolicy(timeout_seconds=5),
                )
                repo = SandboxRepository(memory=mem, ontology=ont)
                repo.save(r, project_id="proj-y")
                fails = mem.find(
                    kind=MemoryKind.FAILURE,
                    scope_type=MemoryScope.PROJECT, scope_id="proj-y",
                )
                assert len(fails) >= 1
            finally:
                s.shutdown()

    check("persist: memory + ontology EXECUTION entity", t_persist)
    check("persist: failures recorded for C33", t_persist_records_failures)

    # ---- e2e: run C14-synthesised code inside sandbox ----
    def t_e2e_run_synthesised_code_in_sandbox() -> None:
        """Compile a small module, run it in the sandbox, verify exit 0."""
        from sebrain.c14 import (
            CodeSynthesisEngine, EntitySpec, FieldSpec, SynthesisRequest,
        )
        from sebrain.c13 import RepoIndex
        from sebrain.c15 import (
            BuildPlan, FileWrite, ProjectBuilder, WriteMode,
        )

        # 1. Synthesise a tiny package
        entity = EntitySpec(
            name="Task",
            fields=[FieldSpec("title", "str", required=True)],
        )
        synth = CodeSynthesisEngine().synthesize(
            SynthesisRequest(
                package_name="pkg", entities=[entity],
                framework="fastapi", model_style="dataclass", mode="fresh",
            ),
            project_id="demo",
            existing_index=RepoIndex(root="<demo>"),
        )

        with tempfile.TemporaryDirectory() as td:
            # 2. Build to disk
            plan = BuildPlan(
                root=td, mode=WriteMode.CREATE_ONLY,
                writes=[FileWrite(f.path, f.content, kind=f.kind.value)
                        for f in synth.files],
            )
            builder = ProjectBuilder()
            br = builder.apply(plan)
            assert br.status.value == "succeeded", br.rationale

            # 3. Run one of the generated files in the sandbox
            # models.py is safe (no external deps at import time except dataclasses)
            r = sb.run(
                [sys.executable, "-I", "-c",
                 "import sys; sys.path.insert(0, '.'); "
                 "from pkg.models import Task, TaskIn; "
                 "t = Task(title='hi'); assert t.title == 'hi'; "
                 "print('ok')"],
                policy=SandboxPolicy(timeout_seconds=10),
                workdir=td,
            )
            assert r.status is ExecutionStatus.SUCCEEDED, r.rationale
            assert "ok" in r.stdout

    check("e2e: sandbox runs C14-synthesised code from C15 build",
          t_e2e_run_synthesised_code_in_sandbox)

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
    print("SE Brain C16 — Execution Sandbox")
    print("=" * 78)
    print(f"platform: {platform.system()}  rlimits: {_HAVE_RESOURCE}")

    sb = Sandbox()

    print("\n[1] Happy path — run Python:")
    r = sb.run_python("print('hello from the sandbox')\n")
    print(r.summary())
    print(f"    stdout: {r.stdout.strip()!r}")

    print("\n[2] Non-zero exit:")
    r = sb.run_python("import sys; sys.exit(42)\n")
    print(r.summary())

    print("\n[3] Wall-clock timeout (0.5s):")
    r = sb.run_python(
        "import time\nprint('working...', flush=True)\ntime.sleep(30)\n",
        policy=SandboxPolicy(timeout_seconds=0.5),
    )
    print(r.summary())
    print(f"    partial stdout: {r.stdout.strip()!r}")

    print("\n[4] Bounded output (1 KB limit):")
    r = sb.run_python(
        "print('x' * 10_000_000)\n",
        policy=SandboxPolicy(max_output_bytes=1000),
    )
    print(f"    status={r.status.value}  "
          f"stdout={len(r.stdout)}B  truncated={r.stdout_truncated}")

    if _HAVE_RESOURCE:
        print("\n[5] CPU limit (1s busy loop):")
        r = sb.run_python(
            "n=0\nwhile True:\n    n+=1\n",
            policy=SandboxPolicy(timeout_seconds=15, max_cpu_seconds=1),
        )
        print(r.summary())

        print("\n[6] File-size limit (100 KB):")
        r = sb.run_python(
            "with open('big.txt','w') as f:\n"
            "    for _ in range(1000): f.write('x'*1024)\n",
            policy=SandboxPolicy(max_file_bytes=100_000, timeout_seconds=10),
        )
        print(r.summary())

    print("\n[7] Shell refused by default:")
    r = sb.run(["sh", "-c", "echo pwned"], policy=SandboxPolicy())
    print(f"    status={r.status.value}  reason={r.rationale}")

    print("\n[8] Env isolation — no inheritance:")
    r = sb.run_python(
        "import os\n"
        "print('PATH=' + os.environ.get('PATH','')[:40])\n"
        "print('HOME=' + os.environ.get('HOME',''))\n"
    )
    print(f"    stdout:\n{r.stdout}")

    print("\n[9] Read-only FS:")
    if platform.system() != "Windows":
        r = sb.run_python(
            "open('forbidden.txt','w').write('x')\n",
            policy=SandboxPolicy(read_only_fs=True),
        )
        print(f"    status={r.status.value}  "
              f"stderr_tail={r.stderr.strip().splitlines()[-1] if r.stderr else ''!r}")

    print("\n[10] Persistence:")
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain", log_level="WARNING")
        app = SEBrainApp(config=cfg)
        app.start()
        try:
            with execution_scope(project_id="demo"):
                mem = MemoryStore(app.storage)
                ont = Ontology(app.storage)
                repo = SandboxRepository(memory=mem, ontology=ont)
                r = sb.run_python("print('persist')\n")
                ent = repo.save(r, project_id="demo")
                print(f"    ontology entity: {ent[:12]}…")
                loaded = repo.load(r.id, project_id="demo")
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
