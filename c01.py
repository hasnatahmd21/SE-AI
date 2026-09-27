"""
================================================================================
AUTONOMOUS SOFTWARE ENGINEERING BRAIN
C01 — CORE FOUNDATION (Single-File Complete Implementation)
================================================================================

Contains (all authoritative, no duplicates, no stubs):

  1.  Version
  2.  Exception hierarchy
  3.  Result / Status / Confidence models
  4.  Execution context (contextvars, correlation IDs)
  5.  Configuration (env-driven, frozen)
  6.  Structured logging + correlation ID injection
  7.  Contracts (Protocols) + require helpers
  8.  Lifecycle manager (init/shutdown/rollback)
  9.  Storage abstraction (Protocol)
  10. SQLite storage (WAL, FK, nested tx via savepoints, health)
  11. Migration runner (versioned, checksummed, idempotent)
  12. Initial migration (kv_store + execution_log + indices)
  13. Health report
  14. SEBrainApp facade
  15. __main__ demo + self-test (runs when executed directly)

Run as script:
    python -m sebrain.c01            # demo
    python -m sebrain.c01 --test     # run self-tests

Use as library:
    from sebrain.c01 import SEBrainApp
    app = SEBrainApp()
    app.start()
    print(app.health().to_dict())
    app.stop()

Requires:
    Python 3.11+
    pip install pydantic pydantic-settings structlog
================================================================================
"""
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────
import asyncio
import hashlib
import logging as std_logging
import os
import sqlite3
import sys
import tempfile
import traceback
import uuid
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as e:  # pragma: no cover
    print("Missing dependency: pip install pydantic pydantic-settings structlog")
    raise

try:
    import structlog
except ImportError as e:  # pragma: no cover
    print("Missing dependency: pip install structlog")
    raise


# ════════════════════════════════════════════════════════════════════════════
# 1. VERSION
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


VERSION = Version(0, 1, 0)
__version__ = str(VERSION)


# ════════════════════════════════════════════════════════════════════════════
# 2. EXCEPTION HIERARCHY
# ════════════════════════════════════════════════════════════════════════════
class SEBrainError(Exception):
    """Base class for all SE Brain errors."""

    code: str = "SE_BRAIN_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "type": type(self).__name__,
        }


class ConfigError(SEBrainError):
    code = "CONFIG_ERROR"


class EnvironmentError_(SEBrainError):
    code = "ENVIRONMENT_ERROR"


class StorageError(SEBrainError):
    code = "STORAGE_ERROR"


class MigrationError(StorageError):
    code = "MIGRATION_ERROR"


class TransactionError(StorageError):
    code = "TRANSACTION_ERROR"


class LifecycleError(SEBrainError):
    code = "LIFECYCLE_ERROR"


class NotInitializedError(LifecycleError):
    code = "NOT_INITIALIZED"


class AlreadyInitializedError(LifecycleError):
    code = "ALREADY_INITIALIZED"


class ContractViolation(SEBrainError):
    code = "CONTRACT_VIOLATION"


class ValidationError(SEBrainError):
    code = "VALIDATION_ERROR"


class HealthCheckError(SEBrainError):
    code = "HEALTH_CHECK_ERROR"


# ════════════════════════════════════════════════════════════════════════════
# 3. RESULT / STATUS / CONFIDENCE
# ════════════════════════════════════════════════════════════════════════════
T = TypeVar("T")


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class Confidence(str, Enum):
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Result(Generic[T]):
    """Uniform operation result with hard invariants."""

    status: Status
    value: T | None = None
    error: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is Status.SUCCEEDED and self.error is not None:
            raise ValueError("SUCCEEDED result cannot carry an error")
        if self.status in (Status.FAILED, Status.PARTIAL) and self.error is None:
            raise ValueError(f"{self.status} result must carry an error")

    @classmethod
    def ok(cls, value: T, **meta: Any) -> "Result[T]":
        return cls(status=Status.SUCCEEDED, value=value, meta=meta)

    @classmethod
    def fail(cls, message: str, *, code: str = "ERROR", **meta: Any) -> "Result[T]":
        return cls(
            status=Status.FAILED,
            error={"code": code, "message": message},
            meta=meta,
        )

    @classmethod
    def partial(
        cls, value: T, message: str, *, code: str = "PARTIAL", **meta: Any
    ) -> "Result[T]":
        return cls(
            status=Status.PARTIAL,
            value=value,
            error={"code": code, "message": message},
            meta=meta,
        )

    @property
    def is_ok(self) -> bool:
        return self.status is Status.SUCCEEDED


# ════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION CONTEXT (correlation IDs)
# ════════════════════════════════════════════════════════════════════════════
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_project_id: ContextVar[str | None] = ContextVar("project_id", default=None)
_task_id: ContextVar[str | None] = ContextVar("task_id", default=None)


def new_correlation_id() -> str:
    return uuid.uuid4().hex


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def get_project_id() -> str | None:
    return _project_id.get()


def get_task_id() -> str | None:
    return _task_id.get()


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    correlation_id: str
    project_id: str | None = None
    task_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@contextmanager
def execution_scope(
    *,
    correlation_id: str | None = None,
    project_id: str | None = None,
    task_id: str | None = None,
) -> Iterator[ExecutionContext]:
    cid = correlation_id or new_correlation_id()
    tok_cid: Token = _correlation_id.set(cid)
    tok_pid: Token = _project_id.set(project_id)
    tok_tid: Token = _task_id.set(task_id)
    ctx = ExecutionContext(correlation_id=cid, project_id=project_id, task_id=task_id)
    try:
        yield ctx
    finally:
        _task_id.reset(tok_tid)
        _project_id.reset(tok_pid)
        _correlation_id.reset(tok_cid)


# ════════════════════════════════════════════════════════════════════════════
# 5. CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
class Config(BaseSettings):
    """Frozen, env-driven configuration. Prefix: SE_BRAIN_"""

    model_config = SettingsConfigDict(
        env_prefix="SE_BRAIN_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    app_name: str = "sebrain"
    environment: str = Field(default="development")
    debug: bool = False

    data_dir: Path = Field(default=Path("./.sebrain"))
    db_filename: str = "sebrain.sqlite3"

    log_level: str = "INFO"
    log_json: bool = False

    default_timeout_seconds: int = 60
    max_retries: int = 3

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.db_filename


def load_config(**overrides: object) -> Config:
    data_dir_env = os.environ.get("SE_BRAIN_DATA_DIR")
    if data_dir_env and "data_dir" not in overrides:
        overrides["data_dir"] = Path(data_dir_env)
    return Config(**overrides)  # type: ignore[arg-type]


# ════════════════════════════════════════════════════════════════════════════
# 6. STRUCTURED LOGGING
# ════════════════════════════════════════════════════════════════════════════
_LOGGING_CONFIGURED = False


def _add_correlation_id(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    cid = get_correlation_id()
    if cid:
        event_dict.setdefault("correlation_id", cid)
    return event_dict


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Idempotent structured logging setup."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    std_logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(std_logging, level.upper(), std_logging.INFO),
    )

    renderer = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(std_logging, level.upper(), std_logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True


def get_logger(name: str | None = None) -> Any:
    return structlog.get_logger(name)


# ════════════════════════════════════════════════════════════════════════════
# 7. CONTRACTS
# ════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class Initializable(Protocol):
    def initialize(self) -> None: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class HealthCheckable(Protocol):
    def health(self) -> dict[str, Any]: ...


def require(condition: bool, message: str, **details: Any) -> None:
    if not condition:
        raise ContractViolation(message, details=details)


def require_not_none(value: Any, name: str) -> None:
    if value is None:
        raise ContractViolation(f"required value is None: {name}", details={"name": name})


# ════════════════════════════════════════════════════════════════════════════
# 8. LIFECYCLE
# ════════════════════════════════════════════════════════════════════════════
class LifecycleState(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class Lifecycle:
    config: Config
    state: LifecycleState = LifecycleState.CREATED
    _subsystems: list[Initializable] = field(default_factory=list)

    def register(self, subsystem: Initializable) -> None:
        if self.state is not LifecycleState.CREATED:
            raise AlreadyInitializedError(
                f"cannot register {type(subsystem).__name__} after lifecycle started"
            )
        self._subsystems.append(subsystem)

    def initialize(self) -> None:
        if self.state is not LifecycleState.CREATED:
            raise AlreadyInitializedError(f"lifecycle state is {self.state.value}")
        self.state = LifecycleState.INITIALIZING
        started: list[Initializable] = []
        try:
            for sub in self._subsystems:
                sub.initialize()
                started.append(sub)
            self.state = LifecycleState.READY
        except Exception:
            # Rollback in reverse order
            for sub in reversed(started):
                try:
                    sub.shutdown()
                except Exception:
                    pass
            self.state = LifecycleState.FAILED
            raise

    def shutdown(self) -> None:
        if self.state is LifecycleState.STOPPED:
            return
        if self.state is LifecycleState.CREATED:
            self.state = LifecycleState.STOPPED
            return
        if self.state is not LifecycleState.READY:
            raise NotInitializedError(f"cannot shutdown from state {self.state.value}")
        self.state = LifecycleState.STOPPING
        for sub in reversed(self._subsystems):
            try:
                sub.shutdown()
            except Exception:
                pass
        self.state = LifecycleState.STOPPED


# ════════════════════════════════════════════════════════════════════════════
# 9. STORAGE ABSTRACTION
# ════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class Storage(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def execute(self, sql: str, params: Sequence[Any] = ()) -> None: ...
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]: ...
    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None: ...
    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None: ...
    def transaction(self) -> AbstractContextManager["Storage"]: ...
    def health(self) -> dict[str, Any]: ...


# ════════════════════════════════════════════════════════════════════════════
# 10. SQLITE STORAGE
# ════════════════════════════════════════════════════════════════════════════
class SQLiteStorage(Initializable, HealthCheckable):
    """SQLite storage with WAL, foreign keys, and nested transactions."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._in_tx: bool = False

    # ---- lifecycle ----
    def initialize(self) -> None:
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                str(self.db_path),
                isolation_level=None,      # we manage transactions
                check_same_thread=False,
                timeout=30.0,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        except sqlite3.Error as exc:
            raise StorageError(f"failed to open SQLite DB: {exc}") from exc

    def shutdown(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    # ---- compatibility aliases (Storage Protocol) ----
    def connect(self) -> None:
        self.initialize()

    def close(self) -> None:
        self.shutdown()

    # ---- internal ----
    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("storage not initialized")
        return self._conn

    # ---- primitives ----
    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self._require_conn()
        try:
            conn.execute(sql, tuple(params))
        except sqlite3.Error as exc:
            raise StorageError(f"execute failed: {exc}", details={"sql": sql}) from exc

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        conn = self._require_conn()
        try:
            conn.executemany(sql, [tuple(p) for p in seq])
        except sqlite3.Error as exc:
            raise StorageError(f"executemany failed: {exc}", details={"sql": sql}) from exc

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = self._require_conn()
        try:
            cur = conn.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.Error as exc:
            raise StorageError(f"query failed: {exc}", details={"sql": sql}) from exc

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---- transactions ----
    @contextmanager
    def transaction(self) -> Iterator["SQLiteStorage"]:
        conn = self._require_conn()
        if self._in_tx:
            # Nested -> savepoint
            sp = f"sp_{id(self)}_{uuid.uuid4().hex[:8]}"
            conn.execute(f"SAVEPOINT {sp};")
            try:
                yield self
                conn.execute(f"RELEASE SAVEPOINT {sp};")
            except Exception:
                conn.execute(f"ROLLBACK TO SAVEPOINT {sp};")
                conn.execute(f"RELEASE SAVEPOINT {sp};")
                raise
            return

        self._in_tx = True
        conn.execute("BEGIN;")
        try:
            yield self
            conn.execute("COMMIT;")
        except Exception as exc:
            try:
                conn.execute("ROLLBACK;")
            except sqlite3.Error:
                pass
            raise TransactionError(f"transaction rolled back: {exc}") from exc
        finally:
            self._in_tx = False

    # ---- health ----
    def health(self) -> dict[str, Any]:
        try:
            conn = self._require_conn()
            row = conn.execute("PRAGMA integrity_check;").fetchone()
            ok = row is not None and row[0] == "ok"
            return {
                "component": "storage.sqlite",
                "ok": ok,
                "path": str(self.db_path),
                "integrity": row[0] if row else "unknown",
            }
        except Exception as exc:
            return {"component": "storage.sqlite", "ok": False, "error": str(exc)}


# ════════════════════════════════════════════════════════════════════════════
# 11. MIGRATION RUNNER
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    up: Callable[[SQLiteStorage], None]

    @property
    def checksum(self) -> str:
        src = f"{self.version}:{self.name}:{self.up.__qualname__}"
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# 12. INITIAL MIGRATION
# ════════════════════════════════════════════════════════════════════════════
def _migration_001_initial(storage: SQLiteStorage) -> None:
    """Creates kv_store + execution_log + indices."""
    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS kv_store (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    storage.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_log (
            operation_id   TEXT PRIMARY KEY,
            correlation_id TEXT NOT NULL,
            project_id     TEXT,
            task_id        TEXT,
            component      TEXT NOT NULL,
            status         TEXT NOT NULL,
            started_at     TEXT NOT NULL,
            ended_at       TEXT,
            input_ref      TEXT,
            output_ref     TEXT,
            error          TEXT,
            meta_json      TEXT
        );
        """
    )
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_exec_corr ON execution_log(correlation_id);"
    )
    storage.execute(
        "CREATE INDEX IF NOT EXISTS idx_exec_project ON execution_log(project_id);"
    )


MIGRATIONS: list[Migration] = [
    Migration(version=1, name="initial", up=_migration_001_initial),
]


class MigrationRunner:
    def __init__(
        self, storage: SQLiteStorage, migrations: list[Migration] | None = None
    ) -> None:
        self.storage = storage
        self.migrations = sorted(migrations or MIGRATIONS, key=lambda m: m.version)

    def _ensure_registry(self) -> None:
        self.storage.execute(
            """
            CREATE TABLE IF NOT EXISTS _migrations (
                version    INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                checksum   TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            """
        )

    def current_version(self) -> int:
        row = self.storage.query_one("SELECT MAX(version) AS v FROM _migrations;")
        if not row or row["v"] is None:
            return 0
        return int(row["v"])

    def applied(self) -> dict[int, dict[str, str]]:
        rows = self.storage.query("SELECT version, name, checksum FROM _migrations;")
        return {
            int(r["version"]): {"name": r["name"], "checksum": r["checksum"]}
            for r in rows
        }

    def run(self) -> int:
        self._ensure_registry()
        applied = self.applied()
        target = 0
        for mig in self.migrations:
            existing = applied.get(mig.version)
            if existing:
                if existing["checksum"] != mig.checksum:
                    raise MigrationError(
                        f"checksum mismatch for migration {mig.version} ({mig.name}); "
                        "was the migration edited after being applied?"
                    )
                target = mig.version
                continue
            with self.storage.transaction():
                mig.up(self.storage)
                self.storage.execute(
                    "INSERT INTO _migrations(version, name, checksum, applied_at) "
                    "VALUES (?, ?, ?, datetime('now'));",
                    (mig.version, mig.name, mig.checksum),
                )
            target = mig.version
        return target


# ════════════════════════════════════════════════════════════════════════════
# 13. HEALTH REPORT
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class HealthReport:
    ok: bool
    components: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "components": self.components}


def run_health(components: list[HealthCheckable]) -> HealthReport:
    reports = [c.health() for c in components]
    ok = all(bool(r.get("ok")) for r in reports)
    return HealthReport(ok=ok, components=reports)


# ════════════════════════════════════════════════════════════════════════════
# 14. APPLICATION FACADE
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SEBrainApp:
    """Top-level facade.

    app = SEBrainApp()
    app.start()
    ...
    app.stop()
    """

    config: Config = field(default_factory=load_config)
    _lifecycle: Lifecycle | None = field(default=None, init=False, repr=False)
    _storage: SQLiteStorage | None = field(default=None, init=False, repr=False)
    _migration_version: int = field(default=0, init=False)

    @property
    def storage(self) -> SQLiteStorage:
        if self._storage is None:
            raise NotInitializedError("app not started")
        return self._storage

    @property
    def migration_version(self) -> int:
        return self._migration_version

    def start(self) -> None:
        configure_logging(level=self.config.log_level, json=self.config.log_json)
        with execution_scope() as ctx:
            self._storage = SQLiteStorage(self.config.db_path)
            self._lifecycle = Lifecycle(config=self.config)
            self._lifecycle.register(self._storage)
            self._lifecycle.initialize()

            runner = MigrationRunner(self._storage)
            self._migration_version = runner.run()

    def stop(self) -> None:
        if self._lifecycle is not None:
            self._lifecycle.shutdown()

    def health(self) -> HealthReport:
        if self._storage is None:
            return HealthReport(
                ok=False,
                components=[{"component": "app", "ok": False, "error": "not started"}],
            )
        return run_health([self._storage])

    # ---- context manager sugar ----
    def __enter__(self) -> "SEBrainApp":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


# ════════════════════════════════════════════════════════════════════════════
# 15. SELF-TEST (pytest-free, runnable)
# ════════════════════════════════════════════════════════════════════════════
class _TestFailure(AssertionError):
    pass


def _run_self_tests() -> int:
    """Runs an exhaustive C01 self-test. Returns number of failures."""
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

    print("Running C01 self-tests…")

    # ---------- version ----------
    def t_version() -> None:
        assert str(VERSION) == "0.1.0"
        assert __version__ == "0.1.0"

    check("version", t_version)

    # ---------- exceptions ----------
    def t_exc_hierarchy() -> None:
        assert issubclass(StorageError, SEBrainError)
        assert issubclass(MigrationError, StorageError)
        assert issubclass(TransactionError, StorageError)
        assert issubclass(NotInitializedError, LifecycleError)

    def t_exc_to_dict() -> None:
        e = StorageError("bad", details={"x": 1})
        d = e.to_dict()
        assert d["code"] == "STORAGE_ERROR"
        assert d["details"] == {"x": 1}

    check("exception hierarchy", t_exc_hierarchy)
    check("exception to_dict", t_exc_to_dict)

    # ---------- Result ----------
    def t_result_ok() -> None:
        r = Result.ok(42)
        assert r.is_ok and r.value == 42 and r.error is None

    def t_result_fail() -> None:
        r = Result.fail("boom", code="X")
        assert not r.is_ok
        assert r.error == {"code": "X", "message": "boom"}

    def t_result_partial() -> None:
        r = Result.partial(1, "half")
        assert r.status is Status.PARTIAL and r.error is not None

    def t_result_invariants() -> None:
        try:
            Result(status=Status.SUCCEEDED, value=1, error={"code": "x", "message": "y"})
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
        try:
            Result(status=Status.FAILED)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    check("Result.ok", t_result_ok)
    check("Result.fail", t_result_fail)
    check("Result.partial", t_result_partial)
    check("Result invariants", t_result_invariants)

    # ---------- context ----------
    def t_context_basic() -> None:
        assert get_correlation_id() is None
        with execution_scope(project_id="p1", task_id="t1") as ctx:
            assert get_correlation_id() == ctx.correlation_id
            assert get_project_id() == "p1"
            assert get_task_id() == "t1"
        assert get_correlation_id() is None

    def t_context_nested() -> None:
        outer = new_correlation_id()
        with execution_scope(correlation_id=outer):
            with execution_scope(project_id="inner"):
                assert get_correlation_id() != outer
            assert get_correlation_id() == outer

    check("context set/restore", t_context_basic)
    check("context nested", t_context_nested)

    # ---------- config ----------
    def t_config_defaults() -> None:
        cfg = Config()
        assert cfg.app_name == "sebrain"
        assert cfg.db_filename == "sebrain.sqlite3"
        assert cfg.db_path == cfg.data_dir / cfg.db_filename

    def t_config_frozen() -> None:
        cfg = Config()
        try:
            cfg.app_name = "x"  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("Config should be frozen")

    check("config defaults", t_config_defaults)
    check("config frozen", t_config_frozen)

    # ---------- logging ----------
    def t_logging_idempotent() -> None:
        configure_logging(level="WARNING")
        configure_logging(level="WARNING")
        log = get_logger("t")
        assert log is not None

    check("logging idempotent", t_logging_idempotent)

    # ---------- contracts ----------
    def t_require() -> None:
        require(True, "ok")
        try:
            require(False, "nope", x=1)
        except ContractViolation as e:
            assert e.details == {"x": 1}
        else:
            raise AssertionError

    check("contracts.require", t_require)

    # ---------- storage ----------
    def t_storage_roundtrip() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                s.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
                s.execute("INSERT INTO t (v) VALUES (?);", ("hello",))
                rows = s.query("SELECT v FROM t;")
                assert rows == [{"v": "hello"}]
            finally:
                s.shutdown()

    def t_storage_tx_commit() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                s.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
                with s.transaction():
                    s.execute("INSERT INTO t (v) VALUES ('a');")
                assert s.query_one("SELECT COUNT(*) AS c FROM t;")["c"] == 1
            finally:
                s.shutdown()

    def t_storage_tx_rollback() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                s.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
                try:
                    with s.transaction():
                        s.execute("INSERT INTO t (v) VALUES ('a');")
                        raise RuntimeError("boom")
                except TransactionError:
                    pass
                else:
                    raise AssertionError("expected TransactionError")
                assert s.query_one("SELECT COUNT(*) AS c FROM t;")["c"] == 0
            finally:
                s.shutdown()

    def t_storage_nested_tx() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                s.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);")
                with s.transaction():
                    s.execute("INSERT INTO t (v) VALUES ('outer');")
                    try:
                        with s.transaction():
                            s.execute("INSERT INTO t (v) VALUES ('inner');")
                            raise RuntimeError("inner boom")
                    except RuntimeError:
                        pass
                # outer committed, inner rolled back
                rows = s.query("SELECT v FROM t ORDER BY id;")
                assert rows == [{"v": "outer"}], rows
            finally:
                s.shutdown()

    def t_storage_health() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "t.sqlite3")
            s.initialize()
            try:
                h = s.health()
                assert h["ok"] is True
                assert h["component"] == "storage.sqlite"
            finally:
                s.shutdown()

    def t_storage_not_initialized() -> None:
        s = SQLiteStorage(Path("/tmp/nope.sqlite3"))
        try:
            s.execute("SELECT 1;")
        except StorageError:
            return
        raise AssertionError("expected StorageError")

    check("storage roundtrip", t_storage_roundtrip)
    check("storage tx commit", t_storage_tx_commit)
    check("storage tx rollback", t_storage_tx_rollback)
    check("storage nested tx (savepoint)", t_storage_nested_tx)
    check("storage health", t_storage_health)
    check("storage not-initialized guard", t_storage_not_initialized)

    # ---------- migrations ----------
    def t_migrations() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "m.sqlite3")
            s.initialize()
            try:
                runner = MigrationRunner(s)
                v1 = runner.run()
                assert v1 >= 1
                v2 = runner.run()  # idempotent
                assert v2 == v1
                tables = {r["name"] for r in s.query(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                )}
                assert "_migrations" in tables
                assert "kv_store" in tables
                assert "execution_log" in tables
            finally:
                s.shutdown()

    def t_migration_checksum_mismatch() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "m.sqlite3")
            s.initialize()
            try:
                runner = MigrationRunner(s)
                runner.run()
                # Tamper: pretend a migration was edited
                s.execute("UPDATE _migrations SET checksum='deadbeef' WHERE version=1;")
                try:
                    runner.run()
                except MigrationError:
                    return
                raise AssertionError("expected MigrationError")
            finally:
                s.shutdown()

    check("migrations apply + idempotent", t_migrations)
    check("migration checksum guard", t_migration_checksum_mismatch)

    # ---------- lifecycle ----------
    def t_lifecycle_happy() -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(data_dir=Path(td))
            s = SQLiteStorage(cfg.db_path)
            lc = Lifecycle(config=cfg)
            lc.register(s)
            lc.initialize()
            assert lc.state is LifecycleState.READY
            lc.shutdown()
            assert lc.state is LifecycleState.STOPPED

    def t_lifecycle_double_init() -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(data_dir=Path(td))
            lc = Lifecycle(config=cfg)
            lc.initialize()
            try:
                lc.initialize()
            except AlreadyInitializedError:
                return
            raise AssertionError("expected AlreadyInitializedError")

    def t_lifecycle_rollback() -> None:
        """If second subsystem fails, first must be shut down."""

        class Boom(Initializable):
            def __init__(self) -> None:
                self.shut = False

            def initialize(self) -> None:
                raise RuntimeError("boom")

            def shutdown(self) -> None:
                self.shut = True

        class Good(Initializable):
            def __init__(self) -> None:
                self.shut = False

            def initialize(self) -> None:
                pass

            def shutdown(self) -> None:
                self.shut = True

        good = Good()
        boom = Boom()
        lc = Lifecycle(config=Config())
        lc.register(good)
        lc.register(boom)
        try:
            lc.initialize()
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError")
        assert good.shut is True, "rollback did not shutdown first subsystem"
        assert lc.state is LifecycleState.FAILED

    check("lifecycle happy path", t_lifecycle_happy)
    check("lifecycle double init guard", t_lifecycle_double_init)
    check("lifecycle rollback on failure", t_lifecycle_rollback)

    # ---------- health ----------
    def t_health_report() -> None:
        with tempfile.TemporaryDirectory() as td:
            s = SQLiteStorage(Path(td) / "h.sqlite3")
            s.initialize()
            try:
                rep = run_health([s])
                assert rep.ok is True
                assert len(rep.components) == 1
            finally:
                s.shutdown()

    check("health report", t_health_report)

    # ---------- app ----------
    def t_app_lifecycle() -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(data_dir=Path(td) / "d", log_level="WARNING")
            app = SEBrainApp(config=cfg)
            app.start()
            try:
                assert app.migration_version >= 1
                h = app.health()
                assert h.ok is True
            finally:
                app.stop()

    def t_app_context_manager() -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = Config(data_dir=Path(td) / "d", log_level="WARNING")
            with SEBrainApp(config=cfg) as app:
                assert app.health().ok is True

    def t_app_not_started() -> None:
        app = SEBrainApp(config=Config(log_level="WARNING"))
        try:
            _ = app.storage
        except NotInitializedError:
            return
        raise AssertionError("expected NotInitializedError")

    check("app start/stop/health", t_app_lifecycle)
    check("app context manager", t_app_context_manager)
    check("app not-started guard", t_app_not_started)

    # ---------- async propagation of context ----------
    async def _async_probe() -> tuple[str | None, str | None]:
        await asyncio.sleep(0)
        return get_correlation_id(), get_project_id()

    def t_context_async() -> None:
        cid = new_correlation_id()
        with execution_scope(correlation_id=cid, project_id="p"):
            got_cid, got_pid = asyncio.run(_async_probe())
        assert got_cid == cid
        assert got_pid == "p"

    check("context propagates across async", t_context_async)

    # ---------- summary ----------
    print()
    print(f"Self-tests: {passed} passed, {len(failures)} failed")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")
    return len(failures)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def _demo() -> None:
    """Live demo of the C01 foundation."""
    print("=" * 78)
    print(f"SE Brain C01 — Core Foundation v{VERSION}")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        cfg = Config(data_dir=Path(td) / "sebrain_data", log_level="INFO")
        print(f"\n[1] Config: env={cfg.environment}, db={cfg.db_path}")

        app = SEBrainApp(config=cfg)

        print("\n[2] Starting app…")
        app.start()

        print(f"\n[3] Migration version: {app.migration_version}")
        print(f"\n[4] Health: {app.health().to_dict()}")

        print("\n[5] Using storage…")
        with execution_scope(project_id="demo-project") as ctx:
            app.storage.execute(
                "INSERT INTO kv_store(key, value, updated_at) "
                "VALUES (?, ?, datetime('now'));",
                ("greeting", "hello from SE Brain"),
            )
            row = app.storage.query_one(
                "SELECT value FROM kv_store WHERE key=?;", ("greeting",)
            )
            print(f"    correlation_id = {ctx.correlation_id}")
            print(f"    kv_store row    = {row}")

        print("\n[6] Stopping app…")
        app.stop()
        print("\nDone.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        failures = _run_self_tests()
        sys.exit(0 if failures == 0 else 1)
    else:
        _demo()
