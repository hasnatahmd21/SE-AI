from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c29 import GovernanceEngine, RollbackReason


def test_c29_rollback_is_atomic_on_restore_failure():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = GovernanceEngine(memory=memory)
        for version, change_id in [(1, "c1"), (2, "c2")]:
            memory.create(
                MemoryKind.PROJECT,
                f"self_change:s:seed-{version}",
                {"change_id": change_id, "version": version,
                 "scope": "s", "final_status": "promoted"},
                scope_type=MemoryScope.PROJECT, scope_id="p",
            )
        original = memory.create
        calls = {"n": 0}
        def fail_on_restore(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return original(*args, **kwargs)
            raise RuntimeError("synthetic restore failure")
        memory.create = fail_on_restore
        try:
            engine.rollback(
                scope="s", reason=RollbackReason.MANUAL,
                project_id="p", to_version=1,
            )
        except RuntimeError as exc:
            assert "synthetic restore failure" in str(exc)
        else:
            raise AssertionError("restore failure was swallowed")
        finally:
            memory.create = original

        entries = engine._entries_for_scope("s", project_id="p")
        active = [e for e in entries if e.status.value == "active"]
        assert any(e.content.get("change_id") == "c2" for e in active)
        assert not any(e.key.endswith(":rollback-active") for e in active)
