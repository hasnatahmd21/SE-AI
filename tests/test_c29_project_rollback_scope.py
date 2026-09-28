from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c29 import GovernanceEngine, RollbackReason


def _seed(engine, memory, project_id, scope, version, change_id):
    memory.create(
        MemoryKind.PROJECT,
        f"self_change:{scope}:seed-{project_id}-{version}",
        {
            "change_id": change_id,
            "version": version,
            "scope": scope,
            "final_status": "promoted",
        },
        scope_type=MemoryScope.PROJECT,
        scope_id=project_id,
    )


def test_c29_rollback_is_project_scoped():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = GovernanceEngine(memory=memory)

        _seed(engine, memory, "project-a", "shared-scope", 1, "a1")
        _seed(engine, memory, "project-a", "shared-scope", 2, "a2")
        _seed(engine, memory, "project-b", "shared-scope", 1, "b1")
        _seed(engine, memory, "project-b", "shared-scope", 2, "b2")

        rec = engine.rollback(
            scope="shared-scope",
            reason=RollbackReason.MANUAL,
            project_id="project-a",
            to_version=1,
        )

        assert rec.from_version == 2
        assert rec.to_version == 1

        a = engine._entries_for_scope("shared-scope", project_id="project-a")
        b = engine._entries_for_scope("shared-scope", project_id="project-b")

        assert any(e.status.value == "active" and e.content.get("change_id") == "a1" for e in a)
        assert any(e.status.value == "active" and e.content.get("change_id") == "b2" for e in b)
        assert not any(e.status.value == "active" and e.content.get("change_id") == "b1" for e in b)
