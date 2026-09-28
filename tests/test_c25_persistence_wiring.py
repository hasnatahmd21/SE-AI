from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryStore
from sebrain.c25 import CodebaseUnderstanding, CodebaseUnderstandingBundle


def test_c25_persist_writes_project_memory():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        cb = CodebaseUnderstanding(memory=memory)

        bundle = CodebaseUnderstandingBundle(
            project_id="p1",
            root="/tmp/repo",
            total_modules=1,
        )
        key = cb.persist(bundle, project_id="p1")

        assert key == f"understanding:{bundle.id}"
        saved = memory.get_current(
            __import__("sebrain.c04", fromlist=["MemoryKind"]).MemoryKind.PROJECT,
            key,
            scope_type=__import__("sebrain.c04", fromlist=["MemoryScope"]).MemoryScope.PROJECT,
            scope_id="p1",
        )
        assert saved is not None
        assert saved.content["id"] == bundle.id


def test_c25_persist_without_memory_fails_explicitly():
    cb = CodebaseUnderstanding()
    bundle = CodebaseUnderstandingBundle()
    try:
        cb.persist(bundle, project_id="p1")
    except Exception as exc:
        assert "persistence is not configured" in str(exc)
    else:
        raise AssertionError("unconfigured persistence reported success")
