from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c31 import CrossProjectKnowledgeStore


def test_c31_history_for_project_is_tenant_scoped():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)

        a = store.put_project("project-a", "k-a", {"v": "a"})
        b = store.put_project("project-b", "k-b", {"v": "b"})
        store.promote("project-a", a.key, actor="a")
        store.promote("project-b", b.key, actor="b")

        a_history = store.history_for_project("project-a", "k-a")
        b_history = store.history_for_project("project-b", "k-b")
        assert a_history
        assert all(x.owner_project_id in ("project-a", "") for x in a_history)
        assert b_history
        assert all(x.owner_project_id in ("project-b", "") for x in b_history)

        assert store.history_for_project("project-a", "k-b") == []
        assert store.history_for_project("project-b", "k-a") == []
