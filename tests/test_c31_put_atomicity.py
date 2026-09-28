from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage, TransactionError
from sebrain.c04 import Confidence
from sebrain.c31 import CrossProjectKnowledgeStore


def test_c31_put_rolls_back_knowledge_when_audit_fails():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)

        original = store._audit
        store._audit = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic audit failure")
        )
        try:
            store.put_project("project-a", "k", {"v": 1})
        except TransactionError as exc:
            assert "synthetic audit failure" in str(exc)
        else:
            raise AssertionError("audit failure was swallowed")
        finally:
            store._audit = original

        assert store.resolve("project-a", "k") is None
