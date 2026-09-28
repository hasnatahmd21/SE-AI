from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import Confidence
from sebrain.c31 import CrossProjectKnowledgeStore


def test_c31_forget_shared_rolls_back_archive_on_transition_failure():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)
        store.put_shared("k", {"v": 1}, confidence=Confidence.HIGH)

        original = store._persist_transition
        store._persist_transition = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("synthetic transition failure")
        )
        try:
            store.forget_shared("k", actor="tester", rationale="cleanup")
        except RuntimeError as exc:
            assert "synthetic transition failure" in str(exc)
        else:
            raise AssertionError("transition failure was swallowed")
        finally:
            store._persist_transition = original

        rec = store.resolve("project-a", "k")
        assert rec is not None
        assert rec.status == "active"
        assert store.history("k") == []
