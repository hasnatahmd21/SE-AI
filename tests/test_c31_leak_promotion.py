from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import Confidence
from sebrain.c31 import CrossProjectKnowledgeStore, LeakVerdict


def test_c31_promotion_rejects_source_project_id_leak():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)
        rec = store.put_project(
            "project-a", "secret.pattern",
            {"owner_project": "project-a", "value": "private"},
            confidence=Confidence.VERIFIED,
        )

        transition, evaluation = store.promote(
            "project-a", rec.key, actor="tester", rationale="share"
        )

        assert evaluation is not None
        assert transition.action.value == "reject"
        assert store.list_shared() == []


def test_c31_leak_checker_scans_source_without_target():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)
        verdict, reasons = store.leak_check(
            {"owner": "project-a"},
            source_project_id="project-a",
            target_project_id="",
        )
        assert verdict is LeakVerdict.LEAK
        assert reasons
