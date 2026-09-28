from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sebrain.c01 import Confidence, SQLiteStorage, StorageError
from sebrain.c31 import CrossProjectKnowledgeStore


class FailingStorage(SQLiteStorage):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.fail_shared_insert = False
        self.fail_project_insert = False

    def execute(self, sql, params=()):
        if self.fail_shared_insert and "INSERT INTO c31_knowledge" in sql and params[1] == "shared":
            raise StorageError("synthetic shared insert failure")
        if self.fail_project_insert and "INSERT INTO c31_knowledge" in sql and params[1] == "project":
            raise StorageError("synthetic project insert failure")
        return super().execute(sql, params)


def test_promote_rolls_back_transition_when_shared_insert_fails():
    with TemporaryDirectory() as td:
        storage = FailingStorage(Path(td) / "db.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)
        store.put_project("p", "k", {"v": 1}, confidence=Confidence.VERIFIED)
        storage.fail_shared_insert = True
        with pytest.raises(StorageError):
            store.promote("p", "k", actor="tester")
        assert store.resolve("p", "k") is not None
        assert store.resolve("other", "k") is None
        assert store.history("k") == []


def test_demote_rolls_back_copy_when_project_insert_fails():
    with TemporaryDirectory() as td:
        storage = FailingStorage(Path(td) / "db.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)
        store.put_shared("k", {"v": 1}, confidence=Confidence.VERIFIED)
        storage.fail_project_insert = True
        with pytest.raises(StorageError):
            store.demote_to_project("k", target_project_id="p", actor="tester")
        assert store.resolve("p", "k") is None
        shared = store.resolve("other", "k")
        assert shared is not None and shared.is_shared()
        assert store.history("k") == []
