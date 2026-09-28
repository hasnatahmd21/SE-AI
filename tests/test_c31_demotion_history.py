from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c31 import CrossProjectKnowledgeStore


def test_c31_demotion_recreates_active_project_record_after_archived_history():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "c31.sqlite3")
        storage.initialize()
        store = CrossProjectKnowledgeStore(storage)

        shared = store.put_shared("pattern", {"v": 1})
        # Seed prior archived project history for the same key.
        storage.execute(
            "INSERT INTO c31_knowledge("
            "id, scope, owner_project_id, key, content, tags, version_req, "
            "confidence, provenance_json, rationale, status, created_at, updated_at"
            ") VALUES (?, 'project', ?, ?, ?, '[]', '{}', 'unknown', '{}', '', "
            "'archived', ?, ?);",
            ("old-project-row", "project-a", "pattern", '{"v": 0}', "t", "t"),
        )

        transition = store.demote_to_project(
            "pattern", target_project_id="project-a", actor="tester"
        )

        assert transition.policy_verdict == "demote_to_project"
        resolved = store.resolve("project-a", "pattern")
        assert resolved is not None
        assert resolved.status == "active"
        assert resolved.content == {"v": 1}
        assert store.resolve("project-b", "pattern").scope.value == "shared"
        assert store.resolve("project-b", "pattern").content == {"v": 1}
        assert storage.query_one(
            "SELECT status FROM c31_knowledge WHERE id=?",
            ("old-project-row",),
        )["status"] == "archived"
