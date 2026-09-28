from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c27 import ExperienceInput, LearningEngine, KnowledgeCandidate, Validator


def test_c27_archived_knowledge_is_not_used_for_contradiction_checks():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        memory.upsert(
            MemoryKind.LONG_TERM,
            "old",
            {
                "problem_pattern": "database index hot path",
                "lesson_pattern": "database index hot path failed regressed",
                "approach_pattern": "database index",
            },
            scope_type=MemoryScope.GLOBAL,
        )
        entry = memory.get_current(
            MemoryKind.LONG_TERM, "old",
            scope_type=MemoryScope.GLOBAL, scope_id=None,
        )
        assert entry is not None
        memory.archive(entry.id)

        engine = LearningEngine(memory=memory)
        exp = ExperienceInput(
            id="a", problem="database index hot path",
            approach="database index",
            lesson="database index hot path worked",
            reliability="reliable", score=0.9,
            verification="supported=2", project_id="p1",
        )
        rep = engine.learn([exp, ExperienceInput(**{**exp.to_dict(), "id": "b"})], project_id="p1")
        assert not any(v.status.value == "contradicted" for v in rep.validations)
