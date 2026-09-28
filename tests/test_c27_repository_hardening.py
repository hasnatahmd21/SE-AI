from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage, ValidationError
from sebrain.c02 import Ontology
from sebrain.c04 import MemoryStore
from sebrain.c27 import LearningReport, LearningRepository


def test_c27_repository_does_not_swallow_ontology_link_failure():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        ontology = Ontology(storage)
        repo = LearningRepository(memory, ontology)

        report = LearningReport(project_id="p1")
        report.candidates = []

        original = ontology.link
        def broken_link(*args, **kwargs):
            raise ValidationError("synthetic ontology link failure")
        ontology.link = broken_link
        try:
            repo.save(report, project_id="p1")
        except ValidationError as exc:
            assert "synthetic ontology link failure" in str(exc)
        else:
            raise AssertionError("ontology persistence failure was swallowed")
        finally:
            ontology.link = original


def test_c27_promoted_patterns_excludes_archived_entries():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        memory.upsert(
            __import__("sebrain.c04", fromlist=["MemoryKind"]).MemoryKind.LONG_TERM,
            "active-key", {"candidate_id": "a"},
            scope_type=__import__("sebrain.c04", fromlist=["MemoryScope"]).MemoryScope.GLOBAL,
        )
        memory.upsert(
            __import__("sebrain.c04", fromlist=["MemoryKind"]).MemoryKind.LONG_TERM,
            "archived-key", {"candidate_id": "b"},
            scope_type=__import__("sebrain.c04", fromlist=["MemoryScope"]).MemoryScope.GLOBAL,
        )
        archived = memory.get_current(
            __import__("sebrain.c04", fromlist=["MemoryKind"]).MemoryKind.LONG_TERM,
            "archived-key",
            scope_type=__import__("sebrain.c04", fromlist=["MemoryScope"]).MemoryScope.GLOBAL,
            scope_id=None,
        )
        assert archived is not None
        memory.archive(archived.id)

        from sebrain.c27 import LearningEngine
        patterns = LearningRepository(memory).promoted_patterns()
        keys = {p["key"] for p in patterns}
        assert "active-key" in keys
        assert "archived-key" not in keys
