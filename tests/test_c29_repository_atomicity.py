from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage, ValidationError
from sebrain.c02 import Ontology
from sebrain.c04 import MemoryStore
from sebrain.c29 import GovernanceRepository, GovernanceReport


def test_c29_repository_rolls_back_memory_when_ontology_save_fails():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        ontology = Ontology(storage)
        repo = GovernanceRepository(memory, ontology)
        report = GovernanceReport(project_id="p1", change_id="c1")

        original = ontology.add
        ontology.add = lambda *a, **k: (_ for _ in ()).throw(
            ValidationError("synthetic ontology failure")
        )
        try:
            repo.save(report, project_id="p1")
        except ValidationError as exc:
            assert "synthetic ontology failure" in str(exc)
        else:
            raise AssertionError("ontology failure was swallowed")
        finally:
            ontology.add = original

        assert repo.load(report.id, project_id="p1") is None
