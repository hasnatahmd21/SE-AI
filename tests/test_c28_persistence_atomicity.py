from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage, ValidationError, TransactionError
from sebrain.c02 import Ontology
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c28 import (
    CheckOutcome, MetaAssessment, MetaRepository, MetaVerdict,
    MetaCheck, MetaQuestion,
)


def test_c28_stop_persistence_rolls_back_if_failure_memory_write_fails():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        repo = MetaRepository(memory)
        assessment = MetaAssessment(
            project_id="p1",
            verdict=MetaVerdict.STOP,
            checks=[
                MetaCheck(
                    question=MetaQuestion.EVIDENCE_SUFFICIENT,
                    outcome=CheckOutcome.BLOCKING,
                    rationale="synthetic stop",
                )
            ],
        )
        original = memory.record_failure
        memory.record_failure = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("synthetic failure")
        )
        try:
            repo.save(assessment, project_id="p1")
        except TransactionError as exc:
            assert "synthetic failure" in str(exc)
        else:
            raise AssertionError("persistence failure was swallowed")
        finally:
            memory.record_failure = original
        assert memory.get_current(
            MemoryKind.PROJECT,
            f"meta_assessment:{assessment.id}",
            scope_type=MemoryScope.PROJECT,
            scope_id="p1",
        ) is None


def test_c28_stop_persistence_rolls_back_if_ontology_write_fails():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        ontology = Ontology(storage)
        repo = MetaRepository(memory, ontology)
        assessment = MetaAssessment(
            project_id="p1",
            verdict=MetaVerdict.STOP,
            checks=[
                MetaCheck(
                    question=MetaQuestion.EVIDENCE_SUFFICIENT,
                    outcome=CheckOutcome.BLOCKING,
                    rationale="synthetic stop",
                )
            ],
        )
        original = ontology.add
        ontology.add = lambda *a, **k: (_ for _ in ()).throw(
            ValidationError("synthetic ontology failure")
        )
        try:
            repo.save(assessment, project_id="p1")
        except TransactionError as exc:
            assert "synthetic ontology failure" in str(exc)
        else:
            raise AssertionError("ontology failure was swallowed")
        finally:
            ontology.add = original
        assert memory.get_current(
            MemoryKind.PROJECT,
            f"meta_assessment:{assessment.id}",
            scope_type=MemoryScope.PROJECT,
            scope_id="p1",
        ) is None
