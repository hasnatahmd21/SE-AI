from pathlib import Path
from tempfile import TemporaryDirectory

from sebrain.c01 import SQLiteStorage, ValidationError
from sebrain.c04 import MemoryKind, MemoryScope, MemoryStore
from sebrain.c26 import (
    ExperienceCandidate,
    ExperienceExtractor,
    ExperienceRecord,
    PromotionDecision,
    PromotionStatus,
    Reliability,
)


def test_c26_memory_harvest_failure_is_not_silent():
    storage = SQLiteStorage(Path("/tmp/nonexistent-c26-test") / "m.sqlite3")
    storage.initialize()
    memory = MemoryStore(storage)

    def broken_find(*args, **kwargs):
        raise RuntimeError("synthetic memory failure")

    memory.find = broken_find
    extractor = ExperienceExtractor(memory=memory)
    try:
        extractor.build_bundles(project_id="p1")
    except ValidationError as exc:
        assert "memory harvest failed" in str(exc)
    else:
        raise AssertionError("memory harvest failure was swallowed")


def test_c26_rejects_non_promoted_memory_persistence():
    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        extractor = ExperienceExtractor(memory=memory)
        candidate = ExperienceCandidate(problem="x")
        decision = PromotionDecision(
            candidate_id=candidate.id,
            status=PromotionStatus.HELD,
            reliability=Reliability.WEAK,
            score=0.3,
        )
        record = ExperienceRecord(candidate=candidate, decision=decision)
        try:
            extractor.promote_to_memory(record, project_id="p1")
        except ValidationError as exc:
            assert "only PROMOTED" in str(exc)
        else:
            raise AssertionError("held experience was persisted")
        assert memory.find(
            kind=MemoryKind.EXPERIENCE,
            scope_type=MemoryScope.PROJECT,
            scope_id="p1",
        ) == []
