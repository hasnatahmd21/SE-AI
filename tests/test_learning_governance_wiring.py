from sebrain.c27 import (
    ExperienceInput,
    LearningEngine,
    PromotionAction,
)


def _exp(i):
    return ExperienceInput(
        id=i,
        kind="optimization",
        reliability="reliable",
        score=0.9,
        problem="database query timeout under heavy load",
        approach="added index on hot column",
        lesson="indexing the hot path improved query performance",
        verification="supported=3 refuted=0",
        project_id="p1",
    )


def test_promotion_cap_is_enforced_before_persistence():
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from sebrain.c01 import SQLiteStorage
    from sebrain.c04 import MemoryStore, MemoryKind

    with TemporaryDirectory() as td:
        storage = SQLiteStorage(Path(td) / "m.sqlite3")
        storage.initialize()
        memory = MemoryStore(storage)
        engine = LearningEngine(memory=memory, max_promoted_per_run=1)

        report = engine.learn([_exp("a"), _exp("b"), _exp("c"), _exp("d")])
        persisted = memory.find(kind=MemoryKind.LONG_TERM, status=None)

        assert len(report.promoted_ids) <= 1
        assert len(persisted) <= 1
        assert all(
            p.action is not PromotionAction.PROMOTE_TO_PATTERN
            and p.action is not PromotionAction.PROMOTE_TO_GENERAL
            or p.candidate_id in report.promoted_ids
            for p in report.promotions
        )


def test_demotion_failure_is_not_reported_as_success():
    class BrokenMemory:
        def get_current(self, *args, **kwargs):
            return type("E", (), {
                "id": "entry-1",
                "content": {"candidate_id": "cand-1"},
            })()

        def archive(self, _id):
            raise RuntimeError("archive failed")

    engine = LearningEngine(memory=BrokenMemory())
    try:
        engine.demote(key="learned:test", reason="contradicted")
    except RuntimeError as exc:
        assert "archive failed" in str(exc)
    else:
        raise AssertionError("demotion failure was silently reported as success")
