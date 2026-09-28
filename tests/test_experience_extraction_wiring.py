from sebrain.c26 import (
    ExperienceExtractor,
    Signal,
    SignalSource,
)


def test_episode_bundle_excludes_unrelated_memory_signals():
    extractor = ExperienceExtractor(memory=object())

    extractor.harvester.from_memory = lambda memory, project_id: [
        Signal(
            source=SignalSource.FAILURE_MEMORY,
            ref="related",
            text="database migration regression missing index",
        ),
        Signal(
            source=SignalSource.DECISION_MEMORY,
            ref="unrelated",
            text="frontend color palette typography choice",
        ),
    ]

    class Debug:
        id = "debug-1"
        category = type("C", (), {"value": "logic"})()
        signature = type(
            "S",
            (),
            {"exception_type": "AssertionError", "exception_message": "database migration regression"},
        )()

    bundles = extractor.build_bundles(
        project_id="p1",
        debug_report=Debug(),
        from_memory=True,
    )

    assert len(bundles) == 1
    refs = {s.ref for s in bundles[0].signals}
    assert "debug-1" in refs
    assert "related" in refs
    assert "unrelated" not in refs
