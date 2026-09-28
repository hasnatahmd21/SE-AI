import pytest

from sebrain.c01 import ValidationError
from sebrain.c33 import (
    FailureEvent, FailureDomain, Harvester, Clusterer, FailureLab,
)


class BrokenMemory:
    def find(self, **kwargs):
        raise RuntimeError("synthetic harvest failure")


def event(i):
    return FailureEvent(
        id=str(i), domain=FailureDomain.UNKNOWN,
        what="cache stampede", root_cause="cache not warmed",
    )


def test_memory_harvest_failure_is_not_silently_empty():
    with pytest.raises(ValidationError, match="harvest failed"):
        Harvester().from_memory(BrokenMemory(), project_id="p")


def test_collect_rejects_event_truncation():
    lab = FailureLab(max_events=1)
    with pytest.raises(ValidationError, match="exceeds max_events"):
        lab.collect_events(project_id="p", extra_events=[event(1), event(2)])


def test_cluster_rejects_silent_cluster_truncation():
    clusterer = Clusterer(max_clusters=1)
    with pytest.raises(ValidationError, match="exceeds max_clusters"):
        clusterer.cluster([
            event("a"),
            FailureEvent(
                id="b", domain=FailureDomain.UNKNOWN,
                what="database migration", root_cause="schema mismatch",
            ),
        ])


def test_pattern_limit_is_not_silently_truncated():
    lab = FailureLab(max_patterns=1)
    events = [
        event("a"),
        FailureEvent(
            id="b", domain=FailureDomain.UNKNOWN,
            what="database migration", root_cause="schema mismatch",
        ),
    ]
    with pytest.raises(ValidationError, match="exceeds max_patterns"):
        lab.analyze(events, project_id="p")
