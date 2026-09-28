from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from sebrain.c01 import SQLiteStorage, ValidationError
from sebrain.c32 import BenchmarkCase, BenchmarkRunner, TaskCategory, CaseOutcome


def test_run_rejects_silent_case_truncation():
    runner = BenchmarkRunner(max_cases_per_run=1)
    for i in range(2):
        runner.add_case(BenchmarkCase(
            id=f"c{i}", name=f"c{i}", category=TaskCategory.TESTING,
            runner=lambda inputs: {"correctness": 1.0},
        ))
    with pytest.raises(ValidationError, match="exceeds max_cases_per_run"):
        runner.run()


def test_missing_outcome_metric_is_not_reported_as_pass():
    runner = BenchmarkRunner()
    runner.add_case(BenchmarkCase(
        id="unknown", name="unknown", category=TaskCategory.TESTING,
        runner=lambda inputs: {"duration_seconds": 0.01},
    ))
    result = runner.run().results[0]
    assert result.outcome is CaseOutcome.SKIPPED
    assert result.error["type"] == "NoOutcomeMetric"


def test_wall_clock_duration_bound_is_enforced():
    runner = BenchmarkRunner(max_duration_per_case=0.01)
    runner.add_case(BenchmarkCase(
        id="slow", name="slow", category=TaskCategory.TESTING,
        timeout_seconds=0.01,
        runner=lambda inputs: __import__("time").sleep(0.03) or {"correctness": 1.0},
    ))
    result = runner.run().results[0]
    assert result.outcome is CaseOutcome.FAILED
    assert result.error["type"] == "DurationExceeded"
