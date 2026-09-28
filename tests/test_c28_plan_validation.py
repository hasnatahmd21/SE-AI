from sebrain.c28 import CheckOutcome, PipelineContext, PlanCheck, RedirectionKind, _FakePlan

def test_c28_plan_validation_failure_is_not_swallowed():
    class BrokenPlan(_FakePlan):
        def validate(self):
            raise RuntimeError("synthetic validation failure")
    result = PlanCheck().evaluate(PipelineContext(plan=BrokenPlan()))
    assert result.outcome is CheckOutcome.NEEDS_ATTENTION
    assert "synthetic validation failure" in result.rationale
    assert any(r.kind is RedirectionKind.REVISE_PLAN for r in result.redirections)
