from sebrain.c28 import PipelineContext, UncertaintyCheck, CheckOutcome


def test_unknown_confidence_uses_unknown_weight_not_fallback():
    result = UncertaintyCheck().evaluate(PipelineContext(
        spec=type("S", (), {
            "confidence": "unknown",
        })()
    ))
    assert result.outcome is CheckOutcome.BLOCKING
    assert "0.10" in result.rationale


def test_missing_confidence_signal_is_insufficient():
    result = UncertaintyCheck().evaluate(PipelineContext())
    assert result.outcome is CheckOutcome.INSUFFICIENT
