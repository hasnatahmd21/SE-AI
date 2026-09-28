from pathlib import Path

from sebrain.c23 import SecurityAnalyzer


def test_security_analysis_reports_rule_errors(tmp_path: Path):
    analyzer = SecurityAnalyzer()

    class BrokenRule:
        rule_id = "BROKEN_RULE"

        def check(self, source, path):
            raise RuntimeError("synthetic rule failure")

    import sebrain.c23 as c23
    original = list(c23._RE_RULES)
    c23._RE_RULES.insert(0, BrokenRule())
    try:
        result = analyzer.analyze_source(
            "value = 1\n",
            filename="example.py",
        )
    finally:
        c23._RE_RULES[:] = original

    assert result.analysis_errors == [{
        "rule_id": "BROKEN_RULE",
        "error": "RuntimeError: synthetic rule failure",
    }]
