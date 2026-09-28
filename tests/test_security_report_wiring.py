from sebrain.c23 import SecurityReport, FileScanResult


def test_security_report_serializes_file_results_and_analysis_errors():
    report = SecurityReport(root="/tmp/repo")
    report.file_results.append(
        FileScanResult(
            path="/tmp/repo/example.py",
            analysis_errors=[{
                "rule_id": "BROKEN_RULE",
                "error": "RuntimeError: boom",
            }],
        )
    )

    payload = report.to_dict()

    assert payload["file_results"][0]["analysis_errors"][0]["rule_id"] == "BROKEN_RULE"
    assert payload["analysis_errors"] == [{
        "path": "/tmp/repo/example.py",
        "rule_id": "BROKEN_RULE",
        "error": "RuntimeError: boom",
    }]
