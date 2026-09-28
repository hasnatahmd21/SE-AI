from pathlib import Path

from sebrain.c24 import PerfAnalyzer


def test_repo_scan_surfaces_max_files_truncation(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("y = 2\n", encoding="utf-8")

    report = PerfAnalyzer(max_files=1).analyze_repo(tmp_path)

    assert report.files_scanned == 1
    assert report.scan_complete is False
    assert "repository scan truncated at max_files" in report.coverage.not_covered
    assert report.to_dict()["scan_complete"] is False


def test_repo_scan_surfaces_oversized_python_file(tmp_path: Path):
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "large.py").write_text("x = 1\n" * 20, encoding="utf-8")

    report = PerfAnalyzer(max_file_bytes=10).analyze_repo(tmp_path)

    assert report.files_skipped == 1
    assert report.scan_complete is True
    assert any("exceeds max_file_bytes" in item for item in report.coverage.not_covered)
