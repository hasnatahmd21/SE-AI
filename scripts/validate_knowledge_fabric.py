"""Validate the repository's real Knowledge Fabric export.

This command intentionally exercises the same C34 -> C36 -> C35 path used by
the application, but reports findings instead of assuming a record-count
target. Quality, validity and dataset coverage are authoritative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sebrain import Config, SEBrain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    parser.add_argument("--data-dir", type=Path, default=Path(".sebrain_validation"))
    args = parser.parse_args()

    with SEBrain(Config(data_dir=args.data_dir)) as brain:
        report = brain.connect_knowledge_fabric(args.datasets)
        stats = brain.fabric_stats()

        result = {
            "files": report["files"],
            "records_inserted": report["records"],
            "found_dataset_ids": report["found_dataset_ids"],
            "missing_dataset_ids": report["missing_dataset_ids"],
            "errors": report["errors"],
            "warnings": report["warnings"],
            "coverage": stats["coverage"],
            "sources": stats["sources"],
        }

        print(json.dumps(result, indent=2, ensure_ascii=False))

        if report["errors"]:
            return 1
        if report["missing_dataset_ids"]:
            return 1
        if not stats["coverage"]["complete"]:
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
