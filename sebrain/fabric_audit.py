"""Knowledge Fabric audit and query entry point.

Examples:
    python -m sebrain.fabric_audit --datasets ./datasets
    python -m sebrain.fabric_audit --datasets ./datasets --strict
    python -m sebrain.fabric_audit --datasets ./datasets --query "database migration regression"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import Config, SEBrain


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and query SE Brain Knowledge Fabric")
    parser.add_argument("--datasets", default="./datasets")
    parser.add_argument("--data-dir", default="./.sebrain")
    parser.add_argument("--query", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when intended dataset coverage is incomplete or source errors exist",
    )
    args = parser.parse_args()

    with SEBrain(Config(data_dir=Path(args.data_dir), log_level="WARNING")) as brain:
        report = brain.connect_knowledge_fabric(args.datasets)
        stats = brain.fabric_stats()

        summary = {
            "files": report["files"],
            "records_inserted": report["records"],
            "found_dataset_ids": report["found_dataset_ids"],
            "missing_dataset_ids": report["missing_dataset_ids"],
            "errors": report["errors"],
            "warnings": report["warnings"],
            "coverage": stats["coverage"],
            "sources": stats["sources"],
        }

        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if args.query.strip():
            response = brain.ask(args.query, top_k=max(1, args.top_k))
            print("\n" + response.to_english())

        if args.strict and (report["errors"] or report["missing_dataset_ids"]):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
