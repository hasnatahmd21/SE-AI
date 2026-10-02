"""Prepare deterministic JSONL training data from the real Knowledge Fabric."""
from __future__ import annotations

import argparse
from pathlib import Path

from sebrain import Config, SEBrain


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    parser.add_argument("--data-dir", type=Path, default=Path(".sebrain_training"))
    parser.add_argument("--output", type=Path, default=Path("training/knowledge_fabric.jsonl"))
    args = parser.parse_args()

    with SEBrain(Config(data_dir=args.data_dir)) as brain:
        report = brain.connect_knowledge_fabric(args.datasets)
        stats = brain.fabric_stats()
        # Gate on the same authoritative coverage check as CI validation.
        # Source-level defects remain visible in the report but do not discard
        # an otherwise complete valid fabric.
        if report["missing_dataset_ids"] or not stats["coverage"]["complete"]:
            print("Knowledge Fabric validation failed; refusing to export training data.")
            return 1
        result = brain.export_training_data(args.output)
        print(result)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
