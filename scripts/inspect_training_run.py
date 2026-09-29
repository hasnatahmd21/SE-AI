"""Inspect and optionally verify a persisted SE Brain training run."""
from __future__ import annotations

import argparse
import json

from sebrain.training_registry import TrainingRunRegistry, sha256_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("--runs", default="runs")
    p.add_argument("--verify-artifacts", action="store_true")
    args = p.parse_args()

    run = TrainingRunRegistry(args.runs).load(args.run_id)
    if args.verify_artifacts:
        failures = []
        for key, artifact in run.manifest.get("artifacts", {}).items():
            path = artifact["path"]
            try:
                actual = sha256_path(path)
            except (FileNotFoundError, OSError) as exc:
                failures.append({"key": key, "error": str(exc)})
                continue
            if actual != artifact["sha256"]:
                failures.append(
                    {
                        "key": key,
                        "expected": artifact["sha256"],
                        "actual": actual,
                    }
                )
        run.manifest["artifact_verification"] = {
            "ok": not failures,
            "failures": failures,
        }
    print(json.dumps(run.manifest, indent=2, sort_keys=True, default=str))
    return 0 if not run.manifest.get("artifact_verification", {}).get("failures") else 2


if __name__ == "__main__":
    raise SystemExit(main())
