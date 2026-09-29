"""Inspect a persisted SE Brain training run."""
from __future__ import annotations
import argparse,json
from sebrain.training_registry import TrainingRunRegistry
def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("run_id"); p.add_argument("--runs",default="runs")
    a=p.parse_args(); print(json.dumps(TrainingRunRegistry(a.runs).load(a.run_id).manifest,indent=2,sort_keys=True,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
