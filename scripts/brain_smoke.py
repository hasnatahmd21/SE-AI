"""Run the real no-LLM SE Brain runtime smoke path against repository datasets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from sebrain import Config, SEBrain

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--datasets", type=Path, default=Path("datasets"))
    p.add_argument("--query", default="How do I build a FastAPI endpoint?")
    args=p.parse_args()
    with SEBrain(Config(data_dir=Path(".sebrain_smoke"))) as brain:
        health=brain.health()
        report=brain.connect_knowledge_fabric(args.datasets)
        if report.get("errors") or report.get("missing_dataset_ids"):
            raise SystemExit(json.dumps({"stage":"knowledge-fabric","report":report}, indent=2, default=str))
        response=brain.ask(args.query, top_k=5)
        analysis=brain.analyze_engineering_task(args.query, top_k=5)
        if not response.evidence:
            raise SystemExit("Brain smoke failed: retrieval returned no evidence")
        if not analysis.plan:
            raise SystemExit("Brain smoke failed: planner returned no plan")
        result={"ok":True,"health":health,"fabric":report,"retrieval":{"method":response.retrieval_method,"evidence_count":len(response.evidence)},"analysis":{"plan_steps":len(analysis.plan.steps)}}
        print(json.dumps(result, indent=2, default=str))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
