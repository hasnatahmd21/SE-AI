"""CLI for validated SE Brain LoRA/PEFT training."""
from __future__ import annotations
import argparse,json
from sebrain.training_config import TrainingConfig
from sebrain.training_engine import train
def main()->int:
    p=argparse.ArgumentParser(description="Train SE Brain with real Hugging Face + PEFT LoRA.")
    p.add_argument("--config",required=True); p.add_argument("--dry-run",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--run-id")
    a=p.parse_args(); cfg=TrainingConfig.from_file(a.config)
    if a.resume:\n        if not a.run_id and not cfg.run_id: raise SystemExit("--resume requires --run-id or run_id in the config")\n        cfg=TrainingConfig.from_dict({**cfg.to_dict(),"resume":True,"run_id":a.run_id or cfg.run_id})\n    elif a.run_id:\n        cfg=TrainingConfig.from_dict({**cfg.to_dict(),"run_id":a.run_id})
    print(json.dumps(train(cfg,dry_run=a.dry_run),indent=2,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
