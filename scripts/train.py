"""CLI for validated SE Brain LoRA/PEFT training."""
from __future__ import annotations
import argparse,json
from sebrain.training_config import TrainingConfig
from sebrain.training_engine import train
def main()->int:
    p=argparse.ArgumentParser(description="Train SE Brain with real Hugging Face + PEFT LoRA.")
    p.add_argument("--config",required=True); p.add_argument("--dry-run",action="store_true"); p.add_argument("--resume",action="store_true")
    a=p.parse_args(); cfg=TrainingConfig.from_file(a.config)
    if a.resume: cfg=TrainingConfig.from_dict({**cfg.to_dict(),"resume":True})
    print(json.dumps(train(cfg,dry_run=a.dry_run),indent=2,default=str)); return 0
if __name__=="__main__": raise SystemExit(main())
