"""Load the configured base model and verify tokenizer/LoRA compatibility."""
from __future__ import annotations
import argparse, json
from sebrain.training_config import TrainingConfig

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a=p.parse_args()
    cfg=TrainingConfig.from_file(a.config)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit("Install requirements-training.txt first") from exc
    if cfg.require_gpu and not torch.cuda.is_available():
        raise SystemExit("Configured GPU requirement not satisfied")
    tok=AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.base_model, revision=cfg.tokenizer_revision, use_fast=True)
    if tok.pad_token is None and tok.eos_token is None:
        raise SystemExit("Tokenizer has neither pad nor eos token")
    model=AutoModelForCausalLM.from_pretrained(cfg.base_model, revision=cfg.model_revision)
    names={n.rsplit(".",1)[-1] for n,_ in model.named_modules()}
    targets=list(cfg.lora.target_modules) or [x for x in ("q_proj","k_proj","v_proj","o_proj") if x in names]
    if not targets:
        raise SystemExit("No compatible LoRA target modules found")
    task=getattr(TaskType,cfg.lora.task_type)
    model=get_peft_model(model,LoraConfig(r=cfg.lora.r,lora_alpha=cfg.lora.lora_alpha,lora_dropout=cfg.lora.lora_dropout,target_modules=targets,bias=cfg.lora.bias,task_type=task,modules_to_save=list(cfg.lora.modules_to_save) or None))
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    if trainable<=0 or trainable>=total:
        raise SystemExit("LoRA parameter constraint failed")
    print(json.dumps({"ok":True,"base_model":cfg.base_model,"tokenizer_class":type(tok).__name__,"targets":targets,"trainable_parameters":trainable,"total_parameters":total,"trainable_percentage":100*trainable/total,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
