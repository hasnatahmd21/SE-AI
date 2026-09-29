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
    model_kwargs={"revision": cfg.model_revision}
    if cfg.quantization_4bit:
        try:
            from transformers import BitsAndBytesConfig
            dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            model_kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
            model_kwargs["device_map"]="auto"
        except ImportError as exc:
            raise SystemExit("4-bit QLoRA requires bitsandbytes") from exc
    model=AutoModelForCausalLM.from_pretrained(cfg.base_model, **model_kwargs)
    if cfg.quantization_4bit:
        from peft import prepare_model_for_kbit_training
        model=prepare_model_for_kbit_training(model)
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
    print(json.dumps({"ok":True,"base_model":cfg.base_model,"tokenizer_class":type(tok).__name__,"targets":targets,"quantization_4bit":cfg.quantization_4bit,"trainable_parameters":trainable,"total_parameters":total,"trainable_percentage":100*trainable/total,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
