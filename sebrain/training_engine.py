"""Real Hugging Face Transformers + PEFT LoRA training engine."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from .training_config import TrainingConfig
from .training_registry import TrainingRunRegistry\n\n# Training dependencies are intentionally lazy-loaded so core SE Brain imports stay lightweight.

class TrainingDependencyError(RuntimeError): pass
class TrainingDataError(ValueError): pass

def _require_training_deps():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    except ImportError as exc:
        raise TrainingDependencyError("Training dependencies are missing. Install requirements-training.txt.") from exc
    return torch,LoraConfig,TaskType,get_peft_model,AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback

def load_eligible_examples(path: str | Path) -> tuple[list[dict],dict[str,Any]]:
    p=Path(path)
    if not p.is_file(): raise TrainingDataError(f"training dataset not found: {p}")
    records=[]; excluded={}
    raw_hash=hashlib.sha256(p.read_bytes()).hexdigest()
    for line_no,line in enumerate(p.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        if not row.get("training_eligible",False):
            reason=f"{row.get('execution_status','')}/{row.get('validation_status','')}"
            excluded[reason]=excluded.get(reason,0)+1; continue
        if not row.get("instruction") or not row.get("output") or not row.get("record_id"):
            raise TrainingDataError(f"eligible record missing required fields at line {line_no}")
        records.append(row)
    if not records: raise TrainingDataError("no training-eligible records are available")
    by_split={s:[r for r in records if r.get("split")==s] for s in ("train","validation","test")}
    if not by_split["train"]: raise TrainingDataError("eligible training split is empty")
    ids=[r["record_id"] for r in records]
    manifest={"source_path":str(p),"source_sha256":raw_hash,"eligible_count":len(records),"excluded_count":sum(excluded.values()),"exclusion_reasons":excluded,"split_counts":{k:len(v) for k,v in by_split.items()},"record_ids":ids}
    manifest["record_ids_sha256"]=hashlib.sha256("\n".join(ids).encode()).hexdigest()
    manifest["split_hashes"]={s:hashlib.sha256("\n".join(r["record_id"] for r in by_split[s]).encode()).hexdigest() for s in ("train","validation","test")}
    manifest["dataset_ids"]=sorted({str(r.get("dataset_id","")) for r in records if r.get("dataset_id")})
    return records,manifest

class _CausalDataset:
    def __init__(self,rows,tokenizer,max_length): self.rows,self.tokenizer,self.max_length=rows,tokenizer,max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        row=self.rows[idx]; text=f"### Instruction\n{row['instruction']}\n\n### Response\n{row['output']}"
        enc=self.tokenizer(text,truncation=True,max_length=self.max_length,padding=False); enc["labels"]=list(enc["input_ids"]); return enc

def _collator(tokenizer):
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

def _precision(cfg,torch):
    if cfg.precision=="fp16": return True,False
    if cfg.precision=="bf16": return False,True
    if cfg.precision=="fp32": return False,False
    if not torch.cuda.is_available(): return False,False
    return bool(torch.cuda.is_bf16_supported()),False

def datetime_run_id():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + __import__("uuid").uuid4().hex[:8]

def train(config: TrainingConfig, *, registry: TrainingRunRegistry|None=None, dry_run: bool=False) -> dict[str,Any]:
    config.validate()
    records,dataset_manifest=load_eligible_examples(config.dataset_path)
    registry=registry or TrainingRunRegistry(config.output_dir)
    run_id=config.run_id or datetime_run_id()
    if config.resume:
        run=registry.load(run_id)
        if run.manifest.get("status") == "COMPLETED": raise TrainingDataError(f"training run {run_id} is already completed; create a new run instead")
        if run.manifest.get("dataset",{}).get("source_sha256") != dataset_manifest["source_sha256"]: raise TrainingDataError("resume refused: dataset hash differs from original run")
        if run.manifest.get("config",{}).get("base_model") != config.base_model: raise TrainingDataError("resume refused: base model differs from original run")
    else:
        run=registry.create(run_id,config.to_dict(),dataset_manifest,{"base_model":config.base_model,"revision":config.model_revision})
    (run.root/"training_config.json").write_text(json.dumps(config.to_dict(),indent=2,sort_keys=True),encoding="utf-8")
    (run.root/"dataset_manifest.json").write_text(json.dumps(dataset_manifest,indent=2,sort_keys=True),encoding="utf-8")
    if dry_run:
        registry.update(run,"DRY_RUN_VALIDATED",hardware=runtime_hardware())
        return run.manifest
    torch,LoraConfig,TaskType,get_peft_model,AutoModelForCausalLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback=_require_training_deps()
    if config.require_gpu and not torch.cuda.is_available():
        registry.update(run,"FAILED",failure_reason="GPU required but CUDA is unavailable")
        raise TrainingDependencyError("GPU is required by configuration but CUDA is unavailable")
    try:
        tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name,revision=config.tokenizer_revision,use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None: raise TrainingDependencyError("tokenizer has neither pad_token nor eos_token")
            tokenizer.pad_token=tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(config.base_model,revision=config.model_revision)\n        if getattr(model.config, "pad_token_id", None) is None: model.config.pad_token_id = tokenizer.pad_token_id
        run.manifest["model"].update({"name_or_path":getattr(getattr(model,"config",None),"_name_or_path",config.base_model),"architectures":list(getattr(getattr(model,"config",None),"architectures",[]) or [])})
        targets=list(config.lora.target_modules)
        if not targets:
            names={name.rsplit(".",1)[-1] for name,_ in model.named_modules()}
            targets=[x for x in ("q_proj","k_proj","v_proj","o_proj") if x in names]
        if not targets: raise TrainingDependencyError("LoRA target_modules must be configured for this model architecture")
        task_type=getattr(TaskType,config.lora.task_type)
        peft_cfg=LoraConfig(r=config.lora.r,lora_alpha=config.lora.lora_alpha,lora_dropout=config.lora.lora_dropout,target_modules=targets,bias=config.lora.bias,task_type=task_type,modules_to_save=list(config.lora.modules_to_save) or None)
        model=get_peft_model(model,peft_cfg)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); total=sum(p.numel() for p in model.parameters())
        if trainable<=0 or trainable>=total: raise TrainingDependencyError("LoRA did not produce a constrained trainable parameter set")
        run.manifest["model"].update({"trainable_parameters":trainable,"total_parameters":total,"trainable_percentage":100*trainable/total})
        run.manifest["lora"]={**config.to_dict()["lora"],"target_modules":targets}
        run.manifest["tokenizer"]={"name":config.tokenizer_name,"revision":config.tokenizer_revision,"class":type(tokenizer).__name__}
        (run.root/"model_manifest.json").write_text(json.dumps(run.manifest["model"],indent=2,sort_keys=True,default=str),encoding="utf-8")
        (run.root/"tokenizer_manifest.json").write_text(json.dumps(run.manifest["tokenizer"],indent=2,sort_keys=True,default=str),encoding="utf-8")
        (run.root/"lora_config.json").write_text(json.dumps(run.manifest["lora"],indent=2,sort_keys=True,default=str),encoding="utf-8")
        registry.update(run,"TRAINING_STARTED")
        train_ds=_CausalDataset([r for r in records if r["split"]=="train"],tokenizer,config.max_seq_length)
        val_rows=[r for r in records if r["split"]=="validation"]; val_ds=_CausalDataset(val_rows,tokenizer,config.max_seq_length) if val_rows else None
        fp16,bf16=_precision(config,torch)
        if val_ds and config.save_steps != config.eval_steps: raise TrainingDataError("save_steps and eval_steps must match when validation is enabled")
        args=TrainingArguments(output_dir=str(run.root/"checkpoints"),num_train_epochs=config.epochs,max_steps=config.max_steps,learning_rate=config.learning_rate,warmup_ratio=config.warmup_ratio,per_device_train_batch_size=config.batch_size,per_device_eval_batch_size=config.batch_size,gradient_accumulation_steps=config.gradient_accumulation_steps,weight_decay=config.weight_decay,lr_scheduler_type=config.scheduler,optim=config.optimizer,logging_steps=config.logging_steps,logging_dir=str(run.root/"logs"),eval_strategy="steps" if val_ds else "no",eval_steps=config.eval_steps,save_strategy="steps",save_steps=config.save_steps,save_total_limit=config.save_total_limit,load_best_model_at_end=bool(val_ds),metric_for_best_model="eval_loss" if val_ds else None,greater_is_better=False if val_ds else None,report_to=[],seed=config.seed,fp16=fp16,bf16=bf16,remove_unused_columns=False)
        class RegistryCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    run.manifest["latest_metrics"]=dict(logs)
                    registry.update(run,"TRAINING_RUNNING",step=state.global_step)
            def on_save(self, args, state, control, **kwargs):
                registry.update(run,"CHECKPOINT_SAVED",step=state.global_step)
        trainer=Trainer(model=model,args=args,train_dataset=train_ds,eval_dataset=val_ds,data_collator=_collator(tokenizer),callbacks=[RegistryCallback()])
        resume_checkpoint=None
        if config.resume:
            checkpoints=sorted((run.root/"checkpoints").glob("checkpoint-*"),key=lambda p:int(p.name.rsplit("-",1)[-1]))
            if not checkpoints: raise TrainingDataError("resume requested but no checkpoint exists")
            resume_checkpoint=str(checkpoints[-1])
        result=trainer.train(resume_from_checkpoint=resume_checkpoint)
        metrics=dict(result.metrics); mp=run.root/"metrics"/"training_metrics.json"; mp.write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
        evaluation=None
        if val_ds:
            evaluation=trainer.evaluate(); ep=run.root/"metrics"/"evaluation_metrics.json"; ep.write_text(json.dumps(evaluation,indent=2,default=str),encoding="utf-8"); run.manifest["evaluation"]=evaluation
        adapter_dir=run.root/"adapter"; model.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)\n        if test_rows:\n            test_ds=_CausalDataset(test_rows,tokenizer,config.max_seq_length)\n            test_metrics=trainer.evaluate(eval_dataset=test_ds,metric_key_prefix="test")\n            tp=run.root/"metrics"/"test_metrics.json"; tp.write_text(json.dumps(test_metrics,indent=2,default=str),encoding="utf-8")\n            run.manifest["test_evaluation"]=test_metrics\n            registry.add_artifact(run,"test_metrics",tp)
        registry.add_artifact(run,"adapter",adapter_dir); registry.add_artifact(run,"training_metrics",mp)
        for key,name in (("training_config","training_config.json"),("dataset_manifest","dataset_manifest.json"),("model_manifest","model_manifest.json"),("tokenizer_manifest","tokenizer_manifest.json"),("lora_config","lora_config.json")):
            registry.add_artifact(run,key,run.root/name)
        if evaluation is not None: registry.add_artifact(run,"evaluation_metrics",run.root/"metrics"/"evaluation_metrics.json")
        run.manifest["results"]=metrics; run.manifest["checkpoint_state"]={"best":getattr(trainer.state,"best_model_checkpoint",None),"global_step":trainer.state.global_step}; run.manifest["best_checkpoint"]=getattr(trainer.state,"best_model_checkpoint",None)
        from datetime import datetime,timezone
        registry.update(run,"COMPLETED",end_time=datetime.now(timezone.utc).isoformat())
        return run.manifest
    except Exception as exc:
        registry.update(run,"FAILED",failure_reason=f"{type(exc).__name__}: {exc}"); raise

def runtime_hardware():
    try:
        import torch
        return {"cuda_available":bool(torch.cuda.is_available()),"cuda_version":getattr(torch.version,"cuda",None),"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError: return {"torch_available":False}
