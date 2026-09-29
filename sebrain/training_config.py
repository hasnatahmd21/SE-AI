"""Validated configuration for reproducible SE Brain LoRA/PEFT training."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ()
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    adapter_name: str = "sebrain"
    modules_to_save: tuple[str, ...] = ()
    def validate(self) -> None:
        if self.r <= 0: raise ValueError("LoRA r must be > 0")
        if self.lora_alpha <= 0: raise ValueError("LoRA lora_alpha must be > 0")
        if not 0 <= self.lora_dropout < 1: raise ValueError("LoRA dropout must be in [0, 1)")
        if self.bias not in {"none", "all", "lora_only"}: raise ValueError("invalid LoRA bias")
        if not self.task_type.strip(): raise ValueError("LoRA task_type is required")

@dataclass(frozen=True)
class TrainingConfig:
    base_model: str
    tokenizer: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    dataset_path: str = "training/knowledge_fabric.jsonl"
    output_dir: str = "runs"
    run_id: str | None = None
    max_seq_length: int = 1024
    epochs: float = 1.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    optimizer: str = "adamw_torch"
    precision: str = "auto"
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 3
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    require_gpu: bool = True
    resume: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    def validate(self) -> None:
        if not self.base_model.strip(): raise ValueError("base_model is required")
        if self.max_seq_length <= 0: raise ValueError("max_seq_length must be > 0")
        if self.epochs <= 0: raise ValueError("epochs must be > 0")
        if self.max_steps == 0 or self.max_steps < -1: raise ValueError("max_steps must be -1 or > 0")
        if self.learning_rate <= 0: raise ValueError("learning_rate must be > 0")
        if not 0 <= self.warmup_ratio < 1: raise ValueError("warmup_ratio must be in [0,1)")
        if self.batch_size <= 0 or self.gradient_accumulation_steps <= 0: raise ValueError("batch sizes must be > 0")
        if not 0 <= self.train_ratio < 1 or not 0 <= self.validation_ratio < 1:
            raise ValueError("split ratios must be in [0,1)")
        if self.train_ratio + self.validation_ratio >= 1: raise ValueError("train_ratio + validation_ratio must be < 1")
        if self.precision not in {"auto", "fp32", "fp16", "bf16"}: raise ValueError("invalid precision")
        if self.save_total_limit <= 0: raise ValueError("save_total_limit must be > 0")
        if self.logging_steps <= 0 or self.eval_steps <= 0 or self.save_steps <= 0: raise ValueError("logging/eval/save steps must be > 0")
        self.lora.validate()
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lora"]["target_modules"] = list(self.lora.target_modules)
        data["lora"]["modules_to_save"] = list(self.lora.modules_to_save)
        return data
    def write(self, path: str | Path) -> None:
        self.validate()
        p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        raw=dict(data); lora=raw.pop("lora", {})
        raw["lora"]=LoRAConfig(**{k: tuple(v) if k in {"target_modules","modules_to_save"} else v for k,v in lora.items()})
        cfg=cls(**raw); cfg.validate(); return cfg
    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingConfig":
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict): raise ValueError("training config must be a JSON object")
        return cls.from_dict(data)
