#!/usr/bin/env python3
"""
SE-BRAIN — Production Kaggle 10K QLoRA Runner

Purpose:
  Train Qwen/Qwen2.5-Coder-7B-Instruct on exactly 10,000 unique,
  training-eligible Knowledge Fabric examples, with:
    - private GitHub access via Kaggle Secret GH_TOKEN
    - repository Knowledge Fabric validation
    - deterministic coverage-aware sampling
    - 4-bit NF4 QLoRA
    - assistant/completion-only loss
    - 95/5 train/validation split
    - checkpoint resume within the Kaggle session
    - post-training generation smoke tests
    - reproducibility metadata + SHA256
    - GitHub adapter publication when the adapter is safely below
      GitHub's per-file size limit
    - safe CUDA cleanup and optional Kaggle process shutdown only
      after every publication/verification step succeeds

IMPORTANT:
  Kaggle /kaggle/working is ephemeral. This runner cannot make an
  interrupted run magically durable across a killed session. For
  cross-session resume, copy checkpoints to a durable Kaggle Dataset
  or another artifact store. The final adapter is published to GitHub
  only after the run completes and is verified.

Kaggle Secret required:
  GH_TOKEN = GitHub PAT with access to hasnatahmd21/SE-AI

Optional Kaggle Secrets:
  HF_TOKEN = Hugging Face token, only if the model requires authentication.

This file is intentionally self-contained so it can be committed to the
repository and run from a Kaggle notebook with:
    !python scripts/kaggle_qlora_10k.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = "hasnatahmd21/SE-AI"
BRANCH = "main"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
TARGET_RECORDS = 10_000
VALIDATION_RECORDS = 500
MAX_LENGTH = 1024
SEED = 42
OUTPUT_ROOT = Path("/kaggle/working/sebrain_qlora_10k")
REPO_ROOT = Path("/kaggle/working/SE-AI")
ADAPTER_REPO_PATH = "lora_adapters/qwen2.5-coder-7b_10k"
AUTO_SHUTDOWN = True
MAX_SINGLE_FILE_MB = 90.0

# QLoRA defaults chosen for a 7B Coder model on Kaggle T4-class GPUs.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 1.0
GRAD_ACCUM = 16
BATCH_SIZE = 1
SAVE_STEPS = 100
EVAL_STEPS = 100
LOG_STEPS = 10


def log(msg: str) -> None:
    print(f"[SE-BRAIN] {msg}", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch>=2.2", "transformers>=4.45", "peft>=0.19.1",
         "accelerate>=0.34", "safetensors>=0.4", "bitsandbytes>=0.43",
         "sentencepiece>=0.2", "huggingface_hub>=0.25"])


def get_secret(name: str, required: bool = False) -> str:
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(name)
    except Exception:
        value = os.environ.get(name, "")
    value = (value or "").strip()
    if required and not value:
        raise RuntimeError(f"Missing required Kaggle Secret: {name}")
    return value


def clone_repo(token: str) -> None:
    if REPO_ROOT.exists():
        shutil.rmtree(REPO_ROOT)
    # ExtraHeader avoids putting the token in the repository URL/config.
    env = os.environ.copy()
    env["GH_TOKEN"] = token
    run([
        "git", "-c", "http.extraheader=AUTHORIZATION: bearer " + token,
        "clone", "--depth", "1", "--branch", BRANCH,
        f"https://github.com/{REPO}.git", str(REPO_ROOT)
    ])
    run(["git", "config", "user.name", "SE-BRAIN Training Bot"], cwd=REPO_ROOT)
    run(["git", "config", "user.email", "sebrain-training@users.noreply.github.com"], cwd=REPO_ROOT)


def validate_repo() -> None:
    run([sys.executable, "scripts/validate_knowledge_fabric.py"], cwd=REPO_ROOT)
    run([sys.executable, "scripts/prepare_training_data.py",
         "--output", "training/knowledge_fabric.jsonl"], cwd=REPO_ROOT)


def load_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Training artifact missing: {path}")
    rows = []
    seen = set()
    excluded = Counter()

    invalid_execution = {"", "NOT_EXECUTED", "UNEXECUTED", "UNKNOWN", "PLANNED"}
    invalid_validation = {
        "", "ILLUSTRATIVE", "REQUIRES_TARGET_VALIDATION",
        "UNVALIDATED", "UNKNOWN", "PLANNED"
    }

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("training_eligible") is not True:
            excluded["training_eligible=false"] += 1
            continue
        if str(row.get("execution_status", "")).upper() in invalid_execution:
            excluded["execution_status"] += 1
            continue
        if str(row.get("validation_status", "")).upper() in invalid_validation:
            excluded["validation_status"] += 1
            continue

        rid = str(row.get("record_id", "")).strip()
        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", "")).strip()
        if not rid or not instruction or not output:
            excluded["missing_required_fields"] += 1
            continue
        if rid in seen:
            excluded["duplicate_record_id"] += 1
            continue
        seen.add(rid)
        rows.append(row)

    if len(rows) < TARGET_RECORDS + VALIDATION_RECORDS:
        raise RuntimeError(
            f"Only {len(rows)} unique eligible records available; "
            f"need at least {TARGET_RECORDS + VALIDATION_RECORDS} "
            f"for 10,000 train + 500 validation. Refusing to duplicate records."
        )

    log(f"Eligible unique records: {len(rows)}")
    log(f"Excluded records: {dict(excluded)}")
    return rows


def coverage_sample(rows: list[dict], n: int) -> list[dict]:
    """
    Deterministic proportional/round-robin sampling by dataset_id.
    This prevents a large dataset from consuming the whole 10K selection.
    """
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = str(row.get("dataset_id") or row.get("knowledge_type") or "UNKNOWN")
        groups[key].append(row)

    for values in groups.values():
        rng.shuffle(values)

    keys = sorted(groups)
    selected = []
    cursor = {k: 0 for k in keys}

    # First pass: one example per group where possible.
    while len(selected) < n:
        progressed = False
        for k in keys:
            i = cursor[k]
            if i < len(groups[k]) and len(selected) < n:
                selected.append(groups[k][i])
                cursor[k] += 1
                progressed = True
        if not progressed:
            break

    if len(selected) != n:
        raise RuntimeError(f"Coverage sampler selected {len(selected)} records, expected {n}")
    return selected


def build_prompt(row: dict) -> tuple[str, str]:
    instruction = str(row.get("instruction", "")).strip()
    output = str(row.get("output", "")).strip()

    context = []
    for field in ("dataset_id", "knowledge_type", "language", "framework", "concept", "objective"):
        value = str(row.get(field, "")).strip()
        if value:
            context.append(f"{field}: {value}")

    if context:
        instruction = (
            "Use the following verified software-engineering knowledge context.\n"
            + "\n".join(context)
            + "\n\nTask:\n"
            + instruction
        )
    return instruction, output


def make_training_jsonl(rows: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "target_train_records": TARGET_RECORDS,
        "validation_records": VALIDATION_RECORDS,
        "seed": SEED,
        "records": [],
    }

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            instruction, output = build_prompt(row)
            item = {
                "record_id": str(row["record_id"]),
                "dataset_id": str(row.get("dataset_id", "")),
                "instruction": instruction,
                "output": output,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            manifest["records"].append(item["record_id"])

    manifest["record_ids_sha256"] = hashlib.sha256(
        "\n".join(manifest["records"]).encode()
    ).hexdigest()
    return manifest


def import_training_stack():
    import torch
    import bitsandbytes
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    return (
        torch, bitsandbytes, LoraConfig, TaskType, get_peft_model,
        prepare_model_for_kbit_training, AutoModelForCausalLM,
        AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
        TrainingArguments,
    )


class CompletionOnlyDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        instruction = row["instruction"]
        output = row["output"]

        # Prefer the native Qwen chat template. We still create an explicit
        # assistant boundary so prompt tokens are masked from the loss.
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]

        if getattr(self.tokenizer, "chat_template", None):
            prompt_messages = [{"role": "user", "content": instruction}]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            prompt_text = f"### Instruction\n{instruction}\n\n### Response\n"
            full_text = prompt_text + output

        full = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length, padding=False
        )
        prompt = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length, padding=False
        )

        labels = list(full["input_ids"])
        prompt_len = min(len(prompt["input_ids"]), len(labels))

        # Completion-only loss: prompt tokens receive -100.
        for i in range(prompt_len):
            labels[i] = -100

        # Refuse samples where truncation removed the complete assistant answer.
        if all(x == -100 for x in labels):
            raise RuntimeError(f"Record {row['record_id']} has no trainable completion tokens")

        full["labels"] = labels
        return full


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for x in features:
            pad = max_len - len(x["input_ids"])
            batch["input_ids"].append(x["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(x["attention_mask"] + [0] * pad)
            batch["labels"].append(x["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def train_model(train_rows, val_rows):
    (
        torch, _bnb, LoraConfig, TaskType, get_peft_model,
        prepare_model_for_kbit_training, AutoModelForCausalLM,
        AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
        TrainingArguments,
    ) = import_training_stack()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Refusing CPU training.")

    log(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    hf_token = get_secret("HF_TOKEN", required=False)
    token_kw = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, **token_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        **token_kw,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    run_dir = OUTPUT_ROOT / "trainer"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CompletionOnlyDataset(train_rows, tokenizer, MAX_LENGTH)
    val_ds = CompletionOnlyDataset(val_rows, tokenizer, MAX_LENGTH)

    fp16 = True
    # Transformers renamed evaluation_strategy to eval_strategy in newer releases.
    # Detect the installed API instead of assuming one version.
    import inspect
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    args_kwargs = dict(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        fp16=fp16,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=LOG_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=SEED,
        remove_unused_columns=False,
        save_safetensors=True,
    )
    args_kwargs[strategy_key] = "steps"
    args = TrainingArguments(**args_kwargs)

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                log(f"step={state.global_step} metrics={logs}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(tokenizer),
        callbacks=[ProgressCallback()],
    )

    # Resume only from checkpoints created in this same Kaggle working session.
    checkpoints = sorted(
        run_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
    )
    resume = str(checkpoints[-1]) if checkpoints else None
    if resume:
        log(f"Resuming from {resume}")

    result = trainer.train(resume_from_checkpoint=resume)

    eval_metrics = trainer.evaluate()
    final_dir = OUTPUT_ROOT / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    metrics = {
        "train": dict(result.metrics),
        "validation": dict(eval_metrics),
        "global_step": trainer.state.global_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
    }
    (OUTPUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    return tokenizer, model, metrics


def generation_eval(tokenizer, model, rows: list[dict]) -> list[dict]:
    import torch

    samples = []
    for row in rows[:8]:
        messages = [{"role": "user", "content": row["instruction"]}]
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"### Instruction\n{row['instruction']}\n\n### Response\n"

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        samples.append({
            "record_id": row["record_id"],
            "prompt": row["instruction"],
            "generated": generated,
        })

    path = OUTPUT_ROOT / "generation_eval.json"
    path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    return samples


def write_manifest(train_rows, val_rows, metrics, generation):
    files = {}
    final_dir = OUTPUT_ROOT / "final_adapter"
    for p in sorted(final_dir.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(final_dir))] = {
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }

    datasets = Counter(str(r.get("dataset_id", "UNKNOWN")) for r in train_rows)
    manifest = {
        "project": "SE-BRAIN",
        "purpose": "10k-record QLoRA training",
        "base_model": MODEL_NAME,
        "train_records": len(train_rows),
        "validation_records": len(val_rows),
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "qlora": {
            "bits": 4,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "float16",
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        },
        "coverage": {
            "dataset_count": len(datasets),
            "records_by_dataset": dict(sorted(datasets.items())),
        },
        "metrics": metrics,
        "generation_eval_count": len(generation),
        "adapter_files": files,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUTPUT_ROOT / "training_metadata.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = [
        f"{info['sha256']}  {name}"
        for name, info in sorted(files.items())
    ]
    (OUTPUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def check_adapter_size() -> None:
    final_dir = OUTPUT_ROOT / "final_adapter"
    for p in final_dir.rglob("*"):
        if p.is_file():
            mb = p.stat().st_size / (1024 * 1024)
            log(f"Artifact {p.name}: {mb:.2f} MB")
            if mb > MAX_SINGLE_FILE_MB:
                raise RuntimeError(
                    f"{p} is {mb:.2f} MB. Refusing GitHub publication above "
                    f"{MAX_SINGLE_FILE_MB} MB. Use Git LFS/artifact storage instead."
                )


def publish_to_github(token: str, manifest: dict) -> str:
    target = REPO_ROOT / ADAPTER_REPO_PATH
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    source = OUTPUT_ROOT / "final_adapter"
    for p in source.rglob("*"):
        if p.is_file():
            dest = target / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    for name in ("training_metadata.json", "SHA256SUMS", "metrics.json", "generation_eval.json"):
        src = OUTPUT_ROOT / name
        if src.exists():
            shutil.copy2(src, target / name)

    check_adapter_size()

    run(["git", "add", ADAPTER_REPO_PATH], cwd=REPO_ROOT)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", ADAPTER_REPO_PATH],
        cwd=str(REPO_ROOT), text=True, capture_output=True, check=True
    )
    if not status.stdout.strip():
        raise RuntimeError("No adapter changes detected; refusing empty publication commit.")

    run([
        "git", "commit", "-m",
        "train: publish Qwen2.5-Coder 7B 10k QLoRA adapter"
    ], cwd=REPO_ROOT)

    env = os.environ.copy()
    # Token is supplied only for this process; it is not written to git config.
    run([
        "git", "-c", "http.extraheader=AUTHORIZATION: bearer " + token,
        "push", "origin", BRANCH
    ], cwd=REPO_ROOT)

    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
    ).strip()
    return sha


def safe_gpu_cleanup() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            log("CUDA cache released.")
    except Exception as exc:
        log(f"CUDA cleanup warning: {exc}")


def shutdown_kaggle() -> None:
    if not AUTO_SHUTDOWN:
        log("AUTO_SHUTDOWN=False; leaving Kaggle session running.")
        return

    log("Training, verification, hashing and GitHub publication all succeeded.")
    log("Requesting Kaggle kernel shutdown now.")

    # Do this only after all important files are closed and pushed.
    # SIGTERM gives the runtime a chance to flush before SIGKILL fallback.
    try:
        os.kill(1, 15)
    except Exception:
        pass


def main() -> int:
    start = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    token = get_secret("GH_TOKEN", required=True)
    install_dependencies()

    clone_repo(token)
    validate_repo()

    source = REPO_ROOT / "training" / "knowledge_fabric.jsonl"
    rows = load_records(source)

    # 10,000 training + 500 held-out validation.
    selected = coverage_sample(rows, TARGET_RECORDS + VALIDATION_RECORDS)
    train_rows = selected[:TARGET_RECORDS]
    val_rows = selected[TARGET_RECORDS:]

    # Sanity checks before GPU allocation.
    train_ids = {r["record_id"] for r in train_rows}
    val_ids = {r["record_id"] for r in val_rows}
    if len(train_ids) != TARGET_RECORDS:
        raise RuntimeError("Training set is not exactly 10,000 unique records.")
    if len(val_ids) != VALIDATION_RECORDS:
        raise RuntimeError("Validation set is not exactly 500 unique records.")
    if train_ids & val_ids:
        raise RuntimeError("Train/validation record leakage detected.")

    train_manifest = make_training_jsonl(train_rows, OUTPUT_ROOT / "train_10k.jsonl")
    val_manifest = make_training_jsonl(val_rows, OUTPUT_ROOT / "validation_500.jsonl")
    (OUTPUT_ROOT / "selection_manifest.json").write_text(
        json.dumps({
            "train": train_manifest,
            "validation": val_manifest,
        }, indent=2),
        encoding="utf-8",
    )

    log("Starting 10,000-record QLoRA training.")
    tokenizer, model, metrics = train_model(train_rows, val_rows)

    generation = generation_eval(tokenizer, model, train_rows)
    manifest = write_manifest(train_rows, val_rows, metrics, generation)

    # Final integrity check before touching GitHub.
    check_adapter_size()
    adapter_sha = sha256_file(OUTPUT_ROOT / "final_adapter" / "adapter_model.safetensors")
    if not adapter_sha:
        raise RuntimeError("Adapter SHA256 calculation failed.")

    commit_sha = publish_to_github(token, manifest)
    (OUTPUT_ROOT / "publication.json").write_text(
        json.dumps({
            "repository": REPO,
            "branch": BRANCH,
            "commit": commit_sha,
            "adapter_path": ADAPTER_REPO_PATH,
            "adapter_sha256": adapter_sha,
        }, indent=2),
        encoding="utf-8",
    )

    # The publication file is local audit evidence; the actual GitHub push
    # already happened. We intentionally do not create a second commit after push.
    elapsed = time.time() - start
    log(f"SUCCESS — 10,000 train + 500 validation records.")
    log(f"GitHub commit: {commit_sha}")
    log(f"Elapsed: {elapsed/3600:.2f} hours")
    safe_gpu_cleanup()
    shutdown_kaggle()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[SE-BRAIN] FAILED: {type(exc).__name__}: {exc}", flush=True)
        print("[SE-BRAIN] Session will NOT be force-shutdown so the failure can be inspected.", flush=True)
        try:
            safe_gpu_cleanup()
        except Exception:
            pass
        raise
