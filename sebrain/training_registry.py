"""Persistent, auditable training-run registry."""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def _git(args: list[str]) -> str:
    try: return subprocess.check_output(["git",*args],stderr=subprocess.DEVNULL,text=True).strip()
    except Exception: return ""

def runtime_metadata() -> dict[str, Any]:
    data={"python":sys.version,"platform":platform.platform(),"git_commit":_git(["rev-parse","HEAD"]),"git_branch":_git(["rev-parse","--abbrev-ref","HEAD"]),"git_dirty":bool(_git(["status","--porcelain"]))}
    try:
        import torch
        data.update({"torch_version":torch.__version__,"cuda_version":getattr(torch.version,"cuda",None),"cuda_available":bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            data["gpu_name"]=torch.cuda.get_device_name(0)
            data["gpu_memory_bytes"]=torch.cuda.get_device_properties(0).total_memory
    except ImportError: data["torch_available"]=False
    for package,key in [("transformers","transformers_version"),("peft","peft_version")]:
        try: data[key]=getattr(__import__(package),"__version__","unknown")
        except ImportError: data[f"{key}_available"]=False
    return data

@dataclass
class TrainingRun:
    run_id: str
    root: Path
    manifest: dict[str, Any]

class TrainingRunRegistry:
    def __init__(self, root: str | Path="runs"):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def create(self, run_id: str, config: dict[str,Any], dataset_manifest: dict[str,Any], model_manifest: dict[str,Any]|None=None) -> TrainingRun:
        if not run_id.strip(): raise ValueError("run_id is required")
        root=self.root/run_id
        if root.exists(): raise FileExistsError(f"training run already exists: {run_id}")
        for name in ("checkpoints","adapter","metrics","logs","evaluation"): (root/name).mkdir()
        manifest={"schema_version":1,"run_id":run_id,"status":"CREATED","created_at":datetime.now(timezone.utc).isoformat(),"config":config,"dataset":dataset_manifest,"model":model_manifest or {},"runtime":runtime_metadata(),"history":[],"artifacts":{}}
        self._write(root,manifest); return TrainingRun(run_id,root,manifest)
    def load(self, run_id: str) -> TrainingRun:
        root=self.root/run_id
        if not root.is_dir(): raise FileNotFoundError(run_id)
        return TrainingRun(run_id,root,json.loads((root/"manifest.json").read_text(encoding="utf-8")))
    def update(self, run: TrainingRun, status: str, **fields: Any) -> None:
        run.manifest["status"]=status; run.manifest.update(fields)
        run.manifest.setdefault("history",[]).append({"timestamp":datetime.now(timezone.utc).isoformat(),"status":status})
        self._write(run.root,run.manifest)
    def add_artifact(self, run: TrainingRun, key: str, path: str | Path) -> None:
        p=Path(path); run.manifest.setdefault("artifacts",{})[key]={"path":str(p),"sha256":sha256_file(p) if p.is_file() else None}
        self._write(run.root,run.manifest)
    @staticmethod
    def _write(root: Path, manifest: dict[str,Any]) -> None:
        tmp=root/"manifest.json.tmp"; tmp.write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str),encoding="utf-8"); os.replace(tmp,root/"manifest.json")
