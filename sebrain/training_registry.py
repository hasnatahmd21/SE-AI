"""Persistent, auditable training-run registry."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: str | Path) -> str:
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    if not p.is_dir():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    for child in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(str(child.relative_to(p)).encode("utf-8"))
        h.update(sha256_file(child).encode("ascii"))
    return h.hexdigest()


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def runtime_metadata() -> dict[str, Any]:
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
    }
    try:
        import torch

        data.update(
            {
                "torch_version": torch.__version__,
                "cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            data["gpu_name"] = torch.cuda.get_device_name(0)
            data["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    except ImportError:
        data["torch_available"] = False

    for package, key in (
        ("transformers", "transformers_version"),
        ("peft", "peft_version"),
    ):
        try:
            data[key] = getattr(__import__(package), "__version__", "unknown")
        except ImportError:
            data[f"{key}_available"] = False
    return data


@dataclass
class TrainingRun:
    run_id: str
    root: Path
    manifest: dict[str, Any]


class TrainingRunRegistry:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        value = run_id.strip()
        if not value or value in {".", ".."}:
            raise ValueError("run_id is required and must be a simple directory name")
        candidate = Path(value)
        if candidate.is_absolute() or len(candidate.parts) != 1:
            raise ValueError("run_id must not contain path separators or parent traversal")
        return value

    def create(
        self,
        run_id: str,
        config: dict[str, Any],
        dataset_manifest: dict[str, Any],
        model_manifest: dict[str, Any] | None = None,
    ) -> TrainingRun:
        run_id = self._validate_run_id(run_id)
        root = self.root / run_id
        if root.exists():
            raise FileExistsError(f"training run already exists: {run_id}")
        for name in ("checkpoints", "adapter", "metrics", "logs", "evaluation"):
            (root / name).mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "CREATED",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "dataset": dataset_manifest,
            "model": model_manifest or {},
            "runtime": runtime_metadata(),
            "history": [],
            "artifacts": {},
        }
        self._write(root, manifest)
        return TrainingRun(run_id, root, manifest)

    def load(self, run_id: str) -> TrainingRun:
        run_id = self._validate_run_id(run_id)
        root = self.root / run_id
        if not root.is_dir():
            raise FileNotFoundError(run_id)
        return TrainingRun(
            run_id,
            root,
            json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        )

    def update(self, run: TrainingRun, status: str, **fields: Any) -> None:
        current = run.manifest.get("status")
        if current == "COMPLETED" and status != "COMPLETED":
            raise RuntimeError(f"completed training run cannot transition to {status}")
        if current == "FAILED" and status not in {"FAILED", "RESUMING"}:
            raise RuntimeError(f"failed training run cannot transition to {status}")
        run.manifest["status"] = status
        run.manifest.update(fields)
        run.manifest.setdefault("history", []).append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }
        )
        self._write(run.root, run.manifest)

    def add_artifact(self, run: TrainingRun, key: str, path: str | Path) -> None:
        p = Path(path).resolve()
        root = run.root.resolve()
        if not (p == root or root in p.parents):
            raise ValueError("artifact path must remain inside the training run directory")
        artifact = {"path": str(p), "sha256": sha256_path(p)}
        existing = run.manifest.setdefault("artifacts", {}).get(key)
        if existing is not None and existing != artifact:
            raise RuntimeError(f"artifact key already registered with different content: {key}")
        run.manifest["artifacts"][key] = artifact
        self._write(run.root, run.manifest)

    @staticmethod
    def _write(root: Path, manifest: dict[str, Any]) -> None:
        tmp = root / "manifest.json.tmp"
        tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, root / "manifest.json")
