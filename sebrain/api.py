"""FastAPI interface for the canonical SE Brain facade."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import Config, ModelUnavailableError, SEBrain


class AnalyzeRequest(BaseModel):
    task: str = Field(min_length=1)
    project_id: str = ""
    task_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    language: str | None = None
    dataset_id: str | None = None
    concept: str | None = None


def create_app(*, datasets_dir: str | Path = "./datasets",
               data_dir: str | Path = "./.sebrain_api") -> FastAPI:
    brain = SEBrain(Config(data_dir=Path(data_dir), log_level="WARNING"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        brain.start()
        report = brain.connect_knowledge_fabric(datasets_dir)
        if report.get("errors") or report.get("missing_dataset_ids"):
            brain.stop()
            raise RuntimeError(
                "Knowledge Fabric startup validation failed: "
                f"errors={report.get('errors')}, "
                f"missing={report.get('missing_dataset_ids')}"
            )
        try:
            yield
        finally:
            brain.stop()

    api = FastAPI(
        title="SE Brain API",
        version="0.1.0",
        description="API boundary for the canonical Software Engineering Brain.",
        lifespan=lifespan,
    )

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "brain": brain.health()}

    @api.get("/knowledge")
    def knowledge() -> dict[str, Any]:
        try:
            return brain.fabric_stats()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @api.post("/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        try:
            result = brain.analyze_engineering_task(
                request.task, project_id=request.project_id,
                task_id=request.task_id, top_k=request.top_k
            )
            return result.to_dict()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/ask")
    def ask(request: AskRequest) -> dict[str, Any]:
        try:
            response = brain.ask(
                request.query, top_k=request.top_k,
                language=request.language, dataset_id=request.dataset_id,
                concept=request.concept
            )
            return response.model_dump() if hasattr(response, "model_dump") else response.to_dict()
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api


app = create_app(
    datasets_dir=os.getenv("SEBRAIN_DATASETS", "./datasets"),
    data_dir=os.getenv("SEBRAIN_DATA_DIR", "./.sebrain_api"),
)
