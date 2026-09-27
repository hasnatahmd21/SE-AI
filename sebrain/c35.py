"""C35 — Brain / Knowledge Fabric Bridge.

Connects query -> requirement understanding -> intent/risk -> C36 retrieval ->
structured English response. No external LLM is required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .c01 import Config, SEBrainApp
from .c05 import RequirementParser
from .c06 import IntentContextEngine
from .c34 import FabricRecord, KnowledgeFabricLoader
from .c36 import RAGContext, RAGPipeline


@dataclass(slots=True)
class BrainResponse:
    query: str = ""
    intent: str = ""
    objective: str = ""
    functional: list[str] = field(default_factory=list)
    non_functional: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    risk_level: str = ""
    knowledge: list[FabricRecord] = field(default_factory=list)
    confidence: str = ""
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    retrieval_method: str = "lexical"

    def to_english(self) -> str: return self.answer
    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "intent": self.intent, "objective": self.objective,
                "functional": self.functional, "non_functional": self.non_functional,
                "priorities": self.priorities, "risk": self.risk_level,
                "knowledge_count": len(self.knowledge), "confidence": self.confidence,
                "sources": self.sources, "retrieval_method": self.retrieval_method}


class BrainDatasetBridge:
    def __init__(self, *, brain: SEBrainApp, loader: KnowledgeFabricLoader,
                 rag: RAGPipeline | None = None):
        self.brain = brain
        self.loader = loader
        self.rag = rag or RAGPipeline(loader)
        self.parser = RequirementParser()
        self.intent_engine = IntentContextEngine()

    def answer(self, query: str, *, top_k: int = 5) -> BrainResponse:
        query = (query or "").strip()
        if not query: return BrainResponse(answer="No query provided.")
        try: spec = self.parser.parse(query)
        except Exception: spec = None
        try: intent_ctx = self.intent_engine.analyze(query, project_id="bridge")
        except Exception: intent_ctx = None
        rag_ctx = self.rag.retrieve(query, top_k=top_k)
        return self._build_response(query, spec, intent_ctx, rag_ctx)

    def _build_response(self, query: str, spec: Any, intent_ctx: Any, rag_ctx: RAGContext) -> BrainResponse:
        resp = BrainResponse(query=query, retrieval_method=rag_ctx.retrieval_method)
        if spec:
            resp.objective = spec.objective.text if spec.objective else ""
            resp.functional = [r.text for r in spec.functional]
            resp.non_functional = [r.text for r in spec.non_functional]
            resp.confidence = spec.confidence.value
        if intent_ctx:
            if intent_ctx.intent: resp.intent = intent_ctx.intent.kind.value
            if intent_ctx.risk: resp.risk_level = intent_ctx.risk.level.value
            resp.priorities = [f"[{p.level.value}] {p.text}" for p in intent_ctx.priorities[:5]]
        resp.knowledge = rag_ctx.records
        resp.sources = sorted({r.dataset_id for r in resp.knowledge})
        resp.answer = self._format_english(resp, rag_ctx)
        return resp

    def _format_english(self, resp: BrainResponse, rag_ctx: RAGContext) -> str:
        L = ["=" * 70, "  SE BRAIN — ANALYSIS", "=" * 70, "", "QUERY:", f"  {resp.query}", "",
             "-" * 70, "UNDERSTANDING", "-" * 70,
             f"  Intent          : {resp.intent or 'unknown'}",
             f"  Objective       : {resp.objective or '(not detected)'}",
             f"  Overall Conf.   : {resp.confidence or 'unknown'}",
             f"  Risk Level      : {resp.risk_level or 'unknown'}", ""]
        if resp.functional:
            L += ["-"*70, f"FUNCTIONAL REQUIREMENTS ({len(resp.functional)})", "-"*70]
            L += [f"  {i:02d}. {x}" for i, x in enumerate(resp.functional, 1)] + [""]
        if resp.non_functional:
            L += ["-"*70, f"NON-FUNCTIONAL REQUIREMENTS ({len(resp.non_functional)})", "-"*70]
            L += [f"  {i:02d}. {x}" for i, x in enumerate(resp.non_functional, 1)] + [""]
        if resp.priorities:
            L += ["-"*70, "PRIORITY ITEMS", "-"*70] + [f"  • {x}" for x in resp.priorities] + [""]
        L += ["-"*70, f"KNOWLEDGE FROM DATASETS ({len(resp.knowledge)} records)", "-"*70]
        if not resp.knowledge:
            L += ["  No relevant knowledge found in datasets.", ""]
        else:
            for i, rec in enumerate(resp.knowledge, 1):
                L += ["", f"  [{i}] {rec.dataset_id} › {rec.concept or rec.topic}"]
                if rec.language: L.append(f"      Language : {rec.language}")
                if rec.framework: L.append(f"      Framework: {rec.framework}")
                if rec.knowledge_type: L.append(f"      Type     : {rec.knowledge_type}")
                if rec.question: L.append(f"      Q: {rec.question}")
                if rec.answer:
                    L.append("      A:"); L += [f"        {x}" for x in self._wrap_text(rec.answer, 60)[:15]]
                if rec.explanation:
                    L.append("      Explanation:"); L += [f"        {x}" for x in self._wrap_text(rec.explanation, 60)[:8]]
                if rec.tags: L.append(f"      Tags     : {', '.join(rec.tags)}")
        L += ["", "-"*70, "SUMMARY", "-"*70]
        if resp.knowledge:
            L += [f"  ✓ Found {len(resp.knowledge)} relevant records in {len(resp.sources)} dataset(s).",
                  f"  ✓ Data sources: {', '.join(resp.sources)}", f"  ✓ Retrieval: {resp.retrieval_method}",
                  f"  ✓ Confidence: {resp.confidence or 'unknown'}"]
        else: L.append("  ✗ No matching knowledge found.")
        L += ["", "="*70]
        return "\n".join(L)

    @staticmethod
    def _wrap_text(text: str, width: int = 60) -> list[str]:
        lines: list[str] = []
        for para in (text or "").splitlines() or [""]:
            words = para.split(); current = ""
            for word in words:
                if len(word) > width:
                    if current: lines.append(current); current = ""
                    while len(word) > width: lines.append(word[:width]); word = word[width:]
                    current = word; continue
                if len(current) + len(word) + 1 <= width: current += (" " if current else "") + word
                else: lines.append(current); current = word
            if current: lines.append(current)
            elif not words: lines.append("")
        return lines

    def ask(self, query: str, *, top_k: int = 5) -> str: return self.answer(query, top_k=top_k).to_english()
    def search_only(self, query: str, *, top_k: int = 5) -> list[FabricRecord]: return self.rag.retrieve(query, top_k=top_k).records


def create_bridge(config: Config | None = None, datasets_dir: str | Path = "./datasets") -> tuple[SEBrainApp, KnowledgeFabricLoader, BrainDatasetBridge, dict[str, Any]]:
    brain = SEBrainApp(config=config)
    brain.start()
    loader = KnowledgeFabricLoader(brain.storage, datasets_dir)
    report = loader.load_all_datasets()
    return brain, loader, BrainDatasetBridge(brain=brain, loader=loader), report
