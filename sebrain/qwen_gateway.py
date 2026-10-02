"""Local Qwen + LoRA inference gateway for SE-Brain.

The heavy ML dependencies are imported lazily so importing :mod:`sebrain`
does not require CUDA, bitsandbytes, transformers, or peft.

Configuration can be supplied explicitly or through environment variables:
SEBRAIN_QWEN_BASE_MODEL
SEBRAIN_QWEN_ADAPTER
SEBRAIN_QWEN_4BIT
SEBRAIN_QWEN_MAX_NEW_TOKENS
"""

from __future__ import annotations

import os
from typing import Any

from .model_gateway import ModelRequest, ModelResponse


class QwenLoRAGateway:
    """Real local Qwen causal-LM gateway with an existing PEFT adapter."""

    def __init__(
        self,
        *,
        base_model: str | None = None,
        adapter_path: str | None = None,
        load_in_4bit: bool | None = None,
        max_new_tokens: int | None = None,
        device: str | None = None,
    ) -> None:
        self.base_model = base_model or os.getenv(
            "SEBRAIN_QWEN_BASE_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct"
        )
        self.adapter_path = adapter_path or os.getenv(
            "SEBRAIN_QWEN_ADAPTER", "models/se_brain_d1_d58_qlora"
        )
        self.load_in_4bit = (
            load_in_4bit
            if load_in_4bit is not None
            else os.getenv("SEBRAIN_QWEN_4BIT", "1").lower() not in {"0", "false", "no"}
        )
        self.max_new_tokens = int(
            max_new_tokens
            if max_new_tokens is not None
            else os.getenv("SEBRAIN_QWEN_MAX_NEW_TOKENS", "384")
        )
        self.device = device
        self._tokenizer = None
        self._model = None

    @property
    def model_id(self) -> str:
        return f"{self.base_model}+{self.adapter_path}"

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen inference requires transformers, peft and torch. "
                "For 4-bit loading, install bitsandbytes as well."
            ) from exc

        adapter = os.path.abspath(self.adapter_path)
        if not os.path.isdir(adapter):
            raise FileNotFoundError(f"LoRA adapter directory not found: {adapter}")

        tokenizer = AutoTokenizer.from_pretrained(
            adapter if os.path.exists(os.path.join(adapter, "tokenizer_config.json"))
            else self.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "4-bit QLoRA inference requires bitsandbytes."
                ) from exc

        if self.device:
            model_kwargs["device_map"] = {"": self.device}
        elif torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)
        model = PeftModel.from_pretrained(model, adapter)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    @staticmethod
    def _context_text(context: tuple[dict[str, Any], ...]) -> str:
        blocks: list[str] = []
        for i, item in enumerate(context, 1):
            record = item.get("record", item)
            if not isinstance(record, dict):
                continue
            parts = [
                f"[Evidence {i}]",
                f"dataset={record.get('dataset_id', '')}",
                f"record_id={record.get('record_id', '')}",
                f"concept={record.get('concept', '')}",
                f"question={record.get('question', '')}",
                f"answer={record.get('answer', '')}",
                f"explanation={record.get('explanation', '')}",
                f"language={record.get('language', '')}",
                f"framework={record.get('framework', '')}",
            ]
            blocks.append("\n".join(p for p in parts if not p.endswith("=")))
        return "\n\n".join(blocks)

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._load()
        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        evidence = self._context_text(request.context)
        system = request.system or (
            "You are SE-Brain, an autonomous software engineering assistant. "
            "Use the supplied Knowledge Fabric evidence as the primary source. "
            "Do not invent facts or claim evidence that is not supplied. "
            "If the evidence is insufficient, say so explicitly."
        )
        user = request.prompt
        if evidence:
            user = (
                "Knowledge Fabric evidence:\n\n"
                + evidence
                + "\n\nUser query:\n"
                + user
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"{system}\n\n{user}"

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max(1, self.max_new_tokens),
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return ModelResponse(
            text=answer,
            model_id=self.model_id,
            finish_reason="stop",
            metadata={
                "base_model": self.base_model,
                "adapter_path": self.adapter_path,
                "load_in_4bit": self.load_in_4bit,
                "evidence_count": len(request.context),
            },
        )
