"""Model gateway boundary for the SE Brain.

The Brain owns orchestration, retrieval, validation, and governance. A model
is an interchangeable provider behind this contract. No provider is bundled
or assumed to be available.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ModelRequest:
    prompt: str
    system: str = ""
    context: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    model_id: str
    finish_reason: str = "stop"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ModelGateway(Protocol):
    """Stable inference boundary used by the Brain."""

    model_id: str

    def generate(self, request: ModelRequest) -> ModelResponse:
        ...


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested without a configured model."""


class UnavailableModelGateway:
    """Explicitly unavailable gateway; never fabricates model output."""

    model_id = "unconfigured"

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise ModelUnavailableError(
            "No model gateway is configured. Connect a trained/local model "
            "adapter before requesting model generation."
        )


class CallableModelGateway:
    """Small adapter for tests and future local/open-weight inference servers.

    The callable must perform real inference and return either a string or a
    ModelResponse. This adapter deliberately contains no provider-specific SDK.
    """

    def __init__(
        self,
        fn: Callable[[ModelRequest], str | ModelResponse],
        *,
        model_id: str,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id is required")
        self._fn = fn
        self.model_id = model_id

    def generate(self, request: ModelRequest) -> ModelResponse:
        result = self._fn(request)
        if isinstance(result, ModelResponse):
            return result
        if not isinstance(result, str):
            raise TypeError("model callable must return str or ModelResponse")
        return ModelResponse(text=result, model_id=self.model_id)
