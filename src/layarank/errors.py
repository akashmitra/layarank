"""Domain exceptions for layarank."""

from typing import Any


class LayaRankError(Exception):
    """Base exception for all layarank errors."""

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class ConfigurationError(LayaRankError, ValueError):
    """Raised when an invalid configuration or argument is provided."""


class ContextLimitError(LayaRankError):
    """Raised when the query and candidates exceed the allowable context budget."""


class ModelExecutionError(LayaRankError):
    """Raised when the underlying Laya model or router fails during prediction."""
