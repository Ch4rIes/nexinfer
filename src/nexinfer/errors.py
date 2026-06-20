from __future__ import annotations


class NexInferError(Exception):
    """Base class for NexInfer errors."""


class ConfigurationError(NexInferError, ValueError):
    """Raised when user-provided configuration is invalid."""


class BackendError(NexInferError, ValueError):
    """Raised when a backend violates the inference contract."""


class SchedulerError(NexInferError, ValueError):
    """Raised when request scheduling cannot proceed."""


class CacheError(NexInferError, MemoryError):
    """Raised when KV-cache allocation fails."""
