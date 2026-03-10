from .llm_backend import ScoringBackend, LLMResponse, MultiMetricResponse
from .dspy_backend import DSPyBackend

__all__ = [
    "ScoringBackend",
    "LLMResponse",
    "MultiMetricResponse",
    "DSPyBackend",
    "create_backend",
]


def create_backend(backend_type: str = "dspy", **kwargs) -> ScoringBackend:
    """Factory for scoring backends.

    Parameters
    ----------
    backend_type : str
        ``"dspy"`` (default) or ``"vllm"``.
    **kwargs
        Forwarded to the backend constructor.
    """
    if backend_type == "dspy":
        return DSPyBackend(**kwargs)
    elif backend_type == "vllm":
        from .vllm_backend import VLLMOfflineBackend
        return VLLMOfflineBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend type: {backend_type!r}")
