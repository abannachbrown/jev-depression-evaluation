"""One-attempt inference interface shared by every backend."""

from typing import Protocol

from jev_eval.tasks import NormalizedResponse


class BackendError(RuntimeError):
    """Sanitized failure; never include response bodies, URLs, or credentials."""

    def __init__(self, code: str, *, retryable=False, retry_after=None):
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after


class InferenceBackend(Protocol):
    mode: str

    def predict(self, state, model, task, generation) -> NormalizedResponse: ...

    def close(self) -> None: ...
