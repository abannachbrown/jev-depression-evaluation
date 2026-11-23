"""Future local inference adapters; import runtime dependencies lazily here."""

from jev_eval.config import PrerequisiteError


def unavailable():
    raise PrerequisiteError(
        "Local inference is not implemented. Select demo or hosted; no weights or GPU packages "
        "will be downloaded. A future local adapter must implement the shared backend interface."
    )
