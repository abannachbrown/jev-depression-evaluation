"""Backend selection with explicit permissions for paid inference."""

from jev_eval.config import PrerequisiteError


def create_backend(project, *, allow_paid_calls=False, adapter=None, transport=None):
    mode = project.workflow.execution.mode
    if mode == "demo":
        from jev_eval.backends.demo import DemoBackend

        return DemoBackend()
    if mode == "local":
        from jev_eval.backends.local import unavailable

        unavailable()
    if not (allow_paid_calls and project.workflow.execution.allow_paid_calls):
        raise PrerequisiteError(
            "Paid inference is disabled. Enable it in configs/notebooks.yaml AND the request cell."
        )
    if getattr(project.backend, "protocol", None) == "inference_sh":
        from jev_eval.backends.inference_sh import InferenceShBackend

        return InferenceShBackend(project.backend, transport=transport)
    from jev_eval.backends.hosted.client import HostedBackend

    return HostedBackend(project, adapter=adapter, transport=transport)
