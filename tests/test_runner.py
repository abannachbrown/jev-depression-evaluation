import json

import httpx
import pytest

from jev_eval.backends import create_backend
from jev_eval.backends.base import BackendError
from jev_eval.backends.demo import DemoBackend
from jev_eval.backends.hosted.client import HostedBackend
from jev_eval.config import PrerequisiteError
from jev_eval.data import load_states, validate_states
from jev_eval.runner import plan_run, run_experiment, safe_output_path
from jev_eval.tasks import NormalizedResponse


def make_plan(project):
    states = validate_states(load_states(project.evaluation.dataset, project.root)).require_valid()
    return plan_run(project, states)


def test_default_demo_writes_nothing_and_never_uses_http(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network allowed")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    result = run_experiment(make_plan(project))
    assert len(result.predictions) == 4
    assert result.manifest["mode"] == "demo"
    assert result.manifest["total_attempts"] == 4
    assert not (project.root / "results").exists()


def test_paid_local_and_write_gates(project):
    plan = make_plan(project)
    with pytest.raises(PrerequisiteError, match="Writing is disabled"):
        run_experiment(plan, allow_writes=True)
    project.workflow.execution.mode = "hosted"
    with pytest.raises(PrerequisiteError, match="permission"):
        run_experiment(plan)
    project.workflow.execution.allow_paid_calls = True
    with pytest.raises(PrerequisiteError, match="Hosted inference is blocked"):
        create_backend(project, allow_paid_calls=True)
    project.workflow.execution.mode = "local"
    with pytest.raises(PrerequisiteError, match="not implemented"):
        create_backend(project)


def test_checkpoint_resume_and_fingerprint_protection(project):
    project.workflow.output.allow_writes = True
    project.workflow.output.run_id = "test-resume"
    plan = make_plan(project)
    first = run_experiment(plan, allow_writes=True)
    second = run_experiment(make_plan(project), allow_writes=True)
    assert second.manifest["total_attempts"] == first.manifest["total_attempts"] == 4
    assert len(second.predictions) == 4
    saved = (second.directory / "predictions.jsonl").read_text().splitlines()
    assert len(saved) == 4, "Resuming must not append duplicate completed records"
    project.evaluation.generation.temperature = 0.1
    with pytest.raises(PrerequisiteError, match="Cannot resume"):
        run_experiment(make_plan(project), allow_writes=True)


def test_unknown_outcome_blocks_resume(project):
    class Interrupted(DemoBackend):
        def predict(self, *args):
            raise KeyboardInterrupt

    project.workflow.output.allow_writes = True
    project.workflow.output.run_id = "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_experiment(make_plan(project), allow_writes=True, backend=Interrupted())
    with pytest.raises(PrerequisiteError, match="unknown outcome"):
        run_experiment(make_plan(project), allow_writes=True)


def test_retry_budget_and_retry_after(project):
    class Once(DemoBackend):
        calls = 0

        def predict(self, *args):
            self.calls += 1
            if self.calls == 1:
                raise BackendError("http_429", retryable=True, retry_after=2)
            return super().predict(*args)

    sleeps = []
    result = run_experiment(make_plan(project), backend=Once(), sleeper=sleeps.append)
    assert sleeps == [2]
    assert result.manifest["total_attempts"] == 5
    assert len(result.failures) == 1
    assert all(row["status"] == "success" for row in result.predictions)

    class Always(DemoBackend):
        def predict(self, *args):
            raise BackendError("http_503", retryable=True)

    project.workflow.execution.max_total_attempts = 4
    result = run_experiment(make_plan(project), backend=Always(), sleeper=lambda _: None)
    assert result.manifest["total_attempts"] == 4
    assert len(result.predictions) == 4
    assert result.manifest["status"] == "partial"


def test_paths_and_logical_limits(project):
    with pytest.raises(PrerequisiteError, match="results"):
        safe_output_path(project, "data/private/accidental.csv")
    with pytest.raises(PrerequisiteError, match="results"):
        safe_output_path(project, "results/../../accidental.csv")
    project.workflow.execution.max_logical_requests = 1
    with pytest.raises(PrerequisiteError, match="exceeds"):
        make_plan(project)


def test_stale_plan_cannot_record_changed_settings(project):
    plan = make_plan(project)
    project.evaluation.generation.temperature = 0.2
    with pytest.raises(PrerequisiteError, match="rebuild the request plan"):
        run_experiment(plan)


def test_hosted_transport_uses_explicit_adapter_and_redacts_failures(project, monkeypatch):
    class TestAdapter:
        protocol = "test-fixture-only"

        def build_request(self, client, **kwargs):
            return client.build_request("POST", "/test", json={"model": kwargs["model_id"]})

        def parse_response(self, response):
            return NormalizedResponse(predictions=response.json()["predictions"])

    monkeypatch.setenv("JEV_API_BASE_URL", "https://fixture.invalid/")
    monkeypatch.setenv("JEV_API_KEY", "".join(["synthetic", "-test-value"]))
    for model in project.models:
        monkeypatch.setenv(model.hosted.model_id_env, model.name)
    plan = make_plan(project)
    project.backend.protocol = TestAdapter.protocol
    project.evaluation.task.prompt_file = "prompts/drafts/binary_noul.json"
    project.evaluation.task.response_schema_file = "test-schema.json"
    project.workflow.execution.mode = "hosted"
    project.workflow.execution.allow_paid_calls = True
    schema = project.root / "test-schema.json"
    schema.write_text(json.dumps({"type": "object"}))
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"predictions": {"retain_for_fulltext": True}})
    )
    backend = HostedBackend(project, TestAdapter(), transport)
    try:
        response = backend.predict(
            plan.states[0], project.models[0], plan.task, project.evaluation.generation
        )
        assert response.predictions == {"retain_for_fulltext": True}
    finally:
        backend.close()

    bad = HostedBackend(
        project,
        TestAdapter(),
        httpx.MockTransport(lambda request: httpx.Response(400, text="PRIVATE PROVIDER BODY")),
    )
    try:
        with pytest.raises(BackendError) as caught:
            bad.predict(plan.states[0], project.models[0], plan.task, project.evaluation.generation)
        assert str(caught.value) == "http_400"
    finally:
        bad.close()
