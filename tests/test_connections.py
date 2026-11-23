import json

import httpx
import pytest

from jev_eval.backends.inference_sh import InferenceShBackend, RawExchange
from jev_eval.config import PrerequisiteError
from jev_eval.connections import (
    ConnectionPersistenceError,
    ConnectionResult,
    connection_response_view,
    load_connection,
    preview_connection,
    run_connection,
    save_connection,
    with_synthetic_example,
)
from jev_eval.data import fingerprint, load_states, validate_states
from jev_eval.environment import sha256_file
from jev_eval.runner import plan_run, run_experiment


def test_default_mock_ignores_credentials_and_writes_nothing(project, monkeypatch):
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)

    def forbid_network(*args, **kwargs):
        raise AssertionError("Actual network transport is forbidden")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
    connection = load_connection(project.root)
    result = run_connection(connection)
    assert result.metadata["mode"] == "mock"
    assert result.metadata["status"] == "completed"
    assert result.response.predictions == {}
    assert result.response.probabilities == {"retain_for_fulltext": 0.8}
    assert result.directory is None
    assert not (project.root / "results").exists()
    assert "Authorization" not in json.dumps(preview_connection(connection))


def test_permissions_fail_before_sending_or_writing(project, monkeypatch):
    connection = load_connection(project.root)
    connection.settings.execution.mode = "live"
    sent = []
    monkeypatch.setattr(httpx.Client, "send", lambda *args, **kwargs: sent.append(True))
    with pytest.raises(PrerequisiteError, match="permission"):
        run_connection(connection)
    with pytest.raises(PrerequisiteError, match="permission"):
        run_connection(connection, allow_paid_calls=True)
    connection.settings.execution.allow_paid_calls = True
    with pytest.raises(PrerequisiteError, match="Writing requires"):
        run_connection(connection, allow_paid_calls=True, allow_writes=True)
    assert not sent


def test_explicit_export_and_threshold(project):
    connection = load_connection(project.root)
    connection.settings.output.allow_writes = True
    connection.settings.screening.binary_threshold = 0.8
    result = run_connection(connection, allow_writes=True)
    assert result.response.predictions == {"retain_for_fulltext": True}
    assert (result.directory / "response.body").read_bytes() == result.exchange.body
    assert set(path.name for path in result.directory.iterdir()) == {
        "response.body",
        "request.json",
        "metadata.json",
        "normalized.json",
    }
    assert "Authorization" not in (result.directory / "request.json").read_text()
    with pytest.raises(PrerequisiteError, match="already saved"):
        save_connection(result, connection, allow_writes=True)


def test_invalid_responses_can_be_saved_without_extraction(project):
    path = project.root / "data/examples/inference_sh_response.json"
    raw = json.loads(path.read_text())
    raw["data"]["output"]["nouls"] = {}
    path.write_text(json.dumps(raw))
    connection = load_connection(project.root)
    connection.settings.output.allow_writes = True
    result = run_connection(connection, allow_writes=True)
    assert result.metadata["status"] == "failed"
    assert result.metadata["error_code"] == "invalid_decision_output"
    assert result.response is None
    assert (result.directory / "response.body").read_bytes() == result.exchange.body
    assert not (result.directory / "normalized.json").exists()


def test_connection_restrictions_and_test_mode(project, monkeypatch):
    path = project.root / "configs/connections/inference_sh.yaml"
    original = path.read_text()
    for before, after in (
        ("max_states: 1", "max_states: 2"),
        ("max_questions: 1", "max_questions: 2"),
        ("data/examples/jev_citation_state.json", "data/private/real.json"),
    ):
        path.write_text(original.replace(before, after))
        with pytest.raises(PrerequisiteError, match="Invalid settings"):
            load_connection(project.root)
    path.write_text(original.replace("mode: mock", "mode: live"))
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    with pytest.raises(PrerequisiteError, match="mock mode"):
        load_connection(project.root)


def test_inference_sh_is_blocked_in_existing_batch_runner(project):
    states = validate_states(load_states(project.evaluation.dataset, project.root)).require_valid()
    plan = plan_run(project, states)
    project.backend = load_connection(project.root).backend
    with pytest.raises(PrerequisiteError, match="batch inference is not enabled"):
        plan_run(project, states)
    with pytest.raises(PrerequisiteError, match="batch inference is not enabled"):
        run_experiment(plan)


def test_model_switch_does_not_require_logic_changes(project):
    first = run_connection(load_connection(project.root))
    path = project.root / "configs/connections/inference_sh.yaml"
    path.write_text(path.read_text().replace("models/jev-9b.yaml", "models/jev-27b.yaml"))
    second = run_connection(load_connection(project.root))
    assert first.response.probabilities == second.response.probabilities
    assert first.metadata["requested_app"] == "autotrust/jev-9b"
    assert second.metadata["requested_app"] == "autotrust/jev-27b-v1-1"
    assert second.response.returned_model == "autotrust/autotrust-jev-27b-v1.1"


def synthetic_example(connection):
    state = {
        "title": "Effects of an antidepressant on depressive-like behaviour in mice",
        "abstract": (
            "Adult mice were randomly assigned to receive an "
            "antidepressant or vehicle control for 14 days. "
            "Depressive-like behaviour was measured using "
            "the forced swim test."
        ),
    }
    question = {
        "id": "screening_eligibility",
        "instructions": (
            "Does this study report an in vivo experiment "
            "in mice or rats evaluating an antidepressant "
            "intervention with a behavioural outcome?"
        ),
    }
    return with_synthetic_example(connection, state=state, question=question), state, question


def test_synthetic_example_exact_request_and_unchanged_base(project, monkeypatch):
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    base = load_connection(project.root)
    original = preview_connection(base)
    example, state, question = synthetic_example(base)
    assert preview_connection(example) == {
        "app": "autotrust/jev-9b",
        "input": {"state": state, "nouls": [question]},
        "wait": True,
    }
    assert preview_connection(base) == original
    assert base.input_origin == "configured_files"
    assert example.input_origin == "notebook_synthetic_example"
    result = run_connection(example)
    assert result.response.probabilities == {"screening_eligibility": 0.8}
    assert result.metadata["input_origin"] == "notebook_synthetic_example"
    assert result.metadata["mode"] == "mock"
    assert connection_response_view(result)["http_status"] == 200
    example.settings.execution.allow_paid_calls = True
    assert not base.settings.execution.allow_paid_calls


@pytest.mark.parametrize(
    "state,question",
    [
        ({"title": "x"}, {"id": "screen", "instructions": "x"}),
        (
            {"title": "x", "abstract": "x", "study_id": "extra"},
            {"id": "screen", "instructions": "x"},
        ),
        ({"title": "x", "abstract": "x"}, {"id": "", "instructions": "x"}),
        ({"title": "x", "abstract": "x"}, {"id": "screen", "instructions": 1}),
        ({"title": "x", "abstract": "x"}, {"id": "screen", "instructions": " "}),
    ],
)
def test_invalid_synthetic_inputs_have_actionable_errors(project, state, question):
    with pytest.raises(PrerequisiteError, match="Synthetic"):
        with_synthetic_example(load_connection(project.root), state=state, question=question)


def test_synthetic_mock_does_not_repair_a_missing_answer(project):
    path = project.root / "data/examples/inference_sh_response.json"
    fixture = json.loads(path.read_text())
    fixture["data"]["output"]["nouls"] = {}
    path.write_text(json.dumps(fixture))
    example, _, _ = synthetic_example(load_connection(project.root))
    result = run_connection(example)
    assert result.response is None
    assert result.metadata["error_code"] == "invalid_decision_output"


def test_response_view_redacts_echoed_credentials_without_altering_raw_exchange(
    project, monkeypatch
):
    import jev_eval.connections as module

    fake = "".join(["synthetic", "-credential-", "0123456789"])
    other = "".join(["another", "-credential"])
    payload = {
        "data": {
            "id": "synthetic-task",
            "status": 10,
            "output": {
                "model": "autotrust/jev-9b",
                "nouls": {"screening_eligibility": {"noul": 0.8}},
            },
        },
        "headers": {"Authorization": "Bearer " + fake},
        "api_key": other,
        "echo": ["prefix " + fake, {fake: fake}],
    }
    calls = []

    def handler(request):
        calls.append(request)
        assert json.loads(request.content)["input"]["nouls"][0]["id"] == "screening_eligibility"
        return httpx.Response(200, json=payload)

    def mocked_backend(settings, **kwargs):
        return InferenceShBackend(settings, api_key=fake, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(module, "InferenceShBackend", mocked_backend)
    example, _, _ = synthetic_example(load_connection(project.root))
    example.settings.execution.mode = "live"
    example.settings.execution.allow_paid_calls = True
    example.settings.output.allow_writes = True
    result = run_connection(example, allow_paid_calls=True)
    original = result.exchange.body
    monkeypatch.setenv("INFERENCE_API_KEY", "changed-after-request")
    view = connection_response_view(result)
    assert view["is_json"]
    assert fake not in json.dumps(view)
    assert other not in json.dumps(view)
    assert view["raw_json"]["headers"]["Authorization"] == "[REDACTED]"
    assert result.exchange.body == original
    assert result.response.raw == payload
    assert len(calls) == 1
    assert "[REDACTED]" in json.dumps(view)
    assert fake not in repr(result)


@pytest.fixture
def live_connection(project, monkeypatch):
    """Exercise live policy using only a synthetic credential and mock transport."""
    import jev_eval.connections as module

    connection = load_connection(project.root)
    connection.settings.execution.mode = "live"
    connection.settings.execution.allow_paid_calls = True
    connection.settings.output.allow_writes = True
    payload = json.loads((project.root / "data/examples/inference_sh_response.json").read_text())
    calls = []

    def forbid_network(*args, **kwargs):
        raise AssertionError("Actual network transport is forbidden")

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)

    def mocked_backend(settings, **kwargs):
        return InferenceShBackend(
            settings, api_key="synthetic-test-only", transport=httpx.MockTransport(handler)
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
    monkeypatch.setattr(module, "InferenceShBackend", mocked_backend)
    return connection, calls, payload


def test_live_call_automatically_saves_raw_response_and_provenance(live_connection):
    import jev_eval.connections as module

    connection, calls, _ = live_connection
    result = run_connection(connection, allow_paid_calls=True)
    assert len(calls) == 1
    assert result.metadata["status"] == "completed"
    assert result.directory is not None
    assert (result.directory / "response.body").read_bytes() == result.exchange.body
    assert json.loads((result.directory / "request.json").read_text()) == result.request
    assert json.loads((result.directory / "normalized.json").read_text()) == (
        result.response.model_dump(mode="json")
    )
    metadata = json.loads((result.directory / "metadata.json").read_text())
    assert metadata == result.metadata
    assert metadata["request_sha256"] == fingerprint(result.request)
    assert metadata["state_sha256"] == fingerprint(connection.state.model_dump())
    assert metadata["task_sha256"] == connection.task.fingerprint
    assert metadata["task_id"] == result.response.request_id
    assert metadata["returned_model"] == result.response.returned_model
    assert metadata["requested_app"] == "autotrust/jev-9b"
    assert metadata["http_status"] == 200
    assert metadata["started_at"] <= metadata["ended_at"]
    assert metadata["environment"]["lockfile_sha256"] == sha256_file(connection.root / "uv.lock")
    assert metadata["source_sha256"]["jev_eval/connections.py"] == sha256_file(
        module.Path(module.__file__)
    )
    assert "synthetic-test-only" not in json.dumps(metadata)
    second = run_connection(connection, allow_paid_calls=True)
    assert len(calls) == 2
    assert second.directory != result.directory
    assert (result.directory / "response.body").read_bytes() == result.exchange.body


@pytest.mark.parametrize("failure", ["disabled", "unwritable"])
def test_live_storage_preflight_stops_before_any_request(live_connection, monkeypatch, failure):
    import jev_eval.connections as module

    connection, calls, _ = live_connection
    if failure == "disabled":
        connection.settings.output.allow_writes = False
        message = "Live calls require automatic result saving"
    else:

        def denied(*args, **kwargs):
            raise PermissionError("synthetic storage failure")

        monkeypatch.setattr(module, "TemporaryFile", denied)
        message = "no request was sent"
    with pytest.raises(PrerequisiteError, match=message):
        run_connection(connection, allow_paid_calls=True)
    assert calls == []


def test_save_failure_retains_live_result_for_save_only_recovery(live_connection, monkeypatch):
    import jev_eval.connections as module

    connection, calls, _ = live_connection

    def fail_save(*args, **kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(module, "save_connection", fail_save)
    with pytest.raises(ConnectionPersistenceError, match="Do not resend") as captured:
        run_connection(connection, allow_paid_calls=True)
    result = captured.value.result
    assert len(calls) == 1
    assert result.metadata["status"] == "completed"
    assert result.response.probabilities == {"retain_for_fulltext": 0.8}
    assert result.exchange.status_code == 200
    assert result.directory is None
    # Call the original writer directly: recovery never calls the provider.
    destination = save_connection(result, connection, allow_writes=True)
    assert (destination / "response.body").read_bytes() == result.exchange.body
    assert json.loads((destination / "metadata.json").read_text()) == result.metadata
    assert len(calls) == 1


def test_invalid_live_response_is_also_automatically_preserved(live_connection):
    connection, calls, payload = live_connection
    payload["data"]["output"]["nouls"] = {}
    result = run_connection(connection, allow_paid_calls=True)
    assert len(calls) == 1
    assert result.metadata["error_code"] == "invalid_decision_output"
    assert result.response is None
    assert (result.directory / "response.body").read_bytes() == result.exchange.body
    assert not (result.directory / "normalized.json").exists()


@pytest.mark.parametrize(
    "status,code,expected",
    [
        (401, "http_401", "Authentication failed"),
        (403, "http_403", "Access denied"),
        (500, "http_500", "HTTP 500"),
        (200, "invalid_task_envelope", "Unexpected response format"),
        (200, "invalid_decision_output", "Unexpected decision response"),
        (200, "task_not_completed", "not completed"),
    ],
)
def test_response_view_messages_for_errors_and_unexpected_json(status, code, expected):
    result = ConnectionResult(
        {"mode": "mock", "status": "failed", "error_code": code},
        {},
        exchange=RawExchange(status, 0, "now", b'{"diagnostic":"synthetic"}'),
    )
    view = connection_response_view(result)
    assert expected in view["message"]
    assert view["http_status"] == status
    assert view["raw_json"] == {"diagnostic": "synthetic"}


def test_non_json_and_missing_http_response_are_safe_and_informative(monkeypatch):
    fake = "".join(["synthetic", "-credential-", "0123456789"])
    monkeypatch.setenv("INFERENCE_API_KEY", fake)
    result = ConnectionResult(
        {"mode": "mock", "status": "failed", "error_code": "invalid_task_envelope"},
        {},
        exchange=RawExchange(200, 0, "now", ("not JSON; echoed " + fake).encode()),
    )
    hidden = connection_response_view(result)
    assert not hidden["is_json"]
    assert "body_text" not in hidden
    detailed = connection_response_view(result, include_non_json=True)
    assert fake not in json.dumps(detailed)
    assert "not valid JSON" in detailed["message"]
    assert "[REDACTED]" in detailed["body_text"]
    result.exchange = None
    view = connection_response_view(result)
    assert view["http_status"] is None
    assert "No HTTP response" in view["message"]
