import json
from copy import deepcopy

import httpx
import pytest
from pydantic import ValidationError

from jev_eval.backends import create_backend
from jev_eval.backends.inference_sh import (
    InferenceShBackend,
    InferenceShError,
    RawExchange,
    validate_and_normalize,
)
from jev_eval.config import PrerequisiteError
from jev_eval.connections import load_connection
from jev_eval.tasks import TaskDefinition


def wire(connection):
    return json.loads((connection.root / "data/examples/inference_sh_response.json").read_text())


def exchange(payload, status=200):
    return RawExchange(status, 0.1, "2026-11-20T00:00:00+00:00", json.dumps(payload).encode())


@pytest.mark.parametrize("header,expected", [("10", 10), ("nonsense", None), ("inf", None)])
def test_retry_header_is_retained_without_transport_retries(project, header, expected):
    connection = load_connection(project.root)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, text="synthetic throttle", headers={"Retry-After": header})

    backend = InferenceShBackend(
        connection.backend, api_key="synthetic-key", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(InferenceShError) as error:
            backend.predict(connection.state, connection.model, connection.task)
        assert error.value.exchange.retry_after_seconds == expected
        assert not error.value.retryable
        assert len(calls) == 1
    finally:
        backend.close()


@pytest.mark.parametrize(
    "revision,expected", [(None, None), ("", None), ("reported-rev", "reported-rev")]
)
def test_revision_is_only_retained_if_explicitly_reported(project, revision, expected):
    connection = load_connection(project.root)
    raw = wire(connection)
    if revision is not None:
        raw["data"]["output"]["model_revision"] = revision
    assert validate_and_normalize(exchange(raw), connection.task).model_revision == expected


@pytest.mark.parametrize(
    "model_file, app, returned",
    [
        ("jev-9b", "autotrust/jev-9b", "autotrust/jev-9b"),
        ("jev-27b", "autotrust/jev-27b-v1-1", "autotrust/autotrust-jev-27b-v1.1"),
    ],
)
def test_documented_request_and_model_selection(project, monkeypatch, model_file, app, returned):
    connection = load_connection(project.root)
    path = project.root / "configs/connections/inference_sh.yaml"
    path.write_text(path.read_text().replace("models/jev-9b.yaml", f"models/{model_file}.yaml"))
    connection = load_connection(project.root)
    monkeypatch.setenv("INFERENCE_API_KEY", "".join(["synthetic", "-only"]))
    calls = []
    raw = wire(connection)
    raw["data"]["output"]["model"] = returned

    def handler(request):
        calls.append(request)
        assert str(request.url) == "https://api.inference.sh/apps/run"
        assert request.method == "POST"
        assert request.headers["Authorization"] == "Bearer " + "".join(["synthetic", "-only"])
        body = json.loads(request.content)
        assert body["app"] == app
        assert body["wait"] is True
        assert body["input"]["state"] == connection.state.model_dump()
        assert set(body["input"]) == {"state", "nouls"}
        assert body["input"]["nouls"][0]["id"] == "retain_for_fulltext"
        assert set(body["input"]["nouls"][0]["criteria"]) == {"true", "false"}
        assert not {"temperature", "max_output_tokens", "seed"} & set(body["input"])
        return httpx.Response(200, json=raw)

    backend = InferenceShBackend(connection.backend, transport=httpx.MockTransport(handler))
    try:
        result = backend.predict(connection.state, connection.model, connection.task)
        assert result.probabilities == {"retain_for_fulltext": 0.8}
        assert result.predictions == {}
        assert result.returned_model == returned
        assert result.raw == raw
        assert result.usage == {"input_tokens": 100}
        assert len(calls) == 1
        with pytest.raises(PrerequisiteError, match="generation"):
            backend.predict(
                connection.state, connection.model, connection.task, project.evaluation.generation
            )
        assert len(calls) == 1
    finally:
        backend.close()


@pytest.mark.parametrize("value", [True, "0.8", -0.1, 1.1, float("nan"), float("inf"), None])
def test_rejects_invalid_probabilities_without_losing_body(project, value):
    connection = load_connection(project.root)
    raw = wire(connection)
    raw["data"]["output"]["nouls"]["retain_for_fulltext"]["noul"] = value
    captured = exchange(raw)
    with pytest.raises(InferenceShError) as error:
        validate_and_normalize(captured, connection.task)
    assert error.value.exchange.body == captured.body
    assert not error.value.retryable


@pytest.mark.parametrize(
    "status, code",
    [(1, "task_not_completed"), (11, "task_failed"), (12, "task_cancelled"), (13, "task_rejected")],
)
def test_unfinished_and_failed_tasks(project, status, code):
    connection = load_connection(project.root)
    raw = wire(connection)
    raw["data"]["status"] = status
    raw["data"]["error"] = "PRIVATE PROVIDER BODY"
    with pytest.raises(InferenceShError) as caught:
        validate_and_normalize(exchange(raw), connection.task)
    assert caught.value.code == code
    assert caught.value.task_id == "synthetic-task-001"
    assert "PRIVATE" not in str(caught.value)
    assert b"PRIVATE" in caught.value.exchange.body


def test_http_invalid_json_and_missing_fields_are_preserved(project, monkeypatch):
    connection = load_connection(project.root)
    for captured in (
        RawExchange(401, 0, "now", b"private-auth-error"),
        RawExchange(200, 0, "now", b"not JSON"),
        exchange({}),
        exchange({"data": {"id": "synthetic", "status": 10, "output": {}}}),
    ):
        with pytest.raises(InferenceShError) as caught:
            validate_and_normalize(captured, connection.task)
        assert caught.value.exchange is captured
        assert "private-auth-error" not in str(caught.value)
    raw = wire(connection)
    raw["data"]["output"]["nouls"] = {"wrong_id": {"noul": 0.8}}
    with pytest.raises(InferenceShError, match="invalid_decision_output"):
        validate_and_normalize(exchange(raw), connection.task)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    with pytest.raises(PrerequisiteError, match="INFERENCE_API_KEY"):
        InferenceShBackend(connection.backend)


def test_choice_and_score_question_shapes_and_distributions(project):
    connection = load_connection(project.root)
    cases = [
        (
            "choice",
            {"options": [{"name": "retain"}, {"name": "exclude"}]},
            {
                "choice": "retain",
                "confidence": 0.8,
                "probabilities": {"retain": 0.8, "exclude": 0.2},
            },
            "choices",
        ),
        (
            "score",
            {"levels": ["clearly excluded", "potentially eligible"]},
            {
                "score": 0.8,
                "normalized": 0.8,
                "confidence": 0.8,
                "probabilities": {"0": 0.2, "1": 0.8},
                "legend": {"0": "clearly excluded", "1": "potentially eligible"},
            },
            "scores",
        ),
    ]
    for kind, extra, answer, group in cases:
        task = TaskDefinition.model_validate(
            {
                "questions": {
                    "screen": {"type": kind, "instructions": "Invented test question", **extra}
                }
            }
        )
        raw = wire(connection)
        raw["data"]["output"]["nouls"] = {}
        raw["data"]["output"][group] = {"screen": answer}
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json=raw)

        backend = InferenceShBackend(
            connection.backend, api_key="mock", transport=httpx.MockTransport(handler)
        )
        try:
            normalized = backend.predict(connection.state, connection.model, task)
            assert "screen" in getattr(normalized, group)
            question = calls[0]["input"][group][0]
            assert question["id"] == "screen"
            assert "type" not in question
        finally:
            backend.close()
        broken = deepcopy(raw)
        broken["data"]["output"][group]["screen"]["probabilities"] = {"bad": 0.8}
        with pytest.raises(InferenceShError, match="invalid_decision_output"):
            validate_and_normalize(exchange(broken), task)
        broken = deepcopy(raw)
        for key in broken["data"]["output"][group]["screen"]["probabilities"]:
            broken["data"]["output"][group]["screen"]["probabilities"][key] = 0.1
        with pytest.raises(InferenceShError, match="invalid_decision_output"):
            validate_and_normalize(exchange(broken), task)


def test_no_post_retries_and_transport_errors_are_sanitized(project):
    connection = load_connection(project.root)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="private provider error")

    backend = InferenceShBackend(
        connection.backend, api_key="mock", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(InferenceShError, match="http_503") as caught:
            backend.predict(connection.state, connection.model, connection.task)
        assert caught.value.exchange.body == b"private provider error"
        assert len(calls) == 1
    finally:
        backend.close()

    def timeout(request):
        raise httpx.ReadTimeout("private exception text", request=request)

    backend = InferenceShBackend(
        connection.backend, api_key="mock", transport=httpx.MockTransport(timeout)
    )
    try:
        with pytest.raises(InferenceShError, match="outcome_unknown") as caught:
            backend.predict(connection.state, connection.model, connection.task)
        assert caught.value.exchange is None
        assert "private" not in str(caught.value)
    finally:
        backend.close()


def test_factory_registration_and_strict_endpoint(project, monkeypatch):
    connection = load_connection(project.root)
    project.backend = connection.backend
    project.workflow.execution.mode = "hosted"
    project.workflow.execution.allow_paid_calls = True
    monkeypatch.setenv("INFERENCE_API_KEY", "mock")
    backend = create_backend(
        project,
        allow_paid_calls=True,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=wire(connection))),
    )
    assert isinstance(backend, InferenceShBackend)
    backend.close()
    with pytest.raises(ValidationError):
        type(connection.backend).model_validate(
            {**connection.backend.model_dump(), "base_url": "https://other.invalid"}
        )
