"""One synthetic state, one decision question, one selected model, no bulk loop."""

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryFile
from uuid import uuid4

import httpx
import pandas as pd
from pydantic import Field, SecretStr, StrictStr, ValidationError

from jev_eval.backends.inference_sh import (
    SUPPORTED_APPS,
    InferenceShBackend,
    InferenceShError,
    RawExchange,
    build_decision_input,
)
from jev_eval.config import (
    ConnectionSettings,
    ConnectionTaskSettings,
    InferenceShSettings,
    ModelSettings,
    PrerequisiteError,
    Settings,
    parse_settings,
    read_yaml,
)
from jev_eval.data import fingerprint, load_json, validate_states
from jev_eval.environment import reproducibility_summary, sha256_file
from jev_eval.runner import safe_output_path
from jev_eval.tasks import NormalizedResponse, TaskDefinition, validate_task_template


@dataclass(repr=False)
class Connection:
    root: Path
    settings: ConnectionSettings
    backend: InferenceShSettings
    model: ModelSettings
    state: object
    task: TaskDefinition
    input_origin: str = "configured_files"


class _SyntheticStudy(Settings):
    title: StrictStr = Field(min_length=1)
    abstract: StrictStr = Field(min_length=1)


@dataclass(repr=False)
class ConnectionResult:
    metadata: dict
    request: dict = field(repr=False)
    response: NormalizedResponse | None = field(default=None, repr=False)
    exchange: RawExchange | None = field(default=None, repr=False)
    directory: Path | None = None
    _display_secrets: tuple[SecretStr, ...] = field(default_factory=tuple, repr=False)

    def __repr__(self):
        return (
            f"ConnectionResult(mode={self.metadata['mode']!r}, status={self.metadata['status']!r})"
        )


class ConnectionPersistenceError(RuntimeError):
    """Keep a received result available when local storage fails after the call."""

    def __init__(self, result: ConnectionResult):
        self.result = result
        super().__init__(
            "The API call finished but saving failed. The result is available as "
            "error.result. Do not resend the request; retry save_connection only."
        )


def load_connection(root, config_path="configs/connections/inference_sh.yaml") -> Connection:
    root = Path(root).resolve()
    settings = parse_settings(ConnectionSettings, read_yaml(root / config_path), config_path)
    if os.getenv("JEV_NOTEBOOK_TEST_MODE") == "1" and (
        settings.execution.mode != "mock"
        or settings.execution.allow_paid_calls
        or settings.output.allow_writes
    ):
        raise PrerequisiteError("Notebook tests require connection mock mode with permissions off.")
    backend = parse_settings(
        InferenceShSettings, read_yaml(root / settings.backend_config), settings.backend_config
    )
    model = parse_settings(
        ModelSettings, read_yaml(root / settings.model_config_file), settings.model_config_file
    )
    if model.hosted.app_id not in SUPPORTED_APPS:
        raise PrerequisiteError("Select a supported JEV app_id from the model YAML.")
    state = validate_states([load_json(root / settings.state_file)]).require_valid()[0]
    selected = parse_settings(
        ConnectionTaskSettings, read_yaml(root / settings.task_config), settings.task_config
    )
    task = validate_task_template(root / selected.template_file)
    if selected.question_id not in task.questions:
        raise PrerequisiteError("Configured question_id is absent from the selected task template.")
    task = TaskDefinition(questions={selected.question_id: task.questions[selected.question_id]})
    connection = Connection(root, settings, backend, model, state, task)
    safe_output_path(connection, settings.output.root)
    return connection


def preview_connection(connection):
    return {
        "app": connection.model.hosted.app_id,
        "input": build_decision_input(connection.state, connection.task),
        "wait": connection.backend.wait,
    }


def with_synthetic_example(connection, *, state: dict, question: dict) -> Connection:
    """Derive a connectivity-only request, preserving the configured connection.

    The two-field study is intentionally separate from evaluation CitationState:
    no generated study identifier or reference decisions enter this request.
    """
    try:
        study = _SyntheticStudy.model_validate(state)
        if not study.title.strip() or not study.abstract.strip():
            raise ValueError
    except (ValidationError, ValueError):
        raise PrerequisiteError(
            "Synthetic state requires only non-empty title and abstract strings."
        ) from None
    if (
        not isinstance(question, dict)
        or set(question) != {"id", "instructions"}
        or not isinstance(question["id"], str)
        or not question["id"].strip()
    ):
        raise PrerequisiteError("Synthetic question requires a non-empty id and instructions only.")
    try:
        task = TaskDefinition.model_validate(
            {
                "questions": {
                    question["id"]: {"type": "noul", "instructions": question["instructions"]}
                }
            }
        )
        if not question["instructions"].strip():
            raise ValueError
    except (ValidationError, ValueError):
        raise PrerequisiteError(
            "Synthetic question instructions must be a non-empty string."
        ) from None
    return replace(
        connection,
        state=study,
        task=task,
        settings=connection.settings.model_copy(deep=True),
        backend=connection.backend.model_copy(deep=True),
        model=connection.model.model_copy(deep=True),
        input_origin="notebook_synthetic_example",
    )


def connection_readiness(connection):
    mock = connection.settings.execution.mode == "mock"
    return pd.DataFrame(
        [
            {"check": "Mode", "value": connection.settings.execution.mode},
            {"check": "Requested app", "value": connection.model.hosted.app_id},
            {"check": "States / questions / POST attempts", "value": "1 / 1 / 1"},
            {"check": "Credential required", "value": not mock},
            {
                "check": "INFERENCE_API_KEY set (value hidden)",
                "value": "not read in mock mode"
                if mock
                else bool(os.getenv("INFERENCE_API_KEY", "").strip()),
            },
            {
                "check": "Paid calls enabled in YAML",
                "value": connection.settings.execution.allow_paid_calls,
            },
            {"check": "Writes enabled in YAML", "value": connection.settings.output.allow_writes},
        ]
    )


def run_connection(connection, *, allow_paid_calls=False, allow_writes=False) -> ConnectionResult:
    """Send once; live attempts always save, while mock exports are optional.

    Storage must be enabled and writable before a live POST. A later storage
    failure raises ConnectionPersistenceError carrying the result for save-only
    recovery, without retrying the provider request.
    """
    settings = connection.settings
    if settings.execution.mode == "live" and not (
        allow_paid_calls and settings.execution.allow_paid_calls
    ):
        raise PrerequisiteError(
            "Live calls require permission in connection YAML AND the request cell."
        )
    should_save = settings.execution.mode == "live" or allow_writes
    if should_save and not settings.output.allow_writes:
        raise PrerequisiteError(
            "Writing requires output.allow_writes: true in connection YAML. "
            "Live calls require automatic result saving."
        )
    if should_save:
        destination = safe_output_path(connection, settings.output.root)
        try:
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryFile(dir=destination) as probe:
                probe.write(b"storage check")
                probe.flush()
                os.fsync(probe.fileno())
        except OSError:
            raise PrerequisiteError(
                "Result storage is not writable. Fix the configured output folder "
                "before making a live call; no request was sent."
            ) from None
    request = preview_connection(connection)
    # Prepare reproducibility before any paid request; do not lose its response to
    # a missing local lockfile after a successful provider call.
    metadata = {
        "schema_version": 1,
        "mode": settings.execution.mode,
        "status": "not_sent",
        "requested_app": connection.model.hosted.app_id,
        "model_name": connection.model.name,
        "input_origin": connection.input_origin,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "state_sha256": fingerprint(connection.state.model_dump()),
        "task_sha256": connection.task.fingerprint,
        "request_sha256": fingerprint(request),
        "configuration": {
            "connection": settings.model_dump(),
            "backend": connection.backend.model_dump(),
            "model": connection.model.model_dump(),
        },
        "environment": reproducibility_summary(connection),
        "source_sha256": {
            "jev_eval/" + path.relative_to(Path(__file__).parent).as_posix(): sha256_file(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "binary_threshold": settings.screening.binary_threshold,
    }
    kwargs = {}
    if settings.execution.mode == "mock":
        fixture = load_json(connection.root / "data/examples/inference_sh_response.json")
        fixture["data"]["output"]["model"] = SUPPORTED_APPS[connection.model.hosted.app_id]
        identifier, selected = next(iter(connection.task.questions.items()))
        # Adapt only the known fabricated NOUL fixture. Missing/wrong answer IDs
        # still fail validation, and live provider responses are never rewritten.
        if (
            len(connection.task.questions) == 1
            and selected.type == "noul"
            and set(fixture["data"]["output"].get("nouls", {})) == {"retain_for_fulltext"}
        ):
            answer = fixture["data"]["output"]["nouls"]["retain_for_fulltext"]
            fixture["data"]["output"]["nouls"] = {identifier: answer}
        kwargs = {
            "api_key": SecretStr("mock"),
            "transport": httpx.MockTransport(lambda _request: httpx.Response(200, json=fixture)),
        }
    result = ConnectionResult(metadata, request)
    backend = InferenceShBackend(connection.backend, **kwargs)
    if settings.execution.mode == "live":
        # Keep only a private masked object for safe display even if the
        # environment key changes later. This field is never exported.
        result._display_secrets = (backend.api_key,)
    try:
        result.response = backend.predict(connection.state, connection.model, connection.task)
        result.exchange = backend.last_exchange
        result.metadata.update(
            status="completed",
            task_id=result.response.request_id,
            returned_model=result.response.returned_model,
        )
        threshold = settings.screening.binary_threshold
        if threshold is not None:
            result.response.predictions = {
                key: probability >= threshold
                for key, probability in result.response.probabilities.items()
            }
    except InferenceShError as error:
        result.exchange = error.exchange
        result.metadata.update(status="failed", error_code=error.code, task_id=error.task_id)
    finally:
        backend.close()
    result.metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
    if result.exchange is not None:
        result.metadata.update(
            http_status=result.exchange.status_code,
            elapsed_seconds=result.exchange.elapsed_seconds,
            response_received_at=result.exchange.received_at,
        )
    if should_save:
        try:
            save_connection(result, connection, allow_writes=True)
        except OSError:
            raise ConnectionPersistenceError(result) from None
    return result


def summarize_connection(result):
    row = {
        key: result.metadata.get(key)
        for key in ("mode", "status", "requested_app", "returned_model", "task_id", "error_code")
    }
    if result.response is not None:
        row.update(
            {f"P(true): {key}": value for key, value in result.response.probabilities.items()}
        )
        row.update(
            {f"decision: {key}": value for key, value in result.response.predictions.items()}
        )
        row.update(result.response.usage)
    return pd.DataFrame([row])


def connection_response_view(result, *, include_non_json=False) -> dict:
    """Return display-only JSON/body with credential redaction and clear errors.

    The exact exchange bytes and normalized response remain unchanged. Never
    display request headers, transport exceptions, or unredacted provider errors.
    """
    secrets = {secret.get_secret_value() for secret in result._display_secrets}
    secrets.add(os.getenv("INFERENCE_API_KEY", "").strip())
    secrets.discard("")
    credential_fields = {
        "authorization",
        "proxy_authorization",
        "api_key",
        "x_api_key",
        "inference_api_key",
        "token",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    }

    def redact_text(value):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
        return value

    def redact(value):
        if isinstance(value, dict):
            return {
                redact_text(key): "[REDACTED]"
                if key.casefold().replace("-", "_") in credential_fields
                or key.casefold().replace("-", "_").endswith("_api_key")
                else redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [redact(item) for item in value]
        return redact_text(value) if isinstance(value, str) else value

    exchange = result.exchange
    status = exchange.status_code if exchange is not None else None
    view = {"http_status": status, "is_json": False, "raw_json": None}
    task_id = result.metadata.get("task_id")
    view["task_id"] = redact_text(task_id) if isinstance(task_id, str) else None
    if exchange is None:
        view["message"] = (
            "No HTTP response received. The task outcome may be unknown; do not automatically resubmit."
        )
        return view

    def invalid_constant(_value):
        raise ValueError("Non-finite JSON number")

    try:
        parsed = json.loads(exchange.body, parse_constant=invalid_constant)
        view.update(is_json=True, raw_json=redact(parsed))
    except (ValueError, UnicodeError):
        if include_non_json:
            view["body_text"] = redact_text(exchange.body.decode("utf-8", errors="replace"))

    code = result.metadata.get("error_code")
    if status == 401:
        message = (
            "Authentication failed (HTTP 401). Check INFERENCE_API_KEY without printing its value."
        )
    elif status == 403:
        message = (
            "Access denied (HTTP 403). Check API-key permissions and access to the selected app."
        )
    elif not 200 <= status < 300:
        message = f"The API returned HTTP {status}. Inspect the credential-redacted response below."
    elif code == "invalid_task_envelope":
        message = "Unexpected response format: the API did not return the expected task envelope."
    elif code == "invalid_decision_output":
        message = (
            "Unexpected decision response: answer IDs or probability fields failed validation."
        )
    elif code == "task_not_completed":
        message = "The task is not completed. Inspect its task ID; no request was resubmitted."
    elif code in {"task_failed", "task_cancelled", "task_rejected"}:
        message = (
            "The provider task failed, was cancelled, or was rejected; no probabilities extracted."
        )
    elif result.metadata.get("status") == "completed":
        message = "Received a completed, validated response. Displayed JSON is credential-redacted."
    else:
        message = "Response retained, but no validated decision is available."
    if not view["is_json"]:
        message += " The response body is not valid JSON; its exact bytes remain retained."
    view["message"] = message
    return view


def save_connection(result, connection, *, allow_writes=False):
    if not (allow_writes and connection.settings.output.allow_writes):
        raise PrerequisiteError(
            "Writing requires permission in connection YAML AND the write cell."
        )
    if result.directory is not None:
        raise PrerequisiteError("This result was already saved; use its existing directory.")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
    destination = safe_output_path(connection, Path(connection.settings.output.root) / identifier)
    destination.mkdir(parents=True, exist_ok=False)
    if result.exchange is not None:
        (destination / "response.body").write_bytes(result.exchange.body)
    (destination / "request.json").write_text(
        json.dumps(result.request, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8"
    )
    if result.response is not None:
        (destination / "normalized.json").write_text(
            result.response.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    result.directory = destination
    return destination
