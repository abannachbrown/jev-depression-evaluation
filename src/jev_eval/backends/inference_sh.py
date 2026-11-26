"""Inference.sh decision REST transport; one state/question and one POST per call.

Contract: https://inference.sh/docs/extend/app-code#decision-models-contract
Run API: https://inference.sh/apps/autotrust/jev-27b-v1-1#api-reference
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictInt, StrictStr, ValidationError

from jev_eval.backends.base import BackendError
from jev_eval.backends.hosted.client import retry_after_seconds
from jev_eval.config import InferenceShSettings, PrerequisiteError
from jev_eval.tasks import (
    HYBRID_QUESTIONS,
    ChoiceAnswer,
    NormalizedResponse,
    Probability,
    ScoreAnswer,
    validate_response,
)

SUPPORTED_APPS = {
    "autotrust/jev-9b": "autotrust/jev-9b",
    "autotrust/jev-27b-v1-1": "autotrust/autotrust-jev-27b-v1.1",
}


@dataclass
class RawExchange:
    status_code: int
    elapsed_seconds: float
    received_at: str
    body: bytes = field(repr=False)
    retry_after_seconds: float | None = None


class InferenceShError(BackendError):
    """Sanitized error with a private raw exchange, even on validation failure."""

    def __init__(self, code, exchange=None, *, task_id=None):
        super().__init__(code, retryable=False)
        self.exchange = exchange
        self.task_id = task_id


class _WireModel(BaseModel):
    # Task metadata may grow; the original bytes and JSON are preserved separately.
    model_config = ConfigDict(extra="ignore", strict=True)


class _NoulAnswer(_WireModel):
    noul: Probability


class _Output(_WireModel):
    model: StrictStr = Field(min_length=1)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    model_revision: StrictStr | None = None
    nouls: dict[str, _NoulAnswer] = Field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = Field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = Field(default_factory=dict)


class _Task(_WireModel):
    id: StrictStr = Field(min_length=1)
    status: StrictInt
    output: dict | None = None


class _Envelope(_WireModel):
    data: _Task


def build_decision_input(state, task, *, max_questions=1):
    if len(task.questions) != 1 and not (
        max_questions == 7
        and set(task.questions) == set(HYBRID_QUESTIONS)
        and all(q.type == "noul" for q in task.questions.values())
    ):
        raise PrerequisiteError("Connection tests require exactly one decision question.")
    groups = {"noul": "nouls", "choice": "choices", "score": "scores"}
    payload = {"state": state.model_dump()}
    for identifier, question in task.questions.items():
        body = question.model_dump(exclude={"type"}, exclude_none=True)
        if not body.get("criteria"):
            body.pop("criteria", None)
        payload.setdefault(groups[question.type], []).append({"id": identifier, **body})
    return payload


def _invalid_constant(_value):
    raise ValueError("Non-finite JSON number")


def validate_and_normalize(exchange, task):
    if not 200 <= exchange.status_code < 300:
        raise InferenceShError(f"http_{exchange.status_code}", exchange)
    try:
        raw = json.loads(exchange.body, parse_constant=_invalid_constant)
        envelope = _Envelope.model_validate(raw)
    except (ValueError, UnicodeError, ValidationError):
        raise InferenceShError("invalid_task_envelope", exchange) from None
    provider_task = envelope.data
    if provider_task.status != 10:
        code = {11: "task_failed", 12: "task_cancelled", 13: "task_rejected"}.get(
            provider_task.status, "task_not_completed"
        )
        raise InferenceShError(code, exchange, task_id=provider_task.id)
    try:
        output = _Output.model_validate(provider_task.output)
        normalized = NormalizedResponse(
            probabilities={key: answer.noul for key, answer in output.nouls.items()},
            choices=output.choices,
            scores=output.scores,
            request_id=provider_task.id,
            returned_model=output.model,
            model_revision=output.model_revision
            if output.model_revision and output.model_revision.strip()
            else None,
            usage={"input_tokens": output.input_tokens} if output.input_tokens is not None else {},
            raw=raw,
        )
        return validate_response(normalized, task)
    except (ValidationError, PrerequisiteError):
        raise InferenceShError(
            "invalid_decision_output", exchange, task_id=provider_task.id
        ) from None


class InferenceShBackend:
    mode = "hosted"
    supports_batch = False

    def __init__(self, settings: InferenceShSettings, *, transport=None, api_key=None):
        self.settings = settings
        value = api_key if api_key is not None else os.getenv(settings.api_key_env, "").strip()
        self.api_key = value if isinstance(value, SecretStr) else SecretStr(value)
        if not self.api_key.get_secret_value().strip():
            raise PrerequisiteError(
                "Set INFERENCE_API_KEY in the environment or ignored .env file."
            )
        self.last_exchange = None
        self.client = httpx.Client(
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=transport if transport is not None else httpx.HTTPTransport(retries=0),
        )

    def predict(self, state, model, task, generation=None):
        self.last_exchange = None
        if generation is not None:
            raise PrerequisiteError("Inference.sh decisions do not use text-generation parameters.")
        if model.hosted.app_id not in SUPPORTED_APPS:
            raise PrerequisiteError("Select a supported JEV app_id from the model YAML.")
        payload = {
            "app": model.hosted.app_id,
            "input": build_decision_input(state, task, max_questions=self.settings.max_questions),
            "wait": self.settings.wait,
        }
        started = time.monotonic()
        try:
            response = self.client.post(
                self.settings.run_path,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
            )
        except httpx.TransportError:
            # No response means there are no bytes to save. The task may have run.
            raise InferenceShError("transport_failure_outcome_unknown") from None
        retry_after = retry_after_seconds(response.headers.get("Retry-After"))
        if retry_after is not None and not math.isfinite(retry_after):
            retry_after = None
        exchange = RawExchange(
            response.status_code,
            time.monotonic() - started,
            datetime.now(timezone.utc).isoformat(),
            response.content,
            retry_after,
        )
        self.last_exchange = exchange
        return validate_and_normalize(exchange, task)

    def close(self):
        self.client.close()
