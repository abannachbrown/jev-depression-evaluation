"""Hosted HTTP transport. An explicit provider adapter is required before any call."""

import os
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from jev_eval.backends.base import BackendError
from jev_eval.config import HostedSettings, PrerequisiteError
from jev_eval.tasks import NormalizedResponse, validate_response


class ProviderAdapter(Protocol):
    """Implement only after the provider's request and response contract is known."""

    protocol: str

    def build_request(
        self, client, *, model_id, state, task, generation, api_key
    ) -> httpx.Request: ...

    def parse_response(self, response: httpx.Response) -> NormalizedResponse: ...


def hosted_readiness(project, adapter=None) -> list[dict]:
    cfg = project.backend
    rows = [{"check": "Hosted backend selected", "ready": isinstance(cfg, HostedSettings)}]
    if not isinstance(cfg, HostedSettings):
        return rows
    rows.extend(
        [
            {"check": "Provider protocol configured", "ready": bool(cfg.protocol)},
            {
                "check": "Matching provider adapter implemented",
                "ready": adapter is not None and adapter.protocol == cfg.protocol,
            },
            {
                "check": "Runnable task configured",
                "ready": bool(project.evaluation.task.prompt_file),
            },
            {
                "check": "Response schema configured",
                "ready": bool(project.evaluation.task.response_schema_file),
            },
        ]
    )
    for name in (
        cfg.base_url_env,
        cfg.api_key_env,
        *(m.hosted.model_id_env for m in project.models),
    ):
        rows.append({"check": f"{name} set", "ready": bool(os.getenv(name, "").strip())})
    return rows


def retry_after_seconds(header):
    if not header:
        return None
    try:
        return max(0, float(header))
    except ValueError:
        try:
            return max(
                0, (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
            )
        except (ValueError, TypeError, OverflowError):
            return None


class HostedBackend:
    mode = "hosted"

    def __init__(self, project, adapter=None, transport=None):
        missing = [row["check"] for row in hosted_readiness(project, adapter) if not row["ready"]]
        if missing:
            raise PrerequisiteError("Hosted inference is blocked: " + "; ".join(missing))
        self.settings = project.backend
        self.adapter = adapter
        base_url = os.environ[self.settings.base_url_env].strip()
        parsed = urlsplit(base_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise PrerequisiteError("Hosted base URL must use HTTPS without embedded credentials.")
        self.api_key = os.environ[self.settings.api_key_env]
        self.client = httpx.Client(
            base_url=base_url,
            timeout=self.settings.timeout_seconds,
            follow_redirects=False,
            transport=transport,
        )

    def predict(self, state, model, task, generation):
        try:
            request = self.adapter.build_request(
                self.client,
                model_id=os.environ[model.hosted.model_id_env],
                state=state,
                task=task,
                generation=generation,
                api_key=self.api_key,
            )
            if (
                request.url.scheme != "https"
                or request.url.host != self.client.base_url.host
                or request.url.port != self.client.base_url.port
            ):
                raise BackendError("adapter_endpoint_mismatch")
            response = self.client.send(request)
            if not response.is_success:
                raise BackendError(
                    f"http_{response.status_code}",
                    retryable=response.status_code in self.settings.retry.status_codes,
                    retry_after=retry_after_seconds(response.headers.get("Retry-After")),
                )
            return validate_response(
                NormalizedResponse.model_validate(self.adapter.parse_response(response)), task
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            raise BackendError("connection_failed", retryable=True) from None
        except httpx.TransportError:
            # Request outcome may be unknown. Do not automatically resend it.
            raise BackendError("transport_failure_outcome_unknown") from None
        except (ValidationError, PrerequisiteError, ValueError, KeyError, TypeError):
            raise BackendError("invalid_provider_response_or_adapter") from None
        except BackendError:
            raise
        except Exception:
            raise BackendError("provider_adapter_failure") from None

    def close(self):
        self.client.close()
