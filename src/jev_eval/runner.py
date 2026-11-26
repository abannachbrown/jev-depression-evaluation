"""Bounded batch inference, private run records, and conservative checkpoint resume."""

import hashlib
import json
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from jev_eval.backends import create_backend
from jev_eval.backends.base import BackendError
from jev_eval.backends.inference_sh import (
    SUPPORTED_APPS,
    InferenceShError,
    build_decision_input,
)
from jev_eval.config import HostedSettings, PrerequisiteError
from jev_eval.data import (
    dataset_fingerprint,
    fingerprint,
    freeze_full_dataset,
    load_full_dataset,
    load_json,
    load_jsonl,
    prepare_pilot_sample,
    validate_states,
)
from jev_eval.environment import reproducibility_summary, sha256_file
from jev_eval.tasks import load_task, primary_question, valid_batch_task, validate_response


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_output_path(project, path: str | Path) -> Path:
    allowed = (project.root / "results").resolve()
    # Reject a results symlink pointing outside the repository as well.
    if not allowed.is_relative_to(project.root.resolve()):
        raise PrerequisiteError("The results directory must remain inside this repository.")
    destination = (project.root / path).resolve()
    if not destination.is_relative_to(allowed):
        raise PrerequisiteError(
            "Generated files must be saved under this repository's results directory."
        )
    return destination


def require_writes(project, allow_writes):
    if not (allow_writes and project.workflow.output.allow_writes):
        raise PrerequisiteError(
            "Writing is disabled. Enable it in the workflow YAML AND the write cell."
        )


def _run_directory(project, run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise PrerequisiteError(
            "Run ID must be 1–80 letters, digits, dots, underscores, or hyphens."
        )
    return safe_output_path(project, Path(project.workflow.output.root) / run_id)


@dataclass
class RunPlan:
    project: object
    states: list
    task: object
    run_id: str
    fingerprint: str
    dataset_sha256: str
    environment: dict
    source_sha256: dict
    cohort_metadata: dict | None = None
    configuration_sha256: dict = field(default_factory=dict)

    def summary(self):
        logical = len(self.states) * len(self.project.models)
        full = self.project.workflow.inference_sh_full
        is_full = bool(full and full.enabled)
        skipped = (
            sum(not state.title.strip() and not state.abstract.strip() for state in self.states)
            if is_full
            else 0
        )
        logical -= skipped
        retry = (
            full.retry.max_attempts
            if is_full
            else (
                self.project.backend.retry.max_attempts
                if isinstance(self.project.backend, HostedSettings)
                else 1
            )
        )
        return pd.DataFrame(
            [
                {
                    "mode": self.project.workflow.execution.mode,
                    "records": len(self.states),
                    "not_inferred_no_text": skipped,
                    "models": len(self.project.models),
                    "approach": self.project.evaluation.task.approach,
                    "questions_per_request": len(self.task.questions),
                    "question_evaluations": logical * len(self.task.questions),
                    "logical_requests": logical,
                    "maximum_attempts": min(
                        logical * retry, self.project.workflow.execution.max_total_attempts
                    ),
                    "paid_calls_enabled_in_yaml": self.project.workflow.execution.allow_paid_calls,
                    "writes_enabled_in_yaml": self.project.workflow.output.allow_writes,
                    "cost_estimate": "not available; provider pricing is not configured",
                    "requested_apps": [m.hosted.app_id for m in self.project.models],
                    "sample_sha256": self.dataset_sha256,
                    "task_sha256": self.task.fingerprint,
                    "run_id": self.run_id,
                    "output_directory": str(_run_directory(self.project, self.run_id)),
                }
            ]
        )


def plan_run(project, states, task=None) -> RunPlan:
    _reject_inference_sh_batch(project)
    if not states:
        raise PrerequisiteError("No validated states selected.")
    if len({state.study_id for state in states}) != len(states):
        raise PrerequisiteError("Selected states contain duplicate identifiers.")
    limits = project.workflow.execution
    if limits.max_records is not None and len(states) > limits.max_records:
        raise PrerequisiteError("Selected records exceed max_records in configs/notebooks.yaml.")
    logical = len(states) * len(project.models)
    if _is_full(project):
        logical -= sum(not state.title.strip() and not state.abstract.strip() for state in states)
    if logical > limits.max_logical_requests or logical > limits.max_total_attempts:
        raise PrerequisiteError(
            "Request plan exceeds configured logical-request or total-attempt limits."
        )
    task = task or load_task(project)
    cohort = None
    if _is_inference_sh(project):
        if _is_full(project):
            cohort = _validate_full_plan(project, states, task)
        else:
            _validate_pilot_plan(project, states, task)
    environment = reproducibility_summary(project)
    source = {
        str(path.relative_to(project.root)): sha256_file(path)
        for path in sorted((project.root / "src/jev_eval").rglob("*.py"))
    }
    models = [model.model_dump() for model in project.models]
    if limits.mode == "hosted":
        for model, settings in zip(models, project.models, strict=True):
            model["requested_id"] = _requested_model_id(project, settings)
    dataset_hash = dataset_fingerprint(states)
    contract = {
        "dataset": dataset_hash,
        "task": task.fingerprint,
        "evaluation": project.evaluation.model_dump(),
        "backend": project.backend.model_dump(),
        "models": models,
        "mode": limits.mode,
        "lockfile": environment["lockfile_sha256"],
        "source": source,
        "python": environment["python"],
    }
    if cohort is not None:
        contract["full_cohort"] = fingerprint(cohort.manifest)
    schema_path = project.evaluation.task.response_schema_file
    if schema_path:
        contract["response_schema"] = sha256_file(project.root / schema_path)
    if limits.mode == "hosted" and isinstance(project.backend, HostedSettings):
        contract["endpoint_sha256"] = fingerprint(os.getenv(project.backend.base_url_env, ""))
    run_id = project.workflow.output.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    _run_directory(project, run_id)
    return RunPlan(
        project,
        list(states),
        task,
        run_id,
        fingerprint(contract),
        dataset_hash,
        environment,
        source,
        cohort.manifest if cohort is not None else None,
        _configuration_checksums(project) if cohort is not None else {},
    )


@dataclass
class RunResult:
    manifest: dict
    predictions: list[dict] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    directory: Path | None = None
    written_counts: dict = field(default_factory=dict, init=False, repr=False)
    _pending_exchange: object = field(default=None, init=False, repr=False)


class PilotPersistenceError(RuntimeError):
    """Preserve an in-memory run if storage fails; never retry the paid POST."""

    def __init__(self, result):
        self.result = result
        scope = "Full-evaluation" if result.manifest.get("run_kind") == "full" else "Pilot"
        super().__init__(
            f"{scope} storage failed. Keep this kernel open: error.result retains the run "
            "and any received exchange. Do not resend; inspect saved pending state first."
        )


def _atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_run_records(result):
    directory = result.directory
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("responses", "predictions", "failures"):
        records = getattr(result, name)
        written = result.written_counts.get(name, 0)
        # Append only new records: rewriting all prior responses at every request
        # would make a 5,000-state experiment perform quadratic amounts of I/O.
        with (directory / f"{name}.jsonl").open("a", encoding="utf-8") as stream:
            for record in records[written:]:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        result.written_counts[name] = len(records)


def _checkpoint(result):
    _append_run_records(result)
    directory = result.directory
    # Write the manifest last: an interrupted checkpoint retains its pending marker.
    _atomic_text(directory / "manifest.json", json.dumps(result.manifest, indent=2) + "\n")


def load_run(project, run_id) -> RunResult:
    directory = _run_directory(project, run_id)
    manifest = load_json(directory / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PrerequisiteError("Unsupported or invalid run manifest.")
    result = RunResult(
        manifest,
        *(
            load_jsonl(directory / f"{name}.jsonl")
            for name in ("predictions", "responses", "failures")
        ),
        directory,
    )
    result.written_counts = {
        name: len(getattr(result, name)) for name in ("predictions", "responses", "failures")
    }
    return result


def validate_run_compatibility(result, project, states):
    if result.manifest.get("dataset_sha256") != dataset_fingerprint(states):
        raise PrerequisiteError(
            "Saved run and selected dataset differ; choose the original dataset."
        )
    if result.manifest.get("model_names") != [model.name for model in project.models]:
        raise PrerequisiteError("Saved run and selected model names differ.")


def _prepare_run(plan, *, writing):
    project = plan.project
    mode = project.workflow.execution.mode
    result = RunResult(
        {
            "schema_version": 1,
            "run_id": plan.run_id,
            "mode": mode,
            "status": "running",
            "fingerprint": plan.fingerprint,
            "dataset_sha256": plan.dataset_sha256,
            "task_sha256": plan.task.fingerprint,
            "approach": project.evaluation.task.approach,
            "prompt_version": project.evaluation.task.prompt_version,
            "primary_question": primary_question(project, plan.task),
            "question_ids": list(plan.task.questions),
            "model_names": [model.name for model in project.models],
            "requested_model_ids": {
                m.name: _requested_model_id(project, m) if mode == "hosted" else m.name
                for m in project.models
            },
            "settings": project.resolved_settings(),
            "environment": plan.environment,
            "source_sha256": plan.source_sha256,
            "generation_applied": not _is_inference_sh(project),
            "started_at": utc_now(),
            "ended_at": None,
            "total_attempts": 0,
            "pending_request": None,
        }
    )
    directory = _run_directory(project, plan.run_id)
    if writing and directory.exists():
        if not project.evaluation.output.resume:
            raise PrerequisiteError(
                "Run already exists and resume is disabled; choose a new run ID."
            )
        result = load_run(project, plan.run_id)
        if result.manifest.get("fingerprint") != plan.fingerprint:
            raise PrerequisiteError(
                "Cannot resume: dataset, task, code, environment, or model settings changed."
            )
        if result.manifest.get("pending_request"):
            raise PrerequisiteError(
                "A prior request has an unknown outcome. Review it before resending."
            )
        result.manifest["status"] = "running"
        result.manifest["ended_at"] = None
    if writing:
        result.directory = directory
    if _is_full(project):
        if result.manifest.get("run_kind", "full") != "full":
            raise PrerequisiteError("Full runs cannot reuse a pilot run directory.")
        result.manifest.update(
            run_kind="full",
            experiment_id=project.workflow.inference_sh_full.experiment_id,
            full_cohort={
                key: value
                for key, value in plan.cohort_metadata.items()
                if key not in {"ordered_study_ids", "validation_issues", "not_inferred_no_text_ids"}
            },
            cohort_manifest=project.workflow.inference_sh_full.cohort_manifest,
            configuration_sha256=plan.configuration_sha256,
            screening_questions=plan.task.model_dump(),
        )
        result.manifest.setdefault("execution_sessions", []).append(
            {
                "at": utc_now(),
                "settings": project.resolved_settings(),
                "configuration_sha256": plan.configuration_sha256,
            }
        )
    return result


def run_experiment(
    plan,
    *,
    allow_paid_calls=False,
    allow_writes=False,
    backend=None,
    adapter=None,
    transport=None,
    sleeper=time.sleep,
    progress=None,
) -> RunResult:
    project = plan.project
    _reject_inference_sh_batch(project)
    if _is_full(project) and project.workflow.inference_sh_full.dry_run:
        raise PrerequisiteError(
            "Full-evaluation dry_run is enabled. Review the execution plan, then explicitly "
            "disable dry_run and enable YAML paid calls and the chosen notebook model flag."
        )
    mode = project.workflow.execution.mode
    if mode == "hosted" and not (allow_paid_calls and project.workflow.execution.allow_paid_calls):
        raise PrerequisiteError(
            "Paid inference requires permission in both YAML and the execution cell."
        )
    if mode == "local":
        from jev_eval.backends.local import unavailable

        unavailable()
    current = plan_run(project, plan.states, plan.task)
    if current.fingerprint != plan.fingerprint or (
        _is_full(project) and current.configuration_sha256 != plan.configuration_sha256
    ):
        raise PrerequisiteError("The planned inputs or settings changed; rebuild the request plan.")
    writing = bool(allow_writes)
    if writing:
        require_writes(project, allow_writes)
    if _is_inference_sh(project):
        if not writing:
            raise PrerequisiteError("Inference.sh pilots require ALLOW_WRITES = True.")
        if not _is_full(project) and project.evaluation.task.prompt_version == "screening-v1":
            prepare_pilot_sample(project, allow_writes=True)
        return _run_inference_sh_pilot(
            plan, backend=backend, transport=transport, sleeper=sleeper, progress=progress
        )
    result = _prepare_run(plan, writing=writing)
    owned_backend = backend is None
    backend = backend or create_backend(
        project, allow_paid_calls=allow_paid_calls, adapter=adapter, transport=transport
    )
    if backend.mode != mode:
        if owned_backend:
            backend.close()
        raise PrerequisiteError("Backend mode does not match the workflow.")
    retry = project.backend.retry if isinstance(project.backend, HostedSettings) else None
    max_attempts = retry.max_attempts if retry else 1
    completed = {(row["study_id"], row["model"]) for row in result.predictions}
    try:
        if writing:
            _checkpoint(result)
        for state in plan.states:
            for model in project.models:
                key = (state.study_id, model.name)
                if key in completed:
                    continue
                record = {
                    "study_id": state.study_id,
                    "model": model.name,
                    "mode": mode,
                    "status": "failed",
                    "predictions": {},
                    "attempts": 0,
                }
                started = time.monotonic()
                for attempt in range(1, max_attempts + 1):
                    if (
                        result.manifest["total_attempts"]
                        >= project.workflow.execution.max_total_attempts
                    ):
                        record["error_code"] = "total_attempt_budget_exhausted"
                        result.failures.append({**record, "at": utc_now()})
                        break
                    result.manifest["total_attempts"] += 1
                    result.manifest["pending_request"] = {
                        "study_id": state.study_id,
                        "model": model.name,
                    }
                    record["attempts"] = attempt
                    if writing:
                        _checkpoint(result)
                    try:
                        response = validate_response(
                            backend.predict(state, model, plan.task, project.evaluation.generation),
                            plan.task,
                        )
                    except (BackendError, PrerequisiteError) as error:
                        code = (
                            error.code
                            if isinstance(error, BackendError)
                            else "invalid_normalized_response"
                        )
                        record["error_code"] = code
                        result.failures.append(
                            {
                                "study_id": state.study_id,
                                "model": model.name,
                                "attempt": attempt,
                                "code": code,
                                "at": utc_now(),
                            }
                        )
                        result.manifest["pending_request"] = None
                        can_retry = (
                            isinstance(error, BackendError)
                            and error.retryable
                            and attempt < max_attempts
                        )
                        if can_retry:
                            if writing:
                                _checkpoint(result)
                            delay = min(
                                retry.initial_backoff_seconds * 2 ** (attempt - 1),
                                retry.max_backoff_seconds,
                            )
                            if retry.respect_retry_after and error.retry_after is not None:
                                delay = max(delay, error.retry_after)
                            sleeper(delay)
                            continue
                        break
                    else:
                        record.update(
                            status="success",
                            predictions=response.predictions,
                            request_id=response.request_id,
                            model_revision=response.model_revision,
                            usage=response.usage,
                        )
                        record.pop("error_code", None)
                        if project.evaluation.output.save_raw_responses:
                            result.responses.append(
                                {
                                    "study_id": state.study_id,
                                    "model": model.name,
                                    "raw": response.raw,
                                }
                            )
                        result.manifest["pending_request"] = None
                        break
                record["latency_seconds"] = time.monotonic() - started
                result.predictions.append(record)
                if writing:
                    _checkpoint(result)
        result.manifest["status"] = (
            "completed"
            if all(row["status"] == "success" for row in result.predictions)
            else "partial"
        )
        result.manifest["ended_at"] = utc_now()
        if writing:
            _checkpoint(result)
    except BaseException:
        result.manifest["status"] = "interrupted"
        if writing:
            _checkpoint(result)
        raise
    finally:
        if owned_backend:
            backend.close()
    return result


def _reject_inference_sh_batch(project):
    pilot = project.workflow.inference_sh_pilot
    full = project.workflow.inference_sh_full
    if pilot and pilot.enabled and full and full.enabled:
        raise PrerequisiteError("Select either the pilot or full-evaluation scope, not both.")
    if full and full.enabled and not _is_inference_sh(project):
        raise PrerequisiteError("Full-evaluation settings require the Inference.sh backend.")
    if pilot is not None and pilot.enabled and not _is_inference_sh(project):
        raise PrerequisiteError("Pilot settings require the Inference.sh backend.")
    if _is_inference_sh(project) and not ((pilot and pilot.enabled) or (full and full.enabled)):
        raise PrerequisiteError(
            "Inference.sh batch inference is not enabled. Select an explicit ten-state "
            "pilot workflow or use 02_model_connection.ipynb for a connection test."
        )


def _is_inference_sh(project):
    return getattr(project.backend, "protocol", None) == "inference_sh"


def _is_full(project):
    settings = project.workflow.inference_sh_full
    return bool(settings is not None and settings.enabled)


def _configuration_checksums(project):
    paths = [
        project.workflow_path,
        project.workflow.evaluation_config,
        project.workflow.scoring_config,
        project.evaluation.backend,
        project.evaluation.task.prompt_file,
        *project.evaluation.models,
    ]
    return {path: sha256_file(project.root / path) for path in paths if path}


def _validate_full_plan(project, states, task):
    limits = project.workflow.execution
    settings = project.workflow.inference_sh_full
    if (
        limits.mode != "hosted"
        or limits.max_records is not None
        or len(project.models) != 1
        or project.models[0].hosted.app_id not in SUPPORTED_APPS
        or not valid_batch_task(project, task)
        or not project.evaluation.output.save_raw_responses
    ):
        raise PrerequisiteError(
            "Full evaluation requires hosted mode, all records, one JEV app, "
            "one NOUL and raw response saving."
        )
    allowed = (project.root / "results/full").resolve()
    if not safe_output_path(project, project.workflow.output.root).is_relative_to(allowed):
        raise PrerequisiteError("Full evaluation outputs must remain under results/full/.")
    if not safe_output_path(project, settings.cohort_manifest).is_relative_to(allowed):
        raise PrerequisiteError("The full cohort manifest must remain under results/full/.")
    cohort = load_full_dataset(project)
    if (
        dataset_fingerprint(states) != cohort.manifest["states_sha256"]
        or task.fingerprint != cohort.manifest["task_sha256"]
    ):
        raise PrerequisiteError(
            "Full plans must use every source state in order and the exact task."
        )
    return cohort


def _requested_model_id(project, model):
    return (
        model.hosted.app_id if _is_inference_sh(project) else os.getenv(model.hosted.model_id_env)
    )


def _validate_pilot_plan(project, states, task):
    limits = project.workflow.execution
    if (
        limits.mode != "hosted"
        or limits.max_records != 10
        or limits.max_logical_requests > 10
        or limits.max_total_attempts > 10
        or len(project.models) != 1
        or project.models[0].hosted.app_id not in SUPPORTED_APPS
        or not valid_batch_task(project, task)
        or not project.evaluation.output.save_raw_responses
    ):
        raise PrerequisiteError(
            "Pilot requires hosted mode, one supported model, one NOUL, raw saving, "
            "max_records: 10 and request/attempt limits no greater than ten."
        )
    validate_states([state.model_dump() for state in states]).require_valid()
    sample = prepare_pilot_sample(
        project, preview_only=project.evaluation.task.prompt_version == "screening-v1"
    )
    if dataset_fingerprint(states) != sample.manifest["sample_sha256"]:
        raise PrerequisiteError("Selected states must match the ten frozen pilot states in order.")
    if task.fingerprint != sample.manifest["task_sha256"]:
        raise PrerequisiteError("The pilot task must match the frozen eligibility criteria.")


def _save_pilot_receipt(project, result, record, request, exchange, response):
    """Save exact bytes and a request/response receipt before completing a row."""
    receipt = {
        "study_id": record["study_id"],
        "model": record["model"],
        "request": request,
        "request_sha256": fingerprint(request),
        "state_sha256": fingerprint(request["input"]["state"]),
        "task_sha256": result.manifest["task_sha256"],
        "request_id": record.get("request_id"),
        "started_at": record["started_at"],
        "submitted_at": record["submitted_at"],
        "ended_at": record["ended_at"],
        "error_code": record.get("error_code"),
        "attempt": record.get("attempts", 1),
        "normalized": response.model_dump(mode="json", exclude={"raw"})
        if response is not None and _is_full(project)
        else response.model_dump(mode="json")
        if response is not None
        else None,
    }
    tag = uuid4().hex
    relative = Path("raw") / tag
    body = safe_output_path(project, result.directory / relative.with_suffix(".body"))
    record["receipt"] = str(relative.with_suffix(".json"))
    if not _is_full(project):
        result.responses.append(receipt)
    body.parent.mkdir(parents=True, exist_ok=True)
    if exchange is not None:
        with body.open("xb") as stream:
            stream.write(exchange.body)
            stream.flush()
            os.fsync(stream.fileno())
        receipt.update(
            http_status=exchange.status_code,
            elapsed_seconds=exchange.elapsed_seconds,
            response_received_at=exchange.received_at,
            response_sha256=hashlib.sha256(exchange.body).hexdigest(),
            response_body=str(relative.with_suffix(".body")),
            retry_after_seconds=exchange.retry_after_seconds,
        )
    receipt_path = safe_output_path(project, result.directory / relative.with_suffix(".json"))
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    if _is_full(project):
        result.responses.append(
            {
                key: value
                for key, value in {**receipt, "receipt": record["receipt"]}.items()
                if key not in {"request", "normalized"}
            }
        )


def _commit_pilot_request(result, *, unresolved):
    # First persist the receipt/row with the pending marker intact. Only a second
    # atomic manifest write may clear it. Interruptions can block, never resend.
    if result.manifest.get("run_kind") == "full":
        # The pre-POST pending manifest stays on disk while the append completes.
        # Do not rewrite the whole manifest twice after every full-run response.
        _append_run_records(result)
    else:
        _checkpoint(result)
    if not unresolved:
        committed = {**result.manifest, "pending_request": None}
        _atomic_text(result.directory / "manifest.json", json.dumps(committed, indent=2) + "\n")
        result.manifest = committed
        result._pending_exchange = None
    elif result.manifest.get("run_kind") == "full":
        _atomic_text(
            result.directory / "manifest.json", json.dumps(result.manifest, indent=2) + "\n"
        )


def _progress_counts(plan, result):
    counts = {status: 0 for status in ("success", "failed", "unknown", "not_inferred_no_text")}
    for row in result.predictions:
        counts[row["status"]] += 1
    return {
        "model": plan.project.models[0].name,
        "records": len(plan.states),
        "recorded": len(result.predictions),
        **counts,
        "attempts": result.manifest["total_attempts"],
        "status": result.manifest["status"],
    }


def _run_inference_sh_pilot(plan, *, backend, transport, sleeper, progress=None):
    """Shared serial decision loop; full-run policy does not relax pilot limits."""
    project = plan.project
    full = project.workflow.inference_sh_full if _is_full(project) else None
    policy = full or project.workflow.inference_sh_pilot
    directory = _run_directory(project, plan.run_id)
    lock = directory.parent / ("." + plan.run_id + ".lock")
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise PrerequisiteError(
            "This run is locked. Investigate the process and pending state "
            "before removing a stale lock; do not start another writer."
        ) from None
    except OSError:
        raise PrerequisiteError("Result storage is not writable; no request was sent.") from None
    os.close(descriptor)
    result = None
    owned_backend = backend is None
    try:
        result = _prepare_run(plan, writing=True)
        if full:
            cohort_path = freeze_full_dataset(
                project, load_full_dataset(project), allow_writes=True
            )
            result.manifest["cohort_manifest_sha256"] = sha256_file(cohort_path)
        else:
            result.manifest["pilot_sample"] = prepare_pilot_sample(project).manifest
        model = project.models[0]
        completed = {(row["study_id"], row["model"]) for row in result.predictions}
        if full:
            expected = {(state.study_id, model.name) for state in plan.states}
            if (
                len(completed) != len(result.predictions)
                or not completed <= expected
                or any(
                    row["status"] not in {"success", "failed", "not_inferred_no_text"}
                    for row in result.predictions
                )
            ):
                raise PrerequisiteError(
                    "Saved full-run rows are duplicate, unexpected or unresolved."
                )
        attempts_by_id = {}
        for receipt in result.responses:
            identifier = receipt["study_id"]
            attempts_by_id[identifier] = max(
                attempts_by_id.get(identifier, 0), receipt.get("attempt", 1)
            )
        # Actual writes test storage before constructing a client or sending.
        _checkpoint(result)
        if progress:
            progress(_progress_counts(plan, result))
        backend = backend or create_backend(project, allow_paid_calls=True, transport=transport)
        if backend.mode != "hosted":
            raise PrerequisiteError("Inference.sh requires a hosted backend.")
        last_submission = None
        consecutive_failures = 0
        stop = False
        for state in plan.states:
            if (state.study_id, model.name) in completed:
                continue
            if full and not state.title.strip() and not state.abstract.strip():
                result.predictions.append(
                    {
                        "study_id": state.study_id,
                        "model": model.name,
                        "mode": "hosted",
                        "requested_app": model.hosted.app_id,
                        "status": "not_inferred_no_text",
                        "predictions": {},
                        "probabilities": {},
                        "attempts": 0,
                        "request_id": None,
                        "returned_model": None,
                        "model_revision": None,
                        "revision_status": "not_reported",
                        "ended_at": utc_now(),
                    }
                )
                _checkpoint(result)
                continue
            attempt = attempts_by_id.get(state.study_id, 0) + 1 if full else 1
            max_attempts = full.retry.max_attempts if full else 1
            if attempt > max_attempts:
                raise PrerequisiteError(
                    "A saved study exhausted its retry limit without a terminal "
                    "record. Inspect its receipts before recovery."
                )
            while attempt <= max_attempts:
                if (
                    result.manifest["total_attempts"]
                    >= project.workflow.execution.max_total_attempts
                ):
                    result.manifest["status"] = "attempt_budget_exhausted"
                    stop = True
                    break
                if full:
                    cooldown = result.manifest.get("retry_not_before")
                    if cooldown and cooldown["study_id"] == state.study_id:
                        delay = max(
                            0,
                            (
                                datetime.fromisoformat(cooldown["at"]) - datetime.now(timezone.utc)
                            ).total_seconds(),
                        )
                        # Chunk long provider waits so progress remains visible and interruptible.
                        while delay > 0:
                            chunk = min(delay, 60)
                            sleeper(chunk)
                            delay -= chunk
                            if progress:
                                progress(_progress_counts(plan, result))
                started = utc_now()
                request = {
                    "app": model.hosted.app_id,
                    "input": build_decision_input(
                        state, plan.task, max_questions=project.backend.max_questions
                    ),
                    "wait": project.backend.wait,
                }
                record = {
                    "study_id": state.study_id,
                    "model": model.name,
                    "mode": "hosted",
                    "requested_app": model.hosted.app_id,
                    "started_at": started,
                    "status": "failed",
                    "attempts": attempt,
                    "predictions": {},
                    "probabilities": {},
                    "request_id": None,
                    "returned_model": None,
                    "model_revision": None,
                    "request_sha256": fingerprint(request),
                }
                result.manifest["total_attempts"] += 1
                result.manifest["pending_request"] = {
                    "study_id": state.study_id,
                    "model": model.name,
                    "attempt": attempt,
                    "requested_app": model.hosted.app_id,
                    "started_at": started,
                    "request_sha256": fingerprint(request),
                }
                _checkpoint(result)
                response, exchange, unresolved = None, None, False
                try:
                    previous = result.manifest.get("last_request_started_at")
                    if previous is not None:
                        elapsed = (
                            time.monotonic() - last_submission
                            if last_submission is not None
                            else (
                                datetime.now(timezone.utc) - datetime.fromisoformat(previous)
                            ).total_seconds()
                        )
                        delay = policy.min_request_interval_seconds - elapsed
                        if delay > 0:
                            sleeper(delay)
                    last_submission = time.monotonic()
                    record["submitted_at"] = utc_now()
                    result.manifest["last_request_started_at"] = record["submitted_at"]
                    response = validate_response(
                        backend.predict(state, model, plan.task), plan.task
                    )
                    exchange = backend.last_exchange
                    record.update(status="success", **response.model_dump(exclude={"raw"}))
                except InferenceShError as error:
                    exchange = error.exchange
                    unresolved = error.code in {
                        "transport_failure_outcome_unknown",
                        "task_not_completed",
                        "invalid_task_envelope",
                    } or (exchange is not None and exchange.status_code >= 500)
                    record.update(
                        status="unknown" if unresolved else "failed",
                        error_code=error.code,
                        request_id=error.task_id,
                    )
                    result.failures.append(
                        {
                            "study_id": state.study_id,
                            "model": model.name,
                            "attempt": attempt,
                            "code": error.code,
                            "request_id": error.task_id,
                            "at": utc_now(),
                        }
                    )
                result._pending_exchange = exchange
                record["ended_at"] = utc_now()
                record["revision_status"] = (
                    "provider_reported_unverified"
                    if record.get("model_revision")
                    else "not_reported"
                )
                if exchange is not None:
                    record.update(
                        http_status=exchange.status_code,
                        latency_seconds=exchange.elapsed_seconds,
                        response_received_at=exchange.received_at,
                    )
                if unresolved:
                    result.manifest["pending_request"].update(
                        request_id=record.get("request_id"), error_code=record["error_code"]
                    )
                retrying = bool(
                    full and record.get("error_code") == "http_429" and attempt < max_attempts
                )
                if retrying:
                    delay = min(
                        full.retry.initial_backoff_seconds * 2 ** (attempt - 1),
                        full.retry.max_backoff_seconds,
                    )
                    if exchange.retry_after_seconds is not None:
                        delay = max(delay, exchange.retry_after_seconds)
                    result.manifest["retry_not_before"] = {
                        "study_id": state.study_id,
                        "at": (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(),
                    }
                else:
                    result.manifest.pop("retry_not_before", None)
                    result.predictions.append(record)
                _save_pilot_receipt(project, result, record, request, exchange, response)
                _commit_pilot_request(result, unresolved=unresolved)
                if retrying:
                    attempt += 1
                    continue
                if record["status"] == "success":
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if not full or unresolved or record.get("http_status") in {401, 402, 403, 429}:
                        result.manifest["status"] = (
                            "needs_review" if unresolved else "paused_after_error"
                        )
                        stop = True
                    elif consecutive_failures >= full.max_consecutive_failures:
                        result.manifest["status"] = "paused_consecutive_failures"
                        stop = True
                if progress and (
                    len(result.predictions) % 25 == 0 or record["status"] != "success"
                ):
                    progress(_progress_counts(plan, result))
                break
            if stop:
                break
        if not stop:
            statuses = {row["status"] for row in result.predictions}
            result.manifest["status"] = (
                "completed"
                if statuses <= {"success"}
                else ("completed_with_exceptions" if full else "partial")
            )
        result.manifest["ended_at"] = utc_now()
        _checkpoint(result)
        if progress:
            progress(_progress_counts(plan, result))
        return result
    except OSError:
        if result is not None:
            raise PilotPersistenceError(result) from None
        raise PrerequisiteError("Storage is unavailable; no new request was sent.") from None
    finally:
        if owned_backend and backend is not None:
            backend.close()
        with suppress(OSError):
            lock.unlink()


def summarize_run(result) -> pd.DataFrame:
    rows = []
    for model in result.manifest["model_names"]:
        records = [row for row in result.predictions if row["model"] == model]
        rows.append(
            {
                "model": model,
                "mode": result.manifest["mode"],
                "success": sum(row["status"] == "success" for row in records),
                "failed": sum(row["status"] != "success" for row in records),
                "attempts": sum(row.get("attempts", 0) for row in records),
            }
        )
    return pd.DataFrame(rows)
