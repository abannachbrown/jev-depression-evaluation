"""Strict configuration loading. Environment values are never placed in reports."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PrerequisiteError(ValueError):
    """An actionable prerequisite is missing; messages must not contain input data."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DatasetSettings(Settings):
    path: str
    root_env: str | None = None
    format: Literal["json", "jsonl"]
    glob: str | None = None
    expected_count: int | None = Field(default=None, ge=1)
    order: Literal["sorted"] = "sorted"
    shuffle: bool = False


class GenerationSettings(Settings):
    temperature: float = Field(ge=0)
    max_output_tokens: int = Field(gt=0)


class TaskSettings(Settings):
    prompt_file: str | None = None
    response_schema_file: str | None = None
    metrics: list[str] = Field(default_factory=list)


class RunOutputSettings(Settings):
    root: str = "results"
    resume: bool = True
    save_raw_responses: bool = True


class EvaluationSettings(Settings):
    schema_version: Literal[1]
    name: str
    seed: int
    dataset: DatasetSettings
    backend: str
    models: list[str] = Field(min_length=1)
    generation: GenerationSettings
    task: TaskSettings
    output: RunOutputSettings


class RetrySettings(Settings):
    max_attempts: int = Field(default=4, ge=1)
    initial_backoff_seconds: float = Field(default=1, ge=0)
    max_backoff_seconds: float = Field(default=30, ge=0)
    respect_retry_after: bool = True
    status_codes: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])


class HostedSettings(Settings):
    schema_version: Literal[1]
    type: Literal["hosted"]
    protocol: str | None = None
    base_url_env: str
    api_key_env: str
    timeout_seconds: float = Field(default=120, gt=0)
    max_concurrency: Literal[1] = 1
    retry: RetrySettings


class InferenceShSettings(Settings):
    schema_version: Literal[1]
    type: Literal["hosted"]
    protocol: Literal["inference_sh"]
    base_url: Literal["https://api.inference.sh"]
    run_path: Literal["/apps/run"]
    api_key_env: Literal["INFERENCE_API_KEY"]
    wait: Literal[True] = True
    timeout_seconds: float = Field(gt=0)
    max_attempts: Literal[1] = 1


class LocalSettings(Settings):
    schema_version: Literal[1]
    type: Literal["local"]
    enabled: bool = False
    engine: str | None = None
    device: str | None = None
    dtype: str | None = None
    model_root: str
    model_root_env: str | None = None
    allow_downloads: Literal[False] = False


class HostedModelSettings(Settings):
    model_id_env: str
    app_id: str | None = None


class LocalModelSettings(Settings):
    relative_path: str
    revision: str | None = None


class ModelSettings(Settings):
    schema_version: Literal[1]
    name: str = Field(min_length=1)
    display_name: str
    hosted: HostedModelSettings
    local: LocalModelSettings


class ExecutionSettings(Settings):
    mode: Literal["demo", "hosted", "local"] = "demo"
    allow_paid_calls: bool = False
    max_records: int | None = Field(default=5, ge=1)
    max_logical_requests: int = Field(default=10, ge=1)
    max_total_attempts: int = Field(default=40, ge=1)


class PrivacySettings(Settings):
    display_record_text: bool = False
    display_raw_responses: bool = False


class NotebookOutputSettings(Settings):
    allow_writes: bool = False
    root: str = "results"
    run_id: str | None = None


class NotebookEvaluationSettings(Settings):
    reference_labels: str
    demo_predictions: str


class InferenceShPilotSettings(Settings):
    enabled: bool = False
    sample_manifest: str
    min_request_interval_seconds: float = Field(default=3.1, ge=3.1, allow_inf_nan=False)


class FullRetrySettings(Settings):
    max_attempts: int = Field(default=3, ge=1, le=3)
    initial_backoff_seconds: float = Field(default=2, ge=0, allow_inf_nan=False)
    max_backoff_seconds: float = Field(default=60, ge=0, allow_inf_nan=False)


class InferenceShFullSettings(Settings):
    enabled: bool = False
    dry_run: bool = True
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    cohort_manifest: str
    missing_text_policy: Literal["record_not_inferred"] = "record_not_inferred"
    max_concurrency: Literal[1] = 1
    min_request_interval_seconds: float = Field(default=3.1, ge=3.1, allow_inf_nan=False)
    max_consecutive_failures: int = Field(default=3, ge=1)
    retry: FullRetrySettings = Field(default_factory=FullRetrySettings)
    decision_threshold: None = None
    label_mapping: dict[str, bool] = Field(default_factory=lambda: {"0": False, "1": True})


class WorkflowSettings(Settings):
    schema_version: Literal[1]
    evaluation_config: str
    scoring_config: str
    execution: ExecutionSettings
    privacy: PrivacySettings
    output: NotebookOutputSettings
    evaluation: NotebookEvaluationSettings
    inference_sh_pilot: InferenceShPilotSettings | None = None
    inference_sh_full: InferenceShFullSettings | None = None


class ScoringSettings(Settings):
    schema_version: Literal[1]
    question: str
    positive_label: Literal[True] = True
    metrics: list[Literal["sensitivity", "specificity", "precision", "f1", "balanced_accuracy"]]
    require_complete_predictions_for_headline_metrics: bool = True


class FullAnalysisSettings(Settings):
    schema_version: Literal[1]
    workflow_9b: str
    workflow_27b: str
    decision_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    report_partial_metrics: bool = False
    threshold_note: str = "No decision threshold has been specified."
    output_root: str = "results/evaluation/full-depression-v1"
    allow_writes: bool = False


class ConnectionExecutionSettings(Settings):
    mode: Literal["mock", "live"] = "mock"
    allow_paid_calls: bool = False
    max_states: Literal[1] = 1
    max_questions: Literal[1] = 1


class ConnectionScreeningSettings(Settings):
    binary_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class ConnectionOutputSettings(Settings):
    allow_writes: bool = False
    root: str = "results/connections"


class ConnectionSettings(Settings):
    schema_version: Literal[1]
    backend_config: str
    model_config_file: str
    state_file: Literal["data/examples/jev_citation_state.json"]
    task_config: str
    execution: ConnectionExecutionSettings
    screening: ConnectionScreeningSettings
    output: ConnectionOutputSettings


class ConnectionTaskSettings(Settings):
    schema_version: Literal[1]
    template_file: str
    question_id: str


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise PrerequisiteError(f"Configuration file missing: {path.name}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeError):
        raise PrerequisiteError(f"Invalid YAML in {path.name}; check its syntax.") from None
    if not isinstance(value, dict):
        raise PrerequisiteError(f"{path.name} must contain a YAML mapping.")
    return value


def parse_settings(cls, value: dict, name: str):
    try:
        return cls.model_validate(value)
    except ValidationError as error:
        fields = sorted({".".join(map(str, item["loc"])) for item in error.errors()})
        raise PrerequisiteError(
            f"Invalid settings in {name}; check fields: {', '.join(fields)}"
        ) from None


@dataclass
class ProjectConfig:
    root: Path
    workflow: WorkflowSettings
    evaluation: EvaluationSettings
    backend: HostedSettings | LocalSettings | InferenceShSettings
    models: list[ModelSettings]
    scoring: ScoringSettings
    workflow_path: str = "configs/notebooks.yaml"

    def resolved_settings(self) -> dict:
        return {
            "workflow": self.workflow.model_dump(),
            "evaluation": self.evaluation.model_dump(),
            "backend": self.backend.model_dump(),
            "models": [model.model_dump() for model in self.models],
            "scoring": self.scoring.model_dump(),
        }


def load_project_config(root: Path, workflow_path: str = "configs/notebooks.yaml") -> ProjectConfig:
    root = root.resolve()
    workflow = parse_settings(WorkflowSettings, read_yaml(root / workflow_path), workflow_path)
    evaluation = parse_settings(
        EvaluationSettings, read_yaml(root / workflow.evaluation_config), workflow.evaluation_config
    )
    backend_raw = read_yaml(root / evaluation.backend)
    if backend_raw.get("type") not in ("hosted", "local"):
        raise PrerequisiteError("Backend type must be hosted or local.")
    if backend_raw.get("protocol") == "inference_sh":
        backend_cls = InferenceShSettings
    else:
        backend_cls = HostedSettings if backend_raw["type"] == "hosted" else LocalSettings
    backend = parse_settings(backend_cls, backend_raw, evaluation.backend)
    models = [parse_settings(ModelSettings, read_yaml(root / p), p) for p in evaluation.models]
    if len({model.name for model in models}) != len(models):
        raise PrerequisiteError("Model names must be unique.")
    scoring = parse_settings(
        ScoringSettings, read_yaml(root / workflow.scoring_config), workflow.scoring_config
    )
    return ProjectConfig(root, workflow, evaluation, backend, models, scoring, workflow_path)
