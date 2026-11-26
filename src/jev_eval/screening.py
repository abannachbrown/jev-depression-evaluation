"""Versioned binary/hybrid campaigns; explicit execution and read-only analysis."""

import csv
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import Field

from jev_eval.config import (
    PrerequisiteError,
    Settings,
    load_project_config,
    parse_settings,
    read_yaml,
)
from jev_eval.data import load_full_dataset, prepare_pilot_sample
from jev_eval.evaluation import (
    load_saved_full_evaluation,
    ranking_readouts,
    select_recall_thresholds,
)
from jev_eval.reporting import (
    full_run_plan_table,
    paired_pilot_plan_table,
    pilot_run_status,
    validate_full_results,
)
from jev_eval.runner import _run_directory, load_run, plan_run, run_experiment
from jev_eval.tasks import PRIMARY_DOMAINS, Probability, load_task, valid_batch_task


class ApproachProfiles(Settings):
    pilot: dict[str, str]
    full: dict[str, str]
    analysis_config: str


class ScreeningSettings(Settings):
    schema_version: Literal[1]
    default_approach: Literal["binary", "hybrid"]
    prompt_version: Literal["screening-v1"]
    profiles: dict[str, ApproachProfiles]
    legacy_analysis_config: str
    hybrid_connection_config: str
    development_split_file: str | None
    planned_validation_records: int = Field(gt=0)
    target_recall: Probability = Field(gt=0)
    r1_policy: Literal["disabled_pending_protocol_review"]
    output_root: str
    allow_writes: bool = False


def load_screening_settings(root, path="configs/screening/v1.yaml"):
    settings = parse_settings(ScreeningSettings, read_yaml(Path(root) / path), path)
    if set(settings.profiles) != {"binary", "hybrid"}:
        raise PrerequisiteError("Screening v1 requires binary and hybrid profiles.")
    for profile in settings.profiles.values():
        if any(set(mapping) != {"jev-9b", "jev-27b"} for mapping in (profile.pilot, profile.full)):
            raise PrerequisiteError("Each screening scope must explicitly configure both models.")
    return settings


def screening_setup_table(root):
    """Validate every v1 profile without reading citation data or making clients."""
    settings = load_screening_settings(root)
    rows = []
    for approach, profile in settings.profiles.items():
        for phase in ("pilot", "full"):
            for model, path in getattr(profile, phase).items():
                project = load_project_config(Path(root), path)
                task = load_task(project)
                if (
                    not valid_batch_task(project, task)
                    or project.evaluation.task.approach != approach
                    or project.evaluation.task.prompt_version != settings.prompt_version
                    or len(project.models) != 1
                    or project.models[0].name != model
                ):
                    raise PrerequisiteError("A screening profile does not match its reviewed task.")
                rows.append(
                    {
                        "approach": approach,
                        "phase": phase,
                        "model": model,
                        "workflow": path,
                        "questions": len(task.questions),
                        "primary_question": project.scoring.question,
                        "paid_calls_enabled": project.workflow.execution.allow_paid_calls,
                        "writes_enabled": project.workflow.output.allow_writes,
                        "task_sha256": task.fingerprint,
                    }
                )
    return pd.DataFrame(rows)


@dataclass
class ScreeningCampaign:
    root: Path
    settings: ScreeningSettings
    approach: str
    phase: str
    plans: tuple

    def summary(self):
        if self.phase == "full":
            return full_run_plan_table(*self.plans)
        return paired_pilot_plan_table(*self.plans)


def prepare_screening_campaign(root, *, approach=None, phase="pilot"):
    root = Path(root).resolve()
    settings = load_screening_settings(root)
    approach = approach or settings.default_approach
    if approach not in settings.profiles or phase not in {"pilot", "full"}:
        raise PrerequisiteError("Select binary/hybrid and pilot/full explicitly.")
    paths = getattr(settings.profiles[approach], phase)
    projects = [load_project_config(root, paths[model]) for model in ("jev-9b", "jev-27b")]
    plans = []
    for project in projects:
        states = (
            prepare_pilot_sample(project, preview_only=True).states
            if phase == "pilot"
            else load_full_dataset(project).states
        )
        plans.append(plan_run(project, states))
    campaign = ScreeningCampaign(root, settings, approach, phase, tuple(plans))
    campaign.summary()
    return campaign


def require_hybrid_execution(plan):
    """Notebook execution guard: never submit a binary or stale single-question plan."""
    if (
        plan.project.evaluation.task.approach != "hybrid"
        or plan.project.scoring.question != "overall_eligibility"
        or not valid_batch_task(plan.project, plan.task)
    ):
        raise PrerequisiteError(
            "Notebook 03 execution is hybrid-only. Reload the hybrid profiles and plans; "
            "binary screening will not be submitted."
        )


def execute_screening_model(
    campaign, model, *, allow_paid_calls=False, allow_writes=False, **kwargs
):
    """Exactly one selected model. Reuse the bounded runner and durable resume policy."""
    selected = [plan for plan in campaign.plans if plan.project.models[0].name == model]
    if len(selected) != 1:
        raise PrerequisiteError("Choose one configured model from this campaign.")
    return run_experiment(
        selected[0], allow_paid_calls=allow_paid_calls, allow_writes=allow_writes, **kwargs
    )


def screening_run_status(campaign):
    """Read archives with original fingerprints; never use this view to resume."""
    rows = []
    for plan in campaign.plans:
        project = plan.project
        path = _run_directory(project, plan.run_id) / "manifest.json"
        result = load_run(project, plan.run_id) if path.is_file() else None
        view = plan
        if result is not None:
            if (
                result.manifest.get("dataset_sha256") != plan.dataset_sha256
                or result.manifest.get("task_sha256") != plan.task.fingerprint
                or result.manifest.get("requested_model_ids")
                != {project.models[0].name: project.models[0].hosted.app_id}
            ):
                raise PrerequisiteError("Saved campaign inputs differ from the selected v1 plan.")
            view = replace(plan, fingerprint=result.manifest["fingerprint"])
        table = (
            validate_full_results(view, result)
            if campaign.phase == "full"
            else pilot_run_status(view, result)
        )
        table["approach"] = campaign.approach
        table["model"] = project.models[0].name
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def hybrid_strategy_views(data):
    """Independent direct/minimum/mean score views; derived scores are not calibrated probabilities."""
    if any(plan.project.evaluation.task.approach != "hybrid" for plan in data.plans):
        raise PrerequisiteError("Hybrid strategy views require archived seven-question runs.")
    views = {"hybrid_direct": data}
    for name, reducer in (("R2_minimum", min), ("R3_mean", lambda x: math.fsum(x) / len(x))):
        rows = data.rows.copy(deep=True)

        def score(probabilities):
            if not all(domain in probabilities for domain in PRIMARY_DOMAINS):
                return None
            return reducer([probabilities[domain] for domain in PRIMARY_DOMAINS])

        rows["probability_true"] = rows.question_probabilities.map(score)
        rows["strategy"] = name
        views[name] = replace(data, rows=rows)
    return views


def screening_question_summary(plans, results):
    """Aggregate raw question scores; not criterion accuracy or inferred human labels."""
    rows = []
    for plan, result in zip(plans, results, strict=True):
        records = result.predictions if result is not None else []
        for question in plan.task.questions:
            values = [
                row["probabilities"][question] for row in records if row.get("status") == "success"
            ]
            rows.append(
                {
                    "model": plan.project.models[0].name,
                    "question_id": question,
                    "scored_studies": len(values),
                    "unavailable": len(plan.states) - len(values),
                    "mean_probability": math.fsum(values) / len(values) if values else None,
                    "minimum_probability": min(values) if values else None,
                    "maximum_probability": max(values) if values else None,
                }
            )
    return pd.DataFrame(rows)


def load_development_split(root, data, settings):
    """Optional original development/tuning membership; never make a new random split."""
    if settings.development_split_file is None:
        return None
    path = (Path(root) / settings.development_split_file).resolve()
    if not path.is_relative_to((Path(root) / "data/private").resolve()):
        raise PrerequisiteError("The original split manifest must remain under data/private/.")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != {"study_id", "dataset_role"}:
                raise ValueError
            records = list(reader)
    except (OSError, ValueError, csv.Error):
        raise PrerequisiteError("Provide an original study_id,dataset_role split CSV.") from None
    seen = set()
    for row in records:
        if (
            None in row
            or row["study_id"] in seen
            or not row["study_id"]
            or row["dataset_role"] not in {"development", "tuning"}
        ):
            raise PrerequisiteError(
                "Original split IDs must be unique with development/tuning roles."
            )
        seen.add(row["study_id"])
    if seen != {state.study_id for state in data.states}:
        raise PrerequisiteError(
            "Resolve the 7,008 versus 7,000 cohort discrepancy before protocol tuning."
        )
    table = pd.DataFrame(records)
    if table.dataset_role.value_counts().to_dict() != {"development": 5749, "tuning": 1251}:
        raise PrerequisiteError("Protocol tuning requires the original 5,749/1,251 assignments.")
    return table


def tune_hybrid_rules(root, data, settings):
    """R1 stays disabled. R2/R3 candidates need the original, reconciled tuning split."""
    split = load_development_split(root, data, settings)
    rows = [
        {
            "model": plan.project.models[0].name,
            "strategy": "R1_conjunction",
            "status": "disabled_pending_protocol_review",
            "threshold": None,
        }
        for plan in data.plans
    ]
    if split is None:
        return pd.DataFrame(
            rows
            + [
                {
                    "model": plan.project.models[0].name,
                    "strategy": strategy,
                    "status": "requires_original_1251_tuning_split",
                    "threshold": None,
                }
                for plan in data.plans
                for strategy in ("R2_minimum", "R3_mean")
            ]
        )
    tuning_ids = set(split.loc[split.dataset_role == "tuning", "study_id"])
    for strategy, view in hybrid_strategy_views(data).items():
        if strategy == "hybrid_direct":
            continue
        subset = replace(view, rows=view.rows[view.rows.study_id.isin(tuning_ids)].copy())
        if subset.rows.human_label.isna().any():
            raise PrerequisiteError("Formal tuning requires every tuning reference label.")
        candidates = select_recall_thresholds(subset, settings.target_recall, basis="all_labelled")
        candidates["strategy"] = strategy
        candidates["scope"] = "Original tuning subset only; final rule selection pending R1 review"
        rows.extend(candidates.to_dict(orient="records"))
    return pd.DataFrame(rows)


@dataclass
class ScreeningAnalysis:
    settings: ScreeningSettings
    availability: pd.DataFrame
    data: dict
    threshold_candidates: pd.DataFrame
    ranking_metrics: pd.DataFrame
    rule_tuning: pd.DataFrame


def load_screening_analysis(root, *, include_legacy=True):
    """Show planned arms honestly and analyse saved v1 runs without clients or resuming."""
    root = Path(root).resolve()
    settings = load_screening_settings(root)
    entries = {name + "_v1": p.analysis_config for name, p in settings.profiles.items()}
    if include_legacy:
        entries = {"legacy_binary": settings.legacy_analysis_config, **entries}
    availability, datasets, candidate_tables, rank_tables, tuning_tables = [], {}, [], [], []
    for label, analysis_path in entries.items():
        config = read_yaml(root / analysis_path)
        projects = [
            load_project_config(root, config[key]) for key in ("workflow_9b", "workflow_27b")
        ]
        present = []
        for project in projects:
            path = _run_directory(project, project.workflow.output.run_id) / "manifest.json"
            exists = path.is_file()
            present.append(exists)
            availability.append(
                {
                    "profile": label,
                    "model": project.models[0].name,
                    "run_id": project.workflow.output.run_id,
                    "status": "saved" if exists else "not_run",
                    "questions": 7 if project.evaluation.task.approach == "hybrid" else 1,
                    "comparison_scope": "legacy wording; not a matched v1 comparison"
                    if label == "legacy_binary"
                    else "matched reviewed v1 wording",
                }
            )
        if not any(present):
            continue
        data = load_saved_full_evaluation(root, analysis_path, allow_missing_runs=True)
        datasets[label] = data
        views = hybrid_strategy_views(data) if label == "hybrid_v1" else {"direct": data}
        for strategy, view in views.items():
            candidates = select_recall_thresholds(
                view, settings.target_recall, basis="all_labelled"
            )
            candidates["profile"], candidates["strategy"] = label, strategy
            candidates["score_semantics"] = (
                "derived criterion score; not a calibrated inclusion probability"
                if strategy in {"R2_minimum", "R3_mean"}
                else "raw overall retention probability"
            )
            candidates["selection_scope"] = (
                "Exploratory pooled development cohort; not formal 1,251 tuning or validation"
            )
            candidate_tables.append(candidates)
            ranks = ranking_readouts(view).table
            ranks["profile"], ranks["strategy"] = label, strategy
            ranks["score_semantics"] = candidates["score_semantics"].iloc[0]
            rank_tables.append(ranks)
        if label == "hybrid_v1":
            tuning_tables.append(tune_hybrid_rules(root, data, settings))

    def combined(tables):
        return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    return ScreeningAnalysis(
        settings,
        pd.DataFrame(availability),
        datasets,
        combined(candidate_tables),
        combined(rank_tables),
        combined(tuning_tables),
    )


def select_primary_analysis(
    root, analysis=None, legacy_path="configs/evaluations/analysis_full.yaml"
):
    """Prefer saved selected v1 results, label legacy fallback, otherwise show planned v1 coverage."""
    root = Path(root).resolve()
    analysis = analysis or load_screening_analysis(root, include_legacy=False)
    name = analysis.settings.default_approach + "_v1"
    selected_path = analysis.settings.profiles[analysis.settings.default_approach].analysis_config
    if name in analysis.data:
        return analysis.data[name], name, selected_path
    legacy_config = read_yaml(root / legacy_path)
    projects = [
        load_project_config(root, legacy_config[key]) for key in ("workflow_9b", "workflow_27b")
    ]
    if all(
        (_run_directory(p, p.workflow.output.run_id) / "manifest.json").is_file() for p in projects
    ):
        return load_saved_full_evaluation(root, legacy_path), "legacy_binary", legacy_path
    return (
        load_saved_full_evaluation(root, selected_path, allow_missing_runs=True),
        "planned_" + name,
        selected_path,
    )


def export_screening_analysis(analysis, *, allow_writes=False):
    """Private v1 analysis bundle with lossless long-format question probabilities."""
    from jev_eval.environment import reproducibility_summary, sha256_file
    from jev_eval.runner import safe_output_path

    if not (allow_writes and analysis.settings.allow_writes):
        raise PrerequisiteError("Screening exports need YAML and cell write permission.")
    if not analysis.data:
        raise PrerequisiteError("No saved v1 results are available to export.")
    data = next(iter(analysis.data.values()))
    root = data.plans[0].project.root
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = safe_output_path(
        data.plans[0].project, Path(analysis.settings.output_root) / "exports" / tag
    )
    folder.mkdir(parents=True, exist_ok=False)
    for name in ("availability", "threshold_candidates", "ranking_metrics", "rule_tuning"):
        getattr(analysis, name).to_csv(folder / f"{name}.csv", index=False)
    records, sources = [], {}
    for profile, view in analysis.data.items():
        for row in view.rows.itertuples():
            plan = next(p for p in view.plans if p.project.models[0].name == row.model_name)
            for question in plan.task.questions:
                records.append(
                    {
                        "profile": profile,
                        "study_id": row.study_id,
                        "model": row.model_name,
                        "run_id": row.run_id,
                        "question_id": question,
                        "noul_score": row.question_probabilities.get(question),
                        "inference_status": row.inference_status,
                        "task_id": row.task_id,
                    }
                )
        sources[profile] = {
            "states_sha256": view.plans[0].dataset_sha256,
            "reference_sha256": view.references.sha256,
            "task_sha256": view.plans[0].task.fingerprint,
            "runs": [
                {
                    "run_id": p.run_id,
                    "fingerprint": r.manifest["fingerprint"] if r else None,
                    "predictions_sha256": sha256_file(
                        _run_directory(p.project, p.run_id) / "predictions.jsonl"
                    )
                    if r
                    else None,
                }
                for p, r in zip(view.plans, view.results, strict=True)
            ],
        }
    pd.DataFrame(records).to_csv(folder / "question_scores.csv", index=False)
    import json

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": analysis.settings.model_dump(),
        "screening_config_sha256": sha256_file(root / "configs/screening/v1.yaml"),
        "analysis_code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("screening.py", "evaluation.py", "reporting.py", "tasks.py", "config.py")
        },
        "analysis_environment": reproducibility_summary(data.plans[0].project),
        "sources": sources,
        "scope": "exploratory development; R1 disabled; validation not executed",
        "output_sha256": {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return folder
