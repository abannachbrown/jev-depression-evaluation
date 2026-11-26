"""Binary screening evaluation with explicit coverage and missingness."""

import csv
import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from jev_eval.config import PrerequisiteError
from jev_eval.data import load_jsonl
from jev_eval.tasks import primary_question, valid_batch_task


@dataclass
class ReferenceCSV:
    table: pd.DataFrame
    path: Path
    sha256: str
    counts: dict

    def summary(self):
        return pd.DataFrame([{**self.counts, "reference_sha256": self.sha256}])


def load_reference_csv(project, states):
    """Read original string IDs and nullable reference labels, never model inputs."""
    path = (project.root / project.workflow.evaluation.reference_labels).resolve()
    mapping = project.workflow.inference_sh_full.label_mapping
    if set(mapping) != {"0", "1"} or set(mapping.values()) != {False, True}:
        raise PrerequisiteError("Reference label_mapping must explicitly map 0/1 to false/true.")
    try:
        raw = path.read_bytes()
        import io

        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if not {"study_id", "Inclusion Status", "ITEM_ID"} <= set(reader.fieldnames or []):
            raise PrerequisiteError(
                "Reference CSV requires study_id, Inclusion Status and ITEM_ID."
            )
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        raise PrerequisiteError("Cannot read the prepared reference CSV.") from None
    expected = {state.study_id for state in states}
    seen, normalized = set(), []
    for row in rows:
        identifier = row.get("study_id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in seen
            or identifier not in expected
            or None in row
        ):
            raise PrerequisiteError(
                "Reference CSV contains missing, duplicate or unexpected IDs/fields."
            )
        seen.add(identifier)
        label = (row.get("Inclusion Status") or "").strip()
        if label and label not in mapping:
            raise PrerequisiteError("Reference CSV contains an unmapped inclusion status.")
        item = row.get("ITEM_ID")
        normalized.append(
            {
                "study_id": identifier,
                "ITEM_ID": item if item and item.strip() else None,
                "human_inclusion_status": label or None,
                "human_label": mapping[label] if label else None,
            }
        )
    table = pd.DataFrame(
        normalized, columns=["study_id", "ITEM_ID", "human_inclusion_status", "human_label"]
    )
    table["human_label"] = table["human_label"].astype("boolean")
    item_counts = table["ITEM_ID"].dropna().value_counts()
    return ReferenceCSV(
        table,
        path,
        hashlib.sha256(raw).hexdigest(),
        {
            "source_studies": len(expected),
            "reference_rows": len(table),
            "missing_reference_ids": len(expected - seen),
            "missing_human_labels": int(table.human_label.isna().sum()) + len(expected - seen),
            "missing_item_ids_in_reference": int(table.ITEM_ID.isna().sum()),
            "shared_item_id_groups": int((item_counts > 1).sum()),
        },
    )


def evaluation_ready_table(plan_9b, result_9b, plan_27b, result_27b, references):
    """Build exactly one row per study/model; preserve all failures and missingness."""
    from jev_eval.reporting import full_run_plan_table, validate_full_results

    full_run_plan_table(plan_9b, plan_27b)
    plans, results = (plan_9b, plan_27b), (result_9b, result_27b)
    for plan, result in zip(plans, results, strict=True):
        check = validate_full_results(plan, result).iloc[0]
        if any(
            check[name]
            for name in [
                "duplicate_ids",
                "unexpected_rows",
                "invalid_status",
                "invalid_probabilities",
                "missing_task_ids",
                "missing_model_metadata",
                "unexpected_returned_model",
                "incompatible_provenance",
            ]
        ):
            raise PrerequisiteError(
                "Inference integrity checks failed; inspect results before export."
            )
    lookups = [
        {row["study_id"]: row for row in result.predictions} if result else {} for result in results
    ]
    question = primary_question(plan_9b.project, plan_9b.task)
    rows = []
    for state in plan_9b.states:
        for plan, result, lookup in zip(plans, results, lookups, strict=True):
            record = lookup.get(state.study_id, {})
            pending = result.manifest.get("pending_request") if result is not None else None
            if (
                not record
                and isinstance(pending, dict)
                and pending.get("study_id") == state.study_id
            ):
                record = {
                    "status": "unknown",
                    "request_id": pending.get("request_id"),
                    "attempts": pending.get("attempt", 1),
                    "started_at": pending.get("started_at"),
                    "error_code": pending.get("error_code", "interrupted_outcome_unknown"),
                }
            rows.append(
                {
                    "study_id": state.study_id,
                    "model_name": plan.project.models[0].name,
                    "requested_app": plan.project.models[0].hosted.app_id,
                    "returned_model": record.get("returned_model"),
                    "model_revision": record.get("model_revision"),
                    "revision_verified": False,
                    "question_id": question,
                    "approach": plan.project.evaluation.task.approach,
                    "prompt_version": plan.project.evaluation.task.prompt_version,
                    "question_probabilities": record.get("probabilities", {})
                    if record.get("status") == "success"
                    else {},
                    "probability_true": record.get("probabilities", {}).get(question)
                    if record.get("status") == "success"
                    else None,
                    "inference_status": record.get("status", "not_recorded"),
                    "task_id": record.get("request_id"),
                    "attempts": record.get("attempts", 0),
                    "error_code": record.get("error_code"),
                    "experiment_id": plan.project.workflow.inference_sh_full.experiment_id,
                    "run_id": plan.run_id,
                    "started_at": record.get("started_at"),
                    "ended_at": record.get("ended_at"),
                    "input_tokens": record.get("usage", {}).get("input_tokens"),
                }
            )
    table = pd.DataFrame(rows)
    if not set(references.table.study_id) <= {state.study_id for state in plan_9b.states}:
        raise PrerequisiteError("Reference IDs do not belong to the full cohort.")
    if references.table.study_id.duplicated().any():
        raise PrerequisiteError("Reference IDs must be unique before joining.")
    merged = table.merge(
        references.table, on="study_id", how="left", validate="many_to_one", sort=False
    )
    if (
        len(merged) != 2 * len(plan_9b.states)
        or merged.duplicated(["study_id", "model_name"]).any()
    ):
        raise PrerequisiteError("Reference join duplicated or lost expected study/model rows.")
    return merged


def export_evaluation_ready(
    plan_9b, result_9b, plan_27b, result_27b, references, *, allow_writes=False
):
    """Export a private, uniquely named bundle with an explicit completeness audit."""
    from jev_eval.reporting import validate_full_results
    from jev_eval.runner import require_writes, safe_output_path

    require_writes(plan_9b.project, allow_writes)
    require_writes(plan_27b.project, allow_writes)
    if hashlib.sha256(references.path.read_bytes()).hexdigest() != references.sha256:
        raise PrerequisiteError("Reference CSV changed after it was loaded.")
    table = evaluation_ready_table(plan_9b, result_9b, plan_27b, result_27b, references)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = safe_output_path(
        plan_9b.project, Path(plan_9b.project.workflow.output.root) / "exports" / tag
    )
    folder.mkdir(parents=True, exist_ok=False)
    target = folder / "evaluation_ready.csv"
    table.to_csv(target, index=False)
    checks = [
        json.loads(validate_full_results(plan, result).to_json(orient="records"))[0]
        for plan, result in [(plan_9b, result_9b), (plan_27b, result_27b)]
    ]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(table),
        "expected_rows": len(plan_9b.states) * 2,
        "reference_sha256": references.sha256,
        "states_sha256": plan_9b.dataset_sha256,
        "task_sha256": plan_9b.task.fingerprint,
        "csv_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "validation": checks,
        "missing_label_rows": int(table.human_label.isna().sum()),
        "missing_probability_rows": int(table.probability_true.isna().sum()),
        "complete_probability_coverage": bool(table.probability_true.notna().all()),
        "label_mapping": plan_9b.project.workflow.inference_sh_full.label_mapping,
        "decision_threshold": None,
    }
    from jev_eval.runner import _run_directory

    manifest["prediction_sources"] = []
    for plan in (plan_9b, plan_27b):
        source = _run_directory(plan.project, plan.run_id) / "predictions.jsonl"
        manifest["prediction_sources"].append(
            {
                "model": plan.project.models[0].name,
                "run_id": plan.run_id,
                "file": str(source.relative_to(plan.project.root)),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                if source.is_file()
                else None,
            }
        )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return folder


def load_reference_labels(path, *, states, question):
    rows = load_jsonl(path)
    expected_ids = {state.study_id for state in states}
    labels = {}
    for row in rows:
        identifier = row.get("study_id")
        value = row.get("labels", {}).get(question) if isinstance(row.get("labels"), dict) else None
        if (
            not isinstance(identifier, str)
            or identifier in labels
            or identifier not in expected_ids
        ):
            raise PrerequisiteError(
                "Reference labels contain duplicate, invalid, or unknown study IDs."
            )
        if type(value) is not bool:
            raise PrerequisiteError(
                "Reference labels must contain an explicit JSON true or false for the task."
            )
        labels[identifier] = value
    if not labels:
        raise PrerequisiteError(
            "No reference labels are available; performance cannot be calculated."
        )
    return labels


@dataclass
class SavedFullEvaluation:
    settings: object
    states: list
    references: ReferenceCSV
    reference_labels: dict[str, bool]
    plans: tuple
    results: tuple
    rows: pd.DataFrame
    coverage: pd.DataFrame


def load_saved_full_evaluation(
    root, settings_path="configs/evaluations/analysis_full.yaml", *, allow_missing_runs=False
):
    """Read archives without creating a backend, sending requests or resuming.

    Original inference fingerprints and code/environment identities are retained.
    Analysis-code changes do not require repeating inference. Scientific inputs,
    requested models, frozen cohort, task and saved output integrity are checked
    independently before constructing read-only views for existing validators.
    With allow_missing_runs, absent arms receive prospective plans labelled
    not_run; their current fingerprints are never presented as historical runs.
    """
    from jev_eval.config import FullAnalysisSettings, load_project_config, parse_settings, read_yaml
    from jev_eval.data import load_full_dataset
    from jev_eval.reporting import validate_full_results
    from jev_eval.runner import RunPlan, _run_directory, load_run, plan_run
    from jev_eval.tasks import TaskDefinition, load_task

    root = Path(root).resolve()
    settings = parse_settings(FullAnalysisSettings, read_yaml(root / settings_path), settings_path)
    projects = tuple(
        load_project_config(root, workflow)
        for workflow in (settings.workflow_9b, settings.workflow_27b)
    )
    cohorts = tuple(load_full_dataset(project) for project in projects)
    if cohorts[0].manifest != cohorts[1].manifest:
        raise PrerequisiteError(
            "The two archived model evaluations must use the same complete cohort."
        )
    plans, results, coverage = [], [], []
    for project, cohort in zip(projects, cohorts, strict=True):
        if len(project.models) != 1 or project.workflow.output.run_id is None:
            raise PrerequisiteError(
                "Analysis requires one saved full run per model with an explicit ID."
            )
        if (
            allow_missing_runs
            and not (
                _run_directory(project, project.workflow.output.run_id) / "manifest.json"
            ).is_file()
        ):
            plans.append(plan_run(project, cohort.states, load_task(project)))
            results.append(None)
            continue
        result = load_run(project, project.workflow.output.run_id)
        manifest = result.manifest
        model = project.models[0]
        stored = manifest.get("full_cohort", {})
        if (
            manifest.get("run_kind") != "full"
            or manifest.get("mode") != "hosted"
            or manifest.get("dataset_sha256") != cohort.manifest["states_sha256"]
            or manifest.get("task_sha256") != cohort.manifest["task_sha256"]
            or manifest.get("model_names") != [model.name]
            or manifest.get("requested_model_ids") != {model.name: model.hosted.app_id}
            or stored.get("source_file_sha256") != cohort.manifest["source_file_sha256"]
            or not isinstance(manifest.get("fingerprint"), str)
        ):
            raise PrerequisiteError(
                "Saved inference inputs/model identity do not match the selected cohort."
            )
        cohort_path = root / project.workflow.inference_sh_full.cohort_manifest
        if not cohort_path.is_file() or hashlib.sha256(
            cohort_path.read_bytes()
        ).hexdigest() != manifest.get("cohort_manifest_sha256"):
            raise PrerequisiteError("Saved cohort manifest is missing or its checksum changed.")
        try:
            task = TaskDefinition.model_validate(manifest["screening_questions"])
        except (KeyError, ValueError):
            raise PrerequisiteError("Saved screening question metadata is invalid.") from None
        if task.fingerprint != cohort.manifest["task_sha256"]:
            raise PrerequisiteError("Archived screening instructions differ from the frozen task.")
        if (
            not valid_batch_task(project, task)
            or primary_question(project, task) not in task.questions
        ):
            raise PrerequisiteError(
                "The scoring question must match the archived screening question."
            )
        # An archive view preserves the original run's fingerprint; run_experiment
        # still independently rebuilds its live fingerprint and retains its guards.
        plan = RunPlan(
            project,
            cohort.states,
            task,
            manifest["run_id"],
            manifest["fingerprint"],
            manifest["dataset_sha256"],
            manifest["environment"],
            manifest["source_sha256"],
            cohort.manifest,
            manifest.get("configuration_sha256", {}),
        )
        check = validate_full_results(plan, result).iloc[0]
        if any(
            check[name]
            for name in [
                "duplicate_ids",
                "unexpected_rows",
                "invalid_status",
                "invalid_probabilities",
                "missing_task_ids",
                "missing_model_metadata",
                "unexpected_returned_model",
                "incompatible_provenance",
            ]
        ):
            raise PrerequisiteError(
                "Archived prediction integrity checks failed; inspect saved outputs."
            )
        plans.append(plan)
        results.append(result)
    refs = load_reference_csv(projects[0], cohorts[0].states)
    rows = evaluation_ready_table(plans[0], results[0], plans[1], results[1], refs)
    labels = {
        row.study_id: bool(row.human_label)
        for row in refs.table.itertuples()
        if not pd.isna(row.human_label)
    }
    if not labels:
        raise PrerequisiteError("No valid human reference labels are available for evaluation.")
    for project in projects:
        model_rows = rows[rows.model_name == project.models[0].name]
        paired = model_rows.human_label.notna() & model_rows.probability_true.notna()
        coverage.append(
            {
                "model": project.models[0].name,
                "mode": "recorded",
                "source_studies": len(cohorts[0].states),
                "labelled": len(labels),
                "missing_human_labels": len(cohorts[0].states) - len(labels),
                "valid_probabilities": int(paired.sum()),
                "not_inferred_no_text": int(
                    (model_rows.inference_status == "not_inferred_no_text").sum()
                ),
                "failed": int((model_rows.inference_status == "failed").sum()),
                "unknown": int((model_rows.inference_status == "unknown").sum()),
                "missing": int((model_rows.inference_status == "not_recorded").sum()),
                "coverage": int(paired.sum()) / len(labels),
            }
        )
    return SavedFullEvaluation(
        settings,
        cohorts[0].states,
        refs,
        labels,
        tuple(plans),
        tuple(results),
        rows,
        pd.DataFrame(coverage),
    )


def evaluate_saved_probabilities(data):
    """Apply an explicit threshold to valid saved probabilities without filling gaps."""
    threshold = data.settings.decision_threshold
    if threshold is None:
        raise PrerequisiteError(
            "Choose and document decision_threshold before computing binary metrics."
        )
    question = primary_question(data.plans[0].project, data.plans[0].task)
    predictions = []
    for row in data.rows.itertuples():
        predictions.append(
            {
                "study_id": row.study_id,
                "model": row.model_name,
                "mode": "hosted",
                "status": row.inference_status,
                "predictions": {question: bool(row.probability_true >= threshold)}
                if row.inference_status == "success" and not pd.isna(row.probability_true)
                else {},
            }
        )
    scoring = data.plans[0].project.scoring.model_copy(
        update={
            "require_complete_predictions_for_headline_metrics": not data.settings.report_partial_metrics
        }
    )
    result = evaluate_predictions(
        predictions,
        data.reference_labels,
        scoring,
        models=[p.project.models[0].name for p in data.plans],
        states=data.states,
    )
    for field in ("failed", "unknown", "not_inferred_no_text", "missing_human_labels"):
        result.table[field] = data.coverage[field].to_numpy()
    result.table["decision_threshold"] = threshold
    result.table["threshold_note"] = data.settings.threshold_note
    result.table["metric_scope"] = (
        "valid labelled pairs only; inspect coverage"
        if data.settings.report_partial_metrics
        else "headline metrics require complete labelled coverage"
    )
    return result


def export_saved_evaluation(data, evaluation, *, allow_writes=False):
    """Write a private analysis bundle, preserving inference and analysis identities."""
    from jev_eval.environment import reproducibility_summary, sha256_file
    from jev_eval.reporting import plot_confusion_matrices
    from jev_eval.runner import _run_directory, safe_output_path

    if not (allow_writes and data.settings.allow_writes):
        raise PrerequisiteError(
            "Evaluation exports require permission in analysis YAML and the cell."
        )
    if hashlib.sha256(data.references.path.read_bytes()).hexdigest() != data.references.sha256:
        raise PrerequisiteError("Reference labels changed after loading.")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = safe_output_path(data.plans[0].project, Path(data.settings.output_root) / tag)
    folder.mkdir(parents=True, exist_ok=False)
    data.coverage.to_csv(folder / "coverage.csv", index=False)
    if evaluation is not None:
        evaluation.table.to_csv(folder / "summary.csv", index=False)
        figure = plot_confusion_matrices(evaluation)
        figure.savefig(folder / "confusion-matrices.png", dpi=200, bbox_inches="tight")
        figure.savefig(folder / "confusion-matrices.pdf", bbox_inches="tight")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_settings": data.settings.model_dump(),
        "analysis_environment": reproducibility_summary(data.plans[0].project),
        "analysis_code_sha256": sha256_file(Path(__file__)),
        "reference_sha256": data.references.sha256,
        "dataset_sha256": data.plans[0].dataset_sha256,
        "task_sha256": data.plans[0].task.fingerprint,
        "source_studies": len(data.states),
        "reference_labelled_studies": len(data.reference_labels),
        "inference_runs": [
            {
                "run_id": plan.run_id,
                "fingerprint": result.manifest["fingerprint"],
                "inference_environment": result.manifest["environment"],
                "inference_source_sha256": result.manifest["source_sha256"],
                "predictions_sha256": sha256_file(
                    _run_directory(plan.project, plan.run_id) / "predictions.jsonl"
                ),
            }
            for plan, result in zip(data.plans, data.results, strict=True)
        ],
        "output_sha256": {
            path.name: sha256_file(path) for path in folder.iterdir() if path.is_file()
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return folder


@dataclass
class RankingEvaluation:
    table: pd.DataFrame
    roc: pd.DataFrame
    precision_recall: pd.DataFrame


def ranking_readouts(data):
    """Threshold-independent discrimination on scored, human-labelled pairs.

    Positive means retain for full-text assessment. Unavailable probabilities
    are never filled with zero. Metrics are withheld unless both human classes
    occur among scored pairs. PR AUC uses trapezoids; AP is non-interpolated.
    """
    records, roc_points, pr_points = [], [], []
    for model, group in data.rows.groupby("model_name", sort=False):
        labelled = group.human_label.notna()
        scored = group.probability_true.notna()
        paired = group[labelled & scored]
        truth = paired.human_label.astype(bool)
        positive = int(truth.sum())
        negative = len(paired) - positive
        row = {
            "model": model,
            "source_studies": len(group),
            "human_labelled": int(labelled.sum()),
            "scored_labelled_pairs": len(paired),
            "positive_pairs": positive,
            "negative_pairs": negative,
            "unscored_studies": int((~scored).sum()),
            "unscored_positive_labels": int((group.human_label.eq(True) & ~scored).sum()),
            "missing_human_labels": int((~labelled).sum()),
            "scored_pair_prevalence": _divide(positive, len(paired)),
            "roc_auc": float("nan"),
            "pr_auc_trapezoidal": float("nan"),
            "average_precision": float("nan"),
            "status": "no_scored_labelled_pairs"
            if paired.empty
            else "requires_both_reference_classes",
            "metric_scope": "scored human-labelled pairs only; unavailable predictions excluded",
        }
        if positive and negative:
            scores = paired.probability_true.astype(float)
            fpr, tpr, _ = roc_curve(truth, scores, drop_intermediate=False)
            precision, recall, _ = precision_recall_curve(truth, scores)
            row.update(
                roc_auc=float(roc_auc_score(truth, scores)),
                pr_auc_trapezoidal=float(auc(recall, precision)),
                average_precision=float(average_precision_score(truth, scores)),
                status="available",
            )
            roc_points.extend(
                {"model": model, "false_positive_rate": float(x), "true_positive_rate": float(y)}
                for x, y in zip(fpr, tpr, strict=True)
            )
            pr_points.extend(
                {"model": model, "recall": float(x), "precision": float(y)}
                for x, y in zip(recall, precision, strict=True)
            )
        records.append(row)
    return RankingEvaluation(
        pd.DataFrame(records),
        pd.DataFrame(roc_points, columns=["model", "false_positive_rate", "true_positive_rate"]),
        pd.DataFrame(pr_points, columns=["model", "recall", "precision"]),
    )


def threshold_readouts(data, thresholds):
    """Observed threshold metrics, preserving unscored studies as a separate queue.

    thresholds maps model name to one or more cutoffs. Cohort recall is a lower
    bound with zero credit for unavailable scores, not an imputed prediction.
    The manual-retention scenario is reported separately and is never applied.
    PPV, specificity and F2 use scored labelled pairs. WSS@95 is the excluded
    fraction minus 0.05, reported only when observed recall reaches 0.95 in
    the stated scope. Cohort WSS keeps all unscored studies in the review queue
    and requires complete reference labels; no scores or labels are imputed.
    """
    rows = []
    for model, cutoffs in thresholds.items():
        group = data.rows[data.rows.model_name == model]
        if group.empty:
            raise PrerequisiteError("Threshold readouts require a model present in the saved data.")
        labelled = group.human_label.notna()
        scored = group.probability_true.notna()
        paired = group[labelled & scored]
        truth = paired.human_label.astype(bool)
        all_positive = int(group.human_label.eq(True).sum())
        scored_positive = int(truth.sum())
        missing_positive = int((group.human_label.eq(True) & ~scored).sum())
        for threshold in cutoffs:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise PrerequisiteError("Cutoffs must be finite numeric probabilities in 0–1.")
            inferred = paired.probability_true.ge(threshold)
            tp, fp = int((truth & inferred).sum()), int((~truth & inferred).sum())
            fn, tn = int((truth & ~inferred).sum()), int((~truth & ~inferred).sum())
            retained = int(group.probability_true.ge(threshold).sum())
            excluded = int((scored & group.probability_true.lt(threshold)).sum())
            unscored = int((~scored).sum())
            recall_scored = _divide(tp, scored_positive)
            recall_cohort = _divide(tp, all_positive)
            rows.append(
                {
                    "model": model,
                    "threshold": float(threshold),
                    "scored_pairs": len(paired),
                    "positive_labels_all": all_positive,
                    "positive_labels_scored": scored_positive,
                    "unscored_studies": unscored,
                    "unscored_positive_labels": missing_positive,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "recall_scored": recall_scored,
                    "recall_cohort_lower_bound": recall_cohort,
                    "recall_if_unscored_retained": _divide(tp + missing_positive, all_positive),
                    "precision_scored": _divide(tp, tp + fp),
                    "ppv_scored": _divide(tp, tp + fp),
                    "specificity_scored": _divide(tn, tn + fp),
                    "f2_scored": _divide(5 * tp, 5 * tp + 4 * fn + fp),
                    "wss95_scored": _divide(tn + fn, len(paired)) - 0.05
                    if recall_scored >= 0.95
                    else float("nan"),
                    "wss95_cohort_conservative": _divide(excluded, len(group)) - 0.05
                    if labelled.all() and recall_cohort >= 0.95
                    else float("nan"),
                    "retained_scored": retained,
                    "review_queue_with_unscored": retained + unscored,
                    "automatic_exclusions": excluded,
                    "automatic_exclusion_fraction": _divide(excluded, len(group)),
                    "selection_scope": "exploratory; cutoff and performance use the same labels",
                }
            )
    return pd.DataFrame(rows)


def select_recall_thresholds(data, target_recall=0.95, *, basis="all_labelled"):
    """Highest observed positive-score cutoff meeting a declared recall target.

    Selection is exact (no interpolated quantiles), includes equality/ties, and
    reports unattainable targets instead of excluding unavailable positives.
    """
    if (
        isinstance(target_recall, bool)
        or not isinstance(target_recall, (int, float))
        or not math.isfinite(target_recall)
        or not 0 < target_recall <= 1
    ):
        raise PrerequisiteError("target_recall must be a finite number above zero and at most one.")
    if basis not in {"all_labelled", "scored"}:
        raise PrerequisiteError("Recall basis must be all_labelled or scored.")
    rows = []
    for model, group in data.rows.groupby("model_name", sort=False):
        positives = group[group.human_label.eq(True)]
        scores = positives.probability_true.dropna().sort_values(ascending=False)
        denominator = len(positives) if basis == "all_labelled" else len(scores)
        required = int(
            (Decimal(str(target_recall)) * denominator).to_integral_value(rounding=ROUND_CEILING)
        )
        if denominator == 0 or required > len(scores):
            rows.append(
                {
                    "model": model,
                    "threshold": None,
                    "target_recall": target_recall,
                    "basis": basis,
                    "status": "no_positive_labels"
                    if not denominator
                    else "unattainable_without_unscored_recovery",
                    "required_true_positives": required,
                    "positive_labels_all": len(positives),
                    "positive_labels_scored": len(scores),
                }
            )
            continue
        threshold = float(scores.iloc[required - 1])
        row = threshold_readouts(data, {model: [threshold]}).iloc[0].to_dict()
        rows.append(
            {
                **row,
                "target_recall": target_recall,
                "basis": basis,
                "status": "attainable",
                "required_true_positives": required,
            }
        )
    return pd.DataFrame(rows)


def threshold_tradeoff_curve(data):
    """Exact observed score breakpoints, using cumulative counts rather than an N² scan."""
    rows = []
    for model, group in data.rows.groupby("model_name", sort=False):
        paired = group[group.human_label.notna() & group.probability_true.notna()].copy()
        paired["positive"] = paired.human_label.astype(int)
        bins = paired.groupby("probability_true").positive.agg(["sum", "size"])
        score_values = bins.index.tolist()
        cumulative_positive = bins["sum"].cumsum().tolist()
        cumulative_negative = (bins["size"] - bins["sum"]).cumsum().tolist()
        positive = int(paired.positive.sum())
        negative = len(paired) - positive
        all_positive = int(group.human_label.eq(True).sum())
        all_scores = group.probability_true.dropna().sort_values()
        unscored = int(group.probability_true.isna().sum())
        for threshold in sorted({0.0, 1.0, *score_values}):
            index = bisect_left(score_values, threshold)
            fn = int(cumulative_positive[index - 1]) if index else 0
            tn = int(cumulative_negative[index - 1]) if index else 0
            tp, fp = positive - fn, negative - tn
            excluded = int(all_scores.searchsorted(threshold, side="left"))
            rows.append(
                {
                    "model": model,
                    "threshold": threshold,
                    "recall_scored": _divide(tp, positive),
                    "recall_cohort_lower_bound": _divide(tp, all_positive),
                    "precision_scored": _divide(tp, tp + fp),
                    "specificity_scored": _divide(tn, negative),
                    "automatic_exclusion_fraction": _divide(excluded, len(group)),
                    "review_queue_with_unscored": len(all_scores) - excluded + unscored,
                }
            )
    return pd.DataFrame(rows)


@dataclass
class EvaluationResult:
    table: pd.DataFrame
    matrices: dict
    demo: bool


def _divide(numerator, denominator):
    return numerator / denominator if denominator else float("nan")


def evaluate_predictions(
    predictions, reference_labels, scoring, *, models, states
) -> EvaluationResult:
    if not reference_labels:
        raise PrerequisiteError("Reference labels are required.")
    expected_ids = {state.study_id for state in states}
    if not set(reference_labels) <= expected_ids:
        raise PrerequisiteError("Reference labels and dataset IDs do not match.")
    if any(type(value) is not bool for value in reference_labels.values()):
        raise PrerequisiteError("Reference labels must be booleans, not strings or integers.")
    seen, lookup = set(), {}
    for row in predictions:
        identifier, model = row.get("study_id"), row.get("model")
        if not isinstance(identifier, str) or not isinstance(model, str):
            raise PrerequisiteError("Every prediction needs string study_id and model fields.")
        key = identifier, model
        if key in seen or model not in models or identifier not in expected_ids:
            raise PrerequisiteError(
                "Predictions have duplicate or unexpected study/model identifiers."
            )
        seen.add(key)
        lookup[key] = row
    demo = bool(predictions) and all(row.get("mode") == "demo" for row in predictions)
    if any(row.get("mode") == "demo" for row in predictions) and not demo:
        raise PrerequisiteError("Do not mix demonstration and real predictions in one comparison.")
    table, matrices = [], {}
    for model in models:
        truth, inferred = [], []
        missing, failed, invalid = 0, 0, 0
        for identifier, label in reference_labels.items():
            record = lookup.get((identifier, model))
            if record is None:
                missing += 1
            elif record.get("status") != "success":
                failed += 1
            else:
                response = record.get("predictions")
                value = response.get(scoring.question) if isinstance(response, dict) else None
                if type(value) is not bool:
                    invalid += 1
                else:
                    truth.append(label)
                    inferred.append(value)
        matrix = (
            confusion_matrix(truth, inferred, labels=[False, True])
            if truth
            else pd.DataFrame(0, index=[0, 1], columns=[0, 1]).to_numpy()
        )
        matrices[model] = matrix
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        sensitivity, specificity = _divide(tp, tp + fn), _divide(tn, tn + fp)
        metrics = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": _divide(tp, tp + fp),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": (sensitivity + specificity) / 2,
        }
        complete = len(truth) == len(reference_labels)
        if not complete and scoring.require_complete_predictions_for_headline_metrics:
            metrics = {name: float("nan") for name in metrics}
        table.append(
            {
                "model": model,
                "mode": "demo" if demo else "recorded",
                "labelled": len(reference_labels),
                "valid": len(truth),
                "missing": missing,
                "failed": failed,
                "invalid": invalid,
                "unlabelled_predictions": sum(
                    (identifier, model) in lookup
                    for identifier in expected_ids - set(reference_labels)
                ),
                "coverage": len(truth) / len(reference_labels),
                "analysis": "complete" if complete else "partial; inspect coverage",
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                **{name: metrics[name] for name in scoring.metrics},
            }
        )
    return EvaluationResult(pd.DataFrame(table), matrices, demo)
