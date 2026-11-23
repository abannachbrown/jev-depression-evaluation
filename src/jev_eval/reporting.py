"""Researcher-facing tables, figures, and explicitly permitted exports."""

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from jev_eval.config import PrerequisiteError
from jev_eval.data import dataset_fingerprint
from jev_eval.runner import require_writes, safe_output_path


def paired_pilot_plan_table(plan_9b, plan_27b):
    """Check the shared ordered sample/task and present both explicit plans."""
    plans = (plan_9b, plan_27b)
    for plan, app in zip(plans, ("autotrust/jev-9b", "autotrust/jev-27b-v1-1"), strict=True):
        project = plan.project
        if (
            project.workflow.execution.mode != "hosted"
            or getattr(project.backend, "protocol", None) != "inference_sh"
            or len(project.models) != 1
            or project.models[0].hosted.app_id != app
            or len(plan.states) != 10
            or len(plan.task.questions) != 1
            or next(iter(plan.task.questions.values())).type != "noul"
            or dataset_fingerprint(plan.states) != plan.dataset_sha256
        ):
            raise PrerequisiteError(
                "Use the JEV-9B and JEV-27B hosted ten-state pilot plans in order."
            )
    if (
        plan_9b.project.root != plan_27b.project.root
        or plan_9b.dataset_sha256 != plan_27b.dataset_sha256
        or plan_9b.task.fingerprint != plan_27b.task.fingerprint
        or plan_9b.project.evaluation.seed != plan_27b.project.evaluation.seed
    ):
        raise PrerequisiteError("Both model plans must use the same ordered sample, seed and task.")
    table = pd.concat([plan.summary() for plan in plans], ignore_index=True)
    if table["output_directory"].nunique() != 2:
        raise PrerequisiteError("The two models need separate output directories.")
    return table


def _pilot_rows(plan, result):
    if result is None:
        return {}
    if (
        result.manifest.get("fingerprint") != plan.fingerprint
        or result.manifest.get("dataset_sha256") != plan.dataset_sha256
        or result.manifest.get("task_sha256") != plan.task.fingerprint
        or result.manifest.get("model_names") != [plan.project.models[0].name]
        or result.manifest.get("mode") != "hosted"
    ):
        raise PrerequisiteError("Results do not match this model's planned inputs and provenance.")
    expected = {state.study_id for state in plan.states}
    question = next(iter(plan.task.questions))
    rows = {}
    for row in result.predictions:
        identifier = row.get("study_id")
        if (
            identifier not in expected
            or identifier in rows
            or row.get("model") != plan.project.models[0].name
            or row.get("status") not in {"success", "failed", "unknown"}
        ):
            raise PrerequisiteError("Pilot results contain duplicate, unexpected or invalid rows.")
        if row["status"] == "success":
            probability = row.get("probabilities", {}).get(question)
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (float, int))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise PrerequisiteError("Successful pilot rows require valid probabilities.")
        rows[identifier] = row
    return rows


def pilot_run_status(plan, result):
    """Aggregate observed coverage without displaying citation text or raw bodies."""
    rows = _pilot_rows(plan, result)
    return pd.DataFrame(
        [
            {
                "requested_app": plan.project.models[0].hosted.app_id,
                "status": result.manifest["status"]
                if result is not None
                else "not executed in this kernel",
                "planned_studies": len(plan.states),
                "successful": sum(row["status"] == "success" for row in rows.values()),
                "failed": sum(row["status"] == "failed" for row in rows.values()),
                "unknown": sum(row["status"] == "unknown" for row in rows.values()),
                "unrecorded": len(plan.states) - len(rows),
                "recorded_attempts": result.manifest["total_attempts"]
                if result is not None
                else None,
                "input_tokens_reported": sum(
                    row.get("usage", {}).get("input_tokens", 0) for row in rows.values()
                )
                if result is not None
                else None,
                "pending_review": bool(result.manifest.get("pending_request"))
                if result is not None
                else False,
                "error_codes": sorted(
                    {row["error_code"] for row in rows.values() if "error_code" in row}
                ),
                "saved_directory": str(result.directory) if result is not None else None,
            }
        ]
    )


def paired_pilot_probability_table(plan_9b, result_9b, plan_27b, result_27b):
    """Join by study ID in frozen order; retain failed/unknown/missing responses."""
    paired_pilot_plan_table(plan_9b, plan_27b)
    first, second = _pilot_rows(plan_9b, result_9b), _pilot_rows(plan_27b, result_27b)
    question = next(iter(plan_9b.task.questions))
    rows = []
    for state in plan_9b.states:
        row = {"study_id": state.study_id}
        for name, lookup in (("jev_9b", first), ("jev_27b", second)):
            record = lookup.get(state.study_id)
            row[f"{name}_status"] = record["status"] if record is not None else "not_recorded"
            row[f"{name}_p_true"] = (
                record["probabilities"][question]
                if (record is not None and record["status"] == "success")
                else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class PilotMethodsSummary:
    text: str
    provenance: pd.DataFrame


def pilot_methods_summary(pilot, plan_9b, plan_27b, result_9b=None, result_27b=None):
    """Describe the frozen protocol; claim completion only for two complete runs."""
    paired_pilot_plan_table(plan_9b, plan_27b)
    if (
        pilot.manifest["sample_sha256"] != plan_9b.dataset_sha256
        or dataset_fingerprint(pilot.states) != plan_9b.dataset_sha256
        or pilot.manifest["task_sha256"] != plan_9b.task.fingerprint
        or pilot.manifest["seed"] != plan_9b.project.evaluation.seed
    ):
        raise PrerequisiteError("The methods summary must use the plans' frozen sample and task.")
    plans, results = (plan_9b, plan_27b), (result_9b, result_27b)
    statuses = [
        pilot_run_status(plan, result).iloc[0] for plan, result in zip(plans, results, strict=True)
    ]
    complete = all(
        row["status"] == "completed" and row["successful"] == 10 and not row["pending_review"]
        for row in statuses
    )
    counts = pilot.manifest["counts"]
    interval = min(
        plan.project.workflow.inference_sh_pilot.min_request_interval_seconds for plan in plans
    )
    question = next(iter(plan_9b.task.questions))
    text = (
        (
            "Completed pilot methods. "
            if complete
            else "Planned pilot protocol — execution incomplete. "
        )
        + f"Ten studies were sampled without replacement from {counts['available_for_inference']} "
        f"records containing a title or abstract among {counts['source_records']} prepared studies. "
        f"Sampling used Python random.Random with seed {pilot.manifest['seed']} and preserved "
        "the source JSONL order as the sampling frame. Records without any citation text were "
        "reported as unavailable for inference and retained in the source dataset. Title-only "
        "records remained eligible for sampling.\n\n"
        + (
            "Each selected study was evaluated by "
            if complete
            else "The protocol specifies evaluating each selected study with "
        )
        + "autotrust/jev-9b and autotrust/jev-27b-v1-1 through the Inference.sh REST API. "
        f"Both models share the identical ordered sample and eligibility question '{question}'. "
        "Model inputs contain the original study ID, title and abstract; human inclusion "
        "decisions and ITEM_ID are withheld. The protocol uses one state and one yes/no "
        f"probability question per request, sequential execution, at least {interval:g} seconds "
        "between request starts within each model run, at most ten attempts per model, and "
        "no automatic retries. Temperature, generation-token limits and an inference seed "
        "are not sent to this decision API.\n\n"
        "Exact HTTP responses, validated probabilities, requested/returned model identifiers, "
        "task IDs, available input-token usage, UTC timestamps and errors are retained with "
        "source, task, code and environment provenance. Unknown outcomes block automatic "
        "resume, and saved terminal records are not automatically submitted again. The pilot "
        "reports probabilities and coverage; binary performance metrics require a separately "
        "specified threshold and reference-label analysis.\n\n"
        "Observed execution in the supplied results:\n"
        + "\n".join(
            f"- {row['requested_app']}: {row['status']}; {row['successful']}/10 successful, "
            f"{row['failed']} failed, {row['unknown']} unknown, {row['unrecorded']} unrecorded."
            for row in statuses
        )
    )
    provenance = [
        {"item": "Prepared JSONL SHA-256", "value": pilot.manifest["source_file_sha256"]},
        {"item": "Ordered sample SHA-256", "value": plan_9b.dataset_sha256},
        {"item": "Eligibility task SHA-256", "value": plan_9b.task.fingerprint},
        {"item": "Sampling seed", "value": pilot.manifest["seed"]},
        {"item": "Question", "value": question},
    ]
    for plan, result in zip(plans, results, strict=True):
        model = plan.project.models[0]
        provenance.extend(
            [
                {"item": f"{model.name}: planned run ID", "value": plan.run_id},
                {
                    "item": f"{model.name}: lockfile SHA-256",
                    "value": plan.environment["lockfile_sha256"],
                },
                {
                    "item": f"{model.name}: Git commit",
                    "value": plan.environment["git"]["commit"] or "uncommitted",
                },
                {
                    "item": f"{model.name}: returned model(s)",
                    "value": sorted(
                        {
                            row["returned_model"]
                            for row in _pilot_rows(plan, result).values()
                            if row.get("returned_model")
                        }
                    ),
                },
                {
                    "item": f"{model.name}: started at",
                    "value": result.manifest["started_at"] if result else None,
                },
                {
                    "item": f"{model.name}: ended at",
                    "value": result.manifest["ended_at"] if result else None,
                },
            ]
        )
    return PilotMethodsSummary(text, pd.DataFrame(provenance))


def load_full_run_if_present(plan):
    """Read saved results without resuming, writing, creating a client or sending."""
    from jev_eval.runner import _run_directory, load_run

    directory = _run_directory(plan.project, plan.run_id)
    if not directory.exists():
        return None
    result = load_run(plan.project, plan.run_id)
    if result.manifest.get("run_kind") != "full":
        raise PrerequisiteError("This directory does not contain a full-evaluation run.")
    return result


def full_run_plan_table(plan_9b, plan_27b):
    """Dry-run comparison with on-disk pending work and attempt budgets."""
    from jev_eval.backends.inference_sh import SUPPORTED_APPS
    from jev_eval.runner import _run_directory

    plans = (plan_9b, plan_27b)
    for plan, app in zip(plans, SUPPORTED_APPS, strict=True):
        if (
            plan.cohort_metadata is None
            or plan.project.models[0].hosted.app_id != app
            or len(plan.project.models) != 1
        ):
            raise PrerequisiteError("Supply the complete JEV-9B and JEV-27B full plans in order.")
    if (
        plan_9b.cohort_metadata != plan_27b.cohort_metadata
        or plan_9b.dataset_sha256 != plan_27b.dataset_sha256
        or plan_9b.task.fingerprint != plan_27b.task.fingerprint
        or _run_directory(plan_9b.project, plan_9b.run_id)
        == _run_directory(plan_27b.project, plan_27b.run_id)
        or plan_9b.project.workflow.inference_sh_full.cohort_manifest
        != plan_27b.project.workflow.inference_sh_full.cohort_manifest
        or plan_9b.project.workflow.evaluation.reference_labels
        != plan_27b.project.workflow.evaluation.reference_labels
        or plan_9b.project.workflow.inference_sh_full.label_mapping
        != plan_27b.project.workflow.inference_sh_full.label_mapping
    ):
        raise PrerequisiteError("Full models require identical cohort/task and separate outputs.")
    rows = []
    for plan in plans:
        result = load_full_run_if_present(plan)
        validation = validate_full_results(plan, result).iloc[0]
        settings = plan.project.workflow.inference_sh_full
        used = result.manifest["total_attempts"] if result is not None else 0
        resolved = (
            {
                row["study_id"]
                for row in result.predictions
                if row["status"] in {"success", "failed"}
            }
            if result is not None
            else set()
        )
        pending = sum(
            bool(state.title.strip() or state.abstract.strip()) and state.study_id not in resolved
            for state in plan.states
        )
        rows.append(
            {
                **plan.summary().iloc[0].to_dict(),
                "dry_run": settings.dry_run,
                "pending_api_studies": pending,
                "attempts_used": used,
                "remaining_attempt_budget": plan.project.workflow.execution.max_total_attempts
                - used,
                "unknown_pending_request": bool(result.manifest.get("pending_request"))
                if result
                else False,
                "resume_fingerprint_matches": not validation.incompatible_provenance,
                "concurrency": settings.max_concurrency,
                "max_attempts_per_study": settings.retry.max_attempts,
                "retry_initial_seconds": settings.retry.initial_backoff_seconds,
                "retry_cap_seconds": settings.retry.max_backoff_seconds,
                "minimum_request_interval_seconds": settings.min_request_interval_seconds,
                "timeout_seconds": plan.project.backend.timeout_seconds,
                "decision_threshold": settings.decision_threshold,
            }
        )
    return pd.DataFrame(rows)


def validate_full_results(plan, result):
    """Count all failures/missingness and malformed successful rows without filtering."""
    from jev_eval.backends.inference_sh import SUPPORTED_APPS

    expected = {state.study_id for state in plan.states}
    model = plan.project.models[0]
    question = next(iter(plan.task.questions))
    records = result.predictions if result is not None else []
    seen, duplicates, unexpected = set(), 0, 0
    counts = {
        name: 0
        for name in [
            "successful",
            "failed",
            "unknown",
            "not_inferred_no_text",
            "invalid_status",
            "invalid_response_errors",
            "invalid_probabilities",
            "missing_task_ids",
            "missing_model_metadata",
            "name_differences",
            "unexpected_returned_model",
            "revision_not_reported",
        ]
    }
    aliases = {model.hosted.app_id.casefold(), SUPPORTED_APPS[model.hosted.app_id].casefold()}
    for record in records:
        if record.get("error_code") in {
            "invalid_task_envelope",
            "invalid_decision_output",
            "invalid_normalized_response",
        }:
            counts["invalid_response_errors"] += 1
        identifier = record.get("study_id")
        if (
            not isinstance(identifier, str)
            or identifier not in expected
            or record.get("model") != model.name
        ):
            unexpected += 1
        if isinstance(identifier, str):
            if identifier in seen:
                duplicates += 1
            seen.add(identifier)
        status = record.get("status")
        if status == "success":
            counts["successful"] += 1
            probs = record.get("probabilities")
            probability = probs.get(question) if isinstance(probs, dict) else None
            if (
                not isinstance(probs, dict)
                or set(probs) != {question}
                or isinstance(probability, bool)
                or not isinstance(probability, (float, int))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                counts["invalid_probabilities"] += 1
            if not isinstance(record.get("request_id"), str) or not record["request_id"].strip():
                counts["missing_task_ids"] += 1
            returned = record.get("returned_model")
            if (
                not isinstance(returned, str)
                or not returned.strip()
                or record.get("requested_app") != model.hosted.app_id
            ):
                counts["missing_model_metadata"] += 1
            if returned != model.hosted.app_id:
                counts["name_differences"] += 1
            if isinstance(returned, str) and returned.casefold() not in aliases:
                counts["unexpected_returned_model"] += 1
            if record.get("model_revision") is None:
                counts["revision_not_reported"] += 1
        elif status in {"failed", "unknown", "not_inferred_no_text"}:
            counts[status] += 1
        else:
            counts["invalid_status"] += 1
    # A killed process can leave only a pending marker, with no terminal JSONL
    # row. Retain that outcome as unknown rather than implying no request ran.
    pending = result.manifest.get("pending_request") if result is not None else None
    if isinstance(pending, dict):
        identifier = pending.get("study_id")
        if not isinstance(identifier, str) or identifier not in expected:
            unexpected += 1
        elif identifier not in seen:
            seen.add(identifier)
            counts["unknown"] += 1
    incompatible = bool(
        result is not None
        and (
            result.manifest.get("fingerprint") != plan.fingerprint
            or result.manifest.get("dataset_sha256") != plan.dataset_sha256
            or result.manifest.get("task_sha256") != plan.task.fingerprint
            or result.manifest.get("model_names") != [model.name]
            or result.manifest.get("run_kind") != "full"
        )
    )
    return pd.DataFrame(
        [
            {
                "model": model.name,
                "expected_records": len(expected),
                "recorded_rows": len(records),
                **counts,
                "missing": len(expected - seen),
                "duplicate_ids": duplicates,
                "unexpected_rows": unexpected,
                "incompatible_provenance": incompatible,
                "status": result.manifest["status"] if result is not None else "not_run",
                "pending_review": bool(result.manifest.get("pending_request")) if result else False,
            }
        ]
    )


def full_run_metadata_table(plans, results):
    """Use stored run metadata when available; prospective plans otherwise."""
    rows = []
    for plan, result in zip(plans, results, strict=True):
        manifest = result.manifest if result is not None else {}
        cohort = manifest.get("full_cohort", plan.cohort_metadata)
        environment = manifest.get("environment", plan.environment)
        records = result.predictions if result is not None else []
        rows.append(
            {
                "experiment_id": manifest.get(
                    "experiment_id", plan.project.workflow.inference_sh_full.experiment_id
                ),
                "run_id": manifest.get("run_id", plan.run_id),
                "requested_app": manifest.get("requested_model_ids", {}).get(
                    plan.project.models[0].name
                )
                if result is not None
                else plan.project.models[0].hosted.app_id,
                "returned_models": sorted(
                    {row["returned_model"] for row in records if row.get("returned_model")}
                ),
                "provider_reported_revisions": sorted(
                    {row["model_revision"] for row in records if row.get("model_revision")}
                ),
                "immutable_revision_verified": False,
                "unreported_revision": None,
                "source_file": cohort["source_file"],
                "source_file_sha256": cohort["source_file_sha256"],
                "states_sha256": cohort["states_sha256"],
                "task_sha256": cohort["task_sha256"],
                "configuration_sha256": manifest.get(
                    "configuration_sha256", plan.configuration_sha256
                ),
                "git_commit": environment["git"]["commit"],
                "python": environment["python"],
                "lockfile_sha256": environment["lockfile_sha256"],
                "packages": environment["packages"],
                "started_at": manifest.get("started_at"),
                "ended_at": manifest.get("ended_at"),
                "recorded_attempts": manifest.get("total_attempts", 0),
                "reported_input_tokens": sum(
                    row.get("usage", {}).get("input_tokens", 0) for row in records
                ),
                "status": manifest.get("status", "not_run"),
            }
        )
    return pd.DataFrame(rows)


def print_full_progress(counts):
    print(
        f"{counts['model']}: {counts['recorded']}/{counts['records']} records saved; "
        f"{counts['success']} success, {counts['failed']} failed, {counts['unknown']} unknown, "
        f"{counts['not_inferred_no_text']} no text; {counts['attempts']} attempts; {counts['status']}"
    )


def comparison_table(evaluation):
    return evaluation.table.copy()


def plot_confusion_matrices(evaluation):
    import matplotlib.pyplot as plt

    count = len(evaluation.matrices)
    figure, axes = plt.subplots(1, count, figsize=(5 * count, 4), squeeze=False)
    for axis, (model, matrix) in zip(axes.flat, evaluation.matrices.items(), strict=True):
        axis.imshow(matrix, cmap="Blues", vmin=0)
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Exclude", "Retain"],
            yticklabels=["Exclude", "Retain"],
            xlabel="Predicted decision",
            ylabel="Reference decision",
            title=model,
        )
    figure.suptitle(
        "SYNTHETIC DEMONSTRATION — not JEV performance"
        if evaluation.demo
        else "Screening decisions (valid paired predictions only; inspect coverage)"
    )
    figure.tight_layout()
    return figure


def plot_threshold_tradeoffs(curve, candidates, target_recall=0.95):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model, group in curve.groupby("model", sort=False):
        line = axes[0].plot(group.threshold, group.recall_cohort_lower_bound, label=model)[0]
        axes[0].plot(
            group.threshold, group.recall_scored, linestyle=":", color=line.get_color(), alpha=0.6
        )
        axes[1].plot(
            group.threshold, group.automatic_exclusion_fraction, color=line.get_color(), label=model
        )
        candidate = candidates[(candidates.model == model) & (candidates.status == "attainable")]
        if not candidate.empty:
            threshold = float(candidate.iloc[0].threshold)
            for axis in axes:
                axis.axvline(threshold, color=line.get_color(), linestyle="--", alpha=0.6)
    axes[0].axhline(
        target_recall,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Target {target_recall:.0%}",
    )
    axes[0].set(
        title="Recall: solid = full-labelled lower bound; dotted = scored pairs", ylabel="Recall"
    )
    axes[1].set(
        title="Automatic exclusions include false negatives",
        ylabel="Fraction of all studies auto-excluded",
    )
    for axis in axes:
        axis.set(xlabel="Retain if P(true) ≥ cutoff", xlim=(0, 1), ylim=(0, 1.01))
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle(
        "EXPLORATORY — thresholds selected and assessed on the same human-labelled cohort"
    )
    figure.tight_layout()
    return figure


def plot_recall_candidate_matrices(candidates):
    """Scored-pair matrices only: unavailable studies never become exclusions."""
    import matplotlib.pyplot as plt

    available = candidates[candidates.status == "attainable"]
    if available.empty:
        return None
    figure, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5), squeeze=False)
    for axis, row in zip(axes.flat, available.itertuples(), strict=True):
        matrix = [[int(row.tn), int(row.fp)], [int(row.fn), int(row.tp)]]
        axis.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                axis.text(
                    j,
                    i,
                    str(matrix[i][j]),
                    ha="center",
                    va="center",
                    color="white"
                    if matrix[i][j] > max(row.tn, row.fp, row.fn, row.tp) / 2
                    else "black",
                )
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Exclude", "Retain"],
            yticklabels=["Exclude", "Retain"],
            xlabel="Model decision",
            ylabel="Human reference",
            title=f"{row.model} | cutoff {row.threshold!r}\n"
            f"Scored recall {row.recall_scored:.2%}; cohort lower bound {row.recall_cohort_lower_bound:.2%}\n"
            f"{int(row.unscored_studies)} unscored studies are outside this matrix",
        )
    figure.suptitle("EXPLORATORY candidate confusion matrices — valid scored/reference pairs only")
    figure.tight_layout()
    return figure


def export_table(table, project, path, *, allow_writes=False):
    require_writes(project, allow_writes)
    destination = safe_output_path(project, path)
    if destination.suffix != ".csv":
        raise ValueError("Table exports must use a .csv extension.")
    if destination.exists():
        raise FileExistsError("Export already exists; choose a new filename.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return destination.name


def export_figure(figure, project, path, *, allow_writes=False):
    require_writes(project, allow_writes)
    destination = safe_output_path(project, path)
    if destination.suffix not in (".png", ".pdf"):
        raise ValueError("Figure exports must use .png or .pdf.")
    if destination.exists():
        raise FileExistsError("Export already exists; choose a new filename.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    return Path(destination).name
