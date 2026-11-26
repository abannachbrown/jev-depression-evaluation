"""Aggregate, read-only interim reports from archived inference and reference data."""

import base64
import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import Field

from jev_eval.config import PrerequisiteError, Settings, parse_settings, read_yaml
from jev_eval.evaluation import (
    RankingEvaluation,
    ranking_readouts,
    select_recall_thresholds,
    threshold_readouts,
)
from jev_eval.reporting import full_run_metadata_table
from jev_eval.tasks import Probability


class InterimSummarySettings(Settings):
    schema_version: Literal[1]
    analysis_config: str
    comparison_config: str | None = None
    target_recall: Probability = Field(gt=0)
    illustrative_cutoffs: dict[str, Probability] = Field(min_length=1)
    planned_validation_records: int = Field(gt=0)
    validation_status: Literal["planned_after_colleague_review"]
    allow_writes: bool = False
    output_root: str = "results/summary/interim"


@dataclass
class InterimSummary:
    data: object
    settings: InterimSummarySettings
    tables: dict[str, pd.DataFrame]
    candidates: pd.DataFrame
    methods: str
    provenance: dict
    ranking: RankingEvaluation
    screening: object = None
    comparison: object = None


def _conversion_tables(data):
    """Use the archived conversion manifest, never reopen or edit the workbook."""
    source = data.plans[0].project.root / data.plans[0].cohort_metadata["source_file"]
    path = source.parent / "manifest.json"
    if not path.is_file():
        unavailable = pd.DataFrame([{"status": "Conversion manifest not available"}])
        return unavailable, unavailable.copy(), unavailable.copy()
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
        expected = data.plans[0].cohort_metadata["conversion_manifest_sha256"]
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError
        if manifest.get("output_sha256", {}).get("reference_labels.csv") not in (
            None,
            data.references.sha256,
        ):
            raise ValueError
        validation = pd.DataFrame(
            [{"check": name, "value": value} for name, value in manifest["validation"].items()]
        )
        audit = manifest["audit"]
        audit_summary = pd.DataFrame(
            [
                {
                    "audit_entries": manifest["records"]["audit_entries"],
                    "unique_study_ids": audit["unique_study_ids"],
                    "label_disagreements": audit["label_disagreements"],
                    "main_label_mismatches": audit["main_label_mismatches"],
                    "unknown_study_ids": audit["unknown_study_ids"],
                    "policy": audit["policy"],
                    "adjudication_applied": audit["adjudication_applied"],
                }
            ]
        )
        audit_issues = pd.DataFrame(
            [{"issue": issue, "entries": count} for issue, count in audit["issue_counts"].items()],
            columns=["issue", "entries"],
        )
    except (OSError, ValueError, KeyError, TypeError):
        raise PrerequisiteError("Cannot verify the archived conversion/audit manifest.") from None
    return validation, audit_summary, audit_issues


def _paired_positive_errors(data, cutoffs):
    """Aggregate paired false negatives; no study IDs or citation text leave the helper."""
    models = list(cutoffs)
    if len(models) != 2:
        raise PrerequisiteError("The interim paired comparison requires exactly two models.")
    positive = data.rows[data.rows.human_label.eq(True)]
    pairs = positive.pivot(
        index="study_id", columns="model_name", values="probability_true"
    ).reindex(columns=models)
    available = pairs[models].notna().all(axis=1)
    paired = pairs.loc[available]
    missed_a = paired[models[0]].lt(cutoffs[models[0]])
    missed_b = paired[models[1]].lt(cutoffs[models[1]])
    return pd.DataFrame(
        [
            {
                "first_model": models[0],
                "second_model": models[1],
                "first_cutoff": cutoffs[models[0]],
                "second_cutoff": cutoffs[models[1]],
                "human_retain_total": len(pairs),
                "both_scored": int(available.sum()),
                "unavailable_in_either_model": int((~available).sum()),
                "both_missed": int((missed_a & missed_b).sum()),
                "only_first_missed": int((missed_a & ~missed_b).sum()),
                "only_second_missed": int((~missed_a & missed_b).sum()),
                "neither_missed": int((~missed_a & ~missed_b).sum()),
            }
        ]
    )


def load_interim_summary(root, settings_path="configs/evaluations/interim_summary.yaml"):
    """Read saved development results only. No API client, .env or validation cohort."""
    root = Path(root).resolve()
    settings = parse_settings(
        InterimSummarySettings, read_yaml(root / settings_path), settings_path
    )
    if settings.comparison_config is not None:
        return _load_comparison_summary(root, settings, settings_path)
    from jev_eval.screening import (
        load_screening_analysis,
        screening_question_summary,
        select_primary_analysis,
    )

    screening = load_screening_analysis(root, include_legacy=False)
    data, primary_profile, selected_analysis_config = select_primary_analysis(
        root, screening, settings.analysis_config
    )
    names = [plan.project.models[0].name for plan in data.plans]
    if set(settings.illustrative_cutoffs) != set(names):
        raise PrerequisiteError("Illustrative cutoffs must name both archived models exactly.")

    flow = []
    for model, group in data.rows.groupby("model_name", sort=False):
        scored = group.probability_true.notna()
        flow.append(
            {
                "model": model,
                "source_studies": len(group),
                "human_labelled": int(group.human_label.notna().sum()),
                "probabilities_available": int(scored.sum()),
                "scored_labelled_pairs": int((scored & group.human_label.notna()).sum()),
                "unscored_manual_queue": int((~scored).sum()),
                "human_retain_total": int(group.human_label.eq(True).sum()),
                "human_retain_unscored": int((group.human_label.eq(True) & ~scored).sum()),
            }
        )
    validation, audit, audit_issues = _conversion_tables(data)
    candidates = select_recall_thresholds(data, settings.target_recall, basis="all_labelled")
    rounded = (
        threshold_readouts(
            data, {model: [cutoff] for model, cutoff in settings.illustrative_cutoffs.items()}
        )
        if primary_profile == "legacy_binary"
        else pd.DataFrame()
    )
    baseline = threshold_readouts(data, {name: [0.5] for name in names})
    ranking = ranking_readouts(data)
    metrics = pd.concat(
        [
            candidates.assign(cutoff_source="Highest cutoff meeting exploratory recall target"),
            rounded.assign(cutoff_source="Illustrative rounded cutoff; not locked"),
            baseline.assign(cutoff_source="0.5 baseline"),
        ],
        ignore_index=True,
    )
    criteria = pd.DataFrame(
        [
            {
                "question_id": name,
                "instructions": question.instructions,
                "retain": question.criteria.get("true"),
                "exclude": question.criteria.get("false"),
            }
            for name, question in data.plans[0].task.questions.items()
        ]
    )
    reference = data.references.summary().assign(
        human_retain=sum(data.reference_labels.values()),
        human_exclude=sum(not label for label in data.reference_labels.values()),
    )
    run_metadata = full_run_metadata_table(data.plans, data.results)
    audit_note = (
        "Workbook audit findings were recorded without automatic adjudication. "
        if "status" not in audit.columns
        else "The workbook conversion/audit manifest is unavailable to this summary. "
    )
    status = pd.DataFrame(
        [
            {
                "stage": "Development cohort inference",
                "status": "; ".join(
                    f"{name}: {result.manifest['status'] if result is not None else 'not_run'}"
                    for name, result in zip(names, data.results, strict=True)
                ),
            },
            {
                "stage": "Threshold analysis",
                "status": "Exploratory; selected and measured on the same labels",
            },
            {"stage": "Colleague review", "status": "Pending; no approval recorded by this report"},
            {
                "stage": "Unseen-study validation",
                "status": f"Planned after colleague review: {settings.planned_validation_records:,} studies; "
                "not loaded, executed or evaluated here",
            },
            {"stage": "Unseen-cohort independence", "status": "Not yet checked"},
            {
                "stage": "Confidence intervals / independent performance claims",
                "status": "Not established in this interim summary",
            },
        ]
    )
    methods = (
        f"This interim retrospective analysis comprises {len(data.states):,} studies, with "
        f"{len(data.reference_labels):,} available human reference labels "
        f"({sum(data.reference_labels.values()):,} retain and "
        f"{sum(not x for x in data.reference_labels.values()):,} exclude). "
        "Prepared model states contain study ID, title and abstract; human decisions and "
        "ITEM_ID are held separately. The configured/saved Inference.sh model apps "
        f"{', '.join(run_metadata.requested_app)} used the same frozen eligibility question "
        "and cohort. Execution status, available probability counts and exceptions are reported per model. "
        "Retain decisions use P(true) >= cutoff, including ties. Candidate thresholds were "
        "selected and assessed on these same reference labels and are exploratory. "
        "Scored-pair metrics exclude unavailable predictions; full-labelled recall gives "
        "them no model credit. Threshold-independent ROC AUC, trapezoidal PR AUC and "
        "non-interpolated average precision use scored human-labelled pairs only. "
        "Workload calculations retain all unscored records in a "
        f"separate manual-review queue. {audit_note}"
        "Human screening/adjudication procedures require author "
        "confirmation before publication. Immutable hosted model-weight revisions are "
        "not verified. Following colleague review, validation on "
        f"{settings.planned_validation_records:,} unseen studies is planned; no such data "
        "or results are included here."
    )
    from jev_eval.environment import reproducibility_summary, sha256_file
    from jev_eval.runner import _run_directory

    provenance = {
        "report_type": "interim_exploratory_development_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_settings": settings.model_dump(),
        "summary_config_sha256": sha256_file(root / settings_path),
        "analysis_settings": data.settings.model_dump(),
        "analysis_config": selected_analysis_config,
        "primary_profile": primary_profile,
        "analysis_config_sha256": sha256_file(root / selected_analysis_config),
        "analysis_environment": reproducibility_summary(data.plans[0].project),
        "analysis_code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("summary.py", "evaluation.py", "reporting.py", "screening.py")
        },
        "reference_sha256": data.references.sha256,
        "states_sha256": data.plans[0].dataset_sha256,
        "task_sha256": data.plans[0].task.fingerprint,
        "workbook_sha256": data.plans[0].cohort_metadata.get("source_workbook_sha256"),
        "conversion_manifest_sha256": data.plans[0].cohort_metadata.get(
            "conversion_manifest_sha256"
        ),
        "inference_runs": [
            {
                "run_id": plan.run_id,
                "status": result.manifest["status"] if result is not None else "not_run",
                "original_fingerprint": result.manifest["fingerprint"]
                if result is not None
                else None,
                "git_commit": plan.environment["git"]["commit"],
                "lockfile_sha256": plan.environment["lockfile_sha256"],
                "configuration_sha256": plan.configuration_sha256,
                "source_sha256": plan.source_sha256,
                "predictions_sha256": sha256_file(
                    _run_directory(plan.project, plan.run_id) / "predictions.jsonl"
                )
                if result is not None
                else None,
            }
            for plan, result in zip(data.plans, data.results, strict=True)
        ],
        "validation_results_present": False,
        "immutable_weight_revision_verified": False,
        "ranking_definitions": {
            "positive_class": "retain for full-text assessment",
            "score": "saved probability_true, without thresholding or imputation",
            "scope": "scored human-labelled pairs; both classes required",
            "roc_auc": "sklearn.metrics.roc_auc_score",
            "pr_auc_trapezoidal": "sklearn.metrics.auc(recall, precision)",
            "average_precision": "sklearn.metrics.average_precision_score; non-interpolated",
        },
    }
    tables = {
        "review_status": status,
        "study_flow": pd.DataFrame(flow),
        "inference_coverage": data.coverage.copy(),
        "reference_summary": reference,
        "preparation_checks": validation,
        "audit_summary": audit,
        "audit_issues": audit_issues,
        "archived_criteria": criteria,
        "model_execution": run_metadata,
        "screening_metrics": metrics,
        "ranking_metrics": ranking.table,
        "paired_positive_errors": (
            _paired_positive_errors(data, settings.illustrative_cutoffs)
            if primary_profile == "legacy_binary"
            else _paired_positive_errors(
                data, dict(zip(candidates.model, candidates.threshold, strict=True))
            )
            if candidates.status.eq("attainable").all()
            else pd.DataFrame([{"status": "requires_target_cutoffs_and_both_models"}])
        ),
    }
    tables.update(
        {
            "v1_run_availability": screening.availability,
            "v1_exploratory_thresholds": screening.threshold_candidates,
            "v1_ranking_metrics": screening.ranking_metrics,
            "hybrid_rule_tuning": screening.rule_tuning,
        }
    )
    for name, view in screening.data.items():
        tables[name + "_question_summary"] = screening_question_summary(view.plans, view.results)
    methods += (
        " Reviewed v1 binary/hybrid runs are reported separately from these legacy results. "
        "R1 remains disabled pending protocol clarification; formal R2/R3 tuning requires "
        "the reconciled original 1,251-record tuning split."
    )
    provenance["screening_plan"] = screening.settings.model_dump()
    provenance["screening_plan_sha256"] = sha256_file(root / "configs/screening/v1.yaml")
    provenance["v1_inference_sources"] = {
        name: [
            {
                "run_id": plan.run_id,
                "fingerprint": result.manifest["fingerprint"] if result else None,
                "task_sha256": plan.task.fingerprint,
                "reference_sha256": view.references.sha256,
            }
            for plan, result in zip(view.plans, view.results, strict=True)
        ]
        for name, view in screening.data.items()
    }
    return InterimSummary(
        data, settings, tables, candidates, methods, provenance, ranking, screening
    )


def _load_comparison_summary(root, settings, settings_path):
    """Summarise all four explicit archives rather than selecting one approach."""
    from jev_eval.comparison import load_screening_comparison
    from jev_eval.environment import reproducibility_summary, sha256_file
    from jev_eval.runner import _run_directory
    from jev_eval.screening import load_screening_settings

    comparison = load_screening_comparison(root, settings.comparison_config)
    if settings.target_recall != comparison.settings.target_recall:
        raise PrerequisiteError("Summary and comparison recall targets must agree.")
    data = comparison.datasets["binary"]
    screening_settings = load_screening_settings(root)
    validation, audit, audit_issues = _conversion_tables(data)
    coverage = comparison.tables["coverage"]
    flow = []
    for arm, group in comparison.data.rows.groupby("model_name", sort=False):
        scored = group.probability_true.notna()
        flow.append(
            {
                "arm": arm,
                "source_studies": len(group),
                "human_labelled": int(group.human_label.notna().sum()),
                "probabilities_available": int(scored.sum()),
                "scored_labelled_pairs": int((scored & group.human_label.notna()).sum()),
                "unscored_manual_queue": int((~scored).sum()),
                "human_retain_total": int(group.human_label.eq(True).sum()),
                "human_retain_unscored": int((group.human_label.eq(True) & ~scored).sum()),
            }
        )
    reference = data.references.summary().assign(
        human_retain=sum(data.reference_labels.values()),
        human_exclude=sum(not label for label in data.reference_labels.values()),
    )
    status = pd.DataFrame(
        [
            {
                "stage": "Four saved inference arms",
                "status": "; ".join(
                    f"{r.arm}: {r.status}" for r in comparison.tables["arms"].itertuples()
                ),
            },
            {
                "stage": "Question wording comparability",
                "status": "; ".join(comparison.tables["comparability"].interpretation.unique()),
            },
            {
                "stage": "Threshold analysis",
                "status": "Exploratory; cutoffs selected and assessed on the same labels",
            },
            {"stage": "R1", "status": screening_settings.r1_policy},
            {"stage": "Colleague review", "status": "Pending; no approval recorded by this report"},
            {
                "stage": "Unseen-study validation",
                "status": f"Planned after colleague review: {settings.planned_validation_records:,} studies; not loaded or evaluated here",
            },
            {
                "stage": "Confidence intervals / independent performance claims",
                "status": "Not established in this interim summary",
            },
        ]
    )
    tables = {
        "review_status": status,
        "study_flow": pd.DataFrame(flow),
        "reference_summary": reference,
        "preparation_checks": validation,
        "audit_summary": audit,
        "audit_issues": audit_issues,
        "inference_coverage": coverage,
        "archived_criteria": comparison.tables["criteria"],
        "model_execution": comparison.tables["execution"],
        **{
            name: comparison.tables[name]
            for name in (
                "arms",
                "comparability",
                "screening_metrics",
                "ranking_metrics",
                "common_scored_ranking",
                "paired_positive_errors",
                "hybrid_derived_scores",
                "question_summary",
                "threshold_candidates",
            )
        },
    }
    methods = (
        f"This interim retrospective comparison comprises {len(data.states):,} studies and "
        f"{len(data.reference_labels):,} human reference labels "
        f"({sum(data.reference_labels.values()):,} retain). Four saved arms cross JEV-9B and "
        "JEV-27B v1.1 with binary and hybrid NOUL question designs. All arms use the same "
        "ordered citation cohort and reference CSV; study ID, title and abstract are model "
        "inputs, while human decisions and ITEM_ID remain separate. Existing responses were "
        "analysed without rerunning inference or relabelling archived prompts. "
        "Archived binary and hybrid overall definitions are compared explicitly; where wording "
        "differs, performance differences cannot be attributed solely to question design. "
        "The primary hybrid score is its overall eligibility answer. R2 minimum and R3 mean "
        "of four primary domains are secondary derived-score analyses, not additional models "
        "or calibrated joint probabilities. R1 remains disabled pending protocol clarification. "
        f"Each arm's highest cutoff meeting {settings.target_recall:.0%} full-cohort recall "
        "is selected and assessed on the same labels; these estimates are exploratory. "
        "Retain means score >= cutoff, including ties. Scored-pair precision, specificity, "
        "F2 and confusion matrices exclude unavailable scores. Cohort recall gives unscored "
        "studies no model credit; all unscored studies remain in the manual-review queue. "
        "WSS@95 uses excluded cohort fraction minus 0.05 only when its recall requirement is "
        "met; its difference from the protocol's rank-based definition needs review. "
        "ROC AUC, trapezoidal PR AUC and non-interpolated AP are shown on available pairs "
        "and additionally on the identical all-four-scored subset. Workbook discrepancies "
        "are reported without automatic adjudication. Immutable hosted weight revisions "
        "remain unverified. Independent validation on "
        f"{settings.planned_validation_records:,} unseen studies is planned after colleague "
        "review; no such studies are loaded, used for tuning or evaluated here."
    )
    sources = []
    for approach, dataset in comparison.datasets.items():
        for plan, result in zip(dataset.plans, dataset.results, strict=True):
            sources.append(
                {
                    "approach": approach,
                    "model": plan.project.models[0].name,
                    "run_id": plan.run_id,
                    "status": result.manifest["status"] if result else "not_run",
                    "original_fingerprint": result.manifest["fingerprint"] if result else None,
                    "task_sha256": plan.task.fingerprint,
                    "source_sha256": plan.source_sha256,
                    "configuration_sha256": plan.configuration_sha256,
                    "git_commit": plan.environment["git"]["commit"],
                    "lockfile_sha256": plan.environment["lockfile_sha256"],
                    "predictions_sha256": sha256_file(
                        _run_directory(plan.project, plan.run_id) / "predictions.jsonl"
                    )
                    if result
                    else None,
                }
            )
    provenance = {
        "report_type": "interim_exploratory_four_arm_comparison",
        "primary_profile": "four_arm_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary_settings": settings.model_dump(),
        "summary_config_sha256": sha256_file(root / settings_path),
        "analysis_config": settings.comparison_config,
        "analysis_config_sha256": sha256_file(root / settings.comparison_config),
        "analysis_environment": reproducibility_summary(data.plans[0].project),
        "analysis_code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "comparison.py",
                "summary.py",
                "evaluation.py",
                "reporting.py",
                "screening.py",
            )
        },
        "states_sha256": data.plans[0].dataset_sha256,
        "reference_sha256": data.references.sha256,
        "task_sha256": {r["approach"] + " / " + r["model"]: r["task_sha256"] for r in sources},
        "workbook_sha256": data.plans[0].cohort_metadata.get("source_workbook_sha256"),
        "conversion_manifest_sha256": data.plans[0].cohort_metadata.get(
            "conversion_manifest_sha256"
        ),
        "inference_runs": sources,
        "comparison_sources": comparison.settings.model_dump(),
        "same_overall_eligibility_definition": bool(
            comparison.tables["comparability"].same_overall_eligibility_definition.all()
        ),
        "validation_results_present": False,
        "immutable_weight_revision_verified": False,
    }
    return InterimSummary(
        data,
        settings,
        tables,
        comparison.candidates,
        methods,
        provenance,
        comparison.ranking,
        comparison=comparison,
    )


def export_interim_summary(summary, figures=None, *, allow_writes=False):
    """Explicitly export aggregate-only HTML/CSV/figures to a new private results folder."""
    from jev_eval.environment import sha256_file
    from jev_eval.runner import safe_output_path

    if not (allow_writes and summary.settings.allow_writes):
        raise PrerequisiteError(
            "Interim exports require allow_writes in summary YAML and the cell."
        )
    figures = figures or {}
    if any(not re.fullmatch(r"[a-z][a-z0-9-]*", name) for name in figures):
        raise PrerequisiteError(
            "Figure names must contain only lowercase letters, digits and hyphens."
        )
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    folder = safe_output_path(
        summary.data.plans[0].project, Path(summary.settings.output_root) / tag
    )
    folder.mkdir(parents=True, exist_ok=False)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<title>JEV interim summary</title><h1>JEV interim summary</h1>",
        "<p>Exploratory development results. Colleague review pending; unseen-study validation not performed.</p>",
        f"<p>{html.escape(summary.methods)}</p>",
    ]
    for name, table in summary.tables.items():
        table.to_csv(folder / f"{name}.csv", index=False)
        parts.extend(
            [
                f"<h2>{html.escape(name.replace('_', ' '))}</h2>",
                table.to_html(index=False, escape=True),
            ]
        )
    for name, figure in figures.items():
        figure.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight")
        figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
        encoded = base64.b64encode((folder / f"{name}.png").read_bytes()).decode("ascii")
        parts.append(
            f"<img alt='{html.escape(name)}' style='max-width:100%' src='data:image/png;base64,{encoded}'>"
        )
    parts.append("</html>")
    (folder / "summary.html").write_text("\n".join(parts), encoding="utf-8")
    (folder / "methods_draft.txt").write_text(summary.methods + "\n", encoding="utf-8")
    manifest = {
        **summary.provenance,
        "output_sha256": {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return folder
