"""Read-only, explicit two-model by two-question-design archive comparison."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field

from jev_eval.config import (
    FullAnalysisSettings,
    PrerequisiteError,
    Settings,
    load_project_config,
    parse_settings,
    read_yaml,
)
from jev_eval.data import fingerprint
from jev_eval.evaluation import (
    load_saved_full_evaluation,
    ranking_readouts,
    select_recall_thresholds,
    threshold_readouts,
)
from jev_eval.reporting import full_run_metadata_table, validate_full_results
from jev_eval.screening import hybrid_strategy_views, screening_question_summary
from jev_eval.tasks import Probability, load_task, primary_question, valid_batch_task


class ComparisonSettings(Settings):
    schema_version: Literal[1]
    binary_analysis: str
    hybrid_analysis: str
    target_recall: Probability = Field(gt=0)
    allow_writes: bool = False
    output_root: str


@dataclass
class ComparisonScores:
    # model_name is a unique presentation label, not a changed provider identity.
    # Original deployment/model fields and all archived files remain unchanged.
    rows: pd.DataFrame


@dataclass
class ScreeningComparison:
    root: Path
    settings: ComparisonSettings
    settings_path: str
    datasets: dict
    data: ComparisonScores
    tables: dict
    ranking: object
    candidates: pd.DataFrame


def _comparison_projects(root, settings_path):
    """Resolve all four profiles without opening private data or constructing clients."""
    settings = parse_settings(ComparisonSettings, read_yaml(root / settings_path), settings_path)
    projects = []
    for approach in ("binary", "hybrid"):
        path = getattr(settings, approach + "_analysis")
        analysis = parse_settings(FullAnalysisSettings, read_yaml(root / path), path)
        for model, workflow in (
            ("jev-9b", analysis.workflow_9b),
            ("jev-27b", analysis.workflow_27b),
        ):
            project = load_project_config(root, workflow)
            task = load_task(project)
            if (
                len(project.models) != 1
                or project.workflow.output.run_id is None
                or project.models[0].name != model
                or project.evaluation.task.approach != approach
                or not valid_batch_task(project, task)
            ):
                raise PrerequisiteError(
                    "Each comparison profile must match its model and NOUL design."
                )
            projects.append((approach, model, project, task))
    return projects


def comparison_setup_table(root, settings_path="configs/evaluations/comparison.yaml"):
    """Configuration-only four-arm readiness; safe before private artifacts are supplied."""
    from jev_eval.runner import _run_directory

    root = Path(root).resolve()
    rows = []
    for approach, model, project, task in _comparison_projects(root, settings_path):
        full = project.workflow.inference_sh_full
        if full is None or not full.enabled:
            raise PrerequisiteError("Comparison sources require full-cohort profiles.")
        rows.append(
            {
                "arm": _arm_label(approach, model),
                "approach": approach,
                "model": model,
                "requested_app": project.models[0].hosted.app_id,
                "questions": len(task.questions),
                "primary_question": primary_question(project, task),
                "configured_task_sha256": task.fingerprint,
                "configured_overall_sha256": fingerprint(
                    task.questions[primary_question(project, task)].model_dump()
                ),
                "workflow": project.workflow_path,
                "states_file": project.evaluation.dataset.path,
                "reference_file": project.workflow.evaluation.reference_labels,
                "input_available": (root / project.evaluation.dataset.path).is_file(),
                "reference_available": (
                    root / project.workflow.evaluation.reference_labels
                ).is_file(),
                "saved_manifest_available": (
                    _run_directory(project, project.workflow.output.run_id) / "manifest.json"
                ).is_file(),
                "run_id": project.workflow.output.run_id,
                "mode": project.workflow.execution.mode,
                "yaml_paid_calls": project.workflow.execution.allow_paid_calls,
                "yaml_writes": project.workflow.output.allow_writes,
                "yaml_dry_run": full.dry_run,
                "seed": project.evaluation.seed,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SharedComparisonInputs:
    projects: tuple
    states: list
    validation: object
    references: object
    table: pd.DataFrame


def load_shared_comparison_inputs(root, settings_path="configs/evaluations/comparison.yaml"):
    """Validate four source selections while retaining one citation cohort and reference.

    Do not read inference results, alter workbooks, create a new conversion,
    freeze samples, load credentials or send requests.
    """
    from jev_eval.data import load_full_dataset, prepared_state_report
    from jev_eval.evaluation import load_reference_csv

    root = Path(root).resolve()
    projects = _comparison_projects(root, settings_path)
    baseline = baseline_reference = None
    rows = []
    for approach, model, project, _task in projects:
        cohort = load_full_dataset(project)
        reference = load_reference_csv(project, cohort.states)
        if baseline is None:
            baseline, baseline_reference = cohort, reference
        if (
            cohort.manifest["states_sha256"] != baseline.manifest["states_sha256"]
            or cohort.manifest["ordered_study_ids"] != baseline.manifest["ordered_study_ids"]
            or reference.sha256 != baseline_reference.sha256
            or not reference.table.equals(baseline_reference.table)
        ):
            raise PrerequisiteError(
                "All four arms must use identical ordered citations and the same mapped human reference."
            )
        rows.append(
            {
                "arm": _arm_label(approach, model),
                "records": len(cohort.states),
                "states_sha256": cohort.manifest["states_sha256"],
                "source_file_sha256": cohort.manifest["source_file_sha256"],
                "reference_sha256": reference.sha256,
                "ordered_ids_match": True,
                "reference_mapping_matches": True,
                "missing_abstracts": cohort.manifest["counts"]["missing_abstracts"],
                "not_inferred_no_text": cohort.manifest["counts"]["not_inferred_no_text"],
                "reference_rows": reference.counts["reference_rows"],
                "missing_human_labels": reference.counts["missing_human_labels"],
                "seed": project.evaluation.seed,
            }
        )
    return SharedComparisonInputs(
        tuple(p for _, _, p, _ in projects),
        baseline.states,
        prepared_state_report(projects[0][2]),
        baseline_reference,
        pd.DataFrame(rows),
    )


def _arm_label(approach, model):
    return f"{approach} / {model}"


def _identify(table, arms):
    return table.merge(
        arms[["arm", "approach", "model_deployment", "source_analysis", "prompt_version"]],
        left_on="model",
        right_on="arm",
        how="left",
        validate="many_to_one",
    )


def _paired_errors(data, candidates):
    lookup = candidates.set_index("model")
    positives = data.rows[data.rows.human_label.eq(True)]
    paired = positives.pivot(
        index="study_id", columns="model_name", values="probability_true"
    ).reindex(columns=lookup.index)
    rows = []
    for first, second in combinations(lookup.index, 2):
        both = paired[[first, second]].notna().all(axis=1)
        row = {
            "first_arm": first,
            "second_arm": second,
            "human_retain_total": len(paired),
            "both_scored": int(both.sum()),
            "unavailable_in_either_arm": int((~both).sum()),
            "status": "requires_attainable_cutoffs",
        }
        if all(lookup.loc[name, "status"] == "attainable" for name in (first, second)):
            missed_a = paired.loc[both, first].lt(lookup.loc[first, "threshold"])
            missed_b = paired.loc[both, second].lt(lookup.loc[second, "threshold"])
            row.update(
                status="exploratory_paired_scored_positives",
                first_cutoff=float(lookup.loc[first, "threshold"]),
                second_cutoff=float(lookup.loc[second, "threshold"]),
                both_missed=int((missed_a & missed_b).sum()),
                only_first_missed=int((missed_a & ~missed_b).sum()),
                only_second_missed=int((~missed_a & missed_b).sum()),
                neither_missed=int((~missed_a & ~missed_b).sum()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def load_screening_comparison(root, settings_path="configs/evaluations/comparison.yaml"):
    """Load all four arms without .env, clients, inference, fallback or artifact writes.

    The selected binary archive can use older wording. Preserve its identity,
    report exact task differences and never present it as a reviewed-v1 rerun.
    Entirely absent arms remain visible with null scores, not fabricated results.
    """
    root = Path(root).resolve()
    settings = parse_settings(ComparisonSettings, read_yaml(root / settings_path), settings_path)
    datasets = {
        approach: load_saved_full_evaluation(root, path, allow_missing_runs=True)
        for approach, path in (
            ("binary", settings.binary_analysis),
            ("hybrid", settings.hybrid_analysis),
        )
    }
    binary, hybrid = datasets.values()
    if (
        binary.plans[0].dataset_sha256 != hybrid.plans[0].dataset_sha256
        or binary.references.sha256 != hybrid.references.sha256
        or binary.reference_labels != hybrid.reference_labels
        or [s.study_id for s in binary.states] != [s.study_id for s in hybrid.states]
    ):
        raise PrerequisiteError(
            "Four-arm comparison requires the identical ordered cohort and reference labels."
        )
    arms, criteria, score_tables, validation, execution, questions = [], [], [], [], [], []
    destinations = set()
    for approach, dataset in datasets.items():
        source_path = getattr(settings, approach + "_analysis")
        for plan, result in zip(dataset.plans, dataset.results, strict=True):
            model = plan.project.models[0].name
            if model not in {"jev-9b", "jev-27b"}:
                raise PrerequisiteError("Comparison requires the JEV-9B and JEV-27B deployments.")
            if plan.project.evaluation.task.approach != approach:
                raise PrerequisiteError(
                    "Comparison source does not match its stated NOUL approach."
                )
            destination = (plan.project.workflow.output.root, plan.run_id)
            if destination in destinations:
                raise PrerequisiteError(
                    "Every comparison arm needs a distinct saved run directory."
                )
            destinations.add(destination)
            label = _arm_label(approach, model)
            primary = primary_question(plan.project, plan.task)
            overall = plan.task.questions[primary]
            arms.append(
                {
                    "arm": label,
                    "approach": approach,
                    "model_deployment": model,
                    "requested_app": plan.project.models[0].hosted.app_id,
                    "source_analysis": source_path,
                    "workflow": getattr(
                        dataset.settings, "workflow_" + ("9b" if model == "jev-9b" else "27b")
                    ),
                    "run_id": plan.run_id,
                    "status": result.manifest["status"] if result else "not_run",
                    "saved": result is not None,
                    "prompt_version": (
                        result.manifest.get("prompt_version")
                        if result
                        else plan.project.evaluation.task.prompt_version
                    )
                    or "archived_pre_v1",
                    "questions": len(plan.task.questions),
                    "primary_question": primary,
                    "task_sha256": plan.task.fingerprint,
                    "overall_definition_sha256": fingerprint(overall.model_dump()),
                    "inference_fingerprint": result.manifest["fingerprint"] if result else None,
                }
            )
            for question_id, question in plan.task.questions.items():
                criteria.append(
                    {
                        "arm": label,
                        "question_id": question_id,
                        "instructions": question.instructions,
                        "retain": question.criteria.get("true"),
                        "exclude": question.criteria.get("false"),
                    }
                )
            validation.append(validate_full_results(plan, result).assign(arm=label))
        scores = dataset.rows.copy(deep=True)
        scores["model_deployment"] = scores.model_name
        scores["approach"] = approach
        scores["model_name"] = scores.model_name.map(lambda model: _arm_label(approach, model))
        score_tables.append(scores)
        execution.append(
            full_run_metadata_table(dataset.plans, dataset.results).assign(approach=approach)
        )
        questions.append(
            screening_question_summary(dataset.plans, dataset.results).assign(approach=approach)
        )
    arms = pd.DataFrame(arms)
    comparability = []
    for model in ("jev-9b", "jev-27b"):
        pair = arms[arms.model_deployment.eq(model)]
        equal = pair.overall_definition_sha256.nunique() == 1
        comparability.append(
            {
                "model": model,
                "same_ordered_cohort": True,
                "same_human_reference": True,
                "same_overall_eligibility_definition": equal,
                "interpretation": "Matched overall wording; descriptive within-cohort comparison"
                if equal
                else "Archived binary and hybrid wording differ; cannot isolate question-design effect",
            }
        )
    data = ComparisonScores(pd.concat(score_tables, ignore_index=True))
    candidates = select_recall_thresholds(data, settings.target_recall, basis="all_labelled")
    baseline = threshold_readouts(data, {name: [0.5] for name in arms.arm})
    ranking = ranking_readouts(data)
    # A common scored subset additionally exposes denominator differences between arms.
    complete_ids = (
        data.rows.pivot(index="study_id", columns="model_name", values="probability_true")
        .dropna()
        .index
    )
    common_rows = data.rows.copy(deep=True)
    common_rows["probability_true"] = common_rows.probability_true.where(
        common_rows.study_id.isin(complete_ids)
    )
    common_ranking = ranking_readouts(ComparisonScores(common_rows))
    common_ranking.table["common_scored_studies"] = len(complete_ids)
    common_ranking.table["metric_scope"] = (
        "Identical studies scored in all four arms; unavailable scores excluded"
    )
    coverage = pd.concat(validation, ignore_index=True).merge(
        arms[["arm", "approach", "model_deployment", "source_analysis", "prompt_version"]],
        on="arm",
        validate="one_to_one",
    )
    metrics = pd.concat(
        [
            candidates.assign(
                cutoff_source=f"Exploratory full-cohort {settings.target_recall:.0%}-target"
            ),
            baseline.assign(cutoff_source="0.5 baseline"),
        ],
        ignore_index=True,
    )
    derived = []
    for strategy, view in hybrid_strategy_views(hybrid).items():
        if strategy == "hybrid_direct":
            continue
        selected = select_recall_thresholds(view, settings.target_recall, basis="all_labelled")
        ranks = ranking_readouts(view).table
        derived.append(
            selected.merge(
                ranks[["model", "roc_auc", "pr_auc_trapezoidal", "average_precision"]],
                on="model",
                validate="one_to_one",
            ).assign(strategy=strategy)
        )
    tables = {
        "arms": arms,
        "comparability": pd.DataFrame(comparability),
        "coverage": coverage,
        "criteria": pd.DataFrame(criteria),
        "execution": pd.concat(execution, ignore_index=True),
        "question_summary": pd.concat(questions, ignore_index=True),
        "threshold_candidates": _identify(candidates, arms),
        "screening_metrics": _identify(metrics, arms),
        "ranking_metrics": _identify(ranking.table, arms),
        "common_scored_ranking": _identify(common_ranking.table, arms),
        "paired_positive_errors": _paired_errors(data, candidates),
        "hybrid_derived_scores": pd.concat(derived, ignore_index=True),
    }
    return ScreeningComparison(
        root, settings, settings_path, datasets, data, tables, ranking, candidates
    )


def reuse_or_run_comparison_arm(comparison, approach, model, *, execute=False, **run_options):
    """Reuse every existing archive, including partial ones; submit only an absent arm.

    An existing artifact directory is never resumed or overwritten here. Recovery
    of partial runs belongs to the original workflow with strict resume checks.
    """
    if approach not in comparison.datasets or model not in {"jev-9b", "jev-27b"}:
        raise PrerequisiteError("Select binary/hybrid and jev-9b/jev-27b explicitly.")
    dataset = comparison.datasets[approach]
    plan = next(p for p in dataset.plans if p.project.models[0].name == model)
    from jev_eval.runner import _run_directory

    directory = _run_directory(plan.project, plan.run_id)
    if directory.exists():
        if not (directory / "manifest.json").is_file():
            raise PrerequisiteError("An existing artifact directory needs review; no request sent.")
        return {
            "arm": _arm_label(approach, model),
            "action": "reuse_saved_run",
            "run_id": plan.run_id,
            "note": "No API request; use the coverage table to identify incomplete records.",
        }
    if not execute:
        return {
            "arm": _arm_label(approach, model),
            "action": "not_run",
            "run_id": plan.run_id,
            "note": "Execution flag is false; no API request or file write.",
        }
    if run_options.keys() & {"allow_paid_calls", "allow_writes"}:
        raise PrerequisiteError(
            "Use the explicit execute flag; YAML permissions are checked separately."
        )
    from jev_eval.environment import initialize_notebook
    from jev_eval.runner import plan_run, run_experiment

    workflow = (
        comparison.tables["arms"].set_index("arm").loc[_arm_label(approach, model), "workflow"]
    )
    project = initialize_notebook(comparison.root, workflow_path=workflow)
    live_plan = plan_run(project, dataset.states)
    return run_experiment(live_plan, allow_paid_calls=True, allow_writes=True, **run_options)


def export_screening_comparison(comparison, figures=None, *, allow_writes=False):
    """Reuse the aggregate report writer; original inference artifacts stay untouched."""
    from types import SimpleNamespace

    from jev_eval.environment import sha256_file
    from jev_eval.runner import _run_directory
    from jev_eval.summary import export_interim_summary

    sources = []
    for approach, data in comparison.datasets.items():
        for plan, result in zip(data.plans, data.results, strict=True):
            sources.append(
                {
                    "approach": approach,
                    "model": plan.project.models[0].name,
                    "run_id": plan.run_id,
                    "task_sha256": plan.task.fingerprint,
                    "original_fingerprint": result.manifest["fingerprint"] if result else None,
                    "predictions_sha256": sha256_file(
                        _run_directory(plan.project, plan.run_id) / "predictions.jsonl"
                    )
                    if result
                    else None,
                }
            )
    data = comparison.datasets["binary"]
    report = SimpleNamespace(
        data=data,
        settings=comparison.settings,
        tables=comparison.tables,
        methods="Four archived model/NOUL arms; no repeated inference. Different overall wording "
        "prevents isolating question-design effects. Cutoffs selected and assessed on the same "
        "labels are exploratory; unavailable scores remain in a manual-review queue. "
        "R1 is disabled; independent validation has not been performed.",
        provenance={
            "report_type": "four_arm_comparison",
            "inference_runs": sources,
            "settings": comparison.settings.model_dump(),
            "comparison_config_sha256": sha256_file(comparison.root / comparison.settings_path),
            "analysis_code_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("comparison.py", "evaluation.py", "reporting.py", "summary.py")
            },
            "states_sha256": data.plans[0].dataset_sha256,
            "reference_sha256": data.references.sha256,
            "validation_results_present": False,
        },
    )
    return export_interim_summary(report, figures, allow_writes=allow_writes)
