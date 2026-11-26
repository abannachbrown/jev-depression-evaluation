import math
from types import SimpleNamespace

import pandas as pd
import pytest

from jev_eval.config import PrerequisiteError
from jev_eval.data import load_jsonl, load_states, validate_states
from jev_eval.evaluation import (
    evaluate_predictions,
    load_reference_labels,
    ranking_readouts,
    select_recall_thresholds,
    threshold_readouts,
    threshold_tradeoff_curve,
)


def inputs(project):
    states = validate_states(load_states(project.evaluation.dataset, project.root)).require_valid()
    labels = load_reference_labels(
        project.root / project.workflow.evaluation.reference_labels,
        states=states,
        question=project.scoring.question,
    )
    predictions = load_jsonl(project.root / project.workflow.evaluation.demo_predictions)
    return states, labels, predictions


def evaluate(project, states, labels, predictions):
    return evaluate_predictions(
        predictions,
        labels,
        project.scoring,
        states=states,
        models=[model.name for model in project.models],
    )


def test_known_confusion_counts_and_undefined_precision(project):
    states, labels, predictions = inputs(project)
    table = evaluate(project, states, labels, predictions).table.set_index("model")
    assert table.loc["jev-9b", "sensitivity"] == 1
    assert table.loc["jev-9b", "specificity"] == 0
    assert table.loc["jev-9b", "precision"] == 0.5
    assert table.loc["jev-27b", "sensitivity"] == 0
    assert table.loc["jev-27b", "specificity"] == 1
    assert math.isnan(table.loc["jev-27b", "precision"])


def test_missing_entire_model_does_not_disappear(project):
    states, labels, predictions = inputs(project)
    table = evaluate(project, states, labels, predictions[:2]).table.set_index("model")
    assert table.loc["jev-27b", "missing"] == 2
    assert table.loc["jev-27b", "coverage"] == 0
    assert math.isnan(table.loc["jev-27b", "sensitivity"])


def test_invalid_and_failed_predictions_are_counted(project):
    states, labels, predictions = inputs(project)
    predictions[0]["status"] = "failed"
    predictions[1]["predictions"]["retain_for_fulltext"] = "true"
    table = evaluate(project, states, labels, predictions).table.set_index("model")
    assert table.loc["jev-9b", "failed"] == 1
    assert table.loc["jev-9b", "invalid"] == 1
    assert math.isnan(table.loc["jev-9b", "f1"])


def test_duplicates_and_mixed_modes_rejected(project):
    states, labels, predictions = inputs(project)
    with pytest.raises(PrerequisiteError, match="duplicate"):
        evaluate(project, states, labels, predictions + predictions[:1])
    predictions[0]["mode"] = "hosted"
    with pytest.raises(PrerequisiteError, match="mix"):
        evaluate(project, states, labels, predictions)


@pytest.fixture
def probability_data():
    # Three reference-positive studies, one unavailable. The 0.8 score is tied
    # across positive and negative studies and must be included as a whole.
    return SimpleNamespace(
        rows=pd.DataFrame(
            {
                "model_name": ["synthetic"] * 5,
                "human_label": [True, True, True, False, False],
                "probability_true": [0.8, 0.8, None, 0.8, 0.2],
            }
        )
    )


def test_recall_cutoff_is_highest_feasible_and_respects_ties(probability_data):
    candidate = select_recall_thresholds(probability_data, 0.5).iloc[0]
    assert candidate.status == "attainable"
    assert candidate.threshold == 0.8
    assert candidate.tp == 2 and candidate.fp == 1
    assert candidate.recall_cohort_lower_bound == pytest.approx(2 / 3)
    higher = threshold_readouts(probability_data, {"synthetic": [0.800001]}).iloc[0]
    assert higher.recall_cohort_lower_bound < 0.5


def test_unavailable_positives_do_not_disappear_or_get_model_credit(probability_data):
    all_labels = select_recall_thresholds(probability_data, 0.95)
    assert all_labels.status.iloc[0] == "unattainable_without_unscored_recovery"
    assert all_labels.threshold.isna().all()
    scored = select_recall_thresholds(probability_data, 0.95, basis="scored").iloc[0]
    assert scored.threshold == 0.8 and scored.recall_scored == 1
    assert scored.positive_labels_all == 3 and scored.positive_labels_scored == 2
    assert scored.unscored_positive_labels == 1
    assert scored.recall_cohort_lower_bound == pytest.approx(2 / 3)
    assert scored.recall_if_unscored_retained == 1
    assert scored.review_queue_with_unscored == 4
    assert scored.automatic_exclusions == 1
    assert scored.tn + scored.fp + scored.fn + scored.tp == 4


def test_exact_curve_matches_readouts_at_all_breakpoints(probability_data):
    curve = threshold_tradeoff_curve(probability_data)
    assert curve.threshold.tolist() == [0, 0.2, 0.8, 1]
    for row in curve.itertuples():
        direct = threshold_readouts(probability_data, {"synthetic": [row.threshold]}).iloc[0]
        for key in [
            "recall_scored",
            "recall_cohort_lower_bound",
            "precision_scored",
            "specificity_scored",
            "automatic_exclusion_fraction",
        ]:
            assert getattr(row, key) == pytest.approx(direct[key], nan_ok=True)
    assert curve.recall_scored.diff().dropna().le(0).all()
    assert curve.automatic_exclusion_fraction.diff().dropna().ge(0).all()


@pytest.mark.parametrize("target", [True, -0.1, 0, 1.1, float("nan")])
def test_invalid_recall_targets_are_rejected(probability_data, target):
    with pytest.raises(PrerequisiteError, match="target_recall"):
        select_recall_thresholds(probability_data, target)


def test_no_reference_positives_and_zero_score_boundary_are_explicit(probability_data):
    probability_data.rows.human_label = False
    assert select_recall_thresholds(probability_data).status.iloc[0] == "no_positive_labels"
    probability_data.rows.loc[0, ["human_label", "probability_true"]] = [True, 0.0]
    result = select_recall_thresholds(probability_data, 1.0).iloc[0]
    assert result.threshold == 0 and result.tp == 1


def test_screening_metrics_use_paired_counts_and_preserve_unavailable_queue(probability_data):
    row = threshold_readouts(probability_data, {"synthetic": [0.8]}).iloc[0]
    assert row.specificity_scored == 0.5  # TN=1, FP=1
    assert row.ppv_scored == row.precision_scored == pytest.approx(2 / 3)
    assert row.f2_scored == pytest.approx(10 / 11)  # TP=2, FN=0, FP=1
    assert row.wss95_scored == pytest.approx(1 / 4 - 0.05)
    assert math.isnan(row.wss95_cohort_conservative)  # Missing positive gets no credit.

    # With all three positives scored and retained, both scopes coincide.
    probability_data.rows.loc[2, "probability_true"] = 0.8
    row = threshold_readouts(probability_data, {"synthetic": [0.8]}).iloc[0]
    assert row.wss95_scored == pytest.approx(1 / 5 - 0.05)
    assert row.wss95_cohort_conservative == pytest.approx(row.wss95_scored)
    assert row.wss95_cohort_conservative == pytest.approx(
        0.95 - row.review_queue_with_unscored / len(probability_data.rows)
    )


def test_wss95_requires_recall_target_and_does_not_clip_negative_values():
    data = SimpleNamespace(
        rows=pd.DataFrame(
            {
                "model_name": ["synthetic"] * 40,
                "human_label": [True] * 20 + [False] * 20,
                "probability_true": [0.9] * 19 + [0.1] + [0.1] * 20,
            }
        )
    )
    row = threshold_readouts(data, {"synthetic": [0.9]}).iloc[0]
    assert row.recall_scored == 0.95
    assert row.f2_scored == pytest.approx(95 / 99)
    assert row.wss95_scored == pytest.approx(21 / 40 - 0.05)
    assert row.wss95_cohort_conservative == pytest.approx(row.wss95_scored)
    data.rows.loc[0, "probability_true"] = 0.1
    row = threshold_readouts(data, {"synthetic": [0.9]}).iloc[0]
    assert row.recall_scored == 0.9
    assert math.isnan(row.wss95_scored)
    assert math.isnan(row.wss95_cohort_conservative)
    row = threshold_readouts(data, {"synthetic": [0]}).iloc[0]
    assert row.wss95_scored == row.wss95_cohort_conservative == -0.05


def test_cohort_wss95_requires_reference_labels_and_denominators_remain_explicit(probability_data):
    probability_data.rows.human_label = probability_data.rows.human_label.astype("boolean")
    probability_data.rows.loc[2, "human_label"] = None
    row = threshold_readouts(probability_data, {"synthetic": [0.8]}).iloc[0]
    assert row.recall_scored == row.recall_cohort_lower_bound == 1
    assert not math.isnan(row.wss95_scored)
    assert math.isnan(row.wss95_cohort_conservative)
    row = threshold_readouts(probability_data, {"synthetic": [1]}).iloc[0]
    assert math.isnan(row.ppv_scored)
    assert row.f2_scored == 0
    assert math.isnan(row.wss95_scored)
    probability_data.rows.human_label = False
    row = threshold_readouts(probability_data, {"synthetic": [1]}).iloc[0]
    assert math.isnan(row.f2_scored)
    assert math.isnan(row.wss95_scored)


def test_feasible_cohort_wss_keeps_unscored_positive_in_review_workload():
    data = SimpleNamespace(
        rows=pd.DataFrame(
            {
                "model_name": ["synthetic"] * 40,
                "human_label": [True] * 20 + [False] * 20,
                "probability_true": [0.9] * 19 + [None] + [0.1] * 20,
            }
        )
    )
    row = threshold_readouts(data, {"synthetic": [0.9]}).iloc[0]
    assert row.recall_scored == 1
    assert row.recall_cohort_lower_bound == 0.95
    assert row.unscored_studies == 1 and row.review_queue_with_unscored == 20
    assert row.wss95_cohort_conservative == pytest.approx(20 / 40 - 0.05)
    assert row.wss95_scored == pytest.approx(20 / 39 - 0.05)


def test_ranking_metrics_handle_ties_and_exclude_unscored_positives(probability_data):
    result = ranking_readouts(probability_data)
    row = result.table.iloc[0]
    assert row.scored_labelled_pairs == 4
    assert row.positive_pairs == row.negative_pairs == 2
    assert row.unscored_studies == row.unscored_positive_labels == 1
    assert row.roc_auc == 0.75
    assert row.average_precision == pytest.approx(2 / 3)
    assert row.pr_auc_trapezoidal == pytest.approx(5 / 6)
    assert row.scored_pair_prevalence == 0.5
    assert row.status == "available"
    assert result.roc.false_positive_rate.iloc[0] == 0
    assert result.roc.true_positive_rate.iloc[-1] == 1
    assert result.precision_recall.recall.iloc[0] == 1
    assert result.precision_recall.recall.iloc[-1] == 0


def test_ranking_uses_continuous_scores_and_distinguishes_ap_from_pr_auc():
    data = SimpleNamespace(
        rows=pd.DataFrame(
            {
                "model_name": ["synthetic"] * 4,
                "human_label": [False, False, True, True],
                "probability_true": [0.1, 0.4, 0.35, 0.8],
            }
        )
    )
    row = ranking_readouts(data).table.iloc[0]
    assert row.roc_auc == 0.75
    assert row.average_precision == pytest.approx(5 / 6)
    assert row.pr_auc_trapezoidal == pytest.approx(19 / 24)
    # Monotone score transformations preserve rankings without selecting a cutoff.
    data.rows.probability_true = data.rows.probability_true**2
    transformed = ranking_readouts(data).table.iloc[0]
    assert transformed.roc_auc == row.roc_auc
    assert transformed.average_precision == row.average_precision


def test_ranking_missing_labels_are_not_negative_references(probability_data):
    probability_data.rows.human_label = probability_data.rows.human_label.astype("boolean")
    probability_data.rows.loc[0, "human_label"] = None
    row = ranking_readouts(probability_data).table.iloc[0]
    assert row.missing_human_labels == 1
    assert row.human_labelled == 4 and row.scored_labelled_pairs == 3
    assert row.positive_pairs == 1 and row.negative_pairs == 2
    assert row.unscored_positive_labels == 1


@pytest.mark.parametrize("no_scores", [False, True])
def test_ranking_unavailable_classes_do_not_produce_fake_curves(probability_data, no_scores):
    from jev_eval.reporting import plot_probability_curves

    probability_data.rows.human_label = False
    if no_scores:
        probability_data.rows.probability_true = float("nan")
    result = ranking_readouts(probability_data)
    row = result.table.iloc[0]
    assert row.status == (
        "no_scored_labelled_pairs" if no_scores else "requires_both_reference_classes"
    )
    assert math.isnan(row.roc_auc) and math.isnan(row.average_precision)
    assert math.isnan(row.pr_auc_trapezoidal)
    assert result.roc.empty and result.precision_recall.empty
    assert plot_probability_curves(result) is None
