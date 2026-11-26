"""Four-arm comparisons reuse archives; every provider call here uses MockTransport."""

import csv
import hashlib
import json
import shutil

import httpx
import nbformat
import pytest
import yaml
from conftest import ROOT, configure_v1_inputs
from nbclient import NotebookClient

from jev_eval.comparison import (
    export_screening_comparison,
    load_screening_comparison,
    reuse_or_run_comparison_arm,
)
from jev_eval.config import PrerequisiteError, load_project_config
from jev_eval.data import load_full_dataset
from jev_eval.runner import plan_run, run_experiment
from jev_eval.summary import export_interim_summary, load_interim_summary


def hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def comparison_root(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real HTTP is forbidden in comparison tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    # Explicitly opted-in hosted unit tests use MockTransport. The notebook-only
    # demo guard is tested separately in fresh kernels with this variable set to 1.
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "0")
    monkeypatch.setenv("INFERENCE_API_KEY", "synthetic-comparison-only")
    root = project.root
    source = root / "data/private/comparison/states.jsonl"
    source.parent.mkdir(parents=True)
    studies = [
        {"study_id": f"{i:03d}", "title": f"Invented study {i}", "abstract": "Invented abstract"}
        for i in range(10)
    ]
    for row in studies[-2:]:
        row.update(title="", abstract="")
    source.write_text("".join(json.dumps(row) + "\n" for row in studies))
    reference = source.with_name("reference_labels.csv")
    with reference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_id", "Inclusion Status", "ITEM_ID"])
        writer.writerows((row["study_id"], int(i < 2), "") for i, row in enumerate(studies))
    configure_v1_inputs(root, source, reference, live=True)
    for model in ("jev_9b", "jev_27b"):
        p = root / f"configs/evaluations/full_{model}.yaml"
        c = yaml.safe_load(p.read_text())
        c["dataset"].update(path=str(source.relative_to(root)), expected_count=None)
        p.write_text(yaml.safe_dump(c))
        p = root / f"configs/workflows/full_{model}.yaml"
        c = yaml.safe_load(p.read_text())
        c["evaluation"]["reference_labels"] = str(reference.relative_to(root))
        c["execution"]["allow_paid_calls"] = True
        c["output"]["allow_writes"] = True
        c["inference_sh_full"]["dry_run"] = False
        p.write_text(yaml.safe_dump(c))
    shutil.copytree(
        ROOT / "src/jev_eval",
        root / "src/jev_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy(ROOT / "pyproject.toml", root / "pyproject.toml")
    return root


def save_arms(root, selected=None):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        index = int(body["input"]["state"]["study_id"])
        score = 0.85 if index < 2 else 0.15
        if body["app"].endswith("v1-1"):
            score += 0.02
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": f"invented-task-{len(calls)}",
                    "status": 10,
                    "output": {
                        "model": body["app"],
                        "input_tokens": 30,
                        "nouls": {q["id"]: {"noul": score} for q in body["input"]["nouls"]},
                    },
                }
            },
        )

    for approach in ("binary", "hybrid"):
        for model in ("jev_9b", "jev_27b"):
            if selected is not None and (approach, model) not in selected:
                continue
            name = f"full_{model}" if approach == "binary" else f"full_hybrid_v1_{model}"
            project = load_project_config(root, f"configs/workflows/{name}.yaml")
            plan = plan_run(project, load_full_dataset(project).states)
            run_experiment(
                plan,
                allow_paid_calls=True,
                allow_writes=True,
                transport=httpx.MockTransport(handler),
                sleeper=lambda _: None,
            )
    return calls


def test_four_arms_have_independent_identity_matching_cohort_and_visible_wording(
    comparison_root, monkeypatch
):
    calls = save_arms(comparison_root)
    assert len(calls) == 32
    before = hashes(comparison_root / "results/full")
    import jev_eval.runner as runner

    monkeypatch.setattr(
        runner, "run_experiment", lambda *a, **k: pytest.fail("Repeated paid inference")
    )
    comparison = load_screening_comparison(comparison_root)
    arms = comparison.tables["arms"]
    assert arms.arm.tolist() == [
        "binary / jev-9b",
        "binary / jev-27b",
        "hybrid / jev-9b",
        "hybrid / jev-27b",
    ]
    assert arms.questions.tolist() == [1, 1, 7, 7]
    assert arms.run_id.nunique() == 4
    assert not comparison.tables["comparability"].same_overall_eligibility_definition.any()
    assert comparison.tables["coverage"].successful.tolist() == [8] * 4
    assert comparison.tables["coverage"].not_inferred_no_text.tolist() == [2] * 4
    assert len(comparison.tables["ranking_metrics"]) == 4
    assert len(comparison.tables["screening_metrics"]) == 8
    assert len(comparison.tables["paired_positive_errors"]) == 6
    assert len(comparison.tables["hybrid_derived_scores"]) == 4
    for approach in ("binary", "hybrid"):
        for model in ("jev-9b", "jev-27b"):
            assert (
                reuse_or_run_comparison_arm(comparison, approach, model, execute=True)["action"]
                == "reuse_saved_run"
            )
    assert hashes(comparison_root / "results/full") == before


def test_missing_arms_stay_visible_without_fallback_or_fabricated_scores(comparison_root):
    save_arms(comparison_root, {("binary", "jev_9b"), ("hybrid", "jev_9b")})
    c = load_screening_comparison(comparison_root)
    assert c.tables["arms"].saved.tolist() == [True, False, True, False]
    assert c.tables["coverage"].successful.tolist() == [8, 0, 8, 0]
    assert (
        c.tables["ranking_metrics"].status.tolist() == ["available", "no_scored_labelled_pairs"] * 2
    )
    assert c.tables["common_scored_ranking"].common_scored_studies.eq(0).all()
    assert c.tables["common_scored_ranking"].roc_auc.isna().all()
    assert reuse_or_run_comparison_arm(c, "binary", "jev-27b")["action"] == "not_run"


def test_common_scored_subset_and_null_scores_preserve_missingness(comparison_root):
    save_arms(comparison_root)
    c = load_screening_comparison(comparison_root)
    result = c.datasets["hybrid"].results[0]
    p = result.directory / "predictions.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    p.write_text("".join(json.dumps(row) + "\n" for row in rows if row["study_id"] != "002"))
    c = load_screening_comparison(comparison_root)
    assert c.tables["coverage"].missing.tolist() == [0, 0, 1, 0]
    assert c.tables["ranking_metrics"].scored_labelled_pairs.tolist() == [8, 8, 7, 8]
    assert c.tables["common_scored_ranking"].scored_labelled_pairs.eq(7).all()
    assert c.tables["common_scored_ranking"].common_scored_studies.eq(7).all()
    assert (
        c.data.rows[c.data.rows.study_id.eq("002") & c.data.rows.model_name.eq("hybrid / jev-9b")]
        .probability_true.isna()
        .all()
    )


def test_different_reference_bundle_blocks_cross_approach_comparison(comparison_root):
    source = comparison_root / "data/private/comparison/reference_labels.csv"
    other = source.with_name("different_reference.csv")
    other.write_text(source.read_text().replace("000,1,", "000,0,"))
    for model in ("jev_9b", "jev_27b"):
        p = comparison_root / f"configs/workflows/full_hybrid_v1_{model}.yaml"
        c = yaml.safe_load(p.read_text())
        c["evaluation"]["reference_labels"] = str(other.relative_to(comparison_root))
        p.write_text(yaml.safe_dump(c))
    with pytest.raises(PrerequisiteError, match="identical ordered cohort and reference"):
        load_screening_comparison(comparison_root)


@pytest.mark.parametrize("approach", ["binary", "hybrid"])
@pytest.mark.parametrize("model", ["jev-9b", "jev-27b"])
def test_absent_arm_requires_opt_in_and_executes_only_chosen_arm(comparison_root, approach, model):
    c = load_screening_comparison(comparison_root)
    assert reuse_or_run_comparison_arm(c, approach, model)["action"] == "not_run"
    assert not (comparison_root / "results").exists()
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": f"invented-{len(calls)}",
                    "status": 10,
                    "output": {
                        "model": body["app"],
                        "input_tokens": 30,
                        "nouls": {q["id"]: {"noul": 0.8} for q in body["input"]["nouls"]},
                    },
                }
            },
        )

    result = reuse_or_run_comparison_arm(
        c,
        approach,
        model,
        execute=True,
        transport=httpx.MockTransport(handler),
        sleeper=lambda _: None,
    )
    assert len(calls) == 8
    expected_app = "autotrust/jev-9b" if model == "jev-9b" else "autotrust/jev-27b-v1-1"
    assert {call["app"] for call in calls} == {expected_app}
    assert {len(call["input"]["nouls"]) for call in calls} == {1 if approach == "binary" else 7}
    assert len(result.predictions) == 10
    assert (
        reuse_or_run_comparison_arm(c, approach, model, execute=True)["action"] == "reuse_saved_run"
    )
    assert len(calls) == 8


def test_summary_and_exports_include_all_four_sources_without_mutating_archives(comparison_root):
    save_arms(comparison_root)
    before = hashes(comparison_root / "results/full")
    summary = load_interim_summary(comparison_root)
    assert summary.provenance["primary_profile"] == "four_arm_comparison"
    assert len(summary.provenance["inference_runs"]) == 4
    assert len(summary.tables["study_flow"]) == 4
    assert len(summary.tables["ranking_metrics"]) == 4
    assert summary.provenance["same_overall_eligibility_definition"] is False
    assert summary.provenance["validation_results_present"] is False
    assert "cannot be attributed solely" in summary.methods
    with pytest.raises(PrerequisiteError):
        export_interim_summary(summary, allow_writes=True)
    with pytest.raises(PrerequisiteError):
        export_screening_comparison(summary.comparison, allow_writes=True)
    summary.settings.allow_writes = True
    output = export_interim_summary(summary, allow_writes=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["inference_runs"]) == 4
    assert len(list(csv.DictReader((output / "ranking_metrics.csv").open()))) == 4
    assert "Invented abstract" not in (output / "summary.html").read_text()
    summary.comparison.settings.allow_writes = True
    output = export_screening_comparison(summary.comparison, allow_writes=True)
    assert len(json.loads((output / "manifest.json").read_text())["inference_runs"]) == 4
    assert hashes(comparison_root / "results/full") == before


@pytest.mark.parametrize("notebook_name", ["03_run_models", "04_evaluation", "05_study_summary"])
@pytest.mark.parametrize("saved", [False, True])
def test_four_arm_notebooks_from_fresh_kernel_never_submit_or_change_archives(
    comparison_root,
    monkeypatch,
    notebook_name,
    saved,
):
    if saved:
        save_arms(comparison_root)
    before = hashes(comparison_root / "results/full")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    for name in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(name, str(comparison_root / name.lower()))
    path = ROOT / "notebooks" / (notebook_name + ".ipynb")
    original = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport httpx\n"
            "import jev_eval.environment as environment\nimport jev_eval.runner as runner\n"
            'def forbidden(*args, **kwargs):\n    raise AssertionError("No inference in four-arm notebook tests")\n'
            "httpx.HTTPTransport.handle_request = forbidden\nhttpx.AsyncHTTPTransport.handle_async_request = forbidden\n"
            "runner.run_experiment = forbidden\nrunner.create_backend = forbidden\n"
            f"environment.find_project_root = lambda: Path({str(comparison_root)!r})\n"
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            'assert len(comparison.tables["arms"]) == 4\n'
            'assert len(comparison.tables["ranking_metrics"]) == 4\n'
            + (
                "assert not RUN_BINARY_JEV_9B and not RUN_BINARY_JEV_27B and not RUN_HYBRID_JEV_9B and not RUN_HYBRID_JEV_27B\n"
                if notebook_name == "03_run_models"
                else ""
            )
        )
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    ).execute()
    assert path.read_bytes() == original
    assert hashes(comparison_root / "results/full") == before
