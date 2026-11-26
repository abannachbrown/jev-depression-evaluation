"""Hybrid tests use invented studies and mocked HTTP only."""

import csv
import hashlib
import json
import shutil

import httpx
import pytest
import yaml
from conftest import ROOT, configure_v1_inputs

from jev_eval.backends.inference_sh import build_decision_input
from jev_eval.config import PrerequisiteError
from jev_eval.connections import (
    load_hybrid_connection,
    preview_connection,
    run_connection,
    with_synthetic_example,
)
from jev_eval.data import prepared_state_report
from jev_eval.screening import (
    execute_screening_model,
    export_screening_analysis,
    hybrid_strategy_views,
    load_screening_analysis,
    prepare_screening_campaign,
    require_hybrid_execution,
    screening_question_summary,
    screening_run_status,
    screening_setup_table,
)
from jev_eval.tasks import HYBRID_QUESTIONS, PRIMARY_DOMAINS


@pytest.fixture
def screening_root(project, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real HTTP is forbidden in screening tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setenv("INFERENCE_API_KEY", "synthetic-screening-only")
    root = project.root
    source = root / "data/private/invented-v1/states.jsonl"
    source.parent.mkdir(parents=True)
    studies = [
        {"study_id": f"{i:03d}", "title": f"Invented study {i}", "abstract": "Invented abstract"}
        for i in range(14)
    ]
    for row in studies[-2:]:
        row.update(title="", abstract="")
    source.write_text("".join(json.dumps(row) + "\n" for row in studies))
    reference = source.with_name("reference_labels.csv")
    with reference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_id", "Inclusion Status", "ITEM_ID"])
        writer.writerows((row["study_id"], int(i % 3 == 0), "") for i, row in enumerate(studies))
    configure_v1_inputs(root, source, reference)
    # Explicit archive-comparison profiles use the same invented inputs.
    for model in ("jev_9b", "jev_27b"):
        path = root / f"configs/evaluations/full_{model}.yaml"
        config = yaml.safe_load(path.read_text())
        config["dataset"].update(path=str(source.relative_to(root)), expected_count=None)
        path.write_text(yaml.safe_dump(config))
        path = root / f"configs/workflows/full_{model}.yaml"
        config = yaml.safe_load(path.read_text())
        config["evaluation"]["reference_labels"] = str(reference.relative_to(root))
        path.write_text(yaml.safe_dump(config))
    summary_path = root / "configs/evaluations/interim_summary.yaml"
    summary_config = yaml.safe_load(summary_path.read_text())
    summary_config["comparison_config"] = None  # Test the retained single-approach API below.
    summary_path.write_text(yaml.safe_dump(summary_config))
    shutil.copytree(
        ROOT / "src/jev_eval",
        root / "src/jev_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.fetch", "*.tmp"),
    )
    return root


def enable(root, approach="hybrid", phase="pilot"):
    for model in ("jev_9b", "jev_27b"):
        p = root / f"configs/workflows/{phase}_{approach}_v1_{model}.yaml"
        config = yaml.safe_load(p.read_text())
        config["execution"]["allow_paid_calls"] = True
        config["output"]["allow_writes"] = True
        if phase == "full":
            config["inference_sh_full"]["dry_run"] = False
        p.write_text(yaml.safe_dump(config))


def handler(calls, *, partial=False, throttle_once=False):
    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        if throttle_once and len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="synthetic throttle")
        questions = {q["id"]: {"noul": 0.8} for q in body["input"]["nouls"]}
        # Secondary scores must not enter min/mean primary combinations.
        for q in ("control_group", "eligible_outcomes"):
            if q in questions:
                questions[q] = {"noul": 0.0}
        if partial:
            questions.pop(next(iter(questions)))
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": f"synthetic-task-{len(calls)}",
                    "status": 10,
                    "output": {"model": body["app"], "input_tokens": 50, "nouls": questions},
                }
            },
        )

    return respond


def execute(campaign, model, calls, **options):
    return execute_screening_model(
        campaign,
        model,
        allow_paid_calls=True,
        allow_writes=True,
        transport=httpx.MockTransport(handler(calls, **options)),
        sleeper=lambda _: None,
    )


def test_setup_and_pilot_preview_are_read_only_and_profile_opt_in_is_explicit(screening_root):
    root = screening_root
    setup = screening_setup_table(root)
    assert len(setup) == 8 and not setup.paid_calls_enabled.any()
    assert not setup.writes_enabled.any()
    campaign = prepare_screening_campaign(root)
    assert campaign.approach == "hybrid" and campaign.phase == "pilot"
    plan = campaign.plans[0]
    assert len(plan.states) == 10 and set(plan.task.questions) == set(HYBRID_QUESTIONS)
    assert campaign.summary().question_evaluations.tolist() == [70, 70]
    assert not (root / "results").exists()
    full = prepare_screening_campaign(root, phase="full")
    report = prepared_state_report(full.plans[0].project)
    assert report.valid and report.total == 14 and len(report.require_valid()) == 14
    assert sum(issue["issue"] == "no citation text" for issue in report.issues) == 2
    with pytest.raises(PrerequisiteError, match="one decision question"):
        build_decision_input(plan.states[0], plan.task)
    body = build_decision_input(plan.states[0], plan.task, max_questions=7)
    assert {q["id"] for q in body["nouls"]} == set(HYBRID_QUESTIONS)
    with pytest.raises(PrerequisiteError, match="permission"):
        execute_screening_model(campaign, "jev-9b", allow_paid_calls=True, allow_writes=True)
    assert not (root / "results").exists()


def test_hybrid_connection_mock_preserves_seven_questions_and_one_post_gate(screening_root):
    connection = load_hybrid_connection(screening_root)
    assert connection.settings.execution.max_states == 1
    assert len(preview_connection(connection)["input"]["nouls"]) == 7
    result = run_connection(connection)
    assert result.metadata["mode"] == "mock" and result.metadata["status"] == "completed"
    assert set(result.response.probabilities) == set(HYBRID_QUESTIONS)
    assert result.directory is None and not (screening_root / "results").exists()
    with pytest.raises(PrerequisiteError, match="seven-question"):
        with_synthetic_example(
            connection,
            state={"title": "Invented", "abstract": "Invented"},
            question={"id": "one", "instructions": "Invented"},
        )


def test_pilot_preserves_all_answers_and_resumes_without_duplicate_successes(screening_root):
    enable(screening_root)
    campaign = prepare_screening_campaign(screening_root)
    calls = []
    first = execute(campaign, "jev-9b", calls)
    second = execute(campaign, "jev-27b", calls)
    assert len(calls) == 20
    assert [x["input"] for x in calls[:10]] == [x["input"] for x in calls[10:]]
    assert first.directory != second.directory
    assert (
        first.manifest["approach"] == "hybrid"
        and first.manifest["primary_question"] == "overall_eligibility"
    )
    assert all(set(row["probabilities"]) == set(HYBRID_QUESTIONS) for row in first.predictions)
    assert all(row["model_revision"] is None for row in first.predictions)
    execute(campaign, "jev-9b", calls)
    assert len(calls) == 20
    assert screening_run_status(campaign).successful.tolist() == [10, 10]
    scores = screening_question_summary(campaign.plans, (first, second))
    assert len(scores) == 14 and scores.scored_studies.eq(10).all()
    before = hashlib.sha256((first.directory / "manifest.json").read_bytes()).hexdigest()
    # Changing the task after planning must stop before a POST.
    campaign.plans[0].task.questions["depression_model"].instructions += " Changed."
    with pytest.raises(PrerequisiteError):
        execute(campaign, "jev-9b", calls)
    assert len(calls) == 20
    assert hashlib.sha256((first.directory / "manifest.json").read_bytes()).hexdigest() == before


def test_incomplete_hybrid_response_is_never_a_success(screening_root):
    enable(screening_root)
    campaign = prepare_screening_campaign(screening_root)
    calls = []
    result = execute(campaign, "jev-9b", calls, partial=True)
    assert len(calls) == 1
    assert result.predictions[0]["status"] == "failed"
    assert result.predictions[0]["error_code"] == "invalid_decision_output"
    assert list((result.directory / "raw").glob("*.body"))
    assert not result.predictions[0]["probabilities"]


def test_full_partial_archive_analysis_derived_rules_and_private_export(screening_root):
    root = screening_root
    enable(root, phase="full")
    campaign = prepare_screening_campaign(root, phase="full")
    calls = []
    result = execute(campaign, "jev-9b", calls, throttle_once=True)
    assert len(calls) == 13  # Twelve text-bearing studies plus one explicit 429 retry.
    assert sum(r["status"] == "not_inferred_no_text" for r in result.predictions) == 2
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in result.directory.rglob("*")
        if p.is_file()
    }
    analysis = load_screening_analysis(root, include_legacy=False)
    assert analysis.availability.status.tolist() == ["not_run", "not_run", "saved", "not_run"]
    data = analysis.data["hybrid_v1"]
    assert data.results[1] is None and len(data.rows) == 28
    assert data.coverage.valid_probabilities.tolist() == [12, 0]
    views = hybrid_strategy_views(data)
    for name in ("R2_minimum", "R3_mean"):
        scored = views[name].rows.probability_true.dropna()
        assert len(scored) == 12 and scored.eq(0.8).all()
    assert analysis.rule_tuning.status.eq("disabled_pending_protocol_review").sum() == 2
    assert analysis.rule_tuning.status.eq("requires_original_1251_tuning_split").sum() == 4
    from jev_eval.summary import load_interim_summary

    summary = load_interim_summary(root)
    assert summary.provenance["primary_profile"] == "hybrid_v1"
    assert summary.provenance["analysis_config"] == "configs/evaluations/analysis_hybrid_v1.yaml"
    assert summary.provenance["inference_runs"][1]["original_fingerprint"] is None
    assert not summary.tables["screening_metrics"].cutoff_source.str.contains("Illustrative").any()
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in result.directory.rglob("*")
        if p.is_file()
    }
    with pytest.raises(PrerequisiteError, match="write permission"):
        export_screening_analysis(analysis, allow_writes=True)
    analysis.settings.allow_writes = True
    folder = export_screening_analysis(analysis, allow_writes=True)
    import pandas as pd

    exported = pd.read_csv(folder / "question_scores.csv", dtype={"study_id": str})
    assert len(exported) == 14 * 2 * 7
    assert "000" in set(exported.study_id)
    assert exported.noul_score.notna().sum() == 12 * 7
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["settings"]["r1_policy"] == "disabled_pending_protocol_review"
    for filename, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest
    assert "synthetic-screening-only" not in (folder / "manifest.json").read_text()


def test_binary_v1_and_hybrid_use_identical_samples_but_separate_task_identities(screening_root):
    binary = prepare_screening_campaign(screening_root, approach="binary")
    hybrid = prepare_screening_campaign(screening_root, approach="hybrid")
    assert binary.plans[0].dataset_sha256 == hybrid.plans[0].dataset_sha256
    assert binary.plans[0].task.fingerprint != hybrid.plans[0].task.fingerprint
    assert (
        binary.plans[0].task.questions["retain_for_fulltext"]
        == hybrid.plans[0].task.questions["overall_eligibility"]
    )
    assert binary.plans[0].run_id != hybrid.plans[0].run_id
    assert len(PRIMARY_DOMAINS) == 4
    require_hybrid_execution(hybrid.plans[0])
    with pytest.raises(PrerequisiteError, match="hybrid-only"):
        require_hybrid_execution(binary.plans[0])


@pytest.mark.parametrize("notebook_name", ["04_evaluation", "05_study_summary"])
@pytest.mark.parametrize("saved", [True, False])
def test_analysis_notebooks_include_partial_hybrid_without_hiding_binary(
    screening_root, monkeypatch, notebook_name, saved
):
    import nbformat
    from nbclient import NotebookClient

    summary_path = screening_root / "configs/evaluations/interim_summary.yaml"
    config = yaml.safe_load(summary_path.read_text())
    config["comparison_config"] = "configs/evaluations/comparison.yaml"
    summary_path.write_text(yaml.safe_dump(config))
    enable(screening_root, phase="full")
    campaign = prepare_screening_campaign(screening_root, phase="full")
    calls = []
    if saved:
        execute(campaign, "jev-9b", calls)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    for name in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(name, str(screening_root / name.lower()))
    path = ROOT / "notebooks" / (notebook_name + ".ipynb")
    original = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport httpx\n"
            "import jev_eval.environment as environment_module\n"
            "import jev_eval.runner as runner_module\n"
            "def forbidden(*args, **kwargs):\n"
            "    raise AssertionError('Read-only analysis cannot make API requests')\n"
            "httpx.HTTPTransport.handle_request = forbidden\n"
            "httpx.AsyncHTTPTransport.handle_async_request = forbidden\n"
            "runner_module.create_backend = forbidden\nrunner_module.run_experiment = forbidden\n"
            f"environment_module.find_project_root = lambda: Path({str(screening_root)!r})\n"
        ),
    )
    expected = [0, 0, 12, 0] if saved else [0, 0, 0, 0]
    checks = (
        "assert len(comparison.tables['arms']) == 4\n"
        f"assert comparison.tables['coverage'].successful.tolist() == {expected!r}\n"
    )
    if notebook_name == "04_evaluation":
        checks += "assert not EXPORT_COMPARISON\n"
    else:
        checks += "assert summary.provenance['primary_profile'] == 'four_arm_comparison'\nassert len(performance) == 8\nassert not EXPORT_INTERIM_REPORT\n"
    notebook.cells.append(nbformat.v4.new_code_cell(checks))
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    ).execute()
    assert path.read_bytes() == original
