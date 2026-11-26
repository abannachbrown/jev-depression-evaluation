"""Full workflows are exercised only against invented data and mocked HTTP."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
import yaml
from conftest import configure_v1_inputs

import jev_eval.runner as runner
from jev_eval.config import PrerequisiteError, load_project_config
from jev_eval.data import load_full_dataset
from jev_eval.evaluation import (
    evaluation_ready_table,
    export_evaluation_ready,
    load_reference_csv,
)
from jev_eval.reporting import (
    full_run_metadata_table,
    full_run_plan_table,
    load_full_run_if_present,
    validate_full_results,
)
from jev_eval.runner import PilotPersistenceError, load_run, plan_run, run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real HTTP is forbidden in full-run tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setenv("INFERENCE_API_KEY", "synthetic-full-only")


@pytest.fixture
def full_projects(project):
    source = project.root / "data/private/full/states.jsonl"
    source.parent.mkdir(parents=True)
    rows = [
        {"study_id": f"00{i}", "title": f"Invented title {i}", "abstract": "Invented full abstract"}
        for i in range(5)
    ]
    rows[1]["abstract"] = ""
    rows[3].update(title="", abstract="")
    rows[4]["title"] = ""
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reference = source.parent / "reference_labels.csv"
    with reference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_id", "Inclusion Status", "ITEM_ID"])
        writer.writerows(
            (
                row["study_id"],
                "" if i == 2 else str(i % 2),
                "shared-item" if i < 2 else "" if i == 3 else str(i),
            )
            for i, row in enumerate(rows)
        )
    for model in ("jev_9b", "jev_27b"):
        path = project.root / f"configs/evaluations/full_{model}.yaml"
        settings = yaml.safe_load(path.read_text())
        settings["dataset"].update(path=str(source.relative_to(project.root)), expected_count=None)
        path.write_text(yaml.safe_dump(settings))
        path = project.root / f"configs/workflows/full_{model}.yaml"
        settings = yaml.safe_load(path.read_text())
        settings["execution"]["allow_paid_calls"] = True
        settings["output"]["allow_writes"] = True
        settings["inference_sh_full"]["dry_run"] = False
        settings["evaluation"]["reference_labels"] = str(reference.relative_to(project.root))
        path.write_text(yaml.safe_dump(settings))
    configure_v1_inputs(project.root, source, reference, live=True)
    shutil.copytree(
        ROOT / "src/jev_eval",
        project.root / "src/jev_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.fetch", "*.tmp"),
    )
    # Retain explicit coverage of the older single-approach summary API.
    # Four-arm default reporting is covered in test_comparison.py.
    summary_config = project.root / "configs/evaluations/interim_summary.yaml"
    summary_settings = yaml.safe_load(summary_config.read_text())
    summary_settings["comparison_config"] = None
    summary_config.write_text(yaml.safe_dump(summary_settings))
    return tuple(
        load_project_config(project.root, f"configs/workflows/full_{model}.yaml")
        for model in ("jev_9b", "jev_27b")
    )


def plans(projects):
    cohort = load_full_dataset(projects[0])
    return cohort, *(plan_run(project, cohort.states) for project in projects)


def successful(calls, revision=None):
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        output = {
            "model": "autotrust/JEV-9B"
            if payload["app"] == "autotrust/jev-9b"
            else "autotrust/AutoTrust-JEV-27B-v1.1",
            "input_tokens": 55,
            "nouls": {"retain_for_fulltext": {"noul": 0.75}},
        }
        if revision is not None:
            output["model_revision"] = revision
        return httpx.Response(
            200, json={"data": {"id": f"synthetic-{len(calls)}", "status": 10, "output": output}}
        )

    return handler


def execute(plan, handler, sleeper=lambda _delay: None, progress=None):
    return run_experiment(
        plan,
        allow_paid_calls=True,
        allow_writes=True,
        transport=httpx.MockTransport(handler),
        sleeper=sleeper,
        progress=progress,
    )


def test_dry_plan_counts_all_records_without_creating_files_or_client(full_projects, monkeypatch):
    cohort, p9, p27 = plans(full_projects)
    assert cohort.manifest["counts"] == {
        "records": 5,
        "api_eligible": 4,
        "not_inferred_no_text": 1,
        "duplicate_ids": 0,
        "missing_ids": 0,
        "missing_titles": 2,
        "missing_abstracts": 2,
    }
    full_projects[0].workflow.inference_sh_full.dry_run = True
    table = full_run_plan_table(p9, p27)
    assert list(table.records) == [5, 5]
    assert list(table.logical_requests) == [4, 4]
    assert list(table.pending_api_studies) == [4, 4]
    assert not (full_projects[0].root / "results").exists()
    monkeypatch.setattr(
        runner, "create_backend", lambda *args, **kwargs: pytest.fail("Client created")
    )
    with pytest.raises(PrerequisiteError, match="dry_run"):
        execute(p9, successful([]))
    assert not (full_projects[0].root / "results").exists()


def test_both_models_process_full_cohort_and_resume_without_duplicate_successes(full_projects):
    cohort, p9, p27 = plans(full_projects)
    calls, progress = [], []
    r9 = execute(p9, successful(calls), progress=progress.append)
    r27 = execute(p27, successful(calls))
    assert len(calls) == 8
    assert [call["input"] for call in calls[:4]] == [call["input"] for call in calls[4:]]
    assert r9.directory != r27.directory
    assert progress[0]["recorded"] == 0 and progress[-1]["recorded"] == 5
    assert full_run_plan_table(p9, p27).pending_api_studies.tolist() == [0, 0]
    for plan, result in [(p9, r9), (p27, r27)]:
        check = validate_full_results(plan, result).iloc[0]
        assert check.successful == 4 and check.not_inferred_no_text == 1 and check.missing == 0
        assert check.unexpected_returned_model == 0 and check.name_differences == 4
        assert result.manifest["status"] == "completed_with_exceptions"
        assert result.manifest["pending_request"] is None
        assert result.manifest["total_attempts"] == 4
        assert result.manifest["configuration_sha256"]
        assert result.manifest["source_sha256"]
        assert result.manifest["full_cohort"]["states_sha256"] == cohort.manifest["states_sha256"]
        assert "ordered_study_ids" not in result.manifest["full_cohort"]
        frozen = plan.project.root / result.manifest["cohort_manifest"]
        assert json.loads(frozen.read_text()) == cohort.manifest
        assert (
            hashlib.sha256(frozen.read_bytes()).hexdigest()
            == result.manifest["cohort_manifest_sha256"]
        )
        assert result.manifest["screening_questions"] == plan.task.model_dump()
        assert all(row.get("model_revision") is None for row in result.predictions)
        serialized = json.dumps(result.responses)
        assert "Invented full abstract" not in serialized
        assert "normalized" not in serialized and "synthetic-full-only" not in serialized
        for entry in result.responses:
            receipt = json.loads((result.directory / entry["receipt"]).read_text())
            body = (result.directory / entry["response_body"]).read_bytes()
            assert hashlib.sha256(body).hexdigest() == entry["response_sha256"]
            assert "raw" not in receipt["normalized"]
        before = (result.directory / "predictions.jsonl").read_bytes()
        execute(plan, successful(calls))
        assert (result.directory / "predictions.jsonl").read_bytes() == before
        assert load_full_run_if_present(plan).manifest["total_attempts"] == 4
    assert len(calls) == 8


def test_429_retry_after_is_recorded_with_bounded_attempt_history(full_projects):
    _, plan, _ = plans(full_projects)
    calls, delays = [], []
    ok = successful(calls)
    counter = 0

    def handler(request):
        nonlocal counter
        counter += 1
        if counter < 3:
            calls.append(json.loads(request.content))
            return httpx.Response(
                429, json={"error": "synthetic throttle"}, headers={"Retry-After": "10"}
            )
        return ok(request)

    result = execute(plan, handler, sleeper=delays.append)
    assert result.manifest["total_attempts"] == 6
    assert result.predictions[0]["attempts"] == 3
    assert len(result.failures) == 2
    assert len(result.responses) == 6
    assert result.responses[0]["retry_after_seconds"] == 10
    assert max(delays) > 9
    assert len({row["study_id"] for row in result.predictions}) == 5


def test_interrupted_backoff_is_safe_to_resume_without_resetting_attempts(full_projects):
    _, plan, _ = plans(full_projects)
    calls = []

    def throttle(request):
        calls.append(request)
        return httpx.Response(429, text="synthetic throttle", headers={"Retry-After": "20"})

    def interrupt(_delay):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(plan, throttle, sleeper=interrupt)
    saved = load_run(plan.project, plan.run_id)
    assert saved.manifest["pending_request"] is None
    assert saved.manifest["retry_not_before"]
    assert saved.manifest["total_attempts"] == 1
    result = execute(plan, successful([]))
    assert result.predictions[0]["attempts"] == 2
    assert result.manifest["total_attempts"] == 5
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["timeout", "unfinished", "server", "malformed"])
def test_ambiguous_outcomes_stop_and_never_automatically_repost(full_projects, failure):
    _, plan, _ = plans(full_projects)
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if failure == "unfinished":
            return httpx.Response(200, json={"data": {"id": "pending-full", "status": 2}})
        return httpx.Response(500 if failure == "server" else 200, text="synthetic invalid body")

    result = execute(plan, handler)
    assert result.manifest["status"] == "needs_review"
    assert result.manifest["pending_request"]
    assert result.predictions[0]["status"] == "unknown"
    with pytest.raises(PrerequisiteError, match="unknown outcome"):
        execute(plan, handler)
    assert len(calls) == 1


def test_record_errors_continue_and_repeated_errors_pause(full_projects):
    _, plan, _ = plans(full_projects)
    calls = []
    ok = successful(calls)

    def one_bad(request):
        if not calls:
            calls.append(json.loads(request.content))
            return httpx.Response(422, json={"error": "synthetic input error"})
        return ok(request)

    result = execute(plan, one_bad)
    assert len(result.predictions) == 5 and result.predictions[0]["status"] == "failed"
    assert len(calls) == 4
    other_plan = plans(full_projects)[2]
    result = execute(other_plan, lambda request: httpx.Response(422, text="synthetic bad input"))
    assert result.manifest["status"] == "paused_consecutive_failures"
    assert result.manifest["total_attempts"] == 3


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_system_errors_pause_and_rate_limit_retries_are_capped(full_projects, status):
    _, plan, _ = plans(full_projects)
    result = execute(plan, lambda _r: httpx.Response(status, text="synthetic error"))
    assert result.manifest["status"] == "paused_after_error"
    assert result.manifest["total_attempts"] == (3 if status == 429 else 1)
    assert result.manifest["pending_request"] is None


def test_attempt_budget_persists_and_terminal_results_are_not_repeated(full_projects):
    project = full_projects[0]
    project.workflow.execution.max_total_attempts = 5
    _, plan, _ = plans(full_projects)
    calls = []
    ok = successful(calls)

    def handler(request):
        if len(calls) < 2:
            calls.append(json.loads(request.content))
            return httpx.Response(429, text="synthetic throttle")
        return ok(request)

    first = execute(plan, handler)
    assert first.manifest["status"] == "attempt_budget_exhausted"
    assert len(calls) == 5
    project.workflow.execution.max_total_attempts = 6
    second = execute(plan_run(project, load_full_dataset(project).states), handler)
    assert second.manifest["total_attempts"] == 6
    assert len(calls) == 6


def test_csv_join_preserves_ids_labels_missingness_and_exports_checksums(full_projects):
    cohort, p9, p27 = plans(full_projects)
    r9 = execute(p9, successful([], revision="synthetic-revision"))
    r27 = execute(p27, successful([]))
    references = load_reference_csv(p9.project, cohort.states)
    assert references.counts["shared_item_id_groups"] == 1
    assert references.counts["missing_human_labels"] == 1
    assert references.table.study_id.iloc[0] == "000"
    table = evaluation_ready_table(p9, r9, p27, r27, references)
    assert len(table) == 10
    assert table.probability_true.notna().sum() == 8
    assert table.human_label.isna().sum() == 2
    assert table.ITEM_ID.isna().sum() == 2
    assert not table.duplicated(["study_id", "model_name"]).any()
    meta = full_run_metadata_table((p9, p27), (r9, r27))
    assert meta.provider_reported_revisions.iloc[0] == ["synthetic-revision"]
    assert not meta.immutable_revision_verified.any()
    destination = export_evaluation_ready(p9, r9, p27, r27, references, allow_writes=True)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["rows"] == 10 and manifest["missing_probability_rows"] == 2
    assert (
        manifest["csv_sha256"]
        == hashlib.sha256((destination / "evaluation_ready.csv").read_bytes()).hexdigest()
    )
    with (destination / "evaluation_ready.csv").open() as stream:
        assert next(csv.DictReader(stream))["study_id"] == "000"


@pytest.mark.parametrize("change", ["duplicate", "bad_label", "unexpected"])
def test_reference_csv_errors_block_the_join(full_projects, change):
    cohort, plan, _ = plans(full_projects)
    path = plan.project.root / plan.project.workflow.evaluation.reference_labels
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "bad_label":
        rows[0]["Inclusion Status"] = "maybe"
    else:
        rows[0]["study_id"] = "unexpected"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(PrerequisiteError):
        load_reference_csv(plan.project, cohort.states)


@pytest.mark.parametrize("change", ["duplicate", "probability", "task_id", "model", "fingerprint"])
def test_malformed_outputs_are_reported_and_block_export(full_projects, change):
    cohort, p9, p27 = plans(full_projects)
    result = execute(p9, successful([]))
    if change == "duplicate":
        result.predictions.append(result.predictions[0])
    elif change == "probability":
        result.predictions[0]["probabilities"]["retain_for_fulltext"] = float("nan")
    elif change == "task_id":
        result.predictions[0]["request_id"] = None
    elif change == "model":
        result.predictions[0]["returned_model"] = "wrong/model"
    else:
        result.manifest["fingerprint"] = "different"
    with pytest.raises(PrerequisiteError, match="integrity"):
        evaluation_ready_table(p9, result, p27, None, load_reference_csv(p9.project, cohort.states))


def test_storage_failure_retains_exchange_and_pending_marker(full_projects, monkeypatch):
    _, plan, _ = plans(full_projects)

    def failed(*args, **kwargs):
        raise OSError("synthetic storage failure")

    monkeypatch.setattr(runner, "_save_pilot_receipt", failed)
    with pytest.raises(PilotPersistenceError) as error:
        execute(plan, successful([]))
    assert error.value.result._pending_exchange.status_code == 200
    assert load_run(plan.project, plan.run_id).manifest["pending_request"]


@pytest.mark.parametrize("change", ["duplicate", "extra_label", "missing_field", "wrong_type"])
def test_full_dataset_strict_fields_and_ids(full_projects, change):
    project = full_projects[0]
    path = project.root / project.evaluation.dataset.path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "duplicate":
        rows[1]["study_id"] = rows[0]["study_id"]
    elif change == "extra_label":
        rows[0]["human_label"] = True
    elif change == "missing_field":
        del rows[0]["title"]
    else:
        rows[0]["abstract"] = 4
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PrerequisiteError):
        load_full_dataset(project)


def test_full_commit_interruption_retains_pending_before_terminal_row_is_durable(
    full_projects, monkeypatch
):
    _, plan, _ = plans(full_projects)
    original = runner._append_run_records

    def interrupted(result):
        if result.predictions:
            assert result.manifest["pending_request"]
            raise KeyboardInterrupt
        return original(result)

    monkeypatch.setattr(runner, "_append_run_records", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(plan, successful([]))
    assert load_run(plan.project, plan.run_id).manifest["pending_request"]


def test_pending_only_interruption_is_explicitly_unknown_in_validation_and_export(full_projects):
    cohort, p9, p27 = plans(full_projects)

    def interrupted(_request):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute(p9, interrupted)
    saved = load_full_run_if_present(p9)
    assert not saved.predictions
    check = validate_full_results(p9, saved).iloc[0]
    assert check.unknown == 1 and check.missing == 4
    table = evaluation_ready_table(
        p9, saved, p27, None, load_reference_csv(p9.project, cohort.states)
    )
    first = table[(table.study_id == "000") & (table.model_name == "jev-9b")].iloc[0]
    assert first.inference_status == "unknown"
    assert first.error_code == "interrupted_outcome_unknown"


def completed_analysis_inputs(full_projects):
    _, p9, p27 = plans(full_projects)
    return execute(p9, successful([])), execute(p27, successful([]))


def test_read_only_analysis_loads_all_labels_and_original_inference_provenance(
    full_projects, monkeypatch
):
    from jev_eval.evaluation import evaluate_saved_probabilities, load_saved_full_evaluation

    completed_analysis_inputs(full_projects)
    root = full_projects[0].root
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (root / "results/full").rglob("*")
        if path.is_file()
    }
    # Read-only analysis remains possible after source changes; no new inference
    # fingerprint is manufactured or written into either original run.
    (root / "src/jev_eval/evaluation.py").write_text("# changed analysis implementation\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must never construct a backend or run inference")

    monkeypatch.setattr(runner, "run_experiment", forbidden)
    monkeypatch.setattr(runner, "create_backend", forbidden)
    data = load_saved_full_evaluation(root)
    assert len(data.states) == 5 and len(data.reference_labels) == 4
    assert len(data.rows) == 10
    assert data.coverage.labelled.tolist() == [4, 4]
    assert data.coverage.valid_probabilities.tolist() == [3, 3]
    assert data.coverage.not_inferred_no_text.tolist() == [1, 1]
    assert data.coverage.missing_human_labels.tolist() == [1, 1]
    assert data.coverage.coverage.tolist() == [0.75, 0.75]
    assert all(type(value) is bool for value in data.reference_labels.values())
    with pytest.raises(PrerequisiteError, match="decision_threshold"):
        evaluate_saved_probabilities(data)
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def test_explicit_threshold_is_inclusive_and_partial_metrics_are_explicit(full_projects):
    from jev_eval.evaluation import evaluate_saved_probabilities, load_saved_full_evaluation

    completed_analysis_inputs(full_projects)
    data = load_saved_full_evaluation(full_projects[0].root)
    data.settings.decision_threshold = 0.75
    data.settings.threshold_note = "Predetermined synthetic test cutoff"
    headline = evaluate_saved_probabilities(data).table
    assert headline.labelled.tolist() == [4, 4]
    assert headline.valid.tolist() == [3, 3]
    assert headline.tp.tolist() == [1, 1]
    assert headline.fp.tolist() == [2, 2]
    assert headline.sensitivity.isna().all()  # Textless labelled record remains visible.
    assert headline.failed.tolist() == [0, 0]
    assert headline.not_inferred_no_text.tolist() == [1, 1]
    data.settings.report_partial_metrics = True
    partial = evaluate_saved_probabilities(data).table
    assert partial.sensitivity.tolist() == [1, 1]
    assert partial.precision.tolist() == [pytest.approx(1 / 3)] * 2
    assert partial.metric_scope.str.contains("valid labelled pairs only").all()
    data.settings.decision_threshold = 0.7501
    assert evaluate_saved_probabilities(data).table.tp.tolist() == [0, 0]


@pytest.mark.parametrize("corruption", ["dataset", "predictions", "cohort"])
def test_saved_evaluation_rejects_changed_scientific_inputs_or_corrupt_outputs(
    full_projects, corruption
):
    from jev_eval.evaluation import load_saved_full_evaluation

    completed_analysis_inputs(full_projects)
    project = full_projects[0]
    if corruption == "dataset":
        path = project.root / project.evaluation.dataset.path
        path.write_text(path.read_text() + "\n")
    elif corruption == "cohort":
        path = project.root / project.workflow.inference_sh_full.cohort_manifest
        path.write_text(path.read_text() + "\n")
    else:
        path = (
            project.root
            / project.workflow.output.root
            / project.workflow.output.run_id
            / "predictions.jsonl"
        )
        path.write_text(path.read_text() + path.read_text().splitlines()[0] + "\n")
    with pytest.raises(PrerequisiteError):
        load_saved_full_evaluation(project.root)


def test_analysis_export_is_opt_in_and_records_threshold_and_original_inference(full_projects):
    from jev_eval.evaluation import export_saved_evaluation, load_saved_full_evaluation

    completed_analysis_inputs(full_projects)
    data = load_saved_full_evaluation(full_projects[0].root)
    with pytest.raises(PrerequisiteError, match="permission"):
        export_saved_evaluation(data, None, allow_writes=True)
    data.settings.allow_writes = True
    folder = export_saved_evaluation(data, None, allow_writes=True)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["reference_labelled_studies"] == 4
    assert manifest["analysis_settings"]["decision_threshold"] is None
    assert len(manifest["inference_runs"]) == 2
    assert (folder / "coverage.csv").is_file()
    assert not (folder / "summary.csv").exists()


@pytest.mark.parametrize("threshold", [None, 0.75])
def test_full_evaluation_notebook_uses_saved_real_format_data_without_any_api(
    full_projects, monkeypatch, threshold
):
    import nbformat
    from nbclient import NotebookClient

    completed_analysis_inputs(full_projects)
    root = full_projects[0].root
    config_path = root / "configs/evaluations/analysis_full.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["decision_threshold"] = threshold
    config["report_partial_metrics"] = threshold is not None
    config["threshold_note"] = "Predetermined synthetic test threshold"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for name in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(name, str(root / name.lower()))
    path = ROOT / "notebooks/04_evaluation.ipynb"
    original = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import httpx\n"
            "import jev_eval.environment as environment_module\n"
            "import jev_eval.runner as runner_module\n"
            f"test_root = Path({str(root)!r})\n"
            "def forbidden(*args, **kwargs):\n"
            "    raise AssertionError('Evaluation cannot call an API or construct a client')\n"
            "httpx.HTTPTransport.handle_request = forbidden\n"
            "runner_module.run_experiment = forbidden\n"
            "runner_module.create_backend = forbidden\n"
            "environment_module.find_project_root = lambda: test_root\n"
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert len(comparison.tables['arms']) == 4\n"
            "assert len(comparison.datasets['binary'].states) == 5\n"
            "assert len(comparison.datasets['binary'].reference_labels) == 4\n"
            "assert comparison.tables['coverage'].successful.tolist() == [4, 4, 0, 0]\n"
            "assert len(performance) == 8 and not EXPORT_COMPARISON\n"
            "assert {'specificity_scored', 'ppv_scored', 'f2_scored', 'wss95_scored', "
            "'wss95_cohort_conservative'} <= set(performance.columns)\n"
            "assert performance.wss95_cohort_conservative.isna().all()\n"
        )
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    ).execute()
    assert path.read_bytes() == original


def test_interim_summary_preserves_archives_and_marks_validation_as_future(
    full_projects, monkeypatch
):
    from jev_eval.summary import load_interim_summary

    completed_analysis_inputs(full_projects)
    root = full_projects[0].root
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "results").rglob("*")
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("An interim summary cannot construct a backend or run inference")

    monkeypatch.setattr(runner, "create_backend", forbidden)
    monkeypatch.setattr(runner, "run_experiment", forbidden)
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    report = load_interim_summary(root)
    assert report.tables["study_flow"].source_studies.tolist() == [5, 5]
    assert report.tables["study_flow"].probabilities_available.tolist() == [4, 4]
    assert report.tables["study_flow"].scored_labelled_pairs.tolist() == [3, 3]
    assert report.tables["study_flow"].unscored_manual_queue.tolist() == [1, 1]
    assert report.tables["reference_summary"].human_retain.iloc[0] == 2
    assert "not available" in report.tables["audit_summary"].status.iloc[0]
    assert report.tables["screening_metrics"].wss95_cohort_conservative.isna().all()
    assert report.tables["ranking_metrics"].scored_labelled_pairs.tolist() == [3, 3]
    assert report.tables["ranking_metrics"].roc_auc.tolist() == [0.5, 0.5]
    assert report.tables["ranking_metrics"].unscored_positive_labels.tolist() == [1, 1]
    paired = report.tables["paired_positive_errors"].iloc[0]
    assert paired.human_retain_total == 2 and paired.both_scored == 1
    assert paired.unavailable_in_either_model == 1 and paired.neither_missed == 1
    assert report.settings.planned_validation_records == 2000
    assert report.settings.validation_status == "planned_after_colleague_review"
    assert report.provenance["validation_results_present"] is False
    assert report.data.settings.decision_threshold is None
    assert all(
        row["original_fingerprint"] == result.manifest["fingerprint"]
        for row, result in zip(
            report.provenance["inference_runs"], report.data.results, strict=True
        )
    )
    assert before == {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "results").rglob("*")
        if p.is_file()
    }


def test_interim_summary_verifies_preparation_audit_and_exports_aggregates(full_projects):
    from jev_eval.summary import export_interim_summary, load_interim_summary

    root = full_projects[0].root
    source = root / "data/private/full/states.jsonl"
    reference = source.with_name("reference_labels.csv")
    conversion = {
        "source": {"sha256": hashlib.sha256(b"synthetic workbook").hexdigest()},
        "validation": {"missing_abstracts": 2, "invalid_labels": 0},
        "records": {"audit_entries": 2},
        "audit": {
            "unique_study_ids": 1,
            "label_disagreements": 1,
            "main_label_mismatches": 0,
            "unknown_study_ids": 0,
            "policy": "report_only",
            "adjudication_applied": False,
            "issue_counts": {"Synthetic label disagreement": 1},
        },
        "output_sha256": {
            "states.jsonl": hashlib.sha256(source.read_bytes()).hexdigest(),
            "reference_labels.csv": hashlib.sha256(reference.read_bytes()).hexdigest(),
        },
    }
    source.with_name("manifest.json").write_text(json.dumps(conversion))
    completed_analysis_inputs(full_projects)
    config = root / "configs/evaluations/interim_summary.yaml"
    settings = yaml.safe_load(config.read_text())
    settings["allow_writes"] = True
    config.write_text(yaml.safe_dump(settings))
    report = load_interim_summary(root)
    assert report.tables["audit_summary"].label_disagreements.iloc[0] == 1
    assert not report.tables["audit_summary"].adjudication_applied.iloc[0]
    with pytest.raises(PrerequisiteError, match="allow_writes"):
        export_interim_summary(report)
    folder = export_interim_summary(report, allow_writes=True)
    assert folder.is_relative_to(root / "results/summary/interim")
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    assert (folder / "summary.html").is_file()
    assert (folder / "ranking_metrics.csv").is_file()
    assert "non-interpolated" in manifest["ranking_definitions"]["average_precision"]
    assert manifest["validation_results_present"] is False
    for path in folder.iterdir():
        text = path.read_text()
        assert "Invented title" not in text and "Invented full abstract" not in text
        assert "synthetic-full-only" not in text
        assert "synthetic-1" not in text  # No provider task IDs.
    # A valid but altered reference file must not be presented as the archived conversion.
    reference.write_text(reference.read_text().replace("shared-item", "changed-item"))
    with pytest.raises(PrerequisiteError, match="conversion/audit"):
        load_interim_summary(root)


def test_interim_summary_invalid_config_and_export_permissions(full_projects):
    from jev_eval.summary import export_interim_summary, load_interim_summary

    completed_analysis_inputs(full_projects)
    root = full_projects[0].root
    report = load_interim_summary(root)
    with pytest.raises(PrerequisiteError, match="allow_writes"):
        export_interim_summary(report, allow_writes=True)
    config = root / "configs/evaluations/interim_summary.yaml"
    settings = yaml.safe_load(config.read_text())
    settings["illustrative_cutoffs"] = {"unexpected-model": 0.5}
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(PrerequisiteError, match="both archived models"):
        load_interim_summary(root)
    settings["target_recall"] = True
    config.write_text(yaml.safe_dump(settings))
    with pytest.raises(PrerequisiteError, match="target_recall"):
        load_interim_summary(root)


def test_interim_summary_notebook_from_fresh_kernel_without_api_or_writes(
    full_projects, monkeypatch
):
    import nbformat
    from nbclient import NotebookClient

    from jev_eval.repository import check_blob

    completed_analysis_inputs(full_projects)
    root = full_projects[0].root
    settings_path = root / "configs/evaluations/interim_summary.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    settings["comparison_config"] = "configs/evaluations/comparison.yaml"
    settings_path.write_text(yaml.safe_dump(settings))
    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    for name in ("IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "MPLCONFIGDIR"):
        monkeypatch.setenv(name, str(root / name.lower()))
    path = ROOT / "notebooks/05_study_summary.ipynb"
    original = path.read_bytes()
    assert not check_blob("notebooks/05_study_summary.ipynb", original)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport httpx\n"
            "import jev_eval.environment as environment_module\n"
            "import jev_eval.runner as runner_module\n"
            "def forbidden(*args, **kwargs):\n"
            "    raise AssertionError('Summary cannot call an API or construct a client')\n"
            "httpx.HTTPTransport.handle_request = forbidden\n"
            "httpx.AsyncHTTPTransport.handle_async_request = forbidden\n"
            "runner_module.run_experiment = forbidden\nrunner_module.create_backend = forbidden\n"
            f"environment_module.find_project_root = lambda: Path({str(root)!r})\n"
        ),
    )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "assert summary.settings.planned_validation_records == 2000\n"
            "assert summary.provenance['validation_results_present'] is False\n"
            "assert len(performance) == 8\nassert not EXPORT_INTERIM_REPORT\n"
            "assert comparison.tables['ranking_metrics'].scored_labelled_pairs.tolist() == [3, 3, 0, 0]\n"
            "assert comparison.tables['ranking_metrics'].roc_auc.iloc[:2].tolist() == [0.5, 0.5]\n"
            "assert 'four-arm-roc-pr' in figures\n"
            "assert not summary.settings.allow_writes\n"
            "assert not (summary.data.plans[0].project.root / 'results/summary').exists()\n"
        )
    )
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    ).execute()
    assert path.read_bytes() == original


def test_config_change_after_planning_is_rejected_before_http(full_projects):
    _, plan, _ = plans(full_projects)
    path = plan.project.root / plan.project.workflow_path
    path.write_text(path.read_text() + "\n# Changed after the review\n")
    with pytest.raises(PrerequisiteError, match="settings changed"):
        execute(plan, successful([]))


# Notebook 03's former appended single-approach UI was replaced by the
# four-arm workflow. Fresh-kernel and explicit-execution coverage now lives in
# test_comparison.py; the full runner safeguards remain covered above.
