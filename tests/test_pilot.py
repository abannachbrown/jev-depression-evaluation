"""Pilot tests use fabricated states, credentials and MockTransport exclusively."""

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
from jev_eval.data import dataset_fingerprint, load_jsonl, prepare_pilot_sample
from jev_eval.runner import PilotPersistenceError, load_run, plan_run, run_experiment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_real_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real HTTP is forbidden in pilot tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setenv("INFERENCE_API_KEY", "synthetic-test-only")


@pytest.fixture
def pilot_project(project):
    source = project.root / "data/private/prepared/synthetic/states.jsonl"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "study_id": f"synthetic-{index}",
            "title": f"Invented title {index}",
            "abstract": "Invented abstract" if index else "",
        }
        for index in range(14)
    ]
    for row in rows[-2:]:
        row.update(title="", abstract="")
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    reference = source.parent / "reference_labels.csv"
    with reference.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["study_id", "Inclusion Status", "ITEM_ID"])
        writer.writerows((row["study_id"], "1", "synthetic-item") for row in rows)
    for model in ("jev_9b", "jev_27b"):
        path = project.root / f"configs/evaluations/pilot_{model}.yaml"
        config = yaml.safe_load(path.read_text())
        config["dataset"].update(path=str(source.relative_to(project.root)), expected_count=14)
        path.write_text(yaml.safe_dump(config))
        full_path = project.root / f"configs/evaluations/full_{model}.yaml"
        full_config = yaml.safe_load(full_path.read_text())
        full_config["dataset"].update(
            path=str(source.relative_to(project.root)), expected_count=None
        )
        full_path.write_text(yaml.safe_dump(full_config))
        workflow_path = project.root / f"configs/workflows/full_{model}.yaml"
        full_workflow = yaml.safe_load(workflow_path.read_text())
        full_workflow["evaluation"]["reference_labels"] = str(reference.relative_to(project.root))
        workflow_path.write_text(yaml.safe_dump(full_workflow))
    configure_v1_inputs(project.root, source, reference, live=True)
    shutil.copytree(
        ROOT / "src/jev_eval",
        project.root / "src/jev_eval",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.fetch", "*.tmp"),
    )
    return load_project_config(project.root, "configs/workflows/pilot_jev_9b.yaml")


def frozen_plan(project):
    sample = prepare_pilot_sample(project, allow_writes=True)
    return plan_run(project, sample.states)


def success_handler(calls):
    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        large = payload["app"] == "autotrust/jev-27b-v1-1"
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": f"synthetic-task-{len(calls)}",
                    "status": 10,
                    "output": {
                        "model": "autotrust/AutoTrust-JEV-27B-v1.1"
                        if large
                        else "autotrust/JEV-9B",
                        "input_tokens": 100,
                        "nouls": {"retain_for_fulltext": {"noul": 0.9 if large else 0.8}},
                    },
                }
            },
        )

    return handler


def execute(plan, handler):
    return run_experiment(
        plan,
        allow_paid_calls=True,
        allow_writes=True,
        transport=httpx.MockTransport(handler),
        sleeper=lambda _delay: None,
    )


def test_sample_freezes_same_ten_and_audits_unavailable_records(pilot_project):
    with pytest.raises(PrerequisiteError, match="ALLOW_SAMPLE_WRITES"):
        prepare_pilot_sample(pilot_project)
    first = prepare_pilot_sample(pilot_project, allow_writes=True)
    original = first.path.read_bytes()
    second_project = load_project_config(pilot_project.root, "configs/workflows/pilot_jev_27b.yaml")
    second = prepare_pilot_sample(second_project)
    assert first.manifest == second.manifest
    assert first.path.read_bytes() == original
    assert len(first.states) == 10
    assert len({state.study_id for state in first.states}) == 10
    assert first.manifest["counts"] == {
        "source_records": 14,
        "available_for_inference": 12,
        "unavailable_for_inference": 2,
        "selected_records": 10,
        "missing_abstract_warnings": 1,
    }
    assert first.manifest["unavailable_for_inference_ids"] == ["synthetic-12", "synthetic-13"]
    assert dataset_fingerprint(first.states) == first.manifest["sample_sha256"]
    assert first.manifest["ordered_study_ids"] == [state.study_id for state in first.states]
    assert all(
        set(state.model_dump()) == {"study_id", "title", "abstract"} for state in first.states
    )


@pytest.mark.parametrize("change", ["seed", "source", "task"])
def test_changed_source_seed_or_criteria_cannot_replace_frozen_sample(pilot_project, change):
    sample = prepare_pilot_sample(pilot_project, allow_writes=True)
    original = sample.path.read_bytes()
    if change == "seed":
        pilot_project.evaluation.seed += 1
    elif change == "source":
        path = pilot_project.root / pilot_project.evaluation.dataset.path
        path.write_text(path.read_text() + "\n")
    else:
        path = pilot_project.root / pilot_project.evaluation.task.prompt_file
        task = json.loads(path.read_text())
        task["questions"]["retain_for_fulltext"]["instructions"] += " Changed criteria."
        path.write_text(json.dumps(task))
    with pytest.raises(PrerequisiteError, match="Frozen pilot sample differs"):
        prepare_pilot_sample(pilot_project, allow_writes=True)
    assert sample.path.read_bytes() == original


@pytest.mark.parametrize("change", ["duplicate", "label", "invalid_text", "too_few"])
def test_sample_blocks_other_errors_and_insufficient_records(pilot_project, change):
    path = pilot_project.root / pilot_project.evaluation.dataset.path
    rows = load_jsonl(path)
    if change == "duplicate":
        rows[1]["study_id"] = rows[0]["study_id"]
    elif change == "label":
        rows[0]["Inclusion Status"] = True
    elif change == "invalid_text":
        rows[0]["abstract"] = 42
    else:
        for row in rows[:4]:
            row.update(title="", abstract="")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(PrerequisiteError, match="validation failed|at least ten"):
        prepare_pilot_sample(pilot_project, allow_writes=True)


def test_conversion_checksum_is_verified(pilot_project):
    source = pilot_project.root / pilot_project.evaluation.dataset.path
    conversion = {
        "source": {"sha256": "synthetic-workbook-hash"},
        "output_sha256": {"states.jsonl": hashlib.sha256(source.read_bytes()).hexdigest()},
    }
    (source.parent / "manifest.json").write_text(json.dumps(conversion))
    sample = prepare_pilot_sample(pilot_project, allow_writes=True)
    assert sample.manifest["source_workbook_sha256"] == "synthetic-workbook-hash"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(PrerequisiteError, match="conversion checksum"):
        prepare_pilot_sample(pilot_project)


def test_both_models_save_full_provenance_separately_and_resume_without_posts(pilot_project):
    calls = []
    first_plan = frozen_plan(pilot_project)
    first = execute(first_plan, success_handler(calls))
    second_project = load_project_config(pilot_project.root, "configs/workflows/pilot_jev_27b.yaml")
    second_plan = frozen_plan(second_project)
    second = execute(second_plan, success_handler(calls))
    assert len(calls) == 20
    assert all(call["app"] == "autotrust/jev-9b" for call in calls[:10])
    assert all(call["app"] == "autotrust/jev-27b-v1-1" for call in calls[10:])
    assert [call["input"] for call in calls[:10]] == [call["input"] for call in calls[10:]]
    assert all(set(call) == {"app", "input", "wait"} and call["wait"] is True for call in calls)
    assert first.directory != second.directory
    assert first.manifest["dataset_sha256"] == second.manifest["dataset_sha256"]
    assert first.manifest["task_sha256"] == second.manifest["task_sha256"]
    for result, probability, plan in [(first, 0.8, first_plan), (second, 0.9, second_plan)]:
        assert result.manifest["status"] == "completed"
        assert result.manifest["total_attempts"] == 10
        assert result.manifest["pending_request"] is None
        assert not result.manifest["generation_applied"]
        assert result.manifest["source_sha256"]
        assert result.manifest["requested_model_ids"] == {
            plan.project.models[0].name: plan.project.models[0].hosted.app_id
        }
        assert len(result.predictions) == len(result.responses) == 10
        for row in result.predictions:
            assert row["probabilities"] == {"retain_for_fulltext": probability}
            assert row["predictions"] == {}  # No implicit eligibility threshold.
            assert row["usage"] == {"input_tokens": 100}
            assert row["request_id"].startswith("synthetic-task-")
            assert row["returned_model"]
            assert row["started_at"] <= row["ended_at"]
            receipt = json.loads((result.directory / row["receipt"]).read_text())
            body = (result.directory / receipt["response_body"]).read_bytes()
            assert hashlib.sha256(body).hexdigest() == receipt["response_sha256"]
            assert json.loads(body) == receipt["normalized"]["raw"]
            assert receipt["http_status"] == 200
        saved = (result.directory / "predictions.jsonl").read_bytes()
        resumed = execute(plan, success_handler(calls))
        assert resumed.manifest["total_attempts"] == 10
        assert (result.directory / "predictions.jsonl").read_bytes() == saved
        for path in result.directory.rglob("*"):
            if path.is_file():
                assert b"synthetic-test-only" not in path.read_bytes()
    assert len(calls) == 20


@pytest.mark.parametrize("paid,writes", [(False, False), (True, False)])
def test_pilot_requires_paid_and_write_permissions_before_http(pilot_project, paid, writes):
    plan = frozen_plan(pilot_project)
    with pytest.raises(PrerequisiteError, match="permission|ALLOW_WRITES"):
        run_experiment(plan, allow_paid_calls=paid, allow_writes=writes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "demo"),
        ("max_records", 11),
        ("max_logical_requests", 20),
        ("max_total_attempts", 20),
    ],
)
def test_pilot_cannot_be_expanded_by_changing_yaml_limits(pilot_project, field, value):
    sample = prepare_pilot_sample(pilot_project, allow_writes=True)
    setattr(pilot_project.workflow.execution, field, value)
    with pytest.raises(PrerequisiteError, match="max_records|Pilot requires"):
        plan_run(pilot_project, sample.states)


def test_pilot_requires_frozen_order_one_model_and_one_question(pilot_project):
    sample = prepare_pilot_sample(pilot_project, allow_writes=True)
    with pytest.raises(PrerequisiteError, match="frozen pilot states"):
        plan_run(pilot_project, list(reversed(sample.states)))
    pilot_project.models.append(pilot_project.models[0])
    with pytest.raises(PrerequisiteError):
        plan_run(pilot_project, sample.states)


def test_interruption_checkpoints_successes_and_blocks_unknown_inflight_resume(pilot_project):
    calls = []
    successful = success_handler(calls)
    plan = frozen_plan(pilot_project)

    def interrupted(request):
        if len(calls) == 3:
            raise KeyboardInterrupt
        return successful(request)

    with pytest.raises(KeyboardInterrupt):
        execute(plan, interrupted)
    saved = load_run(pilot_project, plan.run_id)
    assert len(saved.predictions) == 3
    assert saved.manifest["total_attempts"] == 4
    assert saved.manifest["pending_request"]
    with pytest.raises(PrerequisiteError, match="unknown outcome"):
        execute(plan, successful)
    assert len(calls) == 3


@pytest.mark.parametrize("status", [401, 402, 403, 429])
def test_http_errors_stop_after_one_request_and_preserve_raw_bytes(pilot_project, status):
    calls = []
    plan = frozen_plan(pilot_project)

    def error_response(request):
        calls.append(json.loads(request.content))
        return httpx.Response(status, content=b'{"error":"synthetic denial"}')

    result = execute(plan, error_response)
    assert len(calls) == 1
    assert result.manifest["status"] == "paused_after_error"
    assert result.manifest["pending_request"] is None
    assert result.predictions[0]["error_code"] == f"http_{status}"
    assert (result.directory / result.responses[0]["response_body"]).read_bytes() == (
        b'{"error":"synthetic denial"}'
    )
    # Explicit resume advances to the next unattempted state, never retries the failed one.
    execute(plan, success_handler(calls))
    assert len(calls) == 10
    assert len({call["input"]["state"]["study_id"] for call in calls}) == 10


@pytest.mark.parametrize("failure", ["timeout", "unfinished", "malformed", "server"])
def test_ambiguous_outcomes_preserve_pending_and_block_resume(pilot_project, failure):
    plan = frozen_plan(pilot_project)
    calls = []

    def handler(request):
        calls.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("synthetic timeout", request=request)
        if failure == "unfinished":
            return httpx.Response(200, json={"data": {"id": "pending-task", "status": 2}})
        return httpx.Response(500 if failure == "server" else 200, text="synthetic not JSON")

    result = execute(plan, handler)
    assert len(calls) == 1
    assert result.manifest["status"] == "needs_review"
    assert result.manifest["pending_request"]
    if failure == "unfinished":
        assert result.manifest["pending_request"]["request_id"] == "pending-task"
        assert result.failures[0]["request_id"] == "pending-task"
    with pytest.raises(PrerequisiteError, match="unknown outcome"):
        execute(plan, handler)
    assert len(calls) == 1


def test_storage_check_fails_before_http_and_late_failure_retains_exchange(
    pilot_project, monkeypatch
):
    plan = frozen_plan(pilot_project)
    calls = []

    def storage_failure(*args, **kwargs):
        raise OSError("synthetic storage failure")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_checkpoint", storage_failure)
        with pytest.raises(PilotPersistenceError):
            execute(plan, success_handler(calls))
    assert calls == []
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_save_pilot_receipt", storage_failure)
        with pytest.raises(PilotPersistenceError) as error:
            execute(plan, success_handler(calls))
    assert len(calls) == 1
    retained = error.value.result
    assert retained._pending_exchange.status_code == 200
    assert retained.predictions[0]["probabilities"] == {"retain_for_fulltext": 0.8}
    assert retained.manifest["pending_request"]
    with pytest.raises(PrerequisiteError, match="unknown outcome"):
        execute(plan, success_handler(calls))
    assert len(calls) == 1


def test_interrupt_during_commit_cannot_clear_pending_before_record_is_saved(
    pilot_project, monkeypatch
):
    plan = frozen_plan(pilot_project)
    original = runner._checkpoint

    def interrupted(result):
        if result.predictions:
            assert result.manifest["pending_request"]
            raise KeyboardInterrupt
        original(result)

    monkeypatch.setattr(runner, "_checkpoint", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute(plan, success_handler([]))
    assert load_run(pilot_project, plan.run_id).manifest["pending_request"]


def test_run_lock_prevents_a_second_writer_and_pacing_is_used(pilot_project):
    plan = frozen_plan(pilot_project)
    directory = pilot_project.root / pilot_project.workflow.output.root
    lock = directory / ("." + plan.run_id + ".lock")
    lock.write_text("synthetic active writer")
    with pytest.raises(PrerequisiteError, match="locked"):
        execute(plan, success_handler([]))
    lock.unlink()
    delays = []
    run_experiment(
        plan,
        allow_paid_calls=True,
        allow_writes=True,
        transport=httpx.MockTransport(success_handler([])),
        sleeper=delays.append,
    )
    assert len(delays) == 9
    assert all(0 < delay <= 3.1 for delay in delays)


def test_changed_code_prevents_resume(pilot_project):
    plan = frozen_plan(pilot_project)
    calls = []
    execute(plan, success_handler(calls))
    (pilot_project.root / "src/jev_eval/runner.py").write_text("# changed source\n")
    with pytest.raises(PrerequisiteError, match="Cannot resume"):
        execute(frozen_plan(pilot_project), success_handler(calls))
    assert len(calls) == 10


def paired_results(project):
    first_plan = frozen_plan(project)
    first = execute(first_plan, success_handler([]))
    other = load_project_config(project.root, "configs/workflows/pilot_jev_27b.yaml")
    second_plan = frozen_plan(other)
    second = execute(second_plan, success_handler([]))
    return first_plan, first, second_plan, second


def test_paired_tables_and_completed_methods_use_actual_provenance(pilot_project):
    from jev_eval.reporting import (
        paired_pilot_plan_table,
        paired_pilot_probability_table,
        pilot_methods_summary,
    )

    p9, r9, p27, r27 = paired_results(pilot_project)
    assert len(paired_pilot_plan_table(p9, p27)) == 2
    # Pairing follows study IDs, even if persisted response order changes.
    r27.predictions.reverse()
    table = paired_pilot_probability_table(p9, r9, p27, r27)
    assert list(table.study_id) == [state.study_id for state in p9.states]
    assert list(table.jev_9b_p_true) == [0.8] * 10
    assert list(table.jev_27b_p_true) == [0.9] * 10
    methods = pilot_methods_summary(prepare_pilot_sample(pilot_project), p9, p27, r9, r27)
    assert methods.text.startswith("Completed pilot methods.")
    assert "12 records" in methods.text and "14 prepared" in methods.text
    assert "seed 42" in methods.text
    assert "10/10 successful" in methods.text
    assert p9.dataset_sha256 in str(methods.provenance.to_dict())
    assert p9.task.fingerprint in str(methods.provenance.to_dict())


def test_methods_and_tables_report_unexecuted_and_partial_runs_honestly(pilot_project):
    from jev_eval.reporting import (
        paired_pilot_probability_table,
        pilot_methods_summary,
        pilot_run_status,
    )

    p9, r9, p27, _ = paired_results(pilot_project)
    sample = prepare_pilot_sample(pilot_project)
    planned = pilot_methods_summary(sample, p9, p27)
    assert planned.text.startswith("Planned pilot protocol")
    assert "not executed in this kernel" in planned.text
    assert "Each selected study was evaluated" not in planned.text
    r9.predictions = r9.predictions[:3]
    r9.predictions[0].update(status="unknown", probabilities={})
    r9.predictions[1].update(status="failed", probabilities={})
    r9.manifest.update(status="needs_review", pending_request={"synthetic": True})
    partial = pilot_methods_summary(sample, p9, p27, r9)
    assert partial.text.startswith("Planned pilot protocol")
    assert "1/10 successful, 1 failed, 1 unknown, 7 unrecorded" in partial.text
    table = paired_pilot_probability_table(p9, r9, p27, None)
    assert table.jev_27b_p_true.isna().all()
    assert table.jev_9b_p_true.notna().sum() == 1
    status = pilot_run_status(p9, r9).iloc[0]
    assert status.pending_review and status.unknown == 1 and status.unrecorded == 7


@pytest.mark.parametrize("change", ["duplicate", "wrong_model", "wrong_fingerprint", "nan"])
def test_paired_reporting_rejects_invalid_or_mismatched_results(pilot_project, change):
    from jev_eval.reporting import paired_pilot_probability_table

    p9, r9, p27, r27 = paired_results(pilot_project)
    if change == "duplicate":
        r9.predictions.append(r9.predictions[0])
    elif change == "wrong_model":
        r9.predictions[0]["model"] = "jev-27b"
    elif change == "wrong_fingerprint":
        r9.manifest["fingerprint"] = "different"
    else:
        r9.predictions[0]["probabilities"]["retain_for_fulltext"] = float("nan")
    with pytest.raises(PrerequisiteError):
        paired_pilot_probability_table(p9, r9, p27, r27)


def test_paired_plans_reject_different_order_criteria_or_destinations(pilot_project):
    from jev_eval.reporting import paired_pilot_plan_table

    p9 = frozen_plan(pilot_project)
    other = load_project_config(pilot_project.root, "configs/workflows/pilot_jev_27b.yaml")
    p27 = frozen_plan(other)
    original = list(p27.states)
    p27.states.reverse()
    with pytest.raises(PrerequisiteError):
        paired_pilot_plan_table(p9, p27)
    p27.states = original
    p27.project.workflow.output.run_id = p9.run_id
    p27.run_id = p9.run_id
    with pytest.raises(PrerequisiteError, match="separate output"):
        paired_pilot_plan_table(p9, p27)
    p27.run_id = "other"
    p27.task.questions["retain_for_fulltext"].instructions += " Changed."
    with pytest.raises(PrerequisiteError, match="same ordered sample"):
        paired_pilot_plan_table(p9, p27)


# Notebook 03 now reuses four explicit full archives. Its fresh-kernel checks
# are in test_comparison.py. The bounded pilot runner is tested above; saved
# pilot artifacts are neither changed nor loaded as full-comparison results.
