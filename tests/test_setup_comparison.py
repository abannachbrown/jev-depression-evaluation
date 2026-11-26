"""Shared preparation and synthetic connection checks for the four-arm workflow."""

import hashlib
import json

import httpx
import nbformat
import pytest
import yaml
from conftest import ROOT
from nbclient import NotebookClient
from test_comparison import comparison_root as comparison_root
from test_comparison import hashes

from jev_eval.comparison import comparison_setup_table, load_shared_comparison_inputs
from jev_eval.config import PrerequisiteError
from jev_eval.connections import (
    load_comparison_connections,
    load_connection,
    preview_connection,
    run_connection,
    saved_connection_inventory,
)
from jev_eval.data import dataset_fingerprint, select_sample


def test_configuration_table_needs_no_private_data_client_or_key(project, monkeypatch):
    import jev_eval.runner as runner

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Configuration-only setup cannot open private data or construct clients"
        )

    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(runner, "create_backend", forbidden)
    monkeypatch.setattr("jev_eval.data.load_full_dataset", forbidden)
    table = comparison_setup_table(project.root)
    assert len(table) == 4
    assert table.questions.tolist() == [1, 1, 7, 7]
    assert table.model.tolist() == ["jev-9b", "jev-27b"] * 2
    assert not table.input_available.any()
    assert not table.reference_available.any()
    assert not table.saved_manifest_available.any()
    assert not (project.root / "results").exists()


def test_one_shared_cohort_and_reference_preserve_all_studies_without_writes(comparison_root):
    before = hashes(comparison_root / "data")
    review = load_shared_comparison_inputs(comparison_root)
    assert len(review.states) == review.validation.total == 10
    assert review.table.records.tolist() == [10] * 4
    assert review.table.not_inferred_no_text.tolist() == [2] * 4
    assert review.table.states_sha256.nunique() == review.table.reference_sha256.nunique() == 1
    assert review.table.ordered_ids_match.all() and review.table.reference_mapping_matches.all()
    assert all(
        set(state.model_dump()) == {"study_id", "title", "abstract"} for state in review.states
    )
    assert review.states[0].study_id == "000"
    available = [state for state in review.states if state.title.strip() or state.abstract.strip()]
    preview = select_sample(available, min(10, len(available)), review.projects[0].evaluation.seed)
    assert len(preview) == 8
    assert dataset_fingerprint(review.states) == review.table.states_sha256.iloc[0]
    assert before == hashes(comparison_root / "data")
    assert not (comparison_root / "results").exists()


@pytest.mark.parametrize("difference", ["citations", "reference_mapping"])
def test_shared_preparation_rejects_a_different_arm(comparison_root, difference):
    path = comparison_root / "configs/workflows/full_hybrid_v1_jev_27b.yaml"
    if difference == "reference_mapping":
        config = yaml.safe_load(path.read_text())
        config["inference_sh_full"]["label_mapping"] = {"0": True, "1": False}
        path.write_text(yaml.safe_dump(config))
    else:
        path = comparison_root / "configs/evaluations/full_hybrid_v1_jev_27b.yaml"
        config = yaml.safe_load(path.read_text())
        source = comparison_root / config["dataset"]["path"]
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        rows[0]["title"] = "A different invented title"
        changed = source.with_name("different_states.jsonl")
        changed.write_text("".join(json.dumps(row) + "\n" for row in rows))
        config["dataset"]["path"] = str(changed.relative_to(comparison_root))
        path.write_text(yaml.safe_dump(config))
    with pytest.raises(PrerequisiteError, match="identical ordered citations"):
        load_shared_comparison_inputs(comparison_root)


def test_four_mock_connections_have_distinct_models_correct_limits_and_no_writes(
    project, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mock demonstration must never use real HTTP")

    monkeypatch.delenv("INFERENCE_API_KEY", raising=False)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    original = (project.root / "configs/connections/inference_sh.yaml").read_bytes()
    examples = load_comparison_connections(project.root)
    assert len(examples) == 4
    assert len({json.dumps(c.state.model_dump(), sort_keys=True) for c in examples.values()}) == 1
    for (approach, model), connection in examples.items():
        expected = 1 if approach == "binary" else 7
        request = preview_connection(connection)
        assert request["app"] == (
            "autotrust/jev-9b" if model == "jev-9b" else "autotrust/jev-27b-v1-1"
        )
        assert len(request["input"]["nouls"]) == expected
        assert connection.settings.execution.max_states == connection.backend.max_attempts == 1
        assert connection.settings.execution.max_questions == expected
        result = run_connection(connection)
        assert result.metadata["mode"] == "mock" and result.metadata["status"] == "completed"
        assert len(result.response.probabilities) == expected
        assert result.directory is None
    assert (project.root / "configs/connections/inference_sh.yaml").read_bytes() == original
    assert load_connection(project.root).settings.execution.max_questions == 1
    assert not (project.root / "results").exists()


@pytest.mark.parametrize(
    "field,value", [("mode", "live"), ("allow_paid_calls", True), ("allow_writes", True)]
)
def test_mock_grid_rejects_live_or_paid_or_writable_profiles(project, field, value):
    p = project.root / "configs/connections/comparison_binary.yaml"
    config = yaml.safe_load(p.read_text())
    config["output" if field == "allow_writes" else "execution"][field] = value
    p.write_text(yaml.safe_dump(config))
    with pytest.raises(PrerequisiteError, match="mock mode"):
        load_comparison_connections(project.root)


def test_saved_connection_inventory_uses_allowlist_without_resubmission_or_mutation(project):
    assert saved_connection_inventory(project.root).empty
    folder = project.root / "results/connections/invented"
    folder.mkdir(parents=True)
    metadata = {
        "mode": "live",
        "status": "completed",
        "requested_app": "autotrust/jev-9b",
        "question_ids": ["invented_question"],
        "configuration": {"api_key": ""},
        "authorization": "must-not-appear-in-inventory",
    }
    path = folder / "metadata.json"
    path.write_text(json.dumps(metadata))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = saved_connection_inventory(project.root)
    assert len(inventory) == 1 and inventory.question_count.iloc[0] == 1
    assert "must-not-appear" not in inventory.to_json()
    assert "api_key" not in inventory.to_json()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    path.write_text("{")
    with pytest.raises(PrerequisiteError, match="metadata file needs review"):
        saved_connection_inventory(project.root)


@pytest.mark.parametrize("notebook_name", ["00_setup", "01_prepare_data", "02_model_connection"])
def test_setup_notebooks_from_fresh_kernel_with_shared_private_format_and_no_http(
    comparison_root,
    monkeypatch,
    notebook_name,
):
    before = hashes(comparison_root / "data")
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
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
            "from jev_eval.config import load_project_config\n"
            'def forbidden(*args, **kwargs):\n    raise AssertionError("No HTTP or runner execution in setup notebooks")\n'
            "httpx.HTTPTransport.handle_request = forbidden\nhttpx.AsyncHTTPTransport.handle_async_request = forbidden\n"
            "runner.run_experiment = forbidden\nrunner.create_backend = forbidden\n"
            f"test_root = Path({str(comparison_root)!r})\n"
            "environment.find_project_root = lambda: test_root\n"
            "environment.initialize_notebook = lambda **kw: load_project_config(test_root)\n"
        ),
    )
    checks = {
        "00_setup": "assert len(four_arm_setup) == 4\n",
        "01_prepare_data": "assert len(input_equivalence) == 4\nassert len(states) == 10\nassert not ALLOW_WRITES\n",
        "02_model_connection": "assert len(connection_grid) == 4\nassert saved_connections.empty\nassert not ALLOW_PAID_CALLS and not ALLOW_HYBRID_PAID_CALLS\n",
    }
    notebook.cells.append(nbformat.v4.new_code_cell(checks[notebook_name]))
    NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    ).execute()
    assert path.read_bytes() == original
    assert hashes(comparison_root / "data") == before
    assert not (comparison_root / "results").exists()
