from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from jev_eval.repository import check_blob

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    [
        "00_setup",
        "01_prepare_data",
        "02_model_connection",
        "04_evaluation",
    ],
)
def test_notebook_from_fresh_kernel(name, monkeypatch, tmp_path, project):
    monkeypatch.setenv("JEV_NOTEBOOK_TEST_MODE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "jupyter-runtime"))
    path = ROOT / "notebooks" / f"{name}.ipynb"
    original = path.read_bytes()
    assert not check_blob(f"notebooks/{name}.ipynb", original)
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    if name == "04_evaluation":
        # Full saved-data evaluation is tested independently in test_full_runs.py.
        # This generic walkthrough test intentionally exercises the teaching path.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace('DATA_MODE = "full"', 'DATA_MODE = "demo"')
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "import httpx\n"
            "def forbid_network(*args, **kwargs):\n"
            "    raise AssertionError('Notebook tests must not make HTTP calls')\n"
            "httpx.HTTPTransport.handle_request = forbid_network\n"
            "httpx.AsyncHTTPTransport.handle_async_request = forbid_network\n"
        ),
    )
    if name == "01_prepare_data":
        notebook.cells.insert(
            1,
            nbformat.v4.new_code_cell(
                "import jev_eval.data as data_module\n"
                "original_inspect = data_module.inspect_screening_workbook\n"
                "def synthetic_only(source, *args, **kwargs):\n"
                "    assert isinstance(source, bytes), 'Notebook test must not read the private workbook'\n"
                "    return original_inspect(source, *args, **kwargs)\n"
                "data_module.inspect_screening_workbook = synthetic_only\n"
            ),
        )
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                # This notebook loads prepared JSON states; the Excel converter
                # is independently covered by test_excel_preparation.py.
                "assert project.workflow.execution.mode == 'demo'\n"
                "assert project.evaluation.dataset.path.startswith('data/examples/')\n"
                "assert len(states) == validation.total == len(rows)\n"
                "assert validation.total > 0\n"
                "assert not project.workflow.privacy.display_record_text\n"
                "assert not ALLOW_WRITES\n"
            )
        )
    # The dual-model notebook 03 is exercised in test_pilot.py with synthetic inputs.
    if name == "02_model_connection":
        # Use temporary mock settings, never a live workspace profile.
        notebook.cells.insert(
            1,
            nbformat.v4.new_code_cell(
                "from pathlib import Path\n"
                "import jev_eval.environment as environment_module\n"
                "from jev_eval.config import load_project_config\n"
                f"test_root = Path({str(project.root)!r})\n"
                "environment_module.initialize_notebook = lambda **kwargs: load_project_config(test_root)\n"
            ),
        )
    if name == "02_model_connection":
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "from jev_eval.connections import preview_connection\n"
                "assert example_connection.state.model_dump() == state\n"
                "assert preview_connection(example_connection)['input'] == {'state': state, 'nouls': [question]}\n"
                "assert result.metadata['mode'] == 'mock'\n"
                "assert result.response.probabilities == {'screening_eligibility': 0.8}\n"
                "assert result.metadata['input_origin'] == 'notebook_synthetic_example'\n"
                "assert result.directory is None, 'Mock notebook execution must not write artifacts'\n"
                "assert not RECOVER_UNSAVED_RESULT\n"
            )
        )
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
    )
    client.execute()
    assert path.read_bytes() == original, "Tests must never save outputs into source notebooks"
