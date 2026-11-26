import hashlib
import json
import subprocess

from jev_eval.repository import check_blob, check_repository


def minimal_notebook():
    return {
        "nbformat": 4,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "source": "1 + 1",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
        ],
    }


def test_output_metadata_attachments_and_forced_private_paths():
    name = "notebooks/00_setup.ipynb"
    notebook = minimal_notebook()
    assert not check_blob(name, json.dumps(notebook).encode(), scan_secrets=False)
    notebook["cells"][0]["outputs"] = [{"text": "PRIVATE OUTPUT"}]
    assert check_blob(name, json.dumps(notebook).encode(), scan_secrets=False)
    notebook = minimal_notebook()
    notebook["metadata"]["widgets"] = {"anything": "private"}
    assert check_blob(name, json.dumps(notebook).encode(), scan_secrets=False)
    notebook = minimal_notebook()
    notebook["cells"][0]["attachments"] = {"image": "private"}
    assert check_blob(name, json.dumps(notebook).encode(), scan_secrets=False)
    for path in (
        ".env",
        "data/private/a.json",
        "results/x.csv",
        "models/x/config.json",
        "data/examples/not-reviewed.json",
        "notebooks/extra.ipynb",
        "notebooks/05_study_summary.html",
        "notebooks/05_study_summary.pdf",
    ):
        assert check_blob(path, b"{}", scan_secrets=False), path


def test_interim_summary_template_is_allowed_but_saved_outputs_are_rejected():
    name = "notebooks/05_study_summary.ipynb"
    notebook = minimal_notebook()
    assert not check_blob(name, json.dumps(notebook).encode(), scan_secrets=False)
    notebook["cells"][0]["outputs"] = [{"text": "private interim data"}]
    assert check_blob(name, json.dumps(notebook).encode(), scan_secrets=False)


def test_scans_exact_staged_blob_not_clean_working_copy(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    notebook = minimal_notebook()
    notebook["cells"][0]["outputs"] = [{"text": "private output"}]
    path = tmp_path / "notebooks/00_setup.ipynb"
    path.parent.mkdir()
    path.write_text(json.dumps(notebook))
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    path.write_text(json.dumps(minimal_notebook()))
    assert any("saved output" in problem for problem in check_repository(tmp_path, "staged"))
    assert not check_repository(tmp_path, "working")


def test_credential_findings_do_not_echo_secret():
    value = "".join(["ghp_", "AbCdEf0123456789" * 3])
    findings = check_blob("example.py", ("token = " + repr(value)).encode())
    assert findings
    assert all(value not in finding for finding in findings)


def test_only_exact_environment_reference_is_exempt():
    assert not check_blob("configs/backends/hosted.yaml", b"api_key_env: JEV_API_KEY\n")
    value = "".join(["ghp_", "AbCdEf0123456789" * 3])
    assert check_blob("configs/backends/hosted.yaml", ("api_key_env: " + value).encode())
    assert not check_blob("configs/backends/inference_sh.yaml", b"api_key_env: INFERENCE_API_KEY\n")
    assert check_blob("configs/backends/inference_sh.yaml", ("api_key_env: " + value).encode())
    assert not check_blob(
        "configs/backends/inference_sh_full.yaml", b"api_key_env: INFERENCE_API_KEY\n"
    )
    assert check_blob("configs/backends/inference_sh_full.yaml", ("api_key_env: " + value).encode())
    assert not check_blob(
        "configs/backends/inference_sh_hybrid.yaml", b"api_key_env: INFERENCE_API_KEY\n"
    )
    assert check_blob(
        "configs/backends/inference_sh_hybrid.yaml", ("api_key_env: " + value).encode()
    )


def test_source_checksum_exception_does_not_exempt_credentials():
    digest = hashlib.sha256(b"synthetic workbook").hexdigest()
    path = "configs/data_preparation/depression_excel.yaml"
    assert not check_blob(path, ("expected_sha256: " + digest).encode())
    assert check_blob(path, ("api_key: " + digest).encode())
    assert check_blob("other.yaml", ("expected_sha256: " + digest).encode())
