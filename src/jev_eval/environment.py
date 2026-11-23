"""Notebook initialization and non-sensitive environment summaries."""

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from jev_eval.config import PrerequisiteError, load_project_config


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/jev_eval").is_dir():
            return candidate
    raise PrerequisiteError("Open the notebook from this repository or its notebooks directory.")


def initialize_notebook(start=None, workflow_path="configs/notebooks.yaml", load_env=True):
    root = find_project_root(start)
    testing = os.getenv("JEV_NOTEBOOK_TEST_MODE") == "1"
    if load_env and not testing:
        load_dotenv(root / ".env", override=False)
    project = load_project_config(root, workflow_path)
    if testing and (
        project.workflow.execution.mode != "demo"
        or project.workflow.output.allow_writes
        or project.workflow.execution.allow_paid_calls
        or project.workflow.privacy.display_record_text
        or project.workflow.privacy.display_raw_responses
        or project.workflow.evaluation_config != "configs/evaluations/synthetic.yaml"
    ):
        raise PrerequisiteError(
            "Notebook tests require synthetic demo settings and all permissions off."
        )
    return project


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise PrerequisiteError(f"Required reproducibility file is missing: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_information(root: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "--verify", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def reproducibility_summary(project) -> dict:
    packages = {}
    for name in (
        "httpx",
        "pandas",
        "pydantic",
        "pyyaml",
        "scikit-learn",
        "ipykernel",
        "jupyterlab",
        "nbformat",
        "nbclient",
        "matplotlib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    try:
        uv = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        uv_version = uv.stdout.strip() if uv.returncode == 0 else "unavailable"
    except OSError:
        uv_version = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "uv": uv_version,
        "git": git_information(project.root),
        "lockfile_sha256": sha256_file(project.root / "uv.lock"),
        "packages": packages,
    }


def check_environment(project) -> pd.DataFrame:
    summary = reproducibility_summary(project)
    rows = [
        {"check": "Python >= 3.13", "ready": sys.version_info >= (3, 13)},
        {
            "check": "Project .venv kernel",
            "ready": Path(sys.prefix).resolve() == (project.root / ".venv").resolve(),
        },
    ]
    rows.extend(
        {"check": name, "ready": version != "missing"}
        for name, version in summary["packages"].items()
    )
    for name in (
        "JEV_API_BASE_URL",
        "JEV_API_KEY",
        *(m.hosted.model_id_env for m in project.models),
    ):
        rows.append(
            {"check": f"{name} set (hosted only)", "ready": bool(os.getenv(name, "").strip())}
        )
    return pd.DataFrame(rows)
