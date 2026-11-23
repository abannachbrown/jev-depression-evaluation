import shutil
from pathlib import Path

import pytest
import yaml

from jev_eval.config import load_project_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    for directory in ("configs", "prompts", "data/examples"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    shutil.copy(ROOT / "uv.lock", tmp_path / "uv.lock")
    # Tests must stay offline even when the user's connection is configured live.
    connection = tmp_path / "configs/connections/inference_sh.yaml"
    settings = yaml.safe_load(connection.read_text())
    settings["model_config_file"] = "configs/models/jev-9b.yaml"
    settings["execution"].update(mode="mock", allow_paid_calls=False)
    settings["output"]["allow_writes"] = False
    connection.write_text(yaml.safe_dump(settings, sort_keys=False))
    return load_project_config(tmp_path)
