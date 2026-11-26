import shutil
from pathlib import Path

import pytest
import yaml

from jev_eval.config import load_project_config

ROOT = Path(__file__).resolve().parents[1]


def configure_v1_inputs(root, source, reference, *, live=False):
    """Point every new profile to invented inputs; no real workbook/API access."""
    count = len(source.read_text().splitlines())
    for approach in ("binary", "hybrid"):
        for phase in ("pilot", "full"):
            for model in ("jev_9b", "jev_27b"):
                name = f"{phase}_{approach}_v1_{model}"
                path = root / f"configs/evaluations/{name}.yaml"
                config = yaml.safe_load(path.read_text())
                config["dataset"].update(
                    path=str(source.relative_to(root)),
                    expected_count=count if phase == "pilot" else None,
                )
                path.write_text(yaml.safe_dump(config))
                path = root / f"configs/workflows/{name}.yaml"
                config = yaml.safe_load(path.read_text())
                config["evaluation"]["reference_labels"] = str(reference.relative_to(root))
                config["execution"]["allow_paid_calls"] = live
                config["output"]["allow_writes"] = live
                if phase == "full":
                    config["inference_sh_full"]["dry_run"] = not live
                path.write_text(yaml.safe_dump(config))


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
