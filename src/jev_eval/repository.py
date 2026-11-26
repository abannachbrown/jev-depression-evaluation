"""Inspect Git blobs, not notebook display state, before committing research files."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

NOTEBOOKS = {
    f"notebooks/{name}.ipynb"
    for name in (
        "00_setup",
        "01_prepare_data",
        "02_model_connection",
        "03_run_models",
        "04_evaluation",
        "05_study_summary",
    )
}
DATA_FILES = {
    "data/README.md",
    "data/examples/jev_citation_state.json",
    "data/examples/states.jsonl",
    "data/examples/reference_labels.jsonl",
    "data/examples/synthetic_predictions.jsonl",
    "data/examples/inference_sh_response.json",
}
PROTECTED_ROOTS = {
    "results",
    "outputs",
    "runs",
    "artifacts",
    "logs",
    "models",
    "weights",
    "checkpoints",
}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".ggml", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5"}


def check_blob(path: str, blob: bytes, *, scan_secrets=True) -> list[str]:
    """Return messages containing locations and rule names, never secret values."""
    file = PurePosixPath(path)
    name = file.name
    protected = (
        file.parts[0] in PROTECTED_ROOTS
        or (file.parts[0] == "notebooks" and file.suffix.lower() in {".html", ".pdf"})
        or (file.parts[0] == "data" and path not in DATA_FILES)
        or any(
            part
            in {
                "secrets",
                ".secrets",
                "credentials",
                ".aws",
                ".venv",
                "venv",
                "env",
                "ENV",
                ".cache",
                ".huggingface",
                ".ipynb_checkpoints",
                ".jupyter",
            }
            or part.startswith(".venv")
            for part in file.parts
        )
        or (
            path != ".env.example"
            and (
                name == ".env"
                or name.startswith(".env.")
                or name.endswith(".env")
                or ".env." in name
            )
        )
        or file.suffix.lower() in WEIGHT_SUFFIXES | {".pem", ".key", ".p12", ".pfx"}
        or name in {"credentials.json", "credentials.yaml", "credentials.yml", ".netrc"}
        or name.endswith("-credentials.json")
        or name.endswith("service-account.json")
    )
    if protected:
        return [f"{path}: protected data, credential, model, or generated-artifact location"]
    try:
        text = blob.decode("utf-8")
    except UnicodeError:
        return [f"{path}: binary content requires explicit review before inclusion"]
    problems = []
    if file.suffix == ".ipynb":
        if path not in NOTEBOOKS:
            return [f"{path}: only explicitly reviewed notebook templates may be committed"]
        try:
            notebook = json.loads(text)
            if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                raise ValueError
            if set(notebook.get("metadata", {})) - {"kernelspec", "language_info"}:
                problems.append(f"{path}: remove extra notebook metadata (including widget state)")
            for index, cell in enumerate(notebook["cells"], 1):
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    problems.append(
                        f"{path}: cell {index} contains saved output or an execution count"
                    )
                if cell.get("attachments") or set(cell.get("metadata", {})) - {"tags"}:
                    problems.append(
                        f"{path}: cell {index} contains attachments or execution metadata"
                    )
        except (ValueError, TypeError, AttributeError):
            return [f"{path}: invalid notebook JSON or structure"]
    if scan_secrets:
        from detect_secrets.core.scan import scan_file
        from detect_secrets.settings import default_settings

        with tempfile.TemporaryDirectory(prefix="jev-commit-check-") as directory:
            temporary = Path(directory) / name
            temporary.write_bytes(blob)
            with default_settings():
                for finding in scan_file(str(temporary)):
                    if (
                        path == "configs/data_preparation/depression_excel.yaml"
                        and finding.type == "Hex High Entropy String"
                        and re.fullmatch(
                            r"expected_sha256: [0-9a-f]{64}",
                            text.splitlines()[finding.line_number - 1].strip(),
                        )
                    ):
                        # The declared input checksum is reproducibility metadata.
                        continue
                    # This exact declaration names an environment variable, not a value.
                    # Do not exempt the field generally: a literal credential must still fail.
                    if (path, text.splitlines()[finding.line_number - 1].strip()) in {
                        ("configs/backends/hosted.yaml", "api_key_env: JEV_API_KEY"),
                        ("configs/backends/inference_sh.yaml", "api_key_env: INFERENCE_API_KEY"),
                        (
                            "configs/backends/inference_sh_hybrid.yaml",
                            "api_key_env: INFERENCE_API_KEY",
                        ),
                        (
                            "configs/backends/inference_sh_full.yaml",
                            "api_key_env: INFERENCE_API_KEY",
                        ),
                    } and finding.type == "Secret Keyword":
                        continue
                    problems.append(
                        f"{path}:{finding.line_number}: possible {finding.type}; value withheld"
                    )
    return problems


def check_repository(root: Path, mode="staged") -> list[str]:
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True
        ).stdout

    if mode == "working":
        paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    elif mode == "tracked":
        paths = git("ls-tree", "-r", "--name-only", "-z", "HEAD")
    else:
        paths = git("ls-files", "--cached", "-z")
    problems = []
    for raw in sorted(set(paths.split(b"\0")) - {b""}):
        path = raw.decode("utf-8")
        if mode == "working":
            candidate = root / path
            if not candidate.exists():
                continue
            if candidate.is_symlink():
                problems.append(f"{path}: symlinks require explicit review")
                continue
            blob = candidate.read_bytes()
        else:
            marker = f"HEAD:{path}" if mode == "tracked" else f":{path}"
            blob = git("show", marker)
        problems.extend(check_blob(path, blob))
    return problems


def main():
    parser = argparse.ArgumentParser(
        description="Reject notebook outputs and protected Git content."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--staged", action="store_true", help="Inspect exactly the current Git index."
    )
    mode.add_argument("--tracked", action="store_true", help="Inspect the HEAD commit (for CI).")
    mode.add_argument(
        "--working", action="store_true", help="Inspect tracked and unignored working files."
    )
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    selected = "working" if args.working else "tracked" if args.tracked else "staged"
    problems = check_repository(root, selected)
    if problems:
        print("Commit safety checks failed. No files were changed:")
        print("\n".join(problems))
        return 1
    print(f"Commit safety checks passed ({selected} content).")
    return 0
