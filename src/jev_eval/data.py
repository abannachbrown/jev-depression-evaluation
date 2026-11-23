"""Citation loading and validation without exposing citation text in errors."""

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator

from jev_eval.config import PrerequisiteError, Settings, parse_settings, read_yaml


class CitationState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    study_id: StrictStr
    title: StrictStr = ""
    abstract: StrictStr = ""


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PrerequisiteError(f"Input file missing: {path.name}") from None
    except (json.JSONDecodeError, UnicodeError) as error:
        location = f" at line {error.lineno}" if isinstance(error, json.JSONDecodeError) else ""
        raise PrerequisiteError(
            f"Invalid JSON in {path.name}{location}; no input text displayed."
        ) from None


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise PrerequisiteError(f"Input file missing: {path.name}")
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeError:
        raise PrerequisiteError(f"Expected UTF-8 text in {path.name}.") from None
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            raise PrerequisiteError(f"Invalid JSON in {path.name}, line {number}.") from None
        if not isinstance(item, dict):
            raise PrerequisiteError(f"Expected a JSON object in {path.name}, line {number}.")
        rows.append(item)
    return rows


def load_states(settings, root: Path) -> list[dict]:
    override = os.getenv(settings.root_env, "").strip() if settings.root_env else ""
    path = root / (override or settings.path)
    if settings.format == "jsonl":
        rows = load_jsonl(path)
    elif path.is_dir():
        files = sorted(path.glob(settings.glob or "*.json"))
        rows = [load_json(file) for file in files if file.is_file()]
    elif path.is_file():
        value = load_json(path)
        rows = value if isinstance(value, list) else [value]
    else:
        raise PrerequisiteError(
            "Dataset not found. Check the dataset path or its directory override."
        )
    if not rows:
        raise PrerequisiteError("The dataset contains no states.")
    if settings.expected_count is not None and len(rows) != settings.expected_count:
        raise PrerequisiteError(
            f"Expected {settings.expected_count} states but loaded {len(rows)}."
        )
    if settings.shuffle:
        raise PrerequisiteError(
            "Use select_sample with an explicit seed; dataset shuffle is unsupported."
        )
    return rows


@dataclass
class ValidationReport:
    records: list[CitationState]
    issues: list[dict]
    total: int

    @property
    def valid(self):
        return not any(issue["severity"] == "error" for issue in self.issues)

    def require_valid(self) -> list[CitationState]:
        if not self.valid:
            raise PrerequisiteError(
                "Dataset validation failed. Review the row/field validation report."
            )
        return self.records


def validate_states(rows: list[dict]) -> ValidationReport:
    records, issues, seen = [], [], set()
    for number, row in enumerate(rows, 1):
        try:
            state = CitationState.model_validate(row)
        except ValidationError as error:
            for item in error.errors():
                issues.append(
                    {
                        "row": number,
                        "field": ".".join(map(str, item["loc"])),
                        "severity": "error",
                        "issue": item["type"],
                    }
                )
            continue
        if not state.study_id.strip() or state.study_id in seen:
            issues.append(
                {
                    "row": number,
                    "field": "study_id",
                    "severity": "error",
                    "issue": "empty or duplicate identifier",
                }
            )
        seen.add(state.study_id)
        if not state.title.strip() and not state.abstract.strip():
            issues.append(
                {
                    "row": number,
                    "field": "title/abstract",
                    "severity": "error",
                    "issue": "no citation text",
                }
            )
        elif not state.abstract.strip():
            issues.append(
                {
                    "row": number,
                    "field": "abstract",
                    "severity": "warning",
                    "issue": "abstract missing; citation retained for review",
                }
            )
        records.append(state)
    if not rows:
        issues.append({"row": 0, "field": "dataset", "severity": "error", "issue": "empty dataset"})
    return ValidationReport(records, issues, len(rows))


def summarize_states(report: ValidationReport, include_text=False) -> pd.DataFrame:
    if include_text:
        return pd.DataFrame([state.model_dump() for state in report.records])
    return pd.DataFrame(
        [
            {
                "records": report.total,
                "errors": sum(issue["severity"] == "error" for issue in report.issues),
                "warnings": sum(issue["severity"] == "warning" for issue in report.issues),
                "missing_abstracts": sum(not state.abstract.strip() for state in report.records),
                "mean_title_characters": sum(len(s.title) for s in report.records)
                / max(1, len(report.records)),
                "mean_abstract_characters": sum(len(s.abstract) for s in report.records)
                / max(1, len(report.records)),
            }
        ]
    )


def select_sample(states: list[CitationState], limit: int | None, seed: int) -> list[CitationState]:
    if limit is None or limit >= len(states):
        return list(states)
    return random.Random(seed).sample(states, limit)


def fingerprint(value) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def dataset_fingerprint(states: list[CitationState]) -> str:
    return fingerprint([state.model_dump() for state in states])


@dataclass
class PilotSample:
    states: list[CitationState]
    manifest: dict
    path: Path

    def summary(self):
        return pd.DataFrame(
            [
                {
                    **self.manifest["counts"],
                    "seed": self.manifest["seed"],
                    "sample_sha256": self.manifest["sample_sha256"],
                    "task_sha256": self.manifest["task_sha256"],
                    "manifest": str(self.path),
                }
            ]
        )


def prepare_pilot_sample(project, *, allow_writes=False) -> PilotSample:
    """Freeze ten input-only states for both models; never contact a provider.

    Only absence of all citation text is deferred from inference. Other errors
    still fail validation. The complete source and any conversion manifest are
    fingerprinted; neither screening decisions nor ITEM_ID are loaded as inputs.
    Existing manifests must match exactly and are never overwritten.
    """
    # Local imports avoid the tasks -> data and runner -> data import cycles.
    from jev_eval.runner import require_writes, safe_output_path
    from jev_eval.tasks import load_task

    settings = project.workflow.inference_sh_pilot
    if settings is None or not settings.enabled:
        raise PrerequisiteError("Select an explicitly enabled Inference.sh pilot workflow.")
    dataset = project.evaluation.dataset
    if dataset.format != "jsonl":
        raise PrerequisiteError("The pilot requires a prepared states.jsonl file.")
    override = os.getenv(dataset.root_env, "").strip() if dataset.root_env else ""
    source = (project.root / (override or dataset.path)).resolve()
    source_bytes = source.read_bytes() if source.is_file() else None
    rows = load_states(dataset, project.root)
    if source_bytes is None or source.read_bytes() != source_bytes:
        raise PrerequisiteError("Prepared states changed while loading; review before sampling.")
    report = validate_states(rows)
    blocking = [
        issue
        for issue in report.issues
        if issue["severity"] == "error"
        and not (issue["field"] == "title/abstract" and issue["issue"] == "no citation text")
    ]
    if blocking:
        raise PrerequisiteError(
            "Pilot input validation failed. Fix duplicate IDs, fields or other errors first."
        )
    available = [state for state in report.records if state.title.strip() or state.abstract.strip()]
    unavailable = [
        state.study_id
        for state in report.records
        if not state.title.strip() and not state.abstract.strip()
    ]
    if len(available) < 10:
        raise PrerequisiteError("The pilot requires at least ten records with citation text.")
    sample = select_sample(available, 10, project.evaluation.seed)
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    conversion_path = source.parent / "manifest.json"
    conversion = load_json(conversion_path) if conversion_path.is_file() else None
    if conversion is not None:
        if not isinstance(conversion, dict):
            raise PrerequisiteError("Invalid conversion manifest beside the prepared states.")
        expected = conversion.get("output_sha256", {}).get(source.name)
        if expected is not None and expected != source_hash:
            raise PrerequisiteError("Prepared states do not match their conversion checksum.")
    manifest = {
        "schema_version": 1,
        "policy": "seeded_sample_from_records_with_citation_text; no eligibility decisions changed",
        "source_file": str(source.relative_to(project.root))
        if source.is_relative_to(project.root)
        else str(source),
        "source_file_sha256": source_hash,
        "conversion_manifest_sha256": hashlib.sha256(conversion_path.read_bytes()).hexdigest()
        if conversion is not None
        else None,
        "source_workbook_sha256": conversion.get("source", {}).get("sha256")
        if conversion is not None
        else None,
        "dataset_sha256": dataset_fingerprint(report.records),
        "task_sha256": load_task(project).fingerprint,
        "seed": project.evaluation.seed,
        "ordered_study_ids": [state.study_id for state in sample],
        "sample_sha256": dataset_fingerprint(sample),
        "unavailable_for_inference_ids": unavailable,
        "counts": {
            "source_records": report.total,
            "available_for_inference": len(available),
            "unavailable_for_inference": len(unavailable),
            "selected_records": len(sample),
            "missing_abstract_warnings": sum(
                issue["severity"] == "warning" for issue in report.issues
            ),
        },
        "validation_issues": report.issues,
    }
    path = safe_output_path(project, settings.sample_manifest)
    if not path.exists():
        if not allow_writes:
            raise PrerequisiteError(
                "No frozen pilot sample exists. Enable ALLOW_SAMPLE_WRITES to create it; "
                "this writes a private manifest without making API calls."
            )
        require_writes(project, allow_writes)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive creation prevents another notebook from replacing a sample.
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            pass
    if load_json(path) != manifest:
        raise PrerequisiteError(
            "Frozen pilot sample differs from the source, seed or task. "
            "Use the original inputs; do not overwrite the sample manifest."
        )
    return PilotSample(sample, manifest, path)


@dataclass
class FullDataset:
    states: list[CitationState]
    manifest: dict

    def summary(self):
        return pd.DataFrame(
            [
                {
                    **self.manifest["counts"],
                    "source_sha256": self.manifest["source_file_sha256"],
                    "states_sha256": self.manifest["states_sha256"],
                }
            ]
        )


def load_full_dataset(project) -> FullDataset:
    """Validate the complete cohort without reading labels or excluding studies."""
    from jev_eval.runner import safe_output_path
    from jev_eval.tasks import load_task

    settings = project.workflow.inference_sh_full
    if settings is None or not settings.enabled:
        raise PrerequisiteError("Select an explicitly enabled full-evaluation workflow.")
    dataset = project.evaluation.dataset
    if dataset.format != "jsonl" or dataset.expected_count is not None:
        raise PrerequisiteError("Full evaluation requires JSONL with expected_count: null.")
    override = os.getenv(dataset.root_env, "").strip() if dataset.root_env else ""
    source = (project.root / (override or dataset.path)).resolve()
    try:
        before = source.read_bytes()
        rows = load_states(dataset, project.root)
        if source.read_bytes() != before:
            raise PrerequisiteError("Dataset changed during loading; review it before proceeding.")
    except OSError:
        raise PrerequisiteError("Cannot read the prepared full dataset.") from None
    required = {"study_id", "title", "abstract"}
    if any(set(row) != required for row in rows):
        raise PrerequisiteError(
            "Every full-run state must contain exactly study_id, title and abstract. "
            "Human labels, ITEM_ID, missing fields and other fields are not allowed."
        )
    report = validate_states(rows)
    if any(
        issue["severity"] == "error" and issue["issue"] != "no citation text"
        for issue in report.issues
    ):
        raise PrerequisiteError(
            "Full dataset validation failed: review IDs and citation field types."
        )
    states = report.records
    empty = [
        state.study_id for state in states if not state.title.strip() and not state.abstract.strip()
    ]
    task = load_task(project)
    source_hash = hashlib.sha256(before).hexdigest()
    conversion_path = source.parent / "manifest.json"
    conversion = load_json(conversion_path) if conversion_path.is_file() else None
    if conversion is not None:
        if not isinstance(conversion, dict):
            raise PrerequisiteError("Invalid conversion manifest.")
        if conversion.get("output_sha256", {}).get(source.name) not in (None, source_hash):
            raise PrerequisiteError("Prepared JSONL checksum differs from its conversion manifest.")
    manifest = {
        "schema_version": 1,
        "kind": "full_cohort",
        "experiment_id": settings.experiment_id,
        "source_file": str(source.relative_to(project.root))
        if source.is_relative_to(project.root)
        else str(source),
        "source_file_sha256": source_hash,
        "conversion_manifest_sha256": hashlib.sha256(conversion_path.read_bytes()).hexdigest()
        if conversion is not None
        else None,
        "source_workbook_sha256": conversion.get("source", {}).get("sha256")
        if conversion is not None
        else None,
        "states_sha256": dataset_fingerprint(states),
        "ordered_study_ids": [state.study_id for state in states],
        "task_sha256": task.fingerprint,
        "task_file": project.evaluation.task.prompt_file,
        "task_file_sha256": hashlib.sha256(
            (project.root / project.evaluation.task.prompt_file).read_bytes()
        ).hexdigest(),
        "missing_text_policy": settings.missing_text_policy,
        "not_inferred_no_text_ids": empty,
        "counts": {
            "records": len(states),
            "api_eligible": len(states) - len(empty),
            "not_inferred_no_text": len(empty),
            "duplicate_ids": 0,
            "missing_ids": 0,
            "missing_titles": sum(not state.title.strip() for state in states),
            "missing_abstracts": sum(not state.abstract.strip() for state in states),
        },
        "validation_issues": [
            {**issue, "severity": "warning"} if issue["issue"] == "no citation text" else issue
            for issue in report.issues
        ],
    }
    path = safe_output_path(project, settings.cohort_manifest)
    if path.exists() and load_json(path) != manifest:
        raise PrerequisiteError(
            "Frozen full cohort differs from the source or eligibility criteria."
        )
    return FullDataset(states, manifest)


def freeze_full_dataset(project, cohort, *, allow_writes=False):
    """Create a shared private cohort manifest only with explicit write permission."""
    from jev_eval.runner import require_writes, safe_output_path

    require_writes(project, allow_writes)
    current = load_full_dataset(project)
    if current.manifest != cohort.manifest:
        raise PrerequisiteError("Full dataset changed after planning.")
    path = safe_output_path(project, project.workflow.inference_sh_full.cohort_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(cohort.manifest, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    if load_json(path) != cohort.manifest:
        raise PrerequisiteError("Existing full cohort cannot be overwritten.")
    return path


# Excel preparation is deliberately separate from inference readiness validation.
# It has no inference imports and preserves even studies with no available text.
AUDIT_COLUMNS = (
    "ID",
    "Year",
    "Title",
    "Inclusion Status",
    "Reference LABEL",
    "Reference ITEM_ID",
    "Issue",
)
REFERENCE_COLUMNS = ("study_id", "Inclusion Status", "ITEM_ID")
ISSUE_COLUMNS = ("worksheet", "source_excel_row", "study_id", "field", "severity", "code")


class ExcelSourceSettings(Settings):
    path: str
    worksheet: str
    audit_worksheet: str
    expected_records: int | None = Field(default=None, ge=1)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ExcelColumnSettings(Settings):
    study_id: str
    title: str
    abstract: str
    inclusion_status: str
    item_id: str

    @model_validator(mode="after")
    def unique_mapping(self):
        values = list(self.model_dump().values())
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("Source column names must be non-empty and distinct.")
        return self


class ExcelValidationSettings(Settings):
    missing_abstract: Literal["warn", "error"] = "warn"
    missing_all_text: Literal["warn", "error"] = "warn"
    missing_item_id: Literal["warn", "error"] = "warn"
    shared_item_id: Literal["warn", "error"] = "warn"
    invalid_label: Literal["error"] = "error"
    duplicate_study_id: Literal["error"] = "error"


class ExcelAuditSettings(Settings):
    policy: Literal["report_only"] = "report_only"


class ExcelOutputSettings(Settings):
    root: str = "data/private/prepared/depression"
    allow_writes: bool = False


class ExcelPreparationSettings(Settings):
    schema_version: Literal[1]
    source: ExcelSourceSettings
    columns: ExcelColumnSettings
    validation: ExcelValidationSettings
    audit: ExcelAuditSettings
    output: ExcelOutputSettings


def load_excel_preparation_config(path: Path) -> ExcelPreparationSettings:
    path = Path(path)
    return parse_settings(ExcelPreparationSettings, read_yaml(path), path.name)


def _blank(value):
    return value is None or (isinstance(value, str) and not value.strip())


def _label(value):
    # Excel's numeric 0/1 and textual "0"/"1" are explicit encodings; bool is not.
    if type(value) in (int, float) and value in (0, 1):
        return int(value)
    if isinstance(value, str) and value in ("0", "1"):
        return int(value)
    return None


def _audit_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


@dataclass(repr=False)
class WorkbookInspection:
    settings: ExcelPreparationSettings
    source_path: Path | None
    source_name: str
    source_sha256: str
    source_bytes: int
    worksheets: dict
    main_rows: list
    audit_rows: list
    issues: list


def _source_unchanged(inspection):
    if inspection.source_path is None:
        return
    try:
        digest = hashlib.sha256(inspection.source_path.read_bytes()).hexdigest()
    except OSError:
        raise PrerequisiteError(
            "Source workbook is no longer readable; conversion was not published."
        ) from None
    if digest != inspection.source_sha256:
        raise PrerequisiteError(
            "Source workbook changed during preparation; rerun from the unchanged input."
        )


def inspect_screening_workbook(source, settings, *, root: Path) -> WorkbookInspection:
    """Read a path or synthetic bytes in read-only mode; never save the workbook."""
    try:
        from openpyxl import load_workbook
    except ImportError:
        raise PrerequisiteError(
            "Install notebook dependencies: uv sync --locked --group notebooks"
        ) from None
    source_path = None if isinstance(source, bytes) else (Path(root) / source).resolve()
    try:
        content = source if source_path is None else source_path.read_bytes()
    except OSError:
        raise PrerequisiteError(
            "Source workbook is missing or unreadable; check source.path."
        ) from None
    digest = hashlib.sha256(content).hexdigest()
    if settings.source.expected_sha256 and digest != settings.source.expected_sha256:
        raise PrerequisiteError(
            "Source workbook checksum differs from the configured input; review before converting."
        )
    inspection = WorkbookInspection(
        settings.model_copy(deep=True),
        source_path,
        source_path.name if source_path else "synthetic-example.xlsx",
        digest,
        len(content),
        {},
        [],
        [],
        [],
    )
    try:
        workbook = load_workbook(
            io.BytesIO(content), read_only=True, data_only=False, keep_links=False
        )
    except Exception:
        raise PrerequisiteError(
            "Cannot read the XLSX workbook; verify that it is a valid Excel file."
        ) from None

    def read_sheet(name, required):
        if name not in workbook.sheetnames:
            raise PrerequisiteError(f"Required worksheet is missing: {name}")
        sheet = workbook[name]
        sheet.reset_dimensions()  # The supplied workbook omits worksheet dimension metadata.
        iterator = iter(sheet.iter_rows())
        header = next(
            (row for row in iterator if any(not _blank(cell.value) for cell in row)), None
        )
        if header is None:
            raise PrerequisiteError(f"Worksheet is empty: {name}")
        names = [cell.value for cell in header]
        if any(not isinstance(value, str) or not value.strip() for value in names) or len(
            set(names)
        ) != len(names):
            raise PrerequisiteError(f"Worksheet has empty or duplicate headers: {name}")
        missing = set(required) - set(names)
        if missing:
            raise PrerequisiteError(
                f"Required columns missing in {name}: {', '.join(sorted(missing))}"
            )
        rows, blank_rows = [], 0
        # EmptyCell has no row attribute; enumerate physical rows from the header.
        for row_number, cells in enumerate(iterator, header[0].row + 1):
            if all(_blank(cell.value) for cell in cells):
                blank_rows += 1
                continue
            if any(not _blank(cell.value) for cell in cells[len(names) :]):
                raise PrerequisiteError(
                    f"Worksheet contains populated columns without headers: {name}"
                )
            values = {
                column: cells[index].value if index < len(cells) else None
                for index, column in enumerate(names)
            }
            for index, column in enumerate(names):
                if (
                    column in required
                    and index < len(cells)
                    and cells[index].data_type in ("f", "e")
                ):
                    inspection.issues.append(
                        {
                            "worksheet": name,
                            "source_excel_row": row_number,
                            "study_id": "",
                            "field": column,
                            "severity": "error",
                            "code": "formula_cell"
                            if cells[index].data_type == "f"
                            else "excel_error_cell",
                        }
                    )
            rows.append((row_number, values))
        inspection.worksheets[name] = {
            "headers": names,
            "header_row": header[0].row,
            "records": len(rows),
            "blank_rows_skipped": blank_rows,
        }
        return rows

    try:
        if settings.source.worksheet == settings.source.audit_worksheet:
            raise PrerequisiteError("Study and audit worksheets must be distinct.")
        inspection.main_rows = read_sheet(
            settings.source.worksheet, settings.columns.model_dump().values()
        )
        inspection.audit_rows = read_sheet(settings.source.audit_worksheet, AUDIT_COLUMNS)
    finally:
        workbook.close()
    _source_unchanged(inspection)
    return inspection


@dataclass(repr=False)
class PreparedDataset:
    inspection: WorkbookInspection
    states: list[dict]
    reference_labels: list[dict]
    audit_rows: list[dict]
    validation_issues: list[dict]
    manifest: dict
    payload_sha256: str

    @property
    def settings(self):
        return self.inspection.settings

    @property
    def valid(self):
        return not any(issue["severity"] == "error" for issue in self.validation_issues)

    def require_exportable(self):
        if not self.valid:
            raise PrerequisiteError(
                "Preparation has blocking validation errors; review issues before export."
            )


def _prepared_payload(prepared):
    return {
        "states": prepared.states,
        "reference_labels": prepared.reference_labels,
        "audit": prepared.audit_rows,
        "issues": prepared.validation_issues,
        "manifest": prepared.manifest,
        "scientific_settings": {
            key: value for key, value in prepared.settings.model_dump().items() if key != "output"
        },
    }


def prepare_screening_workbook(source, settings, *, root: Path) -> PreparedDataset:
    inspection = inspect_screening_workbook(source, settings, root=root)
    settings = inspection.settings
    columns = settings.columns
    issues = list(inspection.issues)
    states, references, audit = [], [], []
    counts = Counter()
    id_counts = Counter()
    item_counts = Counter()

    def issue(row, identifier, field, code, severity="error", worksheet=None):
        issues.append(
            {
                "worksheet": worksheet or settings.source.worksheet,
                "source_excel_row": row,
                "study_id": identifier,
                "field": field,
                "severity": "warning" if severity == "warn" else severity,
                "code": code,
            }
        )

    for row_number, values in inspection.main_rows:
        identifier = values[columns.study_id]
        if _blank(identifier):
            counts["missing_ids"] += 1
            issue(row_number, "", columns.study_id, "missing_id")
        elif not isinstance(identifier, str):
            counts["invalid_id_types"] += 1
            issue(row_number, "", columns.study_id, "non_text_id_requires_review")
        else:
            id_counts[identifier] += 1
        identifier = identifier if isinstance(identifier, str) else ""
        text = {}
        for field, column in (("title", columns.title), ("abstract", columns.abstract)):
            value = values[column]
            if _blank(value):
                counts[f"missing_{field}s"] += 1
                severity = settings.validation.missing_abstract if field == "abstract" else "warn"
                issue(row_number, identifier, column, f"missing_{field}", severity)
                text[field] = ""
            elif not isinstance(value, str):
                counts["invalid_text_types"] += 1
                issue(row_number, identifier, column, "non_text_citation_field")
                text[field] = ""
            else:
                text[field] = value
        if not text["title"] and not text["abstract"]:
            counts["missing_all_text"] += 1
            issue(
                row_number,
                identifier,
                "title/abstract",
                "no_text_for_inference",
                settings.validation.missing_all_text,
            )
        label = _label(values[columns.inclusion_status])
        if label is None:
            counts["invalid_labels"] += 1
            issue(row_number, identifier, columns.inclusion_status, "invalid_label")
        item = values[columns.item_id]
        if _blank(item):
            counts["missing_item_ids"] += 1
            issue(
                row_number,
                identifier,
                columns.item_id,
                "missing_item_id",
                settings.validation.missing_item_id,
            )
            item = ""
        elif not isinstance(item, str):
            counts["invalid_item_id_types"] += 1
            issue(row_number, identifier, columns.item_id, "non_text_item_id_requires_review")
            item = ""
        else:
            item_counts[item] += 1
        # Explicit allowlist: never pass screening decisions or linkage IDs to a model.
        states.append(
            {"study_id": identifier, "title": text["title"], "abstract": text["abstract"]}
        )
        references.append({"study_id": identifier, "Inclusion Status": label, "ITEM_ID": item})

    for (row_number, _), reference in zip(inspection.main_rows, references, strict=True):
        identifier, item = reference["study_id"], reference["ITEM_ID"]
        if id_counts[identifier] > 1:
            issue(row_number, identifier, columns.study_id, "duplicate_id")
        if item and item_counts[item] > 1:
            issue(
                row_number,
                identifier,
                columns.item_id,
                "shared_item_id",
                settings.validation.shared_item_id,
            )
    count = len(states)
    if count == 0:
        issue(0, "", "dataset", "empty_dataset")
    if settings.source.expected_records is not None and count != settings.source.expected_records:
        issue(0, "", "dataset", "unexpected_record_count")
    references_by_id = {reference["study_id"]: reference for reference in references}
    audit_disagreements, audit_main_mismatches, audit_unknown_ids = 0, 0, 0
    for row_number, values in inspection.audit_rows:
        audit.append(
            {
                "source_excel_row": row_number,
                **{key: _audit_scalar(value) for key, value in values.items()},
            }
        )
        identifier = values.get("ID")
        identifier = identifier if isinstance(identifier, str) else ""
        issue(
            row_number,
            identifier,
            "Issue",
            "audit_entry_reported",
            "warn",
            settings.source.audit_worksheet,
        )
        main = _label(values.get("Inclusion Status"))
        other = _label(values.get("Reference LABEL"))
        if main is None or other is None:
            issue(
                row_number,
                identifier,
                "Inclusion Status/Reference LABEL",
                "invalid_audit_label",
                "warn",
                settings.source.audit_worksheet,
            )
        elif main != other:
            audit_disagreements += 1
        if identifier not in references_by_id:
            audit_unknown_ids += 1
            issue(
                row_number,
                identifier,
                "ID",
                "audit_unknown_study_id",
                "warn",
                settings.source.audit_worksheet,
            )
        elif main is not None and main != references_by_id[identifier]["Inclusion Status"]:
            audit_main_mismatches += 1
            issue(
                row_number,
                identifier,
                "Inclusion Status",
                "audit_main_label_mismatch",
                "warn",
                settings.source.audit_worksheet,
            )

    validation = {
        name: counts[name]
        for name in (
            "missing_ids",
            "invalid_id_types",
            "missing_titles",
            "missing_abstracts",
            "missing_all_text",
            "missing_item_ids",
            "invalid_item_id_types",
            "invalid_text_types",
            "invalid_labels",
        )
    }
    validation.update(
        duplicate_id_groups=sum(value > 1 for value in id_counts.values()),
        shared_item_id_groups=sum(value > 1 for value in item_counts.values()),
        rows_with_shared_item_ids=sum(value for value in item_counts.values() if value > 1),
        errors=sum(item["severity"] == "error" for item in issues),
        warnings=sum(item["severity"] == "warning" for item in issues),
    )
    issue_counts = Counter(str(row.get("Issue") or "(missing issue)") for row in audit)
    manifest = {
        "schema_version": 1,
        "source": {
            "filename": inspection.source_name,
            "sha256": inspection.source_sha256,
            "bytes": inspection.source_bytes,
            "worksheets": inspection.worksheets,
        },
        "records": {
            "source": count,
            "states": len(states),
            "reference_labels": len(references),
            "audit_entries": len(audit),
            "records_with_title_or_abstract": count - counts["missing_all_text"],
        },
        "label_counts": {
            str(label): sum(row["Inclusion Status"] == label for row in references)
            for label in (0, 1)
        },
        "validation": {
            **validation,
            "exportable": validation["errors"] == 0,
            "all_records_have_inference_text": counts["missing_all_text"] == 0,
        },
        "audit": {
            "policy": "report_only",
            "adjudication_applied": False,
            "reference_label_source": f"{settings.source.worksheet} / {columns.inclusion_status}",
            "unique_study_ids": len({row.get("ID") for row in audit}),
            "issue_counts": dict(issue_counts),
            "label_disagreements": audit_disagreements,
            "main_label_mismatches": audit_main_mismatches,
            "unknown_study_ids": audit_unknown_ids,
        },
        "transformations": {
            "ids": "text preserved exactly; non-text IDs block export",
            "text": "null/whitespace-only cells become empty strings; other text unchanged",
            "labels": "numeric or textual 0/1 canonicalized to integer 0/1; never adjudicated",
            "order": "original worksheet row order",
            "audit_dates": "ISO 8601 text",
        },
        "source_unchanged": True,
    }
    prepared = PreparedDataset(inspection, states, references, audit, issues, manifest, "")
    prepared.payload_sha256 = fingerprint(_prepared_payload(prepared))
    return prepared


def summarize_preparation(prepared) -> pd.DataFrame:
    values = {**prepared.manifest["records"], **prepared.manifest["validation"]}
    return pd.DataFrame([{"check": key, "value": value} for key, value in values.items()])


def summarize_match_audit(prepared) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"issue": key, "entries": count}
            for key, count in prepared.manifest["audit"]["issue_counts"].items()
        ],
        columns=["issue", "entries"],
    )


def synthetic_screening_workbook() -> bytes:
    """Construct an invented XLSX entirely in memory for the shared notebook."""
    from openpyxl import Workbook

    workbook = Workbook()
    main = workbook.active
    main.title = "Human screening decisions"
    main.append(["ID", "Title", "Abstract", "Inclusion Status", "ITEM_ID"])
    main.append(
        ["0001", "Invented title, with comma", "Invented abstract\nSecond line", 1, "000101"]
    )
    main.append(["0002", "Invented title B", None, 0, None])
    main.append(["0003", None, None, 0, "000103"])
    audit = workbook.create_sheet("Match Audit")
    audit.append(list(AUDIT_COLUMNS))
    audit.append(
        [
            "0001",
            "2020",
            "Invented title, with comma",
            "1",
            "0",
            "000101",
            "Title matched; label disagreed",
        ]
    )
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def _preparation_output(root, output_root):
    root = Path(root).resolve()
    allowed = root / "data/private/prepared"
    if allowed.resolve() != allowed:
        raise PrerequisiteError("Preparation output must not use a symlinked private directory.")
    destination = (root / output_root).resolve()
    if not destination.is_relative_to(allowed):
        raise PrerequisiteError("Preparation exports must stay under data/private/prepared/.")
    return destination


def export_prepared_dataset(prepared, output_root=None, *, root: Path, allow_writes=False) -> Path:
    """Publish a complete private bundle; the source workbook is never opened for writing."""
    if not (allow_writes and prepared.settings.output.allow_writes):
        raise PrerequisiteError(
            "Export requires output.allow_writes in preparation YAML AND allow_writes=True."
        )
    prepared.require_exportable()
    if fingerprint(_prepared_payload(prepared)) != prepared.payload_sha256:
        raise PrerequisiteError(
            "Prepared data changed after validation; rerun conversion before exporting."
        )
    output = _preparation_output(root, output_root or prepared.settings.output.root)
    _source_unchanged(prepared.inspection)
    settings = prepared.settings.model_dump()
    scientific_settings = {key: value for key, value in settings.items() if key != "output"}
    identifier = (
        prepared.inspection.source_sha256[:12] + "-" + fingerprint(scientific_settings)[:12]
    )
    destination = output / identifier
    if destination.exists():
        raise FileExistsError(
            "Conversion already exists; no files overwritten. Select another private output directory."
        )

    # Collect environment information before creating any output files.
    from types import SimpleNamespace

    from jev_eval.environment import reproducibility_summary

    manifest = json.loads(json.dumps(prepared.manifest))
    manifest.update(
        conversion_id=identifier,
        output_directory=str(destination.relative_to(Path(root).resolve())),
        converted_at=datetime.now(timezone.utc).isoformat(),
        configuration=settings,
        configuration_sha256=fingerprint(settings),
        converter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=reproducibility_summary(SimpleNamespace(root=Path(root).resolve())),
        reader={
            "name": "openpyxl",
            "version": importlib.metadata.version("openpyxl"),
            "read_only": True,
            "data_only": False,
            "keep_links": False,
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".conversion-", dir=output))

    def write_csv(name, rows, fieldnames):
        with (staging / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    try:
        with (staging / "states.jsonl").open("w", encoding="utf-8", newline="") as stream:
            for state in prepared.states:
                stream.write(
                    json.dumps(state, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
        write_csv("reference_labels.csv", prepared.reference_labels, REFERENCE_COLUMNS)
        audit_headers = [
            "source_excel_row",
            *prepared.inspection.worksheets[prepared.settings.source.audit_worksheet]["headers"],
        ]
        write_csv("match_audit.csv", prepared.audit_rows, audit_headers)
        write_csv("validation_issues.csv", prepared.validation_issues, ISSUE_COLUMNS)
        manifest["output_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staging.iterdir())
        }
        _source_unchanged(prepared.inspection)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
