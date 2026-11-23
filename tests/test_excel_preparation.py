import csv
import hashlib
import io
import json

import httpx
import pytest
from openpyxl import load_workbook

from jev_eval.config import PrerequisiteError
from jev_eval.data import (
    export_prepared_dataset,
    inspect_screening_workbook,
    load_excel_preparation_config,
    prepare_screening_workbook,
    synthetic_screening_workbook,
    validate_states,
)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Data preparation must never make HTTP calls")

    monkeypatch.setattr(httpx.Client, "send", forbidden)


@pytest.fixture
def settings(project):
    value = load_excel_preparation_config(
        project.root / "configs/data_preparation/depression_excel.yaml"
    )
    value.source.expected_records = 3
    value.source.expected_sha256 = None
    return value


def edit_workbook(content, edit):
    workbook = load_workbook(io.BytesIO(content))
    edit(workbook)
    stream = io.BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def prepare(project, settings, content=None):
    return prepare_screening_workbook(
        content if content is not None else synthetic_screening_workbook(),
        settings,
        root=project.root,
    )


def test_original_ids_no_label_leakage_and_one_row_per_study(project, settings):
    def add_private_notes(book):
        sheet = book[settings.source.worksheet]
        sheet["F1"] = "Human Notes"
        sheet["F2"] = "PRIVATE SCREENING ANNOTATION"

    data = prepare(
        project, settings, edit_workbook(synthetic_screening_workbook(), add_private_notes)
    )
    assert data.valid
    assert [state["study_id"] for state in data.states] == ["0001", "0002", "0003"]
    assert len(data.states) == len(data.reference_labels) == 3
    assert all(set(state) == {"study_id", "title", "abstract"} for state in data.states)
    assert "PRIVATE SCREENING ANNOTATION" not in json.dumps(data.states)
    assert data.reference_labels[0] == {
        "study_id": "0001",
        "Inclusion Status": 1,
        "ITEM_ID": "000101",
    }
    assert data.states[0]["abstract"] == "Invented abstract\nSecond line"


def test_missing_text_preserved_but_inference_remains_strict(project, settings):
    data = prepare(project, settings)
    assert data.valid
    assert data.states[1]["abstract"] == ""
    assert data.states[2] == {"study_id": "0003", "title": "", "abstract": ""}
    assert data.manifest["validation"]["missing_abstracts"] == 2
    assert data.manifest["validation"]["missing_all_text"] == 1
    assert data.manifest["validation"]["missing_item_ids"] == 1
    assert data.manifest["records"]["records_with_title_or_abstract"] == 2
    assert not validate_states(data.states).valid
    assert data.reference_labels[1]["ITEM_ID"] == ""


def test_literal_missing_markers_are_not_automatically_converted(project, settings):
    def literal_text(book):
        sheet = book[settings.source.worksheet]
        sheet["A2"] = "NA"
        sheet["B2"] = "NaN"
        sheet["C2"] = "N/A"
        sheet["E2"] = "NA"

    data = prepare(project, settings, edit_workbook(synthetic_screening_workbook(), literal_text))
    assert data.valid
    assert data.states[0] == {"study_id": "NA", "title": "NaN", "abstract": "N/A"}
    assert data.reference_labels[0]["ITEM_ID"] == "NA"


@pytest.mark.parametrize("edit", ["policy", "manifest"])
def test_changed_validation_provenance_cannot_be_exported(project, settings, edit):
    data = prepare(project, settings)
    data.settings.output.allow_writes = True
    if edit == "policy":
        data.settings.validation.missing_all_text = "error"
    else:
        data.manifest["records"]["source"] = 99
    with pytest.raises(PrerequisiteError, match="changed after validation"):
        export_prepared_dataset(data, root=project.root, allow_writes=True)


@pytest.mark.parametrize("label", [None, 2, True, "yes", " 1 "])
def test_invalid_labels_block_export(project, settings, label):
    content = edit_workbook(
        synthetic_screening_workbook(),
        lambda book: setattr(book[settings.source.worksheet]["D2"], "value", label),
    )
    data = prepare(project, settings, content)
    assert data.manifest["validation"]["invalid_labels"] == 1
    settings.output.allow_writes = True
    data.settings.output.allow_writes = True
    with pytest.raises(PrerequisiteError, match="blocking validation"):
        export_prepared_dataset(data, root=project.root, allow_writes=True)


@pytest.mark.parametrize("label", [0, 1, 0.0, 1.0, "0", "1"])
def test_only_explicit_binary_encodings_are_accepted(project, settings, label):
    content = edit_workbook(
        synthetic_screening_workbook(),
        lambda book: setattr(book[settings.source.worksheet]["D2"], "value", label),
    )
    data = prepare(project, settings, content)
    assert data.valid
    assert type(data.reference_labels[0]["Inclusion Status"]) is int


def test_duplicate_missing_and_numeric_ids_are_not_repaired(project, settings):
    def duplicate(book):
        book[settings.source.worksheet]["A3"] = "0001"

    data = prepare(project, settings, edit_workbook(synthetic_screening_workbook(), duplicate))
    assert not data.valid
    assert len(data.states) == 3
    assert data.manifest["validation"]["duplicate_id_groups"] == 1
    assert sum(row["code"] == "duplicate_id" for row in data.validation_issues) == 2

    def invalid_ids(book):
        book[settings.source.worksheet]["A2"] = None
        book[settings.source.worksheet]["A3"] = 7

    data = prepare(project, settings, edit_workbook(synthetic_screening_workbook(), invalid_ids))
    assert not data.valid
    assert data.manifest["validation"]["missing_ids"] == 1
    assert data.manifest["validation"]["invalid_id_types"] == 1


def test_shared_item_ids_and_audit_disagreements_never_change_labels(project, settings):
    def extra_issues(book):
        book[settings.source.worksheet]["E3"] = "000101"
        book[settings.source.audit_worksheet].append(
            ["0001", "2020", "Invented", "1", "0", "000101", "Year discrepancy"]
        )

    data = prepare(project, settings, edit_workbook(synthetic_screening_workbook(), extra_issues))
    assert data.valid
    assert data.manifest["validation"]["shared_item_id_groups"] == 1
    assert data.manifest["validation"]["rows_with_shared_item_ids"] == 2
    assert [row["Inclusion Status"] for row in data.reference_labels] == [1, 0, 0]
    assert len(data.audit_rows) == 2
    assert data.manifest["audit"]["unique_study_ids"] == 1
    assert data.manifest["audit"]["label_disagreements"] == 2
    assert data.manifest["audit"]["adjudication_applied"] is False
    assert [row["source_excel_row"] for row in data.audit_rows] == [2, 3]


@pytest.mark.parametrize("kind", ["missing_sheet", "missing_column", "duplicate_header"])
def test_missing_worksheets_columns_and_duplicate_headers(project, settings, kind):
    def edit(book):
        if kind == "missing_sheet":
            del book[settings.source.audit_worksheet]
        elif kind == "missing_column":
            book[settings.source.worksheet]["A1"] = "Wrong ID column"
        else:
            book[settings.source.worksheet]["A1"] = "Title"

    content = edit_workbook(synthetic_screening_workbook(), edit)
    with pytest.raises(PrerequisiteError, match="worksheet|columns|headers"):
        prepare(project, settings, content)


@pytest.mark.parametrize("value,code", [("=1+1", "formula_cell"), ("#VALUE!", "excel_error_cell")])
def test_formulas_and_excel_errors_are_not_evaluated(project, settings, value, code):
    content = edit_workbook(
        synthetic_screening_workbook(),
        lambda book: setattr(book[settings.source.worksheet]["B2"], "value", value),
    )
    data = prepare(project, settings, content)
    assert not data.valid
    assert any(row["code"] == code for row in data.validation_issues)


def test_physical_rows_and_source_order_with_blank_rows(project, settings):
    def insert_rows(book):
        sheet = book[settings.source.worksheet]
        sheet.insert_rows(1, 2)
        sheet.insert_rows(5)

    data = prepare(project, settings, edit_workbook(synthetic_screening_workbook(), insert_rows))
    assert data.valid
    assert [row["study_id"] for row in data.states] == ["0001", "0002", "0003"]
    assert data.inspection.worksheets[settings.source.worksheet]["header_row"] == 3
    assert data.inspection.worksheets[settings.source.worksheet]["blank_rows_skipped"] == 1
    assert any(
        row["source_excel_row"] == 6 and row["study_id"] == "0002" for row in data.validation_issues
    )


def test_configured_count_checksum_and_warning_policy(project, settings):
    content = synthetic_screening_workbook()
    settings.source.expected_records = 4
    assert not prepare(project, settings, content).valid
    settings.source.expected_records = 3
    settings.source.expected_sha256 = "0" * 64
    with pytest.raises(PrerequisiteError, match="checksum"):
        inspect_screening_workbook(content, settings, root=project.root)
    settings.source.expected_sha256 = None
    settings.validation.missing_abstract = "error"
    assert not prepare(project, settings, content).valid


def test_export_is_deterministic_linked_and_manifest_checksums_match(project, settings):
    source = project.root / "input.xlsx"
    source.write_bytes(synthetic_screening_workbook())
    original = source.read_bytes()
    settings.output.allow_writes = True
    data = prepare_screening_workbook(source, settings, root=project.root)
    one = export_prepared_dataset(
        data, "data/private/prepared/one", root=project.root, allow_writes=True
    )
    two = export_prepared_dataset(
        data, "data/private/prepared/two", root=project.root, allow_writes=True
    )
    assert source.read_bytes() == original
    assert set(path.name for path in one.iterdir()) == {
        "states.jsonl",
        "reference_labels.csv",
        "match_audit.csv",
        "validation_issues.csv",
        "manifest.json",
    }
    manifest = json.loads((one / "manifest.json").read_text())
    assert manifest["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["records"]["states"] == manifest["records"]["reference_labels"] == 3
    assert manifest["reader"]["read_only"] is True
    assert manifest["reader"]["data_only"] is False
    for name, digest in manifest["output_sha256"].items():
        assert (one / name).read_bytes() == (two / name).read_bytes()
        assert hashlib.sha256((one / name).read_bytes()).hexdigest() == digest
    with (one / "reference_labels.csv").open(newline="") as stream:
        references = list(csv.DictReader(stream))
    states = [json.loads(line) for line in (one / "states.jsonl").read_text().splitlines()]
    assert [row["study_id"] for row in states] == [row["study_id"] for row in references]
    assert references[0]["ITEM_ID"] == "000101"
    assert references[1]["ITEM_ID"] == ""
    assert states[0]["abstract"] == "Invented abstract\nSecond line"
    with pytest.raises(FileExistsError, match="no files overwritten"):
        export_prepared_dataset(
            data, "data/private/prepared/one", root=project.root, allow_writes=True
        )


def test_permissions_paths_and_modified_payload_block_export(project, settings):
    data = prepare(project, settings)
    with pytest.raises(PrerequisiteError, match="allow_writes"):
        export_prepared_dataset(data, root=project.root, allow_writes=True)
    data.settings.output.allow_writes = True
    with pytest.raises(PrerequisiteError, match="allow_writes"):
        export_prepared_dataset(data, root=project.root)
    with pytest.raises(PrerequisiteError, match="data/private/prepared"):
        export_prepared_dataset(data, "data/examples", root=project.root, allow_writes=True)
    data.states[0]["human_decision"] = 1
    with pytest.raises(PrerequisiteError, match="changed after validation"):
        export_prepared_dataset(data, root=project.root, allow_writes=True)


def test_source_changes_block_publication_and_failed_exports_leave_no_bundle(
    project, settings, monkeypatch
):
    source = project.root / "input.xlsx"
    original = synthetic_screening_workbook()
    source.write_bytes(original)
    settings.output.allow_writes = True
    data = prepare_screening_workbook(source, settings, root=project.root)
    source.write_bytes(original + b"changed")
    with pytest.raises(PrerequisiteError, match="Source workbook changed"):
        export_prepared_dataset(data, root=project.root, allow_writes=True)
    assert not (project.root / settings.output.root).exists()
    source.write_bytes(original)

    def fail(*args, **kwargs):
        raise OSError("Synthetic export failure")

    monkeypatch.setattr(csv.DictWriter, "writerows", fail)
    with pytest.raises(OSError, match="Synthetic export"):
        export_prepared_dataset(data, root=project.root, allow_writes=True)
    assert not list((project.root / settings.output.root).iterdir())
