import json

import pytest

from jev_eval.config import PrerequisiteError, load_project_config
from jev_eval.data import load_states, select_sample, validate_states
from jev_eval.tasks import validate_task_template


def test_duplicate_and_missing_abstract_behaviour():
    report = validate_states(
        [
            {"study_id": "one", "title": "Invented title"},
            {"study_id": "one", "title": "Invented title", "abstract": ""},
        ]
    )
    assert not report.valid
    assert any(issue["severity"] == "warning" for issue in report.issues)
    with pytest.raises(PrerequisiteError, match="validation failed"):
        report.require_valid()
    valid = validate_states([{"study_id": "one", "title": "Invented title"}])
    assert valid.valid
    assert len(valid.require_valid()) == 1


def test_type_errors_do_not_echo_citation_text():
    text = "PRIVATE CITATION TEXT"
    report = validate_states([{"study_id": "one", "title": {"text": text}}])
    assert not report.valid
    assert text not in json.dumps(report.issues)


def test_dataset_count_environment_override_and_sample(project, monkeypatch, tmp_path):
    dataset = project.evaluation.dataset
    assert len(load_states(dataset, project.root)) == 2
    custom = tmp_path / "override.jsonl"
    custom.write_text('{"study_id":"a","title":"Invented"}\n')
    dataset.root_env = "JEV_TEST_DATA"
    monkeypatch.setenv("JEV_TEST_DATA", str(custom))
    with pytest.raises(PrerequisiteError, match="Expected 2 states but loaded 1"):
        load_states(dataset, project.root)
    dataset.expected_count = 1
    assert load_states(dataset, project.root)[0]["study_id"] == "a"
    sample = select_sample(list(range(100)), 5, 42)
    assert sample == select_sample(list(range(100)), 5, 42)


def test_malformed_hybrid_is_preserved_and_rejected(project):
    with pytest.raises(PrerequisiteError, match="Invalid JSON"):
        validate_task_template(project.root / "prompts/drafts/hybrid_noul.json")
    binary = validate_task_template(project.root / "prompts/drafts/binary_noul.json")
    assert list(binary.questions) == ["retain_for_fulltext"]


def test_reviewed_templates_have_identical_overall_questions_and_seven_nouls(project):
    folder = project.root / "prompts/reviewed/v1"
    binary = validate_task_template(folder / "binary_noul.json")
    hybrid = validate_task_template(folder / "hybrid_noul.json")
    assert list(binary.questions) == ["retain_for_fulltext"]
    assert set(hybrid.questions) == {
        "primary_research",
        "animal_study",
        "in_vivo_experiment",
        "depression_model",
        "control_group",
        "eligible_outcomes",
        "overall_eligibility",
    }
    assert all(question.type == "noul" for question in hybrid.questions.values())
    assert (
        binary.questions["retain_for_fulltext"].model_dump()
        == hybrid.questions["overall_eligibility"].model_dump()
    )


def test_invalid_configuration_is_redacted(project):
    path = project.root / "configs/notebooks.yaml"
    path.write_text(path.read_text() + '\nunexpected: "PRIVATE VALUE"\n')
    with pytest.raises(PrerequisiteError, match="unexpected") as caught:
        load_project_config(project.root)
    assert "PRIVATE VALUE" not in str(caught.value)
